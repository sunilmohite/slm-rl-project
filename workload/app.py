"""
SLM Inference Server
---------------------
Flask app that serves a small language model and exposes Prometheus metrics.
Both the Kubernetes HPA (via CPU) and the RL agent (via slm_* custom metrics)
consume these metrics to make scaling decisions.
"""
import time
import logging
from flask import Flask, request, jsonify, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("slm-inference")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model loading (lazy, so the pod becomes Ready quickly and downloads once)
# ---------------------------------------------------------------------------
MODEL_NAME = "distilgpt2"  # small, CPU-friendly model - good fit for OpenShift lab
_generator = None


def get_generator():
    global _generator
    if _generator is None:
        logger.info(f"Loading model: {MODEL_NAME}")
        from transformers import pipeline
        _generator = pipeline("text-generation", model=MODEL_NAME)
        logger.info("Model loaded")
    return _generator


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "slm_requests_total", "Total number of inference requests", ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "slm_request_latency_seconds", "Inference request latency in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)
IN_PROGRESS = Gauge(
    "slm_requests_in_progress", "Number of requests currently being processed"
)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/generate", methods=["POST"])
def generate():
    IN_PROGRESS.inc()
    start = time.time()
    try:
        payload = request.get_json(force=True, silent=True) or {}
        prompt = payload.get("prompt", "Hello, how are you")
        max_new_tokens = int(payload.get("max_new_tokens", 30))

        gen = get_generator()
        output = gen(prompt, max_new_tokens=max_new_tokens, num_return_sequences=1)
        text = output[0]["generated_text"]

        REQUEST_COUNT.labels(endpoint="/generate", status="success").inc()
        return jsonify({"prompt": prompt, "generated_text": text}), 200
    except Exception as e:
        logger.exception("Inference failed")
        REQUEST_COUNT.labels(endpoint="/generate", status="error").inc()
        return jsonify({"error": str(e)}), 500
    finally:
        REQUEST_LATENCY.observe(time.time() - start)
        IN_PROGRESS.dec()


@app.route("/metrics", methods=["GET"])
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    get_generator()  # warm up so the first real request isn't abnormally slow
    app.run(host="0.0.0.0", port=8080)
