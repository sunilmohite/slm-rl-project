"""
Simple load generator for the SLM inference service.
Sends concurrent POST requests to /generate at a configurable rate and
prints running latency/error stats to the terminal (for your Phase 2
screenshots).
"""
import argparse
import time
import threading
import statistics
import requests

PROMPTS = [
    "Explain Kubernetes autoscaling in one sentence.",
    "What is reinforcement learning?",
    "Describe the benefits of small language models.",
    "Write a short greeting.",
    "Summarize cloud native architecture.",
]

latencies = []
errors = 0
lock = threading.Lock()


def send_request(url, prompt):
    global errors
    start = time.time()
    try:
        r = requests.post(url, json={"prompt": prompt, "max_new_tokens": 20}, timeout=30)
        elapsed = time.time() - start
        with lock:
            if r.status_code == 200:
                latencies.append(elapsed)
            else:
                errors += 1
    except Exception:
        with lock:
            errors += 1


def worker(url, rps, duration):
    end_time = time.time() + duration
    interval = 1.0 / rps if rps > 0 else 1.0
    i = 0
    while time.time() < end_time:
        prompt = PROMPTS[i % len(PROMPTS)]
        t = threading.Thread(target=send_request, args=(url, prompt))
        t.start()
        i += 1
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True,
                         help="Base URL of the inference route, e.g. https://slm-inference-<project>.apps.<cluster-domain>")
    parser.add_argument("--rps", type=float, default=2.0, help="Requests per second, per worker thread")
    parser.add_argument("--duration", type=int, default=300, help="Duration in seconds")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of parallel worker threads issuing traffic")
    args = parser.parse_args()

    endpoint = args.url.rstrip("/") + "/generate"
    print(f"Sending traffic to {endpoint} at ~{args.rps * args.concurrency:.1f} req/s total for {args.duration}s")

    threads = []
    for _ in range(args.concurrency):
        t = threading.Thread(target=worker, args=(endpoint, args.rps, args.duration))
        t.start()
        threads.append(t)

    start = time.time()
    try:
        while time.time() - start < args.duration + 2:
            time.sleep(5)
            with lock:
                n = len(latencies)
                avg = statistics.mean(latencies) if latencies else 0
                p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else avg
            print(f"[{int(time.time()-start)}s] requests={n} errors={errors} avg_latency={avg:.3f}s p95={p95:.3f}s")
    except KeyboardInterrupt:
        pass

    for t in threads:
        t.join(timeout=1)

    print("Load generation complete.")
    print(f"Total requests: {len(latencies)}  Errors: {errors}")
    if latencies:
        print(f"Avg latency: {statistics.mean(latencies):.3f}s  Max: {max(latencies):.3f}s")


if __name__ == "__main__":
    main()
