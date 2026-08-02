FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# /tmp is writable under OpenShift's default restricted-v2 SCC (arbitrary UID),
# so point the HuggingFace cache there instead of the default home directory.
ENV TRANSFORMERS_CACHE=/tmp/hf_cache
ENV HF_HOME=/tmp/hf_cache
RUN mkdir -p /tmp/hf_cache && chmod -R g+rwX /tmp/hf_cache

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120", "app:app"]
