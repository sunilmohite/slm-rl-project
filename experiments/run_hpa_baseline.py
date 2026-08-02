"""
Baseline data collection while the native Kubernetes/OpenShift HPA
(manifests/hpa.yaml) is managing the SLM inference deployment.

Run this alongside workload/load_generator.py while the HPA is applied.
Produces results/hpa.csv for comparison against the RL agent's results/rl.csv.
"""
import argparse
import csv
import time
import os
import requests
from kubernetes import client, config


def get_replicas(apps_api, namespace, deployment):
    dep = apps_api.read_namespaced_deployment(deployment, namespace)
    return dep.status.replicas or 0


def query_prometheus(prometheus_url, promql):
    try:
        resp = requests.get(f"{prometheus_url}/api/v1/query", params={"query": promql}, timeout=10, verify=False)
        resp.raise_for_status()
        result = resp.json()["data"]["result"]
        return float(result[0]["value"][1]) if result else 0.0
    except Exception as e:
        print(f"Prometheus query failed: {e}")
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prometheus-url", default="http://localhost:9090")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--deployment", default="slm-inference")
    parser.add_argument("--duration", type=int, default=600, help="Total collection time in seconds")
    parser.add_argument("--interval", type=int, default=15, help="Sampling interval in seconds")
    parser.add_argument("--out", default="../results/hpa.csv")
    args = parser.parse_args()

    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()
    apps_api = client.AppsV1Api()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "elapsed_s", "replicas", "cpu", "latency", "request_rate"])
        start = time.time()
        while time.time() - start < args.duration:
            replicas = get_replicas(apps_api, args.namespace, args.deployment)
            cpu = query_prometheus(
                args.prometheus_url,
                f'avg(rate(container_cpu_usage_seconds_total{{namespace="{args.namespace}",pod=~"{args.deployment}.*"}}[1m]))'
            )
            latency = query_prometheus(
                args.prometheus_url,
                f'histogram_quantile(0.95, sum(rate(slm_request_latency_seconds_bucket{{namespace="{args.namespace}"}}[1m])) by (le))'
            )
            req_rate = query_prometheus(
                args.prometheus_url,
                f'sum(rate(slm_requests_total{{namespace="{args.namespace}"}}[1m]))'
            )
            elapsed = round(time.time() - start, 1)
            writer.writerow([time.time(), elapsed, replicas, round(cpu, 4), round(latency, 4), round(req_rate, 4)])
            f.flush()
            print(f"[{elapsed}s] replicas={replicas} cpu={cpu:.2f} latency={latency:.2f}s req_rate={req_rate:.2f}")
            time.sleep(args.interval)

    print(f"Baseline collection complete. Results written to {args.out}")


if __name__ == "__main__":
    main()
