"""
Custom Gymnasium environment representing the SLM inference scaling problem
against a LIVE OpenShift cluster. Reads metrics from Prometheus and scales
the Deployment via the Kubernetes API.

Observation: [cpu_utilization (0-1), p95_latency_seconds, request_rate (req/s), replica_count]
Action:      0 = scale down by 1, 1 = no action, 2 = scale up by 1
Reward:      balances SLA latency violation against resource (replica) cost

NOTE: each step() call sleeps for `step_interval_seconds` to let the cluster
settle after a scaling action, so live training is slow. Use sim_env.py to
train quickly, then use this env only for evaluate.py against the real
cluster.
"""
import time
import logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger("slm-env")


class SLMScalingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        prometheus_url,
        namespace,
        deployment_name,
        min_replicas=1,
        max_replicas=5,
        latency_sla_seconds=2.0,
        step_interval_seconds=15,
    ):
        super().__init__()
        self.prometheus_url = prometheus_url.rstrip("/")
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.latency_sla = latency_sla_seconds
        self.step_interval = step_interval_seconds

        self.action_space = spaces.Discrete(3)  # 0=down, 1=noop, 2=up
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, min_replicas], dtype=np.float32),
            high=np.array([1.0, 30.0, 100.0, max_replicas], dtype=np.float32),
        )

        self._current_replicas = min_replicas
        self._k8s_apps = self._init_k8s()

    def _init_k8s(self):
        from kubernetes import client, config
        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()
        return client.AppsV1Api()

    # ------------------------------------------------------------------
    # Prometheus queries
    # ------------------------------------------------------------------
    def _query_prometheus(self, promql):
        import requests
        try:
            resp = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": promql},
                timeout=10,
                verify=False,
            )
            resp.raise_for_status()
            result = resp.json()["data"]["result"]
            if not result:
                return 0.0
            return float(result[0]["value"][1])
        except Exception as e:
            logger.warning(f"Prometheus query failed ({promql}): {e}")
            return 0.0

    def _get_metrics(self):
        cpu = self._query_prometheus(
            f'avg(rate(container_cpu_usage_seconds_total{{namespace="{self.namespace}",pod=~"{self.deployment_name}.*"}}[1m]))'
        )
        latency = self._query_prometheus(
            f'histogram_quantile(0.95, sum(rate(slm_request_latency_seconds_bucket{{namespace="{self.namespace}"}}[1m])) by (le))'
        )
        req_rate = self._query_prometheus(
            f'sum(rate(slm_requests_total{{namespace="{self.namespace}"}}[1m]))'
        )
        return cpu, latency, req_rate

    # ------------------------------------------------------------------
    # Kubernetes scaling
    # ------------------------------------------------------------------
    def _get_replicas(self):
        dep = self._k8s_apps.read_namespaced_deployment(self.deployment_name, self.namespace)
        return dep.spec.replicas

    def _scale_to(self, replicas):
        replicas = max(self.min_replicas, min(self.max_replicas, replicas))
        self._current_replicas = replicas
        self._k8s_apps.patch_namespaced_deployment_scale(
            self.deployment_name,
            self.namespace,
            {"spec": {"replicas": replicas}},
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_replicas = self._get_replicas()
        cpu, latency, req_rate = self._get_metrics()
        obs = np.array([cpu, latency, req_rate, self._current_replicas], dtype=np.float32)
        return obs, {}

    def step(self, action):
        replicas = self._get_replicas()
        if action == 0:
            replicas -= 1
        elif action == 2:
            replicas += 1
        self._scale_to(replicas)

        time.sleep(self.step_interval)  # let the cluster settle before reading fresh metrics

        cpu, latency, req_rate = self._get_metrics()
        replicas_now = self._get_replicas()
        obs = np.array([cpu, latency, req_rate, replicas_now], dtype=np.float32)

        reward = self._compute_reward(cpu, latency, replicas_now)
        info = {"cpu": cpu, "latency": latency, "request_rate": req_rate, "replicas": replicas_now}
        return obs, reward, False, False, info

    def _compute_reward(self, cpu, latency, replicas):
        latency_penalty = -10.0 * max(0.0, latency - self.latency_sla)
        cost_penalty = -0.5 * replicas
        utilization_bonus = 1.0 if 0.4 <= cpu <= 0.75 else 0.0
        return latency_penalty + cost_penalty + utilization_bonus

    def render(self):
        pass
