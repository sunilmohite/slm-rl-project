import time
import logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import urllib3
import requests

urllib3.disable_warnings()

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

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, min_replicas], dtype=np.float32),
            high=np.array([1, 30, 100, max_replicas], dtype=np.float32),
        )

        self.apps = self._init_k8s()

    # ------------------------------------------------------------
    # Kubernetes
    # ------------------------------------------------------------

    def _init_k8s(self):

        from kubernetes import client
        from kubernetes import config

        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()

        configuration = client.Configuration.get_default_copy()

        configuration.verify_ssl = False
        configuration.assert_hostname = False

        client.Configuration.set_default(configuration)

        api = client.ApiClient(configuration)

        return client.AppsV1Api(api)

    # ------------------------------------------------------------
    # Prometheus
    # ------------------------------------------------------------

    def _query_prometheus(self, query):

        try:

            r = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query},
                verify=False,
                timeout=10,
            )

            if r.status_code != 200:

                print("PROMQL FAILED")
                print(query)
                print(r.text)

                return 0.0

            result = r.json()["data"]["result"]

            if len(result) == 0:
                return 0.0

            return float(result[0]["value"][1])

        except Exception as e:

            print("PROMQL ERROR")
            print(query)
            print(e)

            return 0.0

    # ------------------------------------------------------------

    def _get_metrics(self):

        cpu = self._query_prometheus(
            f'avg(rate(container_cpu_usage_seconds_total{{namespace="{self.namespace}",pod=~"{self.deployment_name}.*"}}[1m]))'
        )

        latency = self._query_prometheus(
            f'histogram_quantile(0.95,sum(rate(slm_request_latency_seconds_bucket{{namespace="{self.namespace}"}}[1m])) by (le))'
        )

        requests_rate = self._query_prometheus(
            f'sum(rate(slm_requests_total{{namespace="{self.namespace}"}}[1m]))'
        )

        return cpu, latency, requests_rate

    # ------------------------------------------------------------
    # Deployment Scaling
    # ------------------------------------------------------------

    def _get_replicas(self):

        dep = self.apps.read_namespaced_deployment(
            self.deployment_name,
            self.namespace,
        )

        return dep.spec.replicas

    def _scale(self, replicas):

        replicas = max(
            self.min_replicas,
            min(self.max_replicas, replicas),
        )

        body = {
            "spec": {
                "replicas": replicas
            }
        }

        self.apps.patch_namespaced_deployment_scale(
            self.deployment_name,
            self.namespace,
            body,
        )

    # ------------------------------------------------------------
    # Gym
    # ------------------------------------------------------------

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        cpu, latency, req = self._get_metrics()

        replicas = self._get_replicas()

        obs = np.array(
            [cpu, latency, req, replicas],
            dtype=np.float32,
        )

        return obs, {}

    def step(self, action):

        replicas = self._get_replicas()

        if action == 0:
            replicas -= 1

        elif action == 2:
            replicas += 1

        self._scale(replicas)

        time.sleep(self.step_interval)

        cpu, latency, req = self._get_metrics()

        replicas = self._get_replicas()

        obs = np.array(
            [cpu, latency, req, replicas],
            dtype=np.float32,
        )

        reward = self._reward(cpu, latency, replicas)

        info = {
            "cpu": cpu,
            "latency": latency,
            "request_rate": req,
            "replicas": replicas,
        }

        return obs, reward, False, False, info

    # ------------------------------------------------------------

    def _reward(self, cpu, latency, replicas):

        reward = 0

        if latency > self.latency_sla:
            reward -= 10 * (latency - self.latency_sla)

        reward -= replicas * 0.5

        if 0.40 <= cpu <= 0.75:
            reward += 1

        return reward

    def render(self):
        pass
