"""
Lightweight simulator for the SLM scaling environment. Lets you train the
PPO agent quickly (thousands of steps in seconds) without a live OpenShift
cluster. The trained policy is then evaluated / used against the real
cluster via env.py + evaluate.py.

Same observation/action/reward definition as env.py, so a policy trained
here transfers directly.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SimulatedSLMScalingEnv(gym.Env):
    def __init__(self, min_replicas=1, max_replicas=5, latency_sla_seconds=2.0, episode_length=200):
        super().__init__()
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.latency_sla = latency_sla_seconds
        self.episode_length = episode_length

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, min_replicas], dtype=np.float32),
            high=np.array([1.0, 30.0, 100.0, max_replicas], dtype=np.float32),
        )
        self._t = 0
        self._replicas = min_replicas
        self._capacity_per_replica = 3.0  # requests/sec a single replica can serve near the SLA

    def _traffic(self, t):
        # Diurnal-ish pattern plus noise, in requests/sec
        base = 6 + 5 * np.sin(2 * np.pi * t / self.episode_length)
        noise = np.random.normal(0, 0.5)
        return max(0.0, base + noise)

    def _simulate_metrics(self, req_rate, replicas):
        capacity = replicas * self._capacity_per_replica
        utilization = min(1.0, req_rate / capacity) if capacity > 0 else 1.0
        # latency grows sharply as utilization approaches/exceeds capacity
        if utilization < 0.95:
            latency = 0.3 + 1.5 * (utilization ** 3)
        else:
            latency = 0.3 + 1.5 + 8.0 * (utilization - 0.95)
        return utilization, latency

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self._replicas = self.min_replicas
        req_rate = self._traffic(self._t)
        cpu, latency = self._simulate_metrics(req_rate, self._replicas)
        return np.array([cpu, latency, req_rate, self._replicas], dtype=np.float32), {}

    def step(self, action):
        if action == 0:
            self._replicas -= 1
        elif action == 2:
            self._replicas += 1
        self._replicas = max(self.min_replicas, min(self.max_replicas, self._replicas))

        self._t += 1
        req_rate = self._traffic(self._t)
        cpu, latency = self._simulate_metrics(req_rate, self._replicas)

        latency_penalty = -10.0 * max(0.0, latency - self.latency_sla)
        cost_penalty = -0.5 * self._replicas
        utilization_bonus = 1.0 if 0.4 <= cpu <= 0.75 else 0.0
        reward = latency_penalty + cost_penalty + utilization_bonus

        terminated = self._t >= self.episode_length
        obs = np.array([cpu, latency, req_rate, self._replicas], dtype=np.float32)
        info = {"cpu": cpu, "latency": latency, "request_rate": req_rate, "replicas": self._replicas}
        return obs, reward, terminated, False, info
