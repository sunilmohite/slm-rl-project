"""
Train a PPO agent on the SLM scaling environment.

Use --dry-run (recommended) to train fast against the local simulator
(sim_env.py) - thousands of timesteps in well under a minute.
Omit --dry-run to train directly against a live OpenShift cluster + Prometheus
(env.py) - each step takes step_interval seconds, so keep --timesteps small.
"""
import argparse
from stable_baselines3 import PPO
from env import SLMScalingEnv
from sim_env import SimulatedSLMScalingEnv


def build_env(args):
    if args.dry_run:
        return SimulatedSLMScalingEnv(
            min_replicas=args.min_replicas,
            max_replicas=args.max_replicas,
            latency_sla_seconds=args.latency_sla,
        )
    return SLMScalingEnv(
        prometheus_url=args.prometheus_url,
        namespace=args.namespace,
        deployment_name=args.deployment,
        min_replicas=args.min_replicas,
        max_replicas=args.max_replicas,
        latency_sla_seconds=args.latency_sla,
        step_interval_seconds=args.step_interval,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prometheus-url", default="http://localhost:9090")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--deployment", default="slm-inference")
    parser.add_argument("--min-replicas", type=int, default=1)
    parser.add_argument("--max-replicas", type=int, default=5)
    parser.add_argument("--latency-sla", type=float, default=2.0)
    parser.add_argument("--step-interval", type=int, default=15)
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--dry-run", action="store_true",
                         help="Train against the fast local simulator instead of a live cluster")
    parser.add_argument("--out", default="ppo_slm_scaler.zip")
    args = parser.parse_args()

    env = build_env(args)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.out)
    print(f"Model saved to {args.out}")


if __name__ == "__main__":
    main()
