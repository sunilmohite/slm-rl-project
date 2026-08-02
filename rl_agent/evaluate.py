"""
Evaluate a trained PPO scaling policy and log results to results/rl.csv
for direct comparison against the HPA baseline (results/hpa.csv).

Run this against the LIVE cluster (no --dry-run) while load_generator.py is
sending traffic, so the RL agent is actually scaling the real Deployment.
"""
import argparse
import csv
import time
import os
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
    parser.add_argument("--model", default="ppo_slm_scaler.zip")
    parser.add_argument("--prometheus-url", default="http://localhost:9090")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--deployment", default="slm-inference")
    parser.add_argument("--min-replicas", type=int, default=1)
    parser.add_argument("--max-replicas", type=int, default=5)
    parser.add_argument("--latency-sla", type=float, default=2.0)
    parser.add_argument("--step-interval", type=int, default=15)
    parser.add_argument("--steps", type=int, default=80,
                         help="Evaluation steps (~steps * step_interval seconds on a live cluster)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default="../results/rl.csv")
    args = parser.parse_args()

    env = build_env(args)
    model = PPO.load(args.model)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    obs, _ = env.reset()

    action_names = {0: "scale_down", 1: "no_action", 2: "scale_up"}

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "step", "action", "cpu", "latency", "request_rate", "replicas", "reward"])
        for step in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            writer.writerow([
                time.time(), step, action_names[action],
                round(info["cpu"], 4), round(info["latency"], 4),
                round(info["request_rate"], 4), info["replicas"], round(reward, 4),
            ])
            f.flush()
            print(f"step={step} action={action_names[action]} cpu={info['cpu']:.2f} "
                  f"latency={info['latency']:.2f}s replicas={info['replicas']} reward={reward:.2f}")
            if terminated or truncated:
                obs, _ = env.reset()

    print(f"Evaluation complete. Results written to {args.out}")


if __name__ == "__main__":
    main()
