"""
Generate the HPA vs RL comparison graphs for the dissertation (Chapter 10).
Reads results/hpa.csv and results/rl.csv and produces results/comparison.png.
"""
import argparse
import csv
import matplotlib.pyplot as plt


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpa", default="../results/hpa.csv")
    parser.add_argument("--rl", default="../results/rl.csv")
    parser.add_argument("--out", default="../results/comparison.png")
    args = parser.parse_args()

    hpa = load_csv(args.hpa)
    rl = load_csv(args.rl)

    hpa_x = list(range(len(hpa)))
    rl_x = list(range(len(rl)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(hpa_x, [float(r["latency"]) for r in hpa], label="HPA")
    axes[0, 0].plot(rl_x, [float(r["latency"]) for r in rl], label="RL")
    axes[0, 0].set_title("P95 Latency (s)")
    axes[0, 0].set_xlabel("Sample")
    axes[0, 0].legend()

    axes[0, 1].plot(hpa_x, [float(r["cpu"]) for r in hpa], label="HPA")
    axes[0, 1].plot(rl_x, [float(r["cpu"]) for r in rl], label="RL")
    axes[0, 1].set_title("CPU Utilization")
    axes[0, 1].set_xlabel("Sample")
    axes[0, 1].legend()

    axes[1, 0].plot(hpa_x, [int(float(r["replicas"])) for r in hpa], label="HPA", drawstyle="steps-post")
    axes[1, 0].plot(rl_x, [int(float(r["replicas"])) for r in rl], label="RL", drawstyle="steps-post")
    axes[1, 0].set_title("Replica Count")
    axes[1, 0].set_xlabel("Sample")
    axes[1, 0].legend()

    axes[1, 1].plot(hpa_x, [float(r["request_rate"]) for r in hpa], label="HPA request rate")
    if rl and "reward" in rl[0]:
        ax2 = axes[1, 1].twinx()
        ax2.plot(rl_x, [float(r["reward"]) for r in rl], label="RL reward", color="green", alpha=0.6)
        ax2.set_ylabel("RL reward")
    axes[1, 1].set_title("Request Rate (HPA) / Reward (RL)")
    axes[1, 1].set_xlabel("Sample")
    axes[1, 1].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Comparison graph saved to {args.out}")


if __name__ == "__main__":
    main()
