"""Plot Student performance under same-robot delayed histories and sham control."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.summary.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    rates = (0.0, 0.8, 1.0)
    conditions = (("actual", "actual"), ("delay02", "0.2 s delayed"),
                  ("delay05", "0.5 s delayed"), ("delay10", "1.0 s delayed"))
    metrics = (("survived_post_failure_horizon", "20 s survival (%)", 100),
               ("recovered_stable_window", "Stable recovery (%)", 100),
               ("post_command_vx_rmse", "Command vx RMSE (m/s)", 1),
               ("post_yaw_rate_rmse", "Yaw-rate RMSE (rad/s)", 1))
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for axis, (metric, label, factor) in zip(axes.flat, metrics):
        for condition, name in conditions:
            axis.plot(rates, [summary["by_rate"][str(rate)][condition][metric] * factor
                              for rate in rates], marker="o", label=name)
        axis.set_ylabel(label)
        axis.set_xticks(rates, ("0.0 sham", "0.8", "1.0"))
        axis.grid(alpha=0.25)
    axes[0, 0].legend(title="History at decision")
    for axis in axes[1]:
        axis.set_xlabel("Degradation fraction")
    figure.suptitle("JT71500 Student-only: same-robot history delay after 5 s onset")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
