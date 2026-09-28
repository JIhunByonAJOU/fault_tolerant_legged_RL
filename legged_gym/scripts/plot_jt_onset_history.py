"""Render severity-wise Student history intervention metrics."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.summary.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    rates = [item["degradation_rate"] for item in summary["severity"]]
    metrics = (("survived_post_failure_horizon", "20 s survival", 100),
               ("recovered_stable_window", "Stable recovery", 100),
               ("post_command_vx_rmse", "Command vx RMSE (m/s)", 1),
               ("post_yaw_rate_rmse", "Yaw-rate RMSE (rad/s)", 1))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for axis, (metric, label, scale) in zip(axes.flat, metrics):
        for mode in ("actual", "zero", "shuffled"):
            values = [item["modes"][mode][metric] * scale for item in summary["severity"]]
            axis.plot(rates, values, marker="o", label=mode)
        axis.set_ylabel(label + (" (%)" if scale == 100 else ""))
        axis.grid(alpha=0.3)
    for axis in axes[1]:
        axis.set_xlabel("Degradation fraction d (1.0 = complete joint loss)")
    axes[0, 0].legend(title="Post-onset history")
    fig.suptitle("JT71500 Student-only: paired severity × history intervention (3 seeds, n=432/rate/mode)")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
