"""Render Teacher/Student joint-by-severity P5 heatmaps from validated JSON."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from legged_gym.scripts.run_p5_paired_evaluation import DEFAULT_OUTPUT


def matrix(summary, model_id, phase, metric):
    joint_data = summary["results"][model_id][phase]["by_joint_and_degradation_rate"]
    joint_ids = sorted(int(key) for key in joint_data)
    rates = sorted(float(key) for key in joint_data[str(joint_ids[0])]["by_degradation_rate"])
    values = np.array(
        [
            [
                joint_data[str(joint)]["by_degradation_rate"][str(rate)][metric]["mean"]
                for rate in rates
            ]
            for joint in joint_ids
        ],
        dtype=float,
    )
    names = [joint_data[str(joint)]["joint_name"].replace("_joint", "") for joint in joint_ids]
    return values, names, rates


def plot(summary, output):
    if summary["status"] != "COMPLETE":
        raise ValueError("P5 summary must be complete before plotting")
    fig, axes = plt.subplots(2, 3, figsize=(16, 12), constrained_layout=True)
    specifications = (
        (0, "fixed", "survived_full_horizon", "Fixed-horizon survival"),
        (1, "onset", "recovered_stable_window", "Post-failure stable recovery"),
    )
    for row, phase, metric, title in specifications:
        teacher, names, rates = matrix(summary, "tf43000", phase, metric)
        student, student_names, student_rates = matrix(summary, "jt71500", phase, metric)
        if names != student_names or rates != student_rates:
            raise ValueError("Teacher/Student heatmap axes differ")
        grids = (teacher, student, student - teacher)
        labels = ("TF43000 Teacher", "JT71500 Student", "Student - Teacher")
        for col, (grid, label) in enumerate(zip(grids, labels)):
            axis = axes[row, col]
            if col < 2:
                image = axis.imshow(grid * 100.0, vmin=0.0, vmax=100.0, cmap="viridis", aspect="auto")
                fig.colorbar(image, ax=axis, label="%")
            else:
                bound = max(5.0, float(np.max(np.abs(grid * 100.0))))
                image = axis.imshow(grid * 100.0, vmin=-bound, vmax=bound, cmap="RdBu", aspect="auto")
                fig.colorbar(image, ax=axis, label="percentage points")
            axis.set_title("{} | {}".format(title, label))
            axis.set_xticks(range(len(rates)), ["{:.1f}".format(rate) for rate in rates])
            axis.set_yticks(range(len(names)), names)
            axis.set_xlabel("Degradation rate d")
            for joint in range(len(names)):
                for rate in range(len(rates)):
                    value = grid[joint, rate] * 100.0
                    cell = "{:+.0f}".format(value) if col == 2 else "{:.0f}".format(value)
                    axis.text(rate, joint, cell, ha="center", va="center", fontsize=7, color="white")
    fig.suptitle("P5 matched-condition evaluation: 3 seeds, 12 joints, 16 replicates/cell", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_OUTPUT / "summary.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "p5_joint_severity_heatmaps.png")
    args = parser.parse_args()
    with args.summary.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    plot(summary, args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
