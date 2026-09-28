"""Plot Student action/latent and physical response around an assigned failure."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_trace(path):
    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    metadata = records[0]
    if metadata.get("record_type") != "metadata":
        raise ValueError("first record must contain metadata")
    rows = [record for record in records[1:] if record.get("record_type") == "step"]
    return metadata, rows


def summarize(meta, rows):
    specs = {spec["env_id"]: spec for spec in meta["trace_specs"]}
    summary = []
    for env_id, spec in specs.items():
        own = sorted((row for row in rows if row["env_id"] == env_id), key=lambda r: r["step"])
        if not own:
            continue
        onset = spec["onset_seconds"]
        time = np.array([row["time_seconds"] for row in own])
        latent = np.array([row["student_latent8"] for row in own])
        action = np.array([row["action"] for row in own])
        velocity = np.array([row["vx_mps"] for row in own])
        yaw = np.array([row["yaw_rate_radps"] for row in own])
        pre = (time >= onset - 2.0) & (time < onset)
        baseline_latent = np.median(latent[pre], axis=0) if pre.any() else np.zeros(8)
        baseline_action = np.median(action[pre], axis=0) if pre.any() else np.zeros(12)
        windows = {}
        for label, mask in (("pre_2s", pre), ("post_0_2s", (time >= onset) & (time < onset + 2)),
                            ("post_2_10s", (time >= onset + 2) & (time < onset + 10))):
            if not mask.any():
                windows[label] = None
                continue
            windows[label] = {
                "samples": int(mask.sum()),
                "vx_error_rmse_mps": float(np.sqrt(np.mean((velocity[mask] - meta["command_x"]) ** 2))),
                "abs_yaw_mean_radps": float(np.mean(np.abs(yaw[mask]))),
                "latent_distance_from_pre_median_mean": float(np.mean(np.linalg.norm(latent[mask] - baseline_latent, axis=1))),
                "action_distance_from_pre_median_mean": float(np.mean(np.linalg.norm(action[mask] - baseline_action, axis=1))),
            }
        summary.append({**spec, "observed_until_seconds": float(time[-1]), "windows": windows})
    return summary


def plot(meta, rows, output):
    specs = {spec["env_id"]: spec for spec in meta["trace_specs"]}
    latent_limit = max(1.0, float(np.quantile(np.abs([value for row in rows for value in row["student_latent8"]]), 0.99)))
    action_limit = max(1.0, float(np.quantile(np.abs([value for row in rows for value in row["action"]]), 0.99)))
    for replicate in sorted({spec["replicate"] for spec in specs.values()}):
        group = [spec for spec in specs.values() if spec["replicate"] == replicate]
        figure, axes = plt.subplots(4, len(group), figsize=(5 * len(group), 10), squeeze=False, sharex="col")
        for column, spec in enumerate(group):
            own = sorted((row for row in rows if row["env_id"] == spec["env_id"]), key=lambda r: r["step"])
            if not own:
                continue
            time = np.array([row["time_seconds"] for row in own])
            latent = np.array([row["student_latent8"] for row in own]).T
            action = np.array([row["action"] for row in own]).T
            axes[0, column].plot(time, [row["vx_mps"] for row in own], lw=0.9)
            axes[0, column].axhline(meta["command_x"], color="gray", ls="--", lw=0.8)
            axes[1, column].plot(time, [row["yaw_rate_radps"] for row in own], lw=0.9)
            axes[2, column].imshow(latent, aspect="auto", origin="lower", extent=(time[0], time[-1], -0.5, 7.5), cmap="coolwarm", vmin=-latent_limit, vmax=latent_limit)
            axes[3, column].imshow(action, aspect="auto", origin="lower", extent=(time[0], time[-1], -0.5, 11.5), cmap="coolwarm", vmin=-action_limit, vmax=action_limit)
            for row in range(4):
                axes[row, column].axvline(spec["onset_seconds"], color="red", lw=1.0)
                if row < 2:
                    axes[row, column].grid(alpha=0.2)
            axes[0, column].set_title("{} d={} env={}".format(spec["joint_name"], spec["degradation_rate"], spec["env_id"]))
            axes[3, column].set_xlabel("Time (s), red = fault onset")
        for row, label in enumerate(("vx (m/s)", "yaw rate (rad/s)", "Student latent 0–7", "action 0–11")):
            axes[row, 0].set_ylabel(label)
        delay = meta.get("history_delay_seconds", 0.0)
        figure.suptitle("71500 Student-only, {} history (delay {} s), seed {}, replicate {}".format(
            meta["history_mode"], delay, meta["seed"], replicate))
        figure.tight_layout()
        target = output.with_name("{}_rep{}.png".format(output.stem, replicate))
        figure.savefig(target, dpi=150)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()
    metadata, rows = load_trace(args.trace)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    plot(metadata, rows, args.output_prefix)
    summary = summarize(metadata, rows)
    summary_path = args.output_prefix.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump({"trace": str(args.trace), "summary": summary}, stream, indent=2)
    print(summary_path)


if __name__ == "__main__":
    main()
