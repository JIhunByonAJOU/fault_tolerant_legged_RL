"""Aggregate paired per-robot FailureEnv evaluations."""

import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = (
    "survived_full_horizon",
    "survival_seconds",
    "mean_reward_per_step",
    "command_vx_rmse",
    "yaw_rate_rmse",
    "vertical_velocity_rms",
    "integrated_forward_progress_m",
    "mean_abs_action",
    "action_near_bound_rate",
    "four_feet_contact_rate",
    "airborne_rate",
    "contact_gated_foot_slip_mps",
)


def read_rows(paths):
    metadata, robots = [], []
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.pop("record_type") == "metadata":
                    metadata.append(row)
                else:
                    robots.append(row)
    return metadata, robots


def means(rows):
    return {
        key: statistics.fmean(float(row[key]) for row in rows)
        for key in METRICS
    }


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", nargs="+", required=True)
    parser.add_argument("--adapted", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base_meta, base = read_rows(args.base)
    adapted_meta, adapted = read_rows(args.adapted)
    if len(base) != len(adapted):
        raise ValueError("base/adapted robot row count differs")
    base_keys = {(r["seed"], r["env_id"], r["joint_index"], r["degradation_rate"]) for r in base}
    adapted_keys = {(r["seed"], r["env_id"], r["joint_index"], r["degradation_rate"]) for r in adapted}
    if base_keys != adapted_keys:
        raise ValueError("base/adapted fixed-condition cells differ")

    def grouped(rows, fields):
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[field] for field in fields)].append(row)
        return groups

    cells = []
    bg = grouped(base, ("joint_index", "degradation_rate"))
    ag = grouped(adapted, ("joint_index", "degradation_rate"))
    for key in sorted(bg):
        b, a = means(bg[key]), means(ag[key])
        cells.append(
            {
                "joint_index": key[0],
                "degradation_rate": key[1],
                "samples_per_checkpoint": len(bg[key]),
                "base": b,
                "adapted": a,
                "adapted_minus_base": {metric: a[metric] - b[metric] for metric in METRICS},
            }
        )
    rates = []
    bg = grouped(base, ("degradation_rate",))
    ag = grouped(adapted, ("degradation_rate",))
    for key in sorted(bg):
        b, a = means(bg[key]), means(ag[key])
        rates.append(
            {
                "degradation_rate": key[0],
                "samples_per_checkpoint": len(bg[key]),
                "base": b,
                "adapted": a,
                "adapted_minus_base": {metric: a[metric] - b[metric] for metric in METRICS},
            }
        )
    overall_base, overall_adapted = means(base), means(adapted)
    result = {
        "schema_version": 1,
        "base_inputs": base_meta,
        "adapted_inputs": adapted_meta,
        "paired_robot_rows_per_checkpoint": len(base),
        "overall": {
            "base": overall_base,
            "adapted": overall_adapted,
            "adapted_minus_base": {
                metric: overall_adapted[metric] - overall_base[metric]
                for metric in METRICS
            },
        },
        "by_degradation_rate": rates,
        "by_joint_and_degradation": cells,
    }
    output = Path(args.output).resolve()
    atomic_json(output, result)
    print(json.dumps({"output": str(output), "paired_rows": len(base), "overall": result["overall"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
