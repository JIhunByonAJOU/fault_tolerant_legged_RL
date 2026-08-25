"""Select representative checkpoints from fixed-condition JSONL evaluations."""

import argparse
import hashlib
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = (
    "survived_full_horizon",
    "command_vx_rmse",
    "yaw_rate_rmse",
    "vertical_velocity_rms",
    "integrated_forward_progress_m",
    "contact_gated_foot_slip_mps",
    "four_feet_contact_rate",
    "action_near_bound_rate",
)


def read_rows(paths):
    rows = []
    metadata = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                value = json.loads(line)
                if value["record_type"] == "metadata":
                    metadata.append(value)
                else:
                    rows.append(value)
    return metadata, rows


def means(rows):
    return {
        metric: statistics.fmean(float(row[metric]) for row in rows)
        for metric in METRICS
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.input_dir).resolve()
    grouped = defaultdict(list)
    for path in sorted(root.glob("*_seed*.jsonl")):
        label = path.name.rsplit("_seed", 1)[0]
        grouped[label].append(path)
    if not grouped:
        raise RuntimeError("no evaluation files")

    candidates = []
    for label, paths in sorted(grouped.items()):
        metadata, rows = read_rows(paths)
        if len(metadata) != 3:
            raise RuntimeError("{} requires exactly three seeds".format(label))
        overall = means(rows)
        by_rate = {}
        for rate in sorted({float(row["degradation_rate"]) for row in rows}):
            by_rate[str(rate)] = means(
                [row for row in rows if float(row["degradation_rate"]) == rate]
            )
        checkpoint = metadata[0]["checkpoint"]
        candidates.append(
            {
                "label": label,
                "checkpoint": checkpoint,
                "checkpoint_sha256": metadata[0]["checkpoint_sha256"],
                "rows": len(rows),
                "overall": overall,
                "by_degradation_rate": by_rate,
            }
        )

    # Primary robustness: full-horizon survival overall and at d=1. Secondary
    # quality: tracking, yaw, slip, vertical motion, then progress.
    def rank_key(candidate):
        overall = candidate["overall"]
        severe = candidate["by_degradation_rate"]["1.0"]
        return (
            -overall["survived_full_horizon"],
            -severe["survived_full_horizon"],
            overall["command_vx_rmse"],
            overall["yaw_rate_rmse"],
            overall["contact_gated_foot_slip_mps"],
            overall["vertical_velocity_rms"],
            -overall["integrated_forward_progress_m"],
        )

    lineages = defaultdict(list)
    for candidate in candidates:
        lineages[candidate["label"].rstrip("0123456789")].append(candidate)
    selected_by_lineage = {
        lineage: sorted(values, key=rank_key)[0]["label"]
        for lineage, values in sorted(lineages.items())
    }
    overall_selected = sorted(candidates, key=rank_key)[0]["label"]
    result = {
        "schema_version": 1,
        "protocol": {
            "seeds": [1, 2, 3],
            "joints": 12,
            "degradation_rates": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "replicates_per_condition": 16,
            "steps": 1000,
            "command_x": 0.5,
            "ranking": "survival overall, d1 survival, vx RMSE, yaw RMSE, slip, vertical RMS, progress",
        },
        "selected_by_lineage": selected_by_lineage,
        "selected_tf_candidate": overall_selected,
        "candidates": sorted(candidates, key=rank_key),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({
        "output": str(output),
        "selected_by_lineage": selected_by_lineage,
        "selected_tf_candidate": overall_selected,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
