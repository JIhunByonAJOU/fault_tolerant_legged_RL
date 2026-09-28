"""Summarize paired post-onset history interventions by degradation severity."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


FIELDS = (
    "failed_before_onset",
    "survived_post_failure_horizon",
    "recovered_stable_window",
    "post_command_vx_rmse",
    "post_yaw_rate_rmse",
    "post_action_near_bound_rate",
)


def load(path):
    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    metadata, rows = records[0], records[1:]
    if any(row.get("record_type") != "robot" for row in rows):
        raise ValueError("unexpected record type in {}".format(path))
    if len(rows) != metadata["num_envs"]:
        raise ValueError("incomplete matrix: {}".format(path))
    return metadata, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    grouped = defaultdict(list)
    paired = defaultdict(dict)
    sources = []
    sha = None
    for mode in ("actual", "zero", "shuffled"):
        for seed in (1, 2, 3):
            path = args.directory / "matrix_{}_seed{}.jsonl".format(mode, seed)
            metadata, rows = load(path)
            if metadata["history_mode"] != mode or metadata["seed"] != seed:
                raise ValueError("matrix metadata mismatch: {}".format(path))
            if mode != "actual" and metadata.get("history_intervention_after_onset_only") is not True:
                raise ValueError("history was not isolated to post-onset: {}".format(path))
            if mode == "shuffled" and metadata.get("shuffled_history_offset_envs") != 90:
                raise ValueError("shuffled history did not exchange joint assignment: {}".format(path))
            if sha is not None and metadata["checkpoint_sha256"] != sha:
                raise ValueError("checkpoint mismatch")
            sha = metadata["checkpoint_sha256"]
            sources.append(str(path))
            for row in rows:
                rate = row["degradation_rate"]
                grouped[(rate, mode)].append(row)
                key = (seed, row["joint_index"], rate, row["onset_seconds"], row["replicate"])
                paired[key][mode] = row
    if any(set(modes) != {"actual", "zero", "shuffled"} for modes in paired.values()):
        raise ValueError("missing paired condition")
    pre_mismatch = 0
    terrain_mismatch = 0
    for modes in paired.values():
        actual = modes["actual"]
        for other in (modes["zero"], modes["shuffled"]):
            pre_mismatch += actual["failed_before_onset"] != other["failed_before_onset"]
            terrain_mismatch += (actual["terrain_level"], actual["terrain_type"]) != (other["terrain_level"], other["terrain_type"])
    summary = {"checkpoint_sha256": sha, "files": sources, "paired_cells": len(paired),
               "pre_failure_status_mismatches": pre_mismatch,
               "terrain_assignment_mismatches": terrain_mismatch, "severity": []}
    for rate in sorted({key[0] for key in grouped}):
        item = {"degradation_rate": rate, "modes": {}, "paired_actual_minus_intervention": {}}
        for mode in ("actual", "zero", "shuffled"):
            rows = grouped[(rate, mode)]
            item["modes"][mode] = {"n": len(rows), **{
                field: statistics.mean(float(row[field]) for row in rows) for field in FIELDS
            }}
        rate_pairs = [modes for key, modes in paired.items() if key[2] == rate]
        for mode in ("zero", "shuffled"):
            item["paired_actual_minus_intervention"][mode] = {
                field: statistics.mean(float(modes["actual"][field]) - float(modes[mode][field])
                                       for modes in rate_pairs)
                for field in FIELDS if field != "failed_before_onset"
            }
        summary["severity"].append(item)
    output = args.directory / "severity_history_summary.json"
    with output.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
