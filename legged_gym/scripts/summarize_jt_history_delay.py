"""Summarize Student same-robot history delays against a zero-degradation sham."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


CONDITIONS = {"actual": ("actual", 0), "delay02": ("delayed", 10),
              "delay05": ("delayed", 25), "delay10": ("delayed", 50)}
METRICS = (
    "failed_before_onset",
    "survived_post_failure_horizon",
    "recovered_stable_window",
    "post_command_vx_rmse",
    "post_yaw_rate_rmse",
    "post_forward_progress_m",
)
TARGETS = (("RL_hip_joint", 1.0), ("RL_thigh_joint", 0.8), ("RR_calf_joint", 0.8))


def load(path):
    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    metadata, rows = records[0], records[1:]
    if len(rows) != metadata["num_envs"] or any(row.get("record_type") != "robot" for row in rows):
        raise ValueError("incomplete matrix: {}".format(path))
    return metadata, rows


def means(rows):
    return {"n": len(rows), **{metric: statistics.mean(float(row[metric]) for row in rows)
                             for metric in METRICS}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    paired = defaultdict(dict)
    source_files = []
    sha = None
    for condition, (mode, steps) in CONDITIONS.items():
        for seed in (1, 2, 3):
            path = args.directory / "matrix_{}_seed{}.jsonl".format(condition, seed)
            metadata, rows = load(path)
            if metadata["history_mode"] != mode or metadata["history_delay_steps"] != steps:
                raise ValueError("wrong intervention: {}".format(path))
            if metadata["seed"] != seed or metadata["rates"] != [0.0, 0.8, 1.0] or metadata["onset_seconds"] != [5.0]:
                raise ValueError("protocol mismatch: {}".format(path))
            if metadata["replicates_per_condition"] != 16 or metadata["post_seconds"] != 20.0:
                raise ValueError("exposure mismatch: {}".format(path))
            if sha is not None and metadata["checkpoint_sha256"] != sha:
                raise ValueError("checkpoint mismatch")
            sha = metadata["checkpoint_sha256"]
            source_files.append(str(path))
            for row in rows:
                paired[(seed, row["env_id"])][condition] = row
    if any(set(conditions) != set(CONDITIONS) for conditions in paired.values()):
        raise ValueError("unpaired robot/seed")
    pre_mismatch = terrain_mismatch = 0
    for conditions in paired.values():
        actual = conditions["actual"]
        for condition in ("delay02", "delay05", "delay10"):
            other = conditions[condition]
            if (actual["joint_index"], actual["degradation_rate"], actual["onset_seconds"]) != (
                other["joint_index"], other["degradation_rate"], other["onset_seconds"]
            ):
                raise ValueError("fault assignment mismatch")
            pre_mismatch += actual["failed_before_onset"] != other["failed_before_onset"]
            terrain_mismatch += (actual["terrain_level"], actual["terrain_type"]) != (
                other["terrain_level"], other["terrain_type"]
            )
    summary = {"checkpoint_sha256": sha, "source_files": source_files, "paired_robots": len(paired),
               "pre_failure_status_mismatches": pre_mismatch,
               "terrain_assignment_mismatches": terrain_mismatch, "by_rate": {},
               "rate_excess_delay_effect": {}, "seed_rate_excess_delay_effect": {}, "target_cells": {}}
    for rate in (0.0, 0.8, 1.0):
        rate_pairs = [conditions for conditions in paired.values()
                      if conditions["actual"]["degradation_rate"] == rate]
        summary["by_rate"][str(rate)] = {
            condition: means([conditions[condition] for conditions in rate_pairs])
            for condition in CONDITIONS
        }
    for rate in (0.8, 1.0):
        fault = summary["by_rate"][str(rate)]
        sham = summary["by_rate"]["0.0"]
        summary["rate_excess_delay_effect"][str(rate)] = {
            condition: {
                metric: (fault[condition][metric] - fault["actual"][metric])
                - (sham[condition][metric] - sham["actual"][metric])
                for metric in METRICS if metric != "failed_before_onset"
            }
            for condition in CONDITIONS if condition != "actual"
        }
        summary["seed_rate_excess_delay_effect"][str(rate)] = {}
        for seed in (1, 2, 3):
            fault_pairs = [conditions for (row_seed, _), conditions in paired.items()
                           if row_seed == seed and conditions["actual"]["degradation_rate"] == rate]
            sham_pairs = [conditions for (row_seed, _), conditions in paired.items()
                          if row_seed == seed and conditions["actual"]["degradation_rate"] == 0.0]
            fault_seed = {condition: means([conditions[condition] for conditions in fault_pairs])
                          for condition in CONDITIONS}
            sham_seed = {condition: means([conditions[condition] for conditions in sham_pairs])
                         for condition in CONDITIONS}
            summary["seed_rate_excess_delay_effect"][str(rate)][str(seed)] = {
                condition: {
                    metric: (fault_seed[condition][metric] - fault_seed["actual"][metric])
                    - (sham_seed[condition][metric] - sham_seed["actual"][metric])
                    for metric in METRICS if metric != "failed_before_onset"
                }
                for condition in CONDITIONS if condition != "actual"
            }
    for joint, rate in TARGETS:
        label = "{}:d={}".format(joint, rate)
        fault_pairs = [conditions for conditions in paired.values()
                       if conditions["actual"]["joint_name"] == joint and conditions["actual"]["degradation_rate"] == rate]
        sham_pairs = [conditions for conditions in paired.values()
                      if conditions["actual"]["joint_name"] == joint and conditions["actual"]["degradation_rate"] == 0.0]
        item = {"fault": {condition: means([conditions[condition] for conditions in fault_pairs])
                          for condition in CONDITIONS},
                "sham": {condition: means([conditions[condition] for conditions in sham_pairs])
                         for condition in CONDITIONS}, "excess_delay_effect": {}, "seed_excess_delay_effect": {}}
        for condition in ("delay02", "delay05", "delay10"):
            item["excess_delay_effect"][condition] = {
                metric: (item["fault"][condition][metric] - item["fault"]["actual"][metric])
                - (item["sham"][condition][metric] - item["sham"]["actual"][metric])
                for metric in METRICS if metric != "failed_before_onset"
            }
        for seed in (1, 2, 3):
            seed_fault = [conditions for (row_seed, _), conditions in paired.items()
                          if row_seed == seed and conditions["actual"]["joint_name"] == joint
                          and conditions["actual"]["degradation_rate"] == rate]
            seed_sham = [conditions for (row_seed, _), conditions in paired.items()
                         if row_seed == seed and conditions["actual"]["joint_name"] == joint
                         and conditions["actual"]["degradation_rate"] == 0.0]
            fault_means = {condition: means([conditions[condition] for conditions in seed_fault])
                           for condition in CONDITIONS}
            sham_means = {condition: means([conditions[condition] for conditions in seed_sham])
                          for condition in CONDITIONS}
            item["seed_excess_delay_effect"][str(seed)] = {
                condition: {
                    metric: (fault_means[condition][metric] - fault_means["actual"][metric])
                    - (sham_means[condition][metric] - sham_means["actual"][metric])
                    for metric in METRICS if metric != "failed_before_onset"
                }
                for condition in ("delay02", "delay05", "delay10")
            }
        summary["target_cells"][label] = item
    output = args.directory / "history_delay_summary.json"
    with output.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
