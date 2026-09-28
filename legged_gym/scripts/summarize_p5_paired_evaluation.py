"""Summarize canonical P5 Teacher/Student paired evaluation artifacts."""

import argparse
import json
import math
import os
from pathlib import Path

from legged_gym.scripts.run_p5_paired_evaluation import (
    DEFAULT_OUTPUT,
    DEFAULT_PROTOCOL,
    load_protocol,
    sha256_file,
)


FIXED_BOOL = ("survived_full_horizon",)
FIXED_MEAN = (
    "command_vx_rmse",
    "yaw_rate_rmse",
    "vertical_velocity_rms",
    "contact_gated_foot_slip_mps",
    "integrated_forward_progress_m",
    "action_near_bound_rate",
)
ONSET_BOOL = ("survived_post_failure_horizon", "recovered_stable_window")
ONSET_MEAN = (
    "post_command_vx_rmse",
    "post_yaw_rate_rmse",
    "post_vertical_velocity_rms",
    "post_contact_gated_foot_slip_mps",
    "post_forward_progress_m",
    "post_action_near_bound_rate",
)


def wilson(values, z=1.96):
    n = len(values)
    p = sum(float(value) for value in values) / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    radius = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return {"mean": p, "wilson95": [center - radius, center + radius]}


def aggregate(rows, bool_keys, mean_keys):
    result = {"robot_episodes": len(rows)}
    for key in bool_keys:
        result[key] = wilson([row[key] for row in rows])
    for key in mean_keys:
        result[key] = sum(float(row[key]) for row in rows) / len(rows)
    return result


def load_jsonl(path):
    metadata = None
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type") == "metadata":
                if metadata is not None:
                    raise ValueError("duplicate metadata in {}".format(path))
                metadata = record
            elif record.get("record_type") == "robot":
                rows.append(record)
    if metadata is None or not rows:
        raise ValueError("incomplete result {}".format(path))
    return metadata, rows


def _float_list(values):
    return [float(value) for value in values]


def validate_result(metadata, model, seed, phase, config, command_x=0.5):
    expected = {
        "checkpoint_sha256": model["checkpoint_sha256"],
        "task": model["task"],
        "seed": seed,
        "policy_mode": model["policy_mode"],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError("{} mismatch: {} != {}".format(key, metadata.get(key), value))
    if float(metadata.get("command_x")) != float(command_x):
        raise ValueError("command_x mismatch")
    if metadata.get("replicates_per_condition") != config["replicates_per_condition"]:
        raise ValueError("replicates_per_condition mismatch")
    if _float_list(metadata.get("rates", [])) != _float_list(config["degradation_rates"]):
        raise ValueError("degradation rates mismatch")
    if phase == "fixed":
        if metadata.get("steps") != config["steps"]:
            raise ValueError("fixed steps mismatch")
    else:
        if float(metadata.get("post_seconds")) != float(config["post_seconds"]):
            raise ValueError("onset post_seconds mismatch")
        if _float_list(metadata.get("onset_seconds", [])) != _float_list(config["onset_seconds"]):
            raise ValueError("onset seconds mismatch")
        recovery = metadata.get("recovery_definition", {})
        expected_recovery = {
            "window_seconds": config["recovery_window_seconds"],
            "max_abs_vx_error_mps": config["recovery_vx_error_mps"],
            "max_abs_yaw_rate_radps": config["recovery_yaw_rate_radps"],
        }
        if recovery != expected_recovery:
            raise ValueError("recovery definition mismatch")


def validate_rows(rows, seed, phase, config):
    rates = _float_list(config["degradation_rates"])
    onsets = _float_list(config["onset_seconds"]) if phase == "onset" else [None]
    expected = {
        (joint, rate, onset, replicate)
        for joint in range(12)
        for rate in rates
        for onset in onsets
        for replicate in range(config["replicates_per_condition"])
    }
    actual = set()
    for row in rows:
        if row.get("seed") != seed:
            raise ValueError("robot row seed mismatch")
        rate = float(row["degradation_rate"])
        matched_rate = next((value for value in rates if abs(rate - value) < 1e-5), None)
        if matched_rate is None:
            raise ValueError("robot row degradation rate mismatch")
        if phase == "onset":
            onset = float(row["onset_seconds"])
            matched_onset = next((value for value in onsets if abs(onset - value) < 1e-5), None)
            if matched_onset is None:
                raise ValueError("robot row onset mismatch")
        else:
            matched_onset = None
        key = (int(row["joint_index"]), matched_rate, matched_onset, int(row["replicate"]))
        if key in actual:
            raise ValueError("duplicate robot condition {}".format(key))
        actual.add(key)
    if actual != expected:
        raise ValueError("robot condition grid incomplete: {} != {}".format(len(actual), len(expected)))


def by_rate(rows, bool_keys, mean_keys):
    rates = sorted(set(float(row["degradation_rate"]) for row in rows))
    return {
        str(rate): aggregate(
            [row for row in rows if float(row["degradation_rate"]) == rate],
            bool_keys,
            mean_keys,
        )
        for rate in rates
    }


def by_joint_and_rate(rows, bool_keys, mean_keys):
    joints = sorted(set((int(row["joint_index"]), row["joint_name"]) for row in rows))
    rates = sorted(set(float(row["degradation_rate"]) for row in rows))
    return {
        str(joint_index): {
            "joint_name": joint_name,
            "by_degradation_rate": {
                str(rate): aggregate(
                    [
                        row
                        for row in rows
                        if int(row["joint_index"]) == joint_index
                        and float(row["degradation_rate"]) == rate
                    ],
                    bool_keys,
                    mean_keys,
                )
                for rate in rates
            },
        }
        for joint_index, joint_name in joints
    }


def metric_deltas(reference, candidate, bool_keys, mean_keys):
    deltas = {}
    for key in bool_keys:
        deltas[key] = candidate[key]["mean"] - reference[key]["mean"]
    for key in mean_keys:
        deltas[key] = candidate[key] - reference[key]
    return deltas


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_OUTPUT / "canonical")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "summary.json")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    run_dir = args.run_dir.resolve()
    results = {}
    sources = []
    required_missing = []
    optional_missing = []
    for model in protocol["models"]:
        model_result = {}
        for phase, bool_keys, mean_keys in (
            ("fixed", FIXED_BOOL, FIXED_MEAN),
            ("onset", ONSET_BOOL, ONSET_MEAN),
        ):
            rows = []
            for seed in protocol["seeds"]:
                path = run_dir / "{}_seed{}_{}.jsonl".format(model["id"], seed, phase)
                if not path.is_file():
                    target = required_missing if model.get("required", True) else optional_missing
                    target.append(str(path))
                    continue
                metadata, seed_rows = load_jsonl(path)
                validate_result(
                    metadata,
                    model,
                    seed,
                    phase,
                    protocol[phase],
                    protocol["command_x_mps"],
                )
                validate_rows(seed_rows, seed, phase, protocol[phase])
                rows.extend(seed_rows)
                sources.append({"path": str(path), "sha256": sha256_file(path)})
            if rows:
                model_result[phase] = {
                    "overall": aggregate(rows, bool_keys, mean_keys),
                    "by_degradation_rate": by_rate(rows, bool_keys, mean_keys),
                    "by_joint_and_degradation_rate": by_joint_and_rate(rows, bool_keys, mean_keys),
                }
        if model_result:
            results[model["id"]] = model_result
    if required_missing and not args.allow_incomplete:
        raise RuntimeError("missing {} required canonical result files".format(len(required_missing)))
    comparisons = {}
    reference = results.get("tf43000", {})
    for model_id in ("jt71500", "jt45000"):
        if model_id not in results:
            continue
        comparisons[model_id + "_minus_tf43000"] = {}
        for phase, bool_keys, mean_keys in (
            ("fixed", FIXED_BOOL, FIXED_MEAN),
            ("onset", ONSET_BOOL, ONSET_MEAN),
        ):
            if phase in reference and phase in results[model_id]:
                comparisons[model_id + "_minus_tf43000"][phase] = metric_deltas(
                    reference[phase]["overall"],
                    results[model_id][phase]["overall"],
                    bool_keys,
                    mean_keys,
                )
    payload = {
        "schema_version": 1,
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "status": "INCOMPLETE" if required_missing else "COMPLETE",
        "missing_required_results": required_missing,
        "missing_optional_results": optional_missing,
        "results": results,
        "comparisons": comparisons,
        "delta_convention": "candidate minus TF43000; positive is better for boolean rates and worse for RMSE/slip/stability/action-saturation metrics",
        "source_artifacts": sources,
    }
    atomic_json(args.output.resolve(), payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "missing_required": len(required_missing),
                "missing_optional": len(optional_missing),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
