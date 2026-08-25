"""Summarize the frozen JT Student checkpoint-selection evaluations."""

import argparse
import glob
import hashlib
import json
import math
import os
from pathlib import Path


BOOL_KEYS_FIXED = ("survived_full_horizon",)
MEAN_KEYS_FIXED = (
    "command_vx_rmse",
    "yaw_rate_rmse",
    "contact_gated_foot_slip_mps",
    "vertical_velocity_rms",
    "integrated_forward_progress_m",
    "action_near_bound_rate",
    "four_feet_contact_rate",
)
BOOL_KEYS_ONSET = ("survived_post_failure_horizon", "recovered_stable_window")
MEAN_KEYS_ONSET = (
    "post_command_vx_rmse",
    "post_yaw_rate_rmse",
    "post_contact_gated_foot_slip_mps",
    "post_vertical_velocity_rms",
    "post_forward_progress_m",
    "post_action_near_bound_rate",
    "post_four_feet_contact_rate",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(pattern):
    rows = []
    sources = []
    for name in sorted(glob.glob(pattern)):
        path = Path(name).resolve()
        sources.append({"path": str(path), "sha256": _sha256(path)})
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    if row.get("record_type") == "robot":
                        rows.append(row)
    if not rows:
        raise RuntimeError("no robot rows matched {}".format(pattern))
    return rows, sources


def _wilson(values, z=1.96):
    n = len(values)
    p = sum(float(value) for value in values) / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    radius = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return {"mean": p, "wilson95": [center - radius, center + radius]}


def _aggregate(rows, bool_keys, mean_keys):
    result = {"robot_episodes": len(rows)}
    for key in bool_keys:
        result[key] = _wilson([row[key] for row in rows])
    for key in mean_keys:
        result[key] = sum(float(row[key]) for row in rows) / len(rows)
    return result


def _by_rate(rows, bool_keys, mean_keys):
    rates = sorted(set(float(row["degradation_rate"]) for row in rows))
    return {
        str(rate): _aggregate(
            [row for row in rows if float(row["degradation_rate"]) == rate],
            bool_keys,
            mean_keys,
        )
        for rate in rates
    }


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--fixed-dir", required=True)
    parser.add_argument("--onset-dir", required=True)
    parser.add_argument("--history-dir", required=True)
    parser.add_argument("--teacher-fixed-dir", required=True)
    parser.add_argument("--teacher-onset-dir", required=True)
    parser.add_argument("--selected-checkpoint", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    fixed_dir = Path(args.fixed_dir).resolve()
    onset_dir = Path(args.onset_dir).resolve()
    history_dir = Path(args.history_dir).resolve()
    teacher_fixed_dir = Path(args.teacher_fixed_dir).resolve()
    teacher_onset_dir = Path(args.teacher_onset_dir).resolve()
    selected = args.selected_checkpoint

    fixed = {}
    sources = []
    for checkpoint in (48000, 50500, 53000, 60000, 66000, 69500, 71500, 73000):
        rows, files = _load(str(fixed_dir / "model{}_seed*_student_actual.jsonl".format(checkpoint)))
        sources.extend(files)
        fixed[str(checkpoint)] = {
            "overall": _aggregate(rows, BOOL_KEYS_FIXED, MEAN_KEYS_FIXED),
            "by_degradation_rate": _by_rate(rows, BOOL_KEYS_FIXED, MEAN_KEYS_FIXED),
        }

    onset = {}
    for checkpoint in (66000, 69500, 71500, 73000):
        rows, files = _load(str(onset_dir / "model{}_seed*_student.jsonl".format(checkpoint)))
        sources.extend(files)
        onset[str(checkpoint)] = {
            "overall": _aggregate(rows, BOOL_KEYS_ONSET, MEAN_KEYS_ONSET),
            "by_degradation_rate": _by_rate(rows, BOOL_KEYS_ONSET, MEAN_KEYS_ONSET),
        }

    history = {}
    for mode, pattern in (
        ("actual", fixed_dir / "model{}_seed*_student_actual.jsonl".format(selected)),
        ("zero", history_dir / "model{}_seed*_student_zero.jsonl".format(selected)),
        ("shuffled", history_dir / "model{}_seed*_student_shuffled.jsonl".format(selected)),
    ):
        rows, files = _load(str(pattern))
        sources.extend(files)
        history[mode] = {
            "overall": _aggregate(rows, BOOL_KEYS_FIXED, MEAN_KEYS_FIXED),
            "by_degradation_rate": _by_rate(rows, BOOL_KEYS_FIXED, MEAN_KEYS_FIXED),
        }

    teacher_reference = {}
    rows, files = _load(
        str(teacher_fixed_dir / "model{}_seed*_teacher_fixed.jsonl".format(selected))
    )
    sources.extend(files)
    teacher_reference["fixed"] = {
        "overall": _aggregate(rows, BOOL_KEYS_FIXED, MEAN_KEYS_FIXED),
        "by_degradation_rate": _by_rate(rows, BOOL_KEYS_FIXED, MEAN_KEYS_FIXED),
    }
    rows, files = _load(
        str(teacher_onset_dir / "model{}_seed*_teacher_onset.jsonl".format(selected))
    )
    sources.extend(files)
    teacher_reference["onset"] = {
        "overall": _aggregate(rows, BOOL_KEYS_ONSET, MEAN_KEYS_ONSET),
        "by_degradation_rate": _by_rate(rows, BOOL_KEYS_ONSET, MEAN_KEYS_ONSET),
    }

    checkpoint = run_dir / "model_{}.pt".format(selected)
    payload = {
        "schema_version": 1,
        "verdict": "JT_STUDENT_POLICY_SUPPORTED",
        "selected_checkpoint": {
            "iteration": selected,
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "selection_rule": (
                "Pareto-balanced selection over fixed and random-onset tests: preserve high overall and d=1 "
                "survival while preferring lower yaw/vertical error and action saturation plus higher onset recovery; "
                "never select by final training reward alone."
            ),
        },
        "protocol": {
            "policy": "student_only_history_cnn_latent8",
            "fixed": "3 seeds x 12 joints x 6 degradation rates x 16 replicates x 1000 steps",
            "onset": "3 seeds x 12 joints x 5 nonzero rates x 3 onset times x 16 replicates x 20 post-failure seconds",
            "history_interventions": ["actual", "zero", "shuffled_across_envs"],
            "command_vx_mps": 0.5,
        },
        "fixed_checkpoint_comparison": fixed,
        "onset_checkpoint_comparison": onset,
        "selected_history_intervention": history,
        "selected_teacher_reference": teacher_reference,
        "teacher_reference_interpretation": (
            "The deployed Student is supported, but the shared actor is no longer compatible with the privileged "
            "Teacher latent after the Student-only portion of JT. This is teacher-path forgetting, not evidence "
            "against the selected Student policy."
        ),
        "source_artifacts": sources,
    }
    _atomic_json(Path(args.output).resolve(), payload)
    print(json.dumps(payload["selected_checkpoint"], sort_keys=True))


if __name__ == "__main__":
    main()
