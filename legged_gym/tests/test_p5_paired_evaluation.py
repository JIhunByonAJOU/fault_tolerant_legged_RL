"""CPU-only contract tests for the P5 paired evaluation workflow."""

import copy
import json
import tempfile
from pathlib import Path

import pytest

from legged_gym.scripts.run_p5_paired_evaluation import (
    DEFAULT_PROTOCOL,
    build_jobs,
    command_for_job,
    evaluation_environment,
    load_protocol,
    resolve_models,
)
from legged_gym.scripts.summarize_p5_paired_evaluation import (
    FIXED_BOOL,
    FIXED_MEAN,
    aggregate,
    by_joint_and_rate,
    load_jsonl,
    metric_deltas,
    validate_result,
    validate_rows,
)


def test_protocol_resolves_exact_checkpoint_shas():
    protocol = load_protocol(DEFAULT_PROTOCOL)
    models = resolve_models(protocol)
    assert [model["id"] for model in models] == ["tf43000", "jt71500", "jt45000"]
    assert all(Path(model["checkpoint_path"]).is_file() for model in models)


def test_canonical_job_matrix_and_policy_modes_are_frozen():
    protocol = load_protocol(DEFAULT_PROTOCOL)
    models = resolve_models(protocol, ["tf43000", "jt71500"])
    jobs = build_jobs(protocol, models, ("fixed", "onset"))
    assert len(jobs) == 12
    commands = [command_for_job(job, Path("/tmp") / (job["name"] + ".jsonl")) for job in jobs]
    teacher = next(command for command in commands if "tf43000_seed1_fixed" in " ".join(command))
    student = next(command for command in commands if "jt71500_seed1_fixed" in " ".join(command))
    assert teacher[teacher.index("--policy-mode") + 1] == "teacher"
    assert student[student.index("--policy-mode") + 1] == "student"
    assert teacher[teacher.index("--rates") + 1 :] == ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]


def test_smoke_reduces_scope_without_changing_model_or_policy():
    protocol = load_protocol(DEFAULT_PROTOCOL)
    model = resolve_models(protocol, ["jt71500"])
    jobs = build_jobs(protocol, model, ("fixed", "onset"), smoke=True)
    assert len(jobs) == 2
    fixed = command_for_job(jobs[0], Path("/tmp/fixed.jsonl"))
    assert fixed[fixed.index("--steps") + 1] == "10"
    assert fixed[fixed.index("--replicates-per-condition") + 1] == "1"
    assert fixed[fixed.index("--policy-mode") + 1] == "student"


def test_unknown_model_and_mutated_sha_fail_closed():
    protocol = load_protocol(DEFAULT_PROTOCOL)
    with pytest.raises(ValueError, match="unknown model"):
        resolve_models(protocol, ["missing"])
    broken = copy.deepcopy(protocol)
    broken["models"][0]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA mismatch"):
        resolve_models(broken, ["tf43000"])


def test_evaluation_subprocess_ignores_user_site_packages():
    assert evaluation_environment()["PYTHONNOUSERSITE"] == "1"


def test_jsonl_contract_validation_and_aggregation():
    protocol = load_protocol(DEFAULT_PROTOCOL)
    model = protocol["models"][1]
    metadata = {
        "record_type": "metadata",
        "checkpoint_sha256": model["checkpoint_sha256"],
        "task": model["task"],
        "seed": 1,
        "policy_mode": "student",
        "command_x": protocol["command_x_mps"],
        "rates": protocol["fixed"]["degradation_rates"],
        "replicates_per_condition": protocol["fixed"]["replicates_per_condition"],
        "steps": protocol["fixed"]["steps"],
    }
    row = {
        "record_type": "robot",
        "degradation_rate": 1.0,
        "survived_full_horizon": True,
        "command_vx_rmse": 0.1,
        "yaw_rate_rmse": 0.2,
        "vertical_velocity_rms": 0.3,
        "contact_gated_foot_slip_mps": 0.4,
        "integrated_forward_progress_m": 8.0,
        "action_near_bound_rate": 0.5,
    }
    with tempfile.TemporaryDirectory(prefix="p5-summary-") as temporary:
        path = Path(temporary) / "result.jsonl"
        path.write_text(json.dumps(metadata) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
        loaded_metadata, rows = load_jsonl(path)
    validate_result(loaded_metadata, model, 1, "fixed", protocol["fixed"])
    summary = aggregate(rows, FIXED_BOOL, FIXED_MEAN)
    assert summary["survived_full_horizon"]["mean"] == 1.0
    assert summary["command_vx_rmse"] == 0.1


def test_delta_sign_convention_is_explicit():
    reference = {"survived_full_horizon": {"mean": 0.9}, "command_vx_rmse": 0.08}
    candidate = {"survived_full_horizon": {"mean": 0.95}, "command_vx_rmse": 0.10}
    delta = metric_deltas(reference, candidate, ("survived_full_horizon",), ("command_vx_rmse",))
    assert delta["survived_full_horizon"] == pytest.approx(0.05)
    assert delta["command_vx_rmse"] == pytest.approx(0.02)


def test_joint_rate_aggregation_preserves_failure_location():
    rows = [
        {"joint_index": 0, "joint_name": "FL_hip", "degradation_rate": 1.0, "survived_full_horizon": True},
        {"joint_index": 0, "joint_name": "FL_hip", "degradation_rate": 1.0, "survived_full_horizon": False},
        {"joint_index": 1, "joint_name": "FL_thigh", "degradation_rate": 1.0, "survived_full_horizon": True},
    ]
    result = by_joint_and_rate(rows, ("survived_full_horizon",), ())
    assert result["0"]["by_degradation_rate"]["1.0"]["survived_full_horizon"]["mean"] == 0.5
    assert result["1"]["by_degradation_rate"]["1.0"]["survived_full_horizon"]["mean"] == 1.0


def test_condition_grid_rejects_missing_duplicate_and_wrong_seed():
    config = {"degradation_rates": [0.0, 1.0], "replicates_per_condition": 1}
    rows = [
        {"seed": 1, "joint_index": joint, "degradation_rate": rate, "replicate": 0}
        for joint in range(12)
        for rate in config["degradation_rates"]
    ]
    validate_rows(rows, 1, "fixed", config)
    with pytest.raises(ValueError, match="incomplete"):
        validate_rows(rows[:-1], 1, "fixed", config)
    with pytest.raises(ValueError, match="duplicate"):
        validate_rows(rows + [rows[0]], 1, "fixed", config)
    bad_seed = copy.deepcopy(rows)
    bad_seed[0]["seed"] = 2
    with pytest.raises(ValueError, match="seed mismatch"):
        validate_rows(bad_seed, 1, "fixed", config)
