"""Focused CPU/protocol tests for the frozen BaseEnv checkpoint selector."""

import copy
import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from legged_gym.scripts.evaluate_checkpoint_selector import (
    assert_resolved_contract,
    environment_clip,
)
import torch
from legged_gym.scripts.evaluate_checkpoint_sweep import (
    canonical_numeric_row,
    checkpoint_eligibility,
    inventory_checkpoints,
    load_protocol,
    select_checkpoint,
    verify_rows,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "legged_gym" / "evaluation" / "p1_checkpoint_selector.json"
RUN_DIR = ROOT / "logs" / "managed" / "p01-plane-teacher" / "wim-a1-flat-saving-teacher45-priv45" / "seed1-1500iter-4096env-fromscratch_20260811_131628_34b5e5"
RESOLVED_PATH = RUN_DIR / "resolved_config.json"


def passing_row(iteration=0):
    protocol = load_protocol(PROTOCOL_PATH)
    groups = {}
    for item in protocol["command_groups"]:
        groups[item["name"]] = {
            "command": item["command"],
            "num_envs": 64,
            "finite": True,
            "survival_rate": 1.0,
            "falls": 0,
            "correct_yaw_sign_rate": 1.0,
            "metrics": {
                "planar_command_rmse_mps": {"median": 0.10, "p10": 0.05, "p90": 0.20},
                "yaw_rate_rmse_radps": {"median": 0.08, "p10": 0.04, "p90": 0.16},
                "four_feet_contact_rate": {"median": 0.20, "p10": 0.10, "p90": 0.30},
                "command_direction_progress_ratio": {"median": 0.80, "p10": 0.70, "p90": 0.90},
                "path_efficiency": {"median": 0.90, "p10": 0.75, "p90": 0.95},
                "applied_action_near_bound_rate": {"median": 0.05, "p10": 0.01, "p90": 0.10},
            },
        }
    return {"finite": True, "checkpoint": {"iteration": iteration}, "per_command": groups}


def test_inventory_is_exact_complete_and_numeric_ordered():
    protocol = load_protocol(PROTOCOL_PATH)
    inventory = inventory_checkpoints(RUN_DIR, protocol["expected_inventory"])
    assert [item["iteration"] for item in inventory] == [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1185]
    assert len({item["sha256"] for item in inventory}) == 13


def test_inventory_rejects_missing_duplicate_temporary_and_mutated_files():
    with tempfile.TemporaryDirectory(prefix="selector-inventory-") as temporary:
        root = Path(temporary)
        (root / "model_0.pt").write_bytes(b"a")
        expected = [{"iteration": 0, "file": "model_0.pt", "size_bytes": 1, "sha256": hashlib.sha256(b"a").hexdigest()}]
        assert len(inventory_checkpoints(root, expected)) == 1
        (root / "model_0.pt.tmp-7").write_bytes(b"partial")
        with pytest.raises(ValueError, match="temporary"):
            inventory_checkpoints(root, expected)
        (root / "model_0.pt.tmp-7").unlink()
        (root / "model_00.pt").write_bytes(b"a")
        with pytest.raises(ValueError, match="duplicate"):
            inventory_checkpoints(root)
        (root / "model_00.pt").unlink()
        (root / "model_0.pt").write_bytes(b"b")
        with pytest.raises(ValueError, match="exactly match"):
            inventory_checkpoints(root, expected)
        (root / "model_0.pt").unlink()
        with pytest.raises(ValueError, match="exactly match"):
            inventory_checkpoints(root, expected)


def test_raw_mean_is_not_tanh_and_only_environment_clip_is_applied():
    raw = torch.tensor([[-4.0, -0.5, 0.5, 4.0]])
    applied = environment_clip(raw, 1.0)
    assert torch.equal(applied, torch.tensor([[-1.0, -0.5, 0.5, 1.0]]))
    assert not torch.equal(applied, torch.tanh(raw))
    source = (ROOT / "legged_gym" / "scripts" / "evaluate_checkpoint_selector.py").read_text(encoding="utf-8")
    assert "env.step(raw_actions)" in source
    assert "torch.tanh" not in source


def test_resolved_config_mismatch_fails_closed():
    protocol = load_protocol(PROTOCOL_PATH)
    resolved = json.loads(RESOLVED_PATH.read_text(encoding="utf-8"))
    assert assert_resolved_contract(resolved, protocol)
    broken = copy.deepcopy(resolved)
    broken["environment"]["normalization"]["clip_actions"] = 0.9
    with pytest.raises(ValueError, match="clip_actions mismatch"):
        assert_resolved_contract(broken, protocol)


@pytest.mark.parametrize(
    "metric,quantile,value",
    [("four_feet_contact_rate", "median", 0.26), ("command_direction_progress_ratio", "p10", 0.59)],
)
def test_passive_high_survival_row_is_rejected(metric, quantile, value):
    protocol = load_protocol(PROTOCOL_PATH)
    row = passing_row()
    row["per_command"]["forward_025"]["metrics"][metric][quantile] = value
    decision = checkpoint_eligibility(row, protocol)
    assert not decision["eligible"]
    assert any(metric in failure for failure in decision["failures"])


def test_command_group_identity_survives_sorted_json_round_trip():
    protocol = load_protocol(PROTOCOL_PATH)
    row = passing_row()
    expected_names = [item["name"] for item in protocol["command_groups"]]
    parsed = json.loads(json.dumps(row, sort_keys=True))
    assert list(parsed["per_command"]) != expected_names
    assert set(parsed["per_command"]) == set(expected_names)
    assert checkpoint_eligibility(parsed, protocol) == {"eligible": True, "failures": []}


def test_command_group_identity_rejects_missing_extra_and_nonfinite_data():
    protocol = load_protocol(PROTOCOL_PATH)
    malformed_rows = []

    missing = passing_row()
    missing["per_command"].pop("forward_025")
    malformed_rows.append(missing)

    extra = passing_row()
    extra["per_command"]["unexpected"] = copy.deepcopy(extra["per_command"]["forward_025"])
    malformed_rows.append(extra)

    finite_false = passing_row()
    finite_false["finite"] = False
    malformed_rows.append(finite_false)

    nonfinite = passing_row()
    nonfinite["per_command"]["forward_025"]["survival_rate"] = float("nan")
    malformed_rows.append(nonfinite)

    for row in malformed_rows:
        assert checkpoint_eligibility(row, protocol) == {
            "eligible": False,
            "failures": ["finite_and_complete_command_groups"],
        }


def test_frozen_lexicographic_ranking_and_lower_iteration_tie_break():
    protocol = load_protocol(PROTOCOL_PATH)
    later = passing_row(100)
    earlier = passing_row(0)
    decisions, selected = select_checkpoint([later, earlier], protocol)
    assert all(item["eligible"] for item in decisions)
    assert selected["iteration"] == 0
    earlier["per_command"]["forward_025"]["metrics"]["planar_command_rmse_mps"]["p90"] = 0.21
    _, selected = select_checkpoint([later, earlier], protocol)
    assert selected["iteration"] == 100


def test_verify_rows_rejects_missing_duplicate_and_source_mismatch():
    inventory = [{"iteration": 0, "path": "/x/model_0.pt", "size_bytes": 1, "sha256": "a"}]
    row = passing_row(0)
    row.update({"checkpoint": inventory[0], "protocol": {"sha256": "p"}, "resolved_config": {"sha256": "r"}, "source_provenance": {"sources": {"x": "y"}}})
    assert verify_rows([row], inventory, "p", "r")["valid"]
    assert not verify_rows([], inventory, "p", "r")["valid"]
    duplicate_inventory = inventory + [{"iteration": 0, "path": "/x/model_0b.pt", "size_bytes": 1, "sha256": "b"}]
    assert not verify_rows([row, row], duplicate_inventory, "p", "r")["valid"]
    second = copy.deepcopy(row)
    second["checkpoint"] = {"iteration": 1, "path": "/x/model_1.pt", "size_bytes": 1, "sha256": "b"}
    second["source_provenance"] = {"sources": {"x": "changed"}}
    inventory_two = inventory + [second["checkpoint"]]
    assert "source_mismatch:1" in verify_rows([row, second], inventory_two, "p", "r")["failures"]


def test_canonical_numeric_row_ignores_only_nonnumeric_provenance_fields():
    row = passing_row()
    row["checkpoint"]["path"] = "/tmp/a"
    first = canonical_numeric_row(row)
    row["checkpoint"]["path"] = "/tmp/b"
    assert first == canonical_numeric_row(row)
    row["per_command"]["forward_025"]["survival_rate"] = 0.5
    assert first != canonical_numeric_row(row)
