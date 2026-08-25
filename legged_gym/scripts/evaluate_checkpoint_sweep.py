"""Fail-closed orchestration and frozen selection for BaseEnv checkpoints."""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path


MODEL_RE = re.compile(r"model_(\d+)\.pt")
ZERO_YAW_GROUPS = {"forward_025", "forward_050", "lateral_left", "lateral_right", "diagonal_left", "diagonal_right"}
ARC_GROUPS = {"arc_left", "arc_right"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def load_protocol(path):
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if protocol.get("name") != "p1_baseenv_checkpoint_selector_v1":
        raise ValueError("unexpected selector protocol name")
    if len(protocol.get("command_groups", [])) != 8:
        raise ValueError("selector protocol must define exactly eight command groups")
    if protocol.get("num_envs") != 512 or protocol.get("seed") != 1:
        raise ValueError("selector protocol num_envs/seed are not frozen")
    return protocol


def inventory_checkpoints(run_dir, expected_inventory=None):
    run_dir = Path(run_dir).resolve()
    temporary = sorted(path.name for path in run_dir.glob("model_*") if MODEL_RE.fullmatch(path.name) is None)
    if temporary:
        raise ValueError("temporary or malformed checkpoint files present: {}".format(temporary))
    entries = []
    seen = set()
    for path in run_dir.iterdir():
        match = MODEL_RE.fullmatch(path.name)
        if match is None:
            continue
        iteration = int(match.group(1))
        if iteration in seen:
            raise ValueError("duplicate numeric checkpoint iteration {}".format(iteration))
        seen.add(iteration)
        entries.append({"iteration": iteration, "file": path.name, "path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    entries.sort(key=lambda item: item["iteration"])
    if expected_inventory is not None:
        actual = [{key: item[key] for key in ("iteration", "file", "size_bytes", "sha256")} for item in entries]
        expected = [{key: item[key] for key in ("iteration", "file", "size_bytes", "sha256")} for item in expected_inventory]
        if actual != expected:
            raise ValueError("checkpoint inventory does not exactly match frozen protocol")
    return entries


def _finite(value):
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def canonical_numeric_row(row):
    """Return every numeric/boolean result while excluding strings and paths."""
    if isinstance(row, dict):
        return {key: canonical_numeric_row(value) for key, value in sorted(row.items()) if canonical_numeric_row(value) is not None}
    if isinstance(row, list):
        values = [canonical_numeric_row(value) for value in row]
        return [value for value in values if value is not None]
    if isinstance(row, (bool, int, float)):
        return row
    return None


def checkpoint_eligibility(row, protocol):
    failures = []
    groups = row.get("per_command", {})
    expected_names = [item["name"] for item in protocol["command_groups"]]
    if len(groups) != len(expected_names) or set(groups) != set(expected_names) or not row.get("finite", False) or not _finite(groups):
        return {"eligible": False, "failures": ["finite_and_complete_command_groups"]}
    limits = protocol["eligibility"]
    for name in expected_names:
        group = groups[name]
        metrics = group["metrics"]
        checks = (
            ("survival_rate", group["survival_rate"] >= limits["survival_rate_min"]),
            ("planar_command_rmse_mps.median", metrics["planar_command_rmse_mps"]["median"] <= limits["planar_command_rmse_mps_max"]["median"]),
            ("planar_command_rmse_mps.p90", metrics["planar_command_rmse_mps"]["p90"] <= limits["planar_command_rmse_mps_max"]["p90"]),
            ("four_feet_contact_rate.median", metrics["four_feet_contact_rate"]["median"] <= limits["four_feet_contact_rate_max"]["median"]),
            ("four_feet_contact_rate.p90", metrics["four_feet_contact_rate"]["p90"] <= limits["four_feet_contact_rate_max"]["p90"]),
            ("command_direction_progress_ratio.median", metrics["command_direction_progress_ratio"]["median"] >= limits["command_direction_progress_ratio_min"]["median"]),
            ("command_direction_progress_ratio.p10", metrics["command_direction_progress_ratio"]["p10"] >= limits["command_direction_progress_ratio_min"]["p10"]),
            ("applied_action_near_bound_rate.median", metrics["applied_action_near_bound_rate"]["median"] <= limits["applied_action_near_bound_rate_max"]["median"]),
            ("applied_action_near_bound_rate.p90", metrics["applied_action_near_bound_rate"]["p90"] <= limits["applied_action_near_bound_rate_max"]["p90"]),
        )
        for clause, passed in checks:
            if not passed:
                failures.append("{}:{}".format(name, clause))
        if name in ZERO_YAW_GROUPS:
            for quantile in ("median", "p10"):
                if metrics["path_efficiency"][quantile] < limits["zero_yaw_path_efficiency_min"][quantile]:
                    failures.append("{}:path_efficiency.{}".format(name, quantile))
        if name in ARC_GROUPS:
            for quantile in ("median", "p90"):
                if metrics["yaw_rate_rmse_radps"][quantile] > limits["arc_yaw_rate_rmse_radps_max"][quantile]:
                    failures.append("{}:yaw_rate_rmse_radps.{}".format(name, quantile))
            if group["correct_yaw_sign_rate"] < limits["arc_correct_yaw_sign_rate_min"]:
                failures.append("{}:correct_yaw_sign_rate".format(name))
    return {"eligible": not failures, "failures": failures}


def selection_key(row):
    groups = row["per_command"].values()
    return (
        max(group["metrics"]["planar_command_rmse_mps"]["p90"] for group in groups),
        max(group["metrics"]["yaw_rate_rmse_radps"]["p90"] for group in groups),
        max(group["metrics"]["four_feet_contact_rate"]["p90"] for group in groups),
        -min(group["survival_rate"] for group in groups),
        row["checkpoint"]["iteration"],
    )


def select_checkpoint(rows, protocol):
    decisions = []
    for row in rows:
        decision = checkpoint_eligibility(row, protocol)
        decisions.append({"iteration": row["checkpoint"]["iteration"], **decision})
    eligible = [row for row, decision in zip(rows, decisions) if decision["eligible"]]
    selected = min(eligible, key=selection_key) if eligible else None
    return decisions, (selected["checkpoint"] if selected is not None else None)


def verify_rows(rows, inventory, protocol_sha256, resolved_sha256):
    failures = []
    if len(rows) != len(inventory):
        failures.append("row_count")
    actual_iterations = [row.get("checkpoint", {}).get("iteration") for row in rows]
    if actual_iterations != [item["iteration"] for item in inventory] or len(set(actual_iterations)) != len(actual_iterations):
        failures.append("ordered_unique_iterations")
    provenance = None
    for row, item in zip(rows, inventory):
        if not row.get("finite", False) or not _finite(row):
            failures.append("nonfinite:{}".format(item["iteration"]))
        checkpoint = row.get("checkpoint", {})
        if checkpoint.get("path") != item["path"] or checkpoint.get("sha256") != item["sha256"] or checkpoint.get("size_bytes") != item["size_bytes"]:
            failures.append("checkpoint_binding:{}".format(item["iteration"]))
        if row.get("protocol", {}).get("sha256") != protocol_sha256 or row.get("resolved_config", {}).get("sha256") != resolved_sha256:
            failures.append("config_binding:{}".format(item["iteration"]))
        current = row.get("source_provenance")
        if provenance is None:
            provenance = current
        elif current != provenance:
            failures.append("source_mismatch:{}".format(item["iteration"]))
    return {"valid": not failures, "failures": failures}


def run_fresh_checkpoint_process(args, checkpoint, output):
    command = [sys.executable, "-u", "-m", "legged_gym.scripts.evaluate_checkpoint_selector", "--task", args.task, "--headless", "--sim_device", args.sim_device, "--rl_device", args.rl_device, "--seed", "1", "--num_envs", "512", "--checkpoint-path", checkpoint["path"], "--resolved-config", str(Path(args.resolved_config).resolve()), "--protocol", str(Path(args.protocol).resolve()), "--output", str(output)]
    return subprocess.run(command, text=True).returncode


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="a1_limping_base_v2")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    protocol_path = Path(args.protocol).resolve()
    resolved_path = Path(args.resolved_config).resolve()
    protocol = load_protocol(protocol_path)
    if args.task != protocol["task"]:
        raise ValueError("task does not match frozen protocol")
    inventory = inventory_checkpoints(args.run_dir, protocol["expected_inventory"])
    protocol_sha = sha256_file(protocol_path)
    resolved_sha = sha256_file(resolved_path)
    if args.check_only:
        print(json.dumps({"valid": True, "checkpoints": len(inventory), "protocol_sha256": protocol_sha, "resolved_config_sha256": resolved_sha}, sort_keys=True))
        return 0
    if not args.headless or not args.output_dir:
        raise ValueError("execution requires --headless and a new --output-dir")
    output_dir = Path(args.output_dir).resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise ValueError("output directory already exists: {}".format(output_dir))
    cells_dir = output_dir / "cells"
    cells_dir.mkdir()
    atomic_write_json(output_dir / "inventory.json", {"schema_version": 1, "before": inventory, "after": None})
    rows = []
    valid_execution = True
    for checkpoint in inventory:
        output = cells_dir / "model_{}.json".format(checkpoint["iteration"])
        return_code = run_fresh_checkpoint_process(args, checkpoint, output)
        if return_code != 0 or not output.is_file():
            valid_execution = False
            break
        rows.append(json.loads(output.read_text(encoding="utf-8")))
    after = inventory_checkpoints(args.run_dir, protocol["expected_inventory"])
    hashes_unchanged = inventory == after
    atomic_write_json(output_dir / "inventory.json", {"schema_version": 1, "before": inventory, "after": after, "hashes_unchanged": hashes_unchanged})
    verification = verify_rows(rows, inventory, protocol_sha, resolved_sha)
    valid = valid_execution and hashes_unchanged and verification["valid"]
    decisions, selected = select_checkpoint(rows, protocol) if valid else ([], None)
    with (output_dir / "rows.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    summary = {"schema_version": 1, "valid": valid, "verification": verification, "eligibility": decisions, "selected_checkpoint": selected, "protocol": {"path": str(protocol_path), "sha256": protocol_sha}, "resolved_config": {"path": str(resolved_path), "sha256": resolved_sha}}
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
