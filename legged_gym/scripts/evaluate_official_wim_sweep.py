"""Fail-closed all-checkpoint orchestration for the frozen official WIM protocol."""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from legged_gym.evaluation.official_wim_metrics import wilson_lower_bound


MODEL_RE = re.compile(r"model_(\d+)\.pt")
DIRECT_EVALUATION_STATES = {"COMPLETED", "STOPPED_TREND", "STOPPED_OPERATOR"}
ANALYSIS_EVALUATION_STATES = {"ANALYZING", "READY"}
TERMINAL_ANALYSIS_ORIGIN_STATES = {"COMPLETED", "STOPPED_TREND"}
CELL_METRIC_KEYS = {
    "episodes", "successes", "success_rate", "base_contact_rate",
    "median_progress_ratio", "command_velocity_rmse", "finite",
    "along_track_progress_m", "time_to_cross_s", "mean_lateral_error_m",
    "mean_yaw_drift_rad", "mean_abs_roll_rad", "mean_abs_pitch_rad",
    "mean_base_height_m", "mean_foot_slip_mps", "duty_factors",
    "airborne_fraction", "two_contact_fraction", "diagonal_pair_fraction",
    "four_contact_fraction", "mean_raw_action_abs", "mean_applied_action_abs",
    "applied_action_near_bound_rate", "timeout_rate",
}
CELL_PROVENANCE_KEYS = {
    "checkpoint", "protocol_sha256", "resolved_config_sha256", "terrain",
    "terrain_config", "seed", "cell_index", "child_provenance",
}
CELL_SCALAR_METRIC_KEYS = CELL_METRIC_KEYS - {"episodes", "successes", "finite", "duty_factors"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _strict_origin_version_matches(state, allowed_origin_states):
    origin = state.get("analysis_origin")
    if not isinstance(origin, dict) or origin.get("state") not in allowed_origin_states:
        return False
    origin_version = origin.get("state_version")
    dispatched_version = state.get("analysis_dispatched_version")
    return (
        isinstance(origin_version, int)
        and not isinstance(origin_version, bool)
        and isinstance(dispatched_version, int)
        and not isinstance(dispatched_version, bool)
        and origin_version == dispatched_version
    )


def _successful_exit_matches(candidate, exit_status):
    return (
        isinstance(exit_status, dict)
        and isinstance(candidate, dict)
        and candidate == exit_status
        and isinstance(exit_status.get("return_code"), int)
        and not isinstance(exit_status.get("return_code"), bool)
        and exit_status.get("return_code") == 0
        and exit_status.get("received_signal") is None
    )


def simulator_evaluation_state_allowed(state, exit_status):
    current = state.get("state")
    if current in DIRECT_EVALUATION_STATES:
        return True
    if current not in ANALYSIS_EVALUATION_STATES:
        return False
    if _strict_origin_version_matches(state, TERMINAL_ANALYSIS_ORIGIN_STATES):
        return True
    if current != "READY" or not _strict_origin_version_matches(state, {"ERROR"}):
        return False
    run_id = state.get("run_id")
    if not isinstance(run_id, str) or not run_id or not _successful_exit_matches(state.get("exit"), exit_status):
        return False

    parent = state
    for _ in range(4):
        parent_version = parent.get("state_version")
        superseded = parent.get("superseded_readiness")
        if (
            not isinstance(parent_version, int)
            or isinstance(parent_version, bool)
            or parent_version <= 0
            or not isinstance(superseded, dict)
        ):
            return False
        record = superseded.get("record")
        if not isinstance(record, dict):
            return False
        record_version = record.get("state_version")
        if (
            record.get("state") != "READY"
            or record.get("run_id") != run_id
            or not isinstance(record_version, int)
            or isinstance(record_version, bool)
            or record_version <= 0
            or record_version >= parent_version
            or not _successful_exit_matches(record.get("exit"), exit_status)
        ):
            return False
        if _strict_origin_version_matches(record, TERMINAL_ANALYSIS_ORIGIN_STATES):
            return True
        if not _strict_origin_version_matches(record, {"ERROR"}):
            return False
        parent = record
    return False


def load_protocol(path):
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    choices = protocol.get("implementation_choice_agent_recommendation", {})
    if protocol.get("name") != "official_wim_a1_rough_selector_v1" or protocol.get("task") != "a1_official_wim_rough":
        raise ValueError("unexpected official WIM protocol")
    if len(choices.get("terrain_cells", [])) != 5 or choices.get("seeds") != [1, 2, 3] or choices.get("num_envs_per_cell") != 64:
        raise ValueError("official WIM protocol dimensions drifted")
    if choices.get("episode_horizon_seconds") != 20.0:
        raise ValueError("official WIM horizon drifted")
    return protocol


def inventory_checkpoints(run_dir):
    run_dir = Path(run_dir).resolve()
    malformed = sorted(path.name for path in run_dir.glob("model_*") if MODEL_RE.fullmatch(path.name) is None)
    if malformed:
        raise ValueError("malformed or temporary checkpoints: {}".format(malformed))
    inventory = []
    seen = set()
    for path in run_dir.iterdir():
        match = MODEL_RE.fullmatch(path.name)
        if match is None:
            continue
        iteration = int(match.group(1))
        if iteration in seen or path.stat().st_size == 0:
            raise ValueError("duplicate or empty checkpoint iteration {}".format(iteration))
        seen.add(iteration)
        inventory.append({"iteration": iteration, "name": path.name, "path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": _sha(path)})
    return sorted(inventory, key=lambda item: item["iteration"])


def checkpoint_eligibility(row, protocol):
    cells = row.get("terrain_cells", [])
    if len(cells) != 15 or row.get("finite") is not True:
        return {"eligible": False, "failures": ["finite_complete_15_cells"]}
    if any(not math.isfinite(float(value)) for cell in cells for value in cell.values() if isinstance(value, (int, float)) and not isinstance(value, bool)):
        return {"eligible": False, "failures": ["nonfinite"]}
    total = sum(cell["episodes"] for cell in cells)
    successes = sum(cell["successes"] for cell in cells)
    lower = wilson_lower_bound(successes, total)
    floors = {}
    for cell in cells:
        floors.setdefault(cell["terrain"], [0, 0])
        floors[cell["terrain"]][0] += cell["successes"]
        floors[cell["terrain"]][1] += cell["episodes"]
    limits = protocol["implementation_choice_agent_recommendation"]["eligibility"]
    failures = []
    if lower < limits["aggregate_wilson_lower_bound_min"]:
        failures.append("aggregate_wilson_lower_bound")
    if any(success / total_cell < limits["per_terrain_point_success_min"] for success, total_cell in floors.values()):
        failures.append("per_terrain_point_success")
    return {"eligible": not failures, "failures": failures, "aggregate_wilson_lower_bound": lower}


def selection_key(row):
    return (-row["aggregate_wilson_lower_bound"], row["base_contact_rate"], -row["median_progress_ratio"], row["command_velocity_rmse"], row["checkpoint"]["iteration"])


def verify_rows(rows, inventory, protocol_sha, resolved_sha):
    failures = []
    if len(rows) != len(inventory):
        failures.append("row_count")
    for row, item in zip(rows, inventory):
        if row.get("checkpoint") != item or row.get("protocol_sha256") != protocol_sha or row.get("resolved_config_sha256") != resolved_sha:
            failures.append("provenance_binding:{}".format(item["iteration"]))
    return {"valid": not failures, "failures": failures}


def expected_cell_identities(protocol):
    choices = protocol["implementation_choice_agent_recommendation"]
    return [
        (terrain["name"], seed)
        for terrain in choices["terrain_cells"]
        for seed in choices["seeds"]
    ]


def run_fresh_cell_process(args, checkpoint, terrain, seed, cell_index, output, protocol_sha, resolved_sha):
    command = [
        sys.executable, "-u", "-m", "legged_gym.scripts.evaluate_official_wim_checkpoint",
        "--task", args.task,
        "--checkpoint-path", checkpoint["path"],
        "--expected-checkpoint-sha256", checkpoint["sha256"],
        "--resolved-config", args.resolved_config,
        "--expected-resolved-config-sha256", resolved_sha,
        "--protocol", args.protocol,
        "--expected-protocol-sha256", protocol_sha,
        "--terrain", terrain["name"],
        "--seed", str(seed),
        "--cell-index", str(cell_index),
        "--output", str(output),
        "--headless", "--sim_device", args.sim_device, "--rl_device", args.rl_device,
    ]
    process = subprocess.run(
        command,
        shell=False,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return command, process


def require_cell_process_result(process, output, checkpoint, terrain, seed):
    if process.returncode != 0:
        raise RuntimeError("cell evaluator failed for {} {} seed {}".format(checkpoint["name"], terrain["name"], seed))
    if not output.is_file():
        raise RuntimeError("cell evaluator produced no output for {} {} seed {}".format(checkpoint["name"], terrain["name"], seed))
    return json.loads(output.read_text(encoding="utf-8"))


def _all_finite(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    return False


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_and_order_cells(cells, checkpoint, protocol, protocol_sha, resolved_sha, evaluator_sha=None):
    choices = protocol["implementation_choice_agent_recommendation"]
    terrains = {terrain["name"]: terrain for terrain in choices["terrain_cells"]}
    identities = expected_cell_identities(protocol)
    if len(cells) != len(identities):
        raise ValueError("checkpoint requires exactly 15 cell records")
    accepted = {}
    for cell in cells:
        if not isinstance(cell, dict) or set(cell) != CELL_METRIC_KEYS | CELL_PROVENANCE_KEYS:
            raise ValueError("cell schema mismatch")
        identity = (cell.get("terrain"), cell.get("seed"))
        if identity not in identities or identity in accepted:
            raise ValueError("duplicate, extra, or unknown cell identity")
        expected_index = identities.index(identity)
        if cell.get("cell_index") != expected_index or isinstance(cell.get("cell_index"), bool):
            raise ValueError("cell identity order provenance mismatch")
        if cell.get("terrain_config") != terrains[identity[0]]:
            raise ValueError("terrain configuration mismatch")
        if (
            cell.get("checkpoint") != checkpoint
            or cell.get("protocol_sha256") != protocol_sha
            or cell.get("resolved_config_sha256") != resolved_sha
        ):
            raise ValueError("cell provenance binding mismatch")
        provenance = cell.get("child_provenance")
        if (
            not isinstance(provenance, dict)
            or set(provenance) != {"pid", "argv", "evaluator_sha256"}
            or not isinstance(provenance.get("pid"), int)
            or isinstance(provenance.get("pid"), bool)
            or provenance["pid"] <= 0
            or not isinstance(provenance.get("argv"), list)
            or not provenance["argv"]
            or not isinstance(provenance.get("evaluator_sha256"), str)
            or len(provenance["evaluator_sha256"]) != 64
            or (evaluator_sha is not None and provenance["evaluator_sha256"] != evaluator_sha)
        ):
            raise ValueError("child provenance mismatch")
        argv = provenance["argv"]
        required_argv = {
            "--checkpoint-path": checkpoint["path"],
            "--expected-checkpoint-sha256": checkpoint["sha256"],
            "--expected-protocol-sha256": protocol_sha,
            "--expected-resolved-config-sha256": resolved_sha,
            "--terrain": identity[0],
            "--seed": str(identity[1]),
            "--cell-index": str(expected_index),
        }
        for flag, expected in required_argv.items():
            if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv) or argv[argv.index(flag) + 1] != expected:
                raise ValueError("child argv provenance mismatch")
        if (
            cell.get("episodes") != choices["num_envs_per_cell"]
            or isinstance(cell.get("episodes"), bool)
            or not isinstance(cell.get("successes"), int)
            or isinstance(cell.get("successes"), bool)
            or not 0 <= cell["successes"] <= cell["episodes"]
            or cell.get("finite") is not True
        ):
            raise ValueError("cell episode or finite contract mismatch")
        if (
            any(not _finite_number(cell[key]) for key in CELL_SCALAR_METRIC_KEYS)
            or not isinstance(cell.get("duty_factors"), list)
            or len(cell["duty_factors"]) != 4
            or any(not _finite_number(value) for value in cell["duty_factors"])
            or not _all_finite(cell)
        ):
            raise ValueError("cell contains malformed or nonfinite metrics")
        accepted[identity] = cell
    if set(accepted) != set(identities):
        raise ValueError("cell identity set is incomplete")
    return [accepted[identity] for identity in identities]


def aggregate_checkpoint_cells(checkpoint, cells, protocol_sha, resolved_sha):
    total = sum(cell["episodes"] for cell in cells)
    successes = sum(cell["successes"] for cell in cells)
    ordered = sorted(cell["median_progress_ratio"] for cell in cells)
    return {
        "checkpoint": checkpoint,
        "protocol_sha256": protocol_sha,
        "resolved_config_sha256": resolved_sha,
        "terrain_cells": cells,
        "success_rate": successes / total,
        "base_contact_rate": sum(cell["base_contact_rate"] * cell["episodes"] for cell in cells) / total,
        "median_progress_ratio": ordered[len(ordered) // 2],
        "command_velocity_rmse": sum(cell["command_velocity_rmse"] for cell in cells) / len(cells),
        "finite": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--sim-device", "--sim_device", dest="sim_device", default="cuda:0")
    parser.add_argument("--rl-device", "--rl_device", dest="rl_device", default="cuda:0")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    protocol = load_protocol(args.protocol)
    if args.task != protocol["task"]:
        raise ValueError("task does not match protocol")
    resolved = json.loads(Path(args.resolved_config).read_text(encoding="utf-8"))
    if resolved.get("task") != args.task or resolved.get("resume") is not False:
        raise ValueError("resolved configuration is incompatible")
    inventory = inventory_checkpoints(args.run_dir)
    protocol_sha, resolved_sha = _sha(args.protocol), _sha(args.resolved_config)
    if args.check_only:
        print(json.dumps({"valid": True, "protocol_sha256": protocol_sha, "resolved_config_sha256": resolved_sha, "checkpoint_count": len(inventory)}, sort_keys=True))
        return 0
    state = json.loads((Path(args.run_dir) / "state.json").read_text(encoding="utf-8"))
    exit_status = json.loads((Path(args.run_dir) / "exit_status.json").read_text(encoding="utf-8"))
    if not simulator_evaluation_state_allowed(state, exit_status):
        raise ValueError("simulator evaluation requires a terminal full run")
    if not args.headless or not args.output_dir:
        raise ValueError("evaluation requires headless mode and a new output directory")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    _atomic(output_dir / "inventory.json", {"before": inventory})
    rows = []
    evaluator_sha = _sha(Path(__file__).with_name("evaluate_official_wim_checkpoint.py"))
    for item in inventory:
        cells = []
        for cell_index, (terrain, seed) in enumerate(
            (terrain, seed)
            for terrain in protocol["implementation_choice_agent_recommendation"]["terrain_cells"]
            for seed in protocol["implementation_choice_agent_recommendation"]["seeds"]
        ):
            output = output_dir / "cells" / item["name"] / ("{:02d}_{}_seed{}.json".format(cell_index, terrain["name"], seed))
            command, process = run_fresh_cell_process(
                args, item, terrain, seed, cell_index, output, protocol_sha, resolved_sha
            )
            diagnostic = {
                "complete": process.returncode == 0 and output.is_file(),
                "argv": command,
                "return_code": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
                "checkpoint": item,
                "terrain": terrain,
                "seed": seed,
                "cell_index": cell_index,
            }
            _atomic(Path(str(output) + ".diagnostic.json"), diagnostic)
            cells.append(require_cell_process_result(process, output, item, terrain, seed))
        ordered_cells = validate_and_order_cells(
            cells, item, protocol, protocol_sha, resolved_sha, evaluator_sha
        )
        rows.append(aggregate_checkpoint_cells(item, ordered_cells, protocol_sha, resolved_sha))
    verification = verify_rows(rows, inventory, protocol_sha, resolved_sha)
    after = inventory_checkpoints(args.run_dir)
    if after != inventory or not verification["valid"]:
        raise RuntimeError("checkpoint inventory/provenance changed during evaluation")
    eligible = []
    for row in rows:
        decision = checkpoint_eligibility(row, protocol)
        row.update(decision)
        if decision["eligible"]:
            eligible.append(row)
    selected = min(eligible, key=selection_key)["checkpoint"] if eligible else None
    _atomic(output_dir / "rows.json", rows)
    _atomic(output_dir / "summary.json", {"selected_checkpoint": selected, "baseline_reproduced": selected is not None, "inventory_after": after})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
