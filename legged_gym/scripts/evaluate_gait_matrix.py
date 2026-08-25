"""Launch fresh read-only evaluator processes and assemble the exact P1 matrix."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from legged_gym.evaluation.gait_metrics import (
    SCENARIO_COMMANDS,
    atomic_write_json,
    gate_cell,
    load_gate,
    sha256_file,
    verify_matrix_cells,
)


def assemble(input_dir, gate_path):
    gate_path = Path(gate_path).resolve()
    gate = load_gate(gate_path)
    gate_sha256 = sha256_file(gate_path)
    cells = []
    for path in sorted(Path(input_dir).glob("*.json")):
        if path.name == "matrix.json":
            continue
        cells.append(json.loads(path.read_text(encoding="utf-8")))
    verification = verify_matrix_cells(cells, gate, gate_sha256)
    gate_results = [gate_cell(cell, gate) for cell in cells] if verification["valid"] else []
    return {
        "schema_version": 1,
        "gate_sha256": gate_sha256,
        "cells": cells,
        "verification": verification,
        "passed": verification["valid"] and all(item["passed"] for item in gate_results),
        "cell_gate_results": gate_results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="a1_limping_base_v2")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--num_envs", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--load_run", required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--assemble-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.headless:
        raise ValueError("evaluate_gait_matrix requires --headless")
    gate = load_gate(args.gate)
    if args.num_envs != gate["num_envs"] or args.seeds != gate["seeds"]:
        raise ValueError("matrix CLI must exactly match gate num_envs and ordered seeds")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.assemble_only:
        for scenario in SCENARIO_COMMANDS:
            for seed in gate["seeds"]:
                output = output_dir / "{}-seed{}.json".format(scenario, seed)
                command = [
                    sys.executable,
                    "-m",
                    "legged_gym.scripts.evaluate_teacher",
                    "--task",
                    args.task,
                    "--headless",
                    "--sim_device",
                    args.sim_device,
                    "--rl_device",
                    args.rl_device,
                    "--num_envs",
                    str(args.num_envs),
                    "--seed",
                    str(seed),
                    "--load_run",
                    args.load_run,
                    "--checkpoint",
                    str(args.checkpoint),
                    "--scenario",
                    scenario,
                    "--gate",
                    str(Path(args.gate).resolve()),
                    "--output",
                    str(output),
                ]
                process = subprocess.run(command, text=True)
                if process.returncode != 0:
                    atomic_write_json(
                        output,
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "num_envs": args.num_envs,
                            "gate_sha256": sha256_file(args.gate),
                            "return_code": process.returncode,
                            "finite": False,
                            "protocol": gate["protocol"],
                        },
                    )
    matrix = assemble(output_dir, args.gate)
    atomic_write_json(output_dir / "matrix.json", matrix)
    print(json.dumps(matrix["verification"], indent=2, sort_keys=True))
    return 0 if matrix["verification"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
