"""CPU tests for the precommitted P1 gait protocol and viewer separation."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from legged_gym.evaluation.gait_metrics import (
    SCENARIO_COMMANDS,
    ContactCycleAccumulator,
    atomic_write_json,
    gate_cell,
    heldout_command_sequence,
    load_gate,
    normalized_trajectory_rmse,
    quantile_summary,
    sha256_file,
    verify_matrix_cells,
)
from legged_gym.evaluation.viewer_paths import target_polyline, viewer_command
from legged_gym.scripts.evaluate_gait_matrix import assemble


ROOT = Path(__file__).resolve().parents[2]
GATE_PATH = ROOT / "legged_gym" / "evaluation" / "p1_normal_gait_gate.json"


def _cycle_result(trace, dt=0.02):
    trace = np.asarray(trace, dtype=float)[:, None, :] * 2.0
    accumulator = ContactCycleAccumulator(1, 4, dt)
    for contacts in trace:
        accumulator.update(contacts)
    return accumulator.result()


def _passing_cell(scenario, seed, gate):
    metrics = {}
    maximum_metrics = set(gate["thresholds"]["stand_max"])
    maximum_metrics.update(gate["thresholds"]["moving_max"])
    maximum_metrics.update(gate["thresholds"]["zero_yaw_max"])
    minimum_metrics = set(gate["thresholds"]["zero_yaw_min"])
    maximum_metrics.add("abs_integrated_yaw_error_rad")
    maximum_metrics.add("normalized_trajectory_rmse")
    for metric in maximum_metrics:
        metrics[metric] = {"median": 0.0, "p10": 0.0, "p90": 0.0}
    for metric in minimum_metrics:
        metrics[metric] = {"median": 1.0, "p10": 1.0, "p90": 1.0}
    gait = {
        "duty_factor_per_foot": [
            {"median": 0.60, "p10": 0.50, "p90": 0.70} for _ in range(4)
        ],
        "valid_swing_count_per_foot": [
            {"median": 8.0, "p10": 5.0, "p90": 10.0} for _ in range(4)
        ],
        "env_foot_duty_in_0p30_0p90_rate": 1.0,
        "four_foot_cycle_pass_rate": 1.0,
    }
    return {
        "scenario": scenario,
        "seed": seed,
        "num_envs": gate["num_envs"],
        "return_code": 0,
        "finite": True,
        "survival_rate": 1.0,
        "correct_yaw_sign_rate": 1.0,
        "metrics": metrics,
        "gait_cycles": gait,
        "protocol": gate["protocol"],
        "gate_sha256": sha256_file(GATE_PATH),
    }


class GaitMetricsTest(unittest.TestCase):
    def test_quantiles_are_population_values(self):
        summary = quantile_summary(np.arange(10, dtype=float))
        self.assertAlmostEqual(summary["median"], 4.5)
        self.assertAlmostEqual(summary["p10"], 0.9)
        self.assertAlmostEqual(summary["p90"], 8.1)

    def test_synthetic_normal_trace_passes_cycle_clauses(self):
        trace = []
        for step in range(1000):
            phase = (step // 25) % 2
            trace.append([phase == 0, phase == 1, phase == 1, phase == 0])
        result = _cycle_result(trace)
        self.assertEqual(result["four_foot_cycle_pass_rate"], 1.0)
        self.assertTrue(all(0.45 <= foot["median"] <= 0.85 for foot in result["duty_factor_per_foot"]))
        self.assertTrue(all(foot["p10"] >= 2 for foot in result["valid_swing_count_per_foot"]))

    def test_synthetic_stomp_drag_and_chatter_fail(self):
        stomp = _cycle_result([[1, 1, 1, 1]] * 300)
        self.assertEqual(stomp["four_foot_cycle_pass_rate"], 0.0)
        drag_trace = []
        chatter_trace = []
        for step in range(500):
            phase = (step // 25) % 2
            drag_trace.append([1, phase, not phase, phase])
            chatter_trace.append([step % 2, (step + 1) % 2, step % 2, (step + 1) % 2])
        self.assertEqual(_cycle_result(drag_trace)["four_foot_cycle_pass_rate"], 0.0)
        self.assertEqual(_cycle_result(chatter_trace)["four_foot_cycle_pass_rate"], 0.0)

    def test_gate_is_strict_per_cell(self):
        gate = load_gate(GATE_PATH)
        cell = _passing_cell("forward_nominal", 1, gate)
        self.assertTrue(gate_cell(cell, gate)["passed"])
        cell["metrics"]["foot_slip_cost"]["p90"] = 0.36
        self.assertEqual(gate_cell(cell, gate)["failures"], ["foot_slip_cost"])

    def test_matrix_requires_exact_cartesian_product_and_identical_metadata(self):
        gate = load_gate(GATE_PATH)
        cells = [
            _passing_cell(scenario, seed, gate)
            for scenario in SCENARIO_COMMANDS
            for seed in gate["seeds"]
        ]
        valid = verify_matrix_cells(cells, gate, sha256_file(GATE_PATH))
        self.assertTrue(valid["valid"])
        self.assertEqual(valid["cell_count"], 24)
        self.assertFalse(
            verify_matrix_cells(cells[:-1], gate, sha256_file(GATE_PATH))["valid"]
        )
        cells[-1]["protocol"] = {"warmup_seconds": 3.0}
        self.assertFalse(
            verify_matrix_cells(cells, gate, sha256_file(GATE_PATH))["valid"]
        )

    def test_cpu_assembler_reads_complete_existing_matrix_without_launching(self):
        gate = load_gate(GATE_PATH)
        with tempfile.TemporaryDirectory(prefix="gait-matrix-") as temporary:
            directory = Path(temporary)
            for scenario in SCENARIO_COMMANDS:
                for seed in gate["seeds"]:
                    atomic_write_json(
                        directory / "{}-seed{}.json".format(scenario, seed),
                        _passing_cell(scenario, seed, gate),
                    )
            matrix = assemble(directory, GATE_PATH)
        self.assertTrue(matrix["verification"]["valid"])
        self.assertTrue(matrix["passed"])
        self.assertEqual(matrix["verification"]["cell_count"], 24)

    def test_heldout_commands_are_deterministic_distinct_and_inside_training_envelope(self):
        generator = load_gate(GATE_PATH)["protocol"]["heldout_command_generator"]
        first = heldout_command_sequence(seed=1, num_envs=16, generator=generator)
        second = heldout_command_sequence(seed=1, num_envs=16, generator=generator)
        other_seed = heldout_command_sequence(seed=2, num_envs=16, generator=generator)
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, other_seed))
        self.assertGreater(len({tuple(row) for row in first[0]}), 8)
        self.assertTrue(np.all((first[..., 0] >= 0.0) & (first[..., 0] <= 0.8)))
        self.assertTrue(np.all(np.abs(first[..., 1]) <= 0.3))
        self.assertTrue(np.all(np.abs(first[..., 2]) <= 0.5))
        invalid = dict(generator)
        invalid["namespace"] = "training-seed"
        with self.assertRaisesRegex(ValueError, "namespace"):
            heldout_command_sequence(seed=1, num_envs=1, generator=invalid)

    def test_heldout_path_adherence_gate_has_synthetic_pass_and_fail(self):
        gate = load_gate(GATE_PATH)
        cell = _passing_cell("heldout_randomized", 1, gate)
        target = np.zeros((100, 4, 2), dtype=float)
        target[:, :, 0] = np.linspace(0.0, 4.0, 100)[:, None]
        exact = normalized_trajectory_rmse(target, target, np.full(4, 4.0))
        diverted = target.copy()
        diverted[:, :, 1] = np.linspace(0.0, 2.0, 100)[:, None]
        failed = normalized_trajectory_rmse(diverted, target, np.full(4, 4.0))
        self.assertTrue(np.allclose(exact, 0.0))
        self.assertTrue(np.all(failed > 0.25))
        cell["metrics"]["normalized_trajectory_rmse"] = {
            "median": 0.10,
            "p10": 0.05,
            "p90": 0.20,
        }
        self.assertTrue(gate_cell(cell, gate)["passed"])
        cell["metrics"]["normalized_trajectory_rmse"]["p90"] = 0.26
        self.assertIn("normalized_trajectory_rmse", gate_cell(cell, gate)["failures"])

    def test_gate_file_freezes_arc_and_every_gait_cycle_hard_clause(self):
        thresholds = load_gate(GATE_PATH)["thresholds"]
        self.assertEqual(thresholds["arc"]["correct_yaw_sign_rate_min"], 0.95)
        self.assertEqual(
            thresholds["arc"]["abs_integrated_yaw_error_rad_max"],
            {"median": 1.0, "p90": 1.5},
        )
        gait = thresholds["gait_cycle"]
        self.assertEqual(gait["valid_swing_duration_seconds"], {"min": 0.08, "max": 1.5})
        self.assertEqual(gait["env_foot_duty_range"], {"min": 0.3, "max": 0.9})
        self.assertEqual(gait["valid_swing_count_median_min"], 4)
        self.assertEqual(gait["valid_swing_count_p10_min"], 2)
        self.assertEqual(
            thresholds["heldout_randomized_max"]["normalized_trajectory_rmse"],
            {"median": 0.15, "p90": 0.25},
        )

    def test_atomic_json_has_no_temporary_residue(self):
        with tempfile.TemporaryDirectory(prefix="gait-atomic-") as temporary:
            path = Path(temporary) / "result.json"
            atomic_write_json(path, {"finite": True})
            self.assertEqual(json.loads(path.read_text()), {"finite": True})
            self.assertEqual(list(Path(temporary).glob("*.tmp-*")), [])

    def test_viewer_paths_are_distinct_seeded_and_within_command_envelope(self):
        paths = [target_polyline(index, seed=11) for index in range(8)]
        endpoints = {tuple(path[-1, :2].round(3)) for path in paths}
        self.assertEqual(len(endpoints), 8)
        self.assertTrue(all(np.isfinite(path).all() for path in paths))
        for index in range(8):
            for elapsed in (0.0, 5.0, 10.0, 15.0):
                vx, vy, yaw = viewer_command(index, elapsed, seed=11)
                self.assertTrue(0.0 <= vx <= 0.8)
                self.assertTrue(abs(vy) <= 0.3)
                self.assertTrue(abs(yaw) <= 0.5)

    def test_headless_training_and_viewer_sources_are_separated(self):
        evaluator = (ROOT / "legged_gym" / "scripts" / "evaluate_teacher.py").read_text()
        viewer = (ROOT / "legged_gym" / "scripts" / "view_teacher.py").read_text()
        runner = (ROOT / "legged_gym" / "learning" / "teacher_runner.py").read_text()
        for api in (
            "create_viewer(",
            "step_graphics(",
            "draw_viewer(",
            "sync_frame_time(",
            "poll_viewer_events(",
            "add_lines(",
            "clear_lines(",
        ):
            self.assertNotIn(api, evaluator)
        self.assertIn("add_lines(", viewer)
        self.assertIn("clear_lines(", viewer)
        self.assertIn("if args.num_envs is None:", viewer)
        self.assertIn("args.num_envs = 8", viewer)
        self.assertIn("_frame_all_robots(env, targets)", viewer)
        self.assertNotIn("legged_gym.evaluation", runner)
        self.assertNotIn("gait_metrics", runner)
        self.assertNotIn("viewer_paths", runner)


if __name__ == "__main__":
    unittest.main()
