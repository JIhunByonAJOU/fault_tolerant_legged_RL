import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from legged_gym.evaluation.official_wim_metrics import base_contact_failure, crossed_finish_plane, wilson_lower_bound
from legged_gym.scripts.evaluate_official_wim_checkpoint import _acquire_rigid_body_states, _refresh_rigid_body_states, main as checkpoint_main
from legged_gym.scripts.evaluate_official_wim_sweep import (
    CELL_METRIC_KEYS,
    aggregate_checkpoint_cells,
    checkpoint_eligibility,
    expected_cell_identities,
    inventory_checkpoints,
    load_protocol,
    require_cell_process_result,
    run_fresh_cell_process,
    selection_key,
    simulator_evaluation_state_allowed,
    validate_and_order_cells,
)


PROTOCOL = Path(__file__).resolve().parents[1] / "evaluation/official_wim_a1_rough_selector_v1.json"


class OfficialWimSweepTest(unittest.TestCase):
    SUCCESSFUL_EXIT = {"finished_at": "2026-08-11T17:57:19+09:00", "received_signal": None, "return_code": 0}

    def test_selected_terrain_constructs_all_protocol_cells_fresh_process(self):
        protocol_bytes = PROTOCOL.read_bytes()
        protocol = json.loads(protocol_bytes)
        expected_protocol_sha256 = __import__("hashlib").sha256(protocol_bytes).hexdigest()
        child_source = r'''
import copy
import hashlib
import json
import sys

import isaacgym
import torch
import legged_gym.envs
from isaacgym import terrain_utils
from legged_gym.envs.a1_official_wim import A1OfficialWimRoughCfg
from legged_gym.utils import terrain as terrain_module
from legged_gym.utils.terrain import Terrain

cell = json.loads(sys.argv[1])
protocol_path = sys.argv[2]
expected_protocol_sha256 = sys.argv[3]
assert hashlib.sha256(open(protocol_path, "rb").read()).hexdigest() == expected_protocol_sha256

class TerrainArguments(dict):
    @property
    def terrain_kwargs(self):
        return self

cfg = copy.deepcopy(A1OfficialWimRoughCfg().terrain)
cfg.num_rows = 1
cfg.num_cols = 1
cfg.selected = True
cfg.curriculum = False
generator = cell["generator"]
arguments = copy.deepcopy(cell["arguments"])
calls = []

if generator == "pyramid_sloped_plus_random_uniform":
    def selected_generator(surface, **kwargs):
        calls.append({"generator": generator, "arguments": kwargs})
        terrain_utils.pyramid_sloped_terrain(surface, slope=kwargs["slope"], platform_size=kwargs["platform_size"])
        terrain_utils.random_uniform_terrain(surface, min_height=kwargs["min_height"], max_height=kwargs["max_height"], step=kwargs["step"], downsampled_scale=kwargs["downsampled_scale"])
    terrain_module.official_wim_rough_slope = selected_generator
    selected_name = "official_wim_rough_slope"
else:
    original_generator = getattr(terrain_utils, generator)
    def selected_generator(surface, _original=original_generator, **kwargs):
        calls.append({"generator": generator, "arguments": kwargs})
        return _original(surface, **kwargs)
    setattr(terrain_utils, generator, selected_generator)
    selected_name = "terrain_utils." + generator

cfg.terrain_kwargs = TerrainArguments(type=selected_name, **arguments)
subterrain_calls = []
original_subterrain = terrain_utils.SubTerrain
def recording_subterrain(*args, **kwargs):
    subterrain_calls.append(dict(kwargs))
    return original_subterrain(*args, **kwargs)
terrain_utils.SubTerrain = recording_subterrain

terrain = Terrain(cfg, 1)
expected_shape = (
    cfg.num_rows * int(cfg.terrain_length / cfg.horizontal_scale) + 2 * int(cfg.border_size / cfg.horizontal_scale),
    cfg.num_cols * int(cfg.terrain_width / cfg.horizontal_scale) + 2 * int(cfg.border_size / cfg.horizontal_scale),
)
assert len(subterrain_calls) == 1
assert subterrain_calls[0]["vertical_scale"] == 0.005 == cfg.vertical_scale
assert subterrain_calls[0]["horizontal_scale"] == 0.1 == cfg.horizontal_scale
assert calls == [{"generator": generator, "arguments": arguments}]
assert terrain.height_field_raw.shape == expected_shape
assert terrain.env_origins.shape == (1, 1, 3)
assert bool(torch.isfinite(torch.from_numpy(terrain.height_field_raw)).all())
assert bool(torch.isfinite(torch.from_numpy(terrain.env_origins)).all())
assert hashlib.sha256(open(protocol_path, "rb").read()).hexdigest() == expected_protocol_sha256
print(json.dumps({"name": cell["name"], "generator": generator, "arguments": arguments,
                  "scales": [cfg.vertical_scale, cfg.horizontal_scale],
                  "height_field_shape": list(terrain.height_field_raw.shape),
                  "env_origins_shape": list(terrain.env_origins.shape), "finite": True}, sort_keys=True))
'''
        records = []
        env = dict(os.environ)
        for cell in protocol["implementation_choice_agent_recommendation"]["terrain_cells"]:
            process = subprocess.run(
                [sys.executable, "-c", child_source, json.dumps(cell, sort_keys=True), str(PROTOCOL), expected_protocol_sha256],
                cwd=str(PROTOCOL.parents[2]), env=env, text=True, encoding="utf-8",
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(process.returncode, 0, {"cell": cell["name"], "stdout": process.stdout, "stderr": process.stderr})
            records.append(json.loads(process.stdout.strip().splitlines()[-1]))
        self.assertEqual([record["name"] for record in records],
                         [cell["name"] for cell in protocol["implementation_choice_agent_recommendation"]["terrain_cells"]])
        self.assertTrue(all(record["finite"] for record in records))
        self.assertEqual(__import__("hashlib").sha256(PROTOCOL.read_bytes()).hexdigest(), expected_protocol_sha256)

    def test_direct_completed_state_is_allowed(self):
        self.assertTrue(simulator_evaluation_state_allowed({"state": "COMPLETED"}, self.SUCCESSFUL_EXIT))

    def test_analysis_states_require_matching_completed_origin(self):
        for state in ("ANALYZING", "READY"):
            with self.subTest(state=state):
                self.assertTrue(simulator_evaluation_state_allowed({
                    "state": state,
                    "analysis_dispatched_version": 3,
                    "analysis_origin": {"state": "COMPLETED", "state_version": 3},
                }, self.SUCCESSFUL_EXIT))
        self.assertTrue(simulator_evaluation_state_allowed({
            "state": "READY",
            "analysis_dispatched_version": 3,
            "analysis_origin": {"state": "STOPPED_TREND", "state_version": 3},
        }, self.SUCCESSFUL_EXIT))

    def test_analysis_state_rejects_mismatched_origin_version(self):
        self.assertFalse(simulator_evaluation_state_allowed({
            "state": "ANALYZING",
            "analysis_dispatched_version": 4,
            "analysis_origin": {"state": "COMPLETED", "state_version": 3},
        }, self.SUCCESSFUL_EXIT))

    def test_analysis_state_rejects_absent_origin(self):
        self.assertFalse(simulator_evaluation_state_allowed({
            "state": "READY",
            "analysis_dispatched_version": 3,
        }, self.SUCCESSFUL_EXIT))

    def test_analysis_state_rejects_error_origin(self):
        for origin_state in ("ERROR", "BLOCKED", "STOPPED_OPERATOR"):
            with self.subTest(origin_state=origin_state):
                self.assertFalse(simulator_evaluation_state_allowed({
                    "state": "READY",
                    "analysis_dispatched_version": 3,
                    "analysis_origin": {"state": origin_state, "state_version": 3},
                }, self.SUCCESSFUL_EXIT))

    def test_superseded_ready_accepts_bounded_terminal_lineage_with_immutable_successful_exit(self):
        terminal_ready = {
            "state": "READY",
            "state_version": 7,
            "run_id": "run-1",
            "analysis_dispatched_version": 3,
            "analysis_origin": {"state": "COMPLETED", "state_version": 3},
            "exit": self.SUCCESSFUL_EXIT,
        }
        current_ready = {
            "state": "READY",
            "state_version": 15,
            "run_id": "run-1",
            "analysis_dispatched_version": 11,
            "analysis_origin": {"state": "ERROR", "state_version": 11},
            "exit": self.SUCCESSFUL_EXIT,
            "superseded_readiness": {"record": terminal_ready},
        }
        self.assertTrue(simulator_evaluation_state_allowed(current_ready, self.SUCCESSFUL_EXIT))

        post_reopen_ready = {
            "state": "READY",
            "state_version": 19,
            "run_id": "run-1",
            "analysis_dispatched_version": 17,
            "analysis_origin": {"state": "ERROR", "state_version": 17},
            "exit": self.SUCCESSFUL_EXIT,
            "superseded_readiness": {"record": current_ready},
        }
        self.assertTrue(simulator_evaluation_state_allowed(post_reopen_ready, self.SUCCESSFUL_EXIT))

        third_reopen_ready = {
            "state": "READY",
            "state_version": 23,
            "run_id": "run-1",
            "analysis_dispatched_version": 21,
            "analysis_origin": {"state": "ERROR", "state_version": 21},
            "exit": self.SUCCESSFUL_EXIT,
            "superseded_readiness": {"record": post_reopen_ready},
        }
        self.assertTrue(simulator_evaluation_state_allowed(third_reopen_ready, self.SUCCESSFUL_EXIT))

        fourth_reopen_ready = {
            "state": "READY",
            "state_version": 27,
            "run_id": "run-1",
            "analysis_dispatched_version": 25,
            "analysis_origin": {"state": "ERROR", "state_version": 25},
            "exit": self.SUCCESSFUL_EXIT,
            "superseded_readiness": {"record": third_reopen_ready},
        }
        self.assertTrue(simulator_evaluation_state_allowed(fourth_reopen_ready, self.SUCCESSFUL_EXIT))

    def test_superseded_ready_rejects_arbitrary_mismatched_and_malformed_lineage(self):
        import copy

        _DELETE = object()

        terminal_ready = {
            "state": "READY",
            "state_version": 7,
            "run_id": "run-1",
            "analysis_dispatched_version": 3,
            "analysis_origin": {"state": "COMPLETED", "state_version": 3},
            "exit": self.SUCCESSFUL_EXIT,
        }
        valid = {
            "state": "READY",
            "state_version": 15,
            "run_id": "run-1",
            "analysis_dispatched_version": 11,
            "analysis_origin": {"state": "ERROR", "state_version": 11},
            "exit": self.SUCCESSFUL_EXIT,
            "superseded_readiness": {"record": terminal_ready},
        }

        def mutate(path, value):
            candidate = copy.deepcopy(valid)
            target = candidate
            for key in path[:-1]:
                target = target[key]
            if value is _DELETE:
                del target[path[-1]]
            else:
                target[path[-1]] = value
            return candidate

        malformed_cases = {
            "root_arbitrary_error_version": mutate(("analysis_dispatched_version",), 12),
            "root_missing_run_id": mutate(("run_id",), _DELETE),
            "root_empty_run_id": mutate(("run_id",), ""),
            "root_bool_version": mutate(("state_version",), True),
            "absent_wrapper": mutate(("superseded_readiness",), _DELETE),
            "non_dict_wrapper": mutate(("superseded_readiness",), []),
            "absent_record": mutate(("superseded_readiness", "record"), _DELETE),
            "non_dict_record": mutate(("superseded_readiness", "record"), "READY"),
            "wrong_record_state": mutate(("superseded_readiness", "record", "state"), "ANALYZING"),
            "wrong_record_run_id": mutate(("superseded_readiness", "record", "run_id"), "run-2"),
            "bool_record_version": mutate(("superseded_readiness", "record", "state_version"), True),
            "nondecreasing_record_version": mutate(("superseded_readiness", "record", "state_version"), 15),
            "mismatched_origin_version": mutate(("superseded_readiness", "record", "analysis_dispatched_version"), 4),
            "bool_origin_version": mutate(("superseded_readiness", "record", "analysis_origin", "state_version"), True),
            "nonterminal_origin": mutate(("superseded_readiness", "record", "analysis_origin", "state"), "BLOCKED"),
            "missing_root_exit": mutate(("exit",), _DELETE),
            "non_dict_root_exit": mutate(("exit",), []),
            "different_root_exit": mutate(("exit", "finished_at"), "different"),
            "missing_record_exit": mutate(("superseded_readiness", "record", "exit"), _DELETE),
            "non_dict_record_exit": mutate(("superseded_readiness", "record", "exit"), None),
            "different_record_exit": mutate(("superseded_readiness", "record", "exit", "finished_at"), "different"),
        }
        for name, candidate in malformed_cases.items():
            with self.subTest(name=name):
                self.assertFalse(simulator_evaluation_state_allowed(candidate, self.SUCCESSFUL_EXIT))

        bad_exit_statuses = {
            "missing": None,
            "non_dict": [],
            "nonzero": {**self.SUCCESSFUL_EXIT, "return_code": 1},
            "bool_zero": {**self.SUCCESSFUL_EXIT, "return_code": False},
            "signaled": {**self.SUCCESSFUL_EXIT, "received_signal": 15},
            "different": {**self.SUCCESSFUL_EXIT, "finished_at": "different"},
        }
        for name, exit_status in bad_exit_statuses.items():
            with self.subTest(name=name):
                candidate = copy.deepcopy(valid)
                if isinstance(exit_status, dict) and name != "different":
                    candidate["exit"] = exit_status
                    candidate["superseded_readiness"]["record"]["exit"] = exit_status
                self.assertFalse(simulator_evaluation_state_allowed(candidate, exit_status))

        nested_valid = copy.deepcopy(valid)
        first = nested_valid["superseded_readiness"]["record"]
        first["analysis_origin"] = {"state": "ERROR", "state_version": 6}
        first["analysis_dispatched_version"] = 6
        second = copy.deepcopy(terminal_ready)
        second["state_version"] = 5
        first["superseded_readiness"] = {"record": second}
        nested_malformed = {
            "absent_nested_wrapper": copy.deepcopy(nested_valid),
            "non_dict_nested_wrapper": copy.deepcopy(nested_valid),
            "absent_nested_record": copy.deepcopy(nested_valid),
            "non_dict_nested_record": copy.deepcopy(nested_valid),
        }
        del nested_malformed["absent_nested_wrapper"]["superseded_readiness"]["record"]["superseded_readiness"]
        nested_malformed["non_dict_nested_wrapper"]["superseded_readiness"]["record"]["superseded_readiness"] = []
        del nested_malformed["absent_nested_record"]["superseded_readiness"]["record"]["superseded_readiness"]["record"]
        nested_malformed["non_dict_nested_record"]["superseded_readiness"]["record"]["superseded_readiness"]["record"] = None
        for name, candidate in nested_malformed.items():
            with self.subTest(name=name):
                self.assertFalse(simulator_evaluation_state_allowed(candidate, self.SUCCESSFUL_EXIT))

        error_only = copy.deepcopy(valid)
        error_record = error_only["superseded_readiness"]["record"]
        error_record["analysis_origin"] = {"state": "ERROR", "state_version": 3}
        error_record["superseded_readiness"] = {"record": copy.deepcopy(error_record)}
        error_record["superseded_readiness"]["record"]["state_version"] = 5
        self.assertFalse(simulator_evaluation_state_allowed(error_only, self.SUCCESSFUL_EXIT))

        terminal = copy.deepcopy(terminal_ready)
        terminal["state_version"] = 1
        terminal["analysis_origin"] = {"state": "COMPLETED", "state_version": 0}
        terminal["analysis_dispatched_version"] = 0
        overdepth = terminal
        for version in (2, 3, 4, 5, 6):
            overdepth = {
                "state": "READY", "state_version": version, "run_id": "run-1",
                "analysis_dispatched_version": version - 1,
                "analysis_origin": {"state": "ERROR", "state_version": version - 1},
                "exit": self.SUCCESSFUL_EXIT,
                "superseded_readiness": {"record": overdepth},
            }
        self.assertFalse(simulator_evaluation_state_allowed(overdepth, self.SUCCESSFUL_EXIT))

    def _fake_checkpoint_and_cells(self):
        protocol = load_protocol(PROTOCOL)
        protocol_sha = __import__("hashlib").sha256(PROTOCOL.read_bytes()).hexdigest()
        resolved_sha = "b" * 64
        evaluator_sha = "e" * 64
        checkpoint = {
            "iteration": 0, "name": "model_0.pt", "path": "/run/model_0.pt",
            "size_bytes": 123, "sha256": "c" * 64,
        }
        cells = []
        for index, (terrain_name, seed) in enumerate(expected_cell_identities(protocol)):
            terrain = next(item for item in protocol["implementation_choice_agent_recommendation"]["terrain_cells"] if item["name"] == terrain_name)
            metrics = {key: 0.0 for key in CELL_METRIC_KEYS}
            metrics.update({
                "episodes": 64, "successes": 32, "success_rate": 0.5,
                "base_contact_rate": 0.25, "median_progress_ratio": float(index),
                "command_velocity_rmse": 0.5, "finite": True,
                "duty_factors": [0.1, 0.2, 0.3, 0.4],
            })
            argv = [
                "child", "--checkpoint-path", checkpoint["path"],
                "--expected-checkpoint-sha256", checkpoint["sha256"],
                "--expected-protocol-sha256", protocol_sha,
                "--expected-resolved-config-sha256", resolved_sha,
                "--terrain", terrain_name, "--seed", str(seed),
                "--cell-index", str(index),
            ]
            metrics.update({
                "checkpoint": checkpoint, "protocol_sha256": protocol_sha,
                "resolved_config_sha256": resolved_sha, "terrain": terrain_name,
                "terrain_config": terrain, "seed": seed, "cell_index": index,
                "child_provenance": {"pid": 1000 + index, "argv": argv, "evaluator_sha256": evaluator_sha},
            })
            cells.append(metrics)
        return protocol, protocol_sha, resolved_sha, evaluator_sha, checkpoint, cells

    def test_exact_15_cell_provenance_and_deterministic_aggregation(self):
        protocol, protocol_sha, resolved_sha, evaluator_sha, checkpoint, cells = self._fake_checkpoint_and_cells()
        ordered = validate_and_order_cells(list(reversed(cells)), checkpoint, protocol, protocol_sha, resolved_sha, evaluator_sha)
        self.assertEqual([(cell["terrain"], cell["seed"]) for cell in ordered], expected_cell_identities(protocol))
        expected = aggregate_checkpoint_cells(checkpoint, cells, protocol_sha, resolved_sha)
        actual = aggregate_checkpoint_cells(checkpoint, ordered, protocol_sha, resolved_sha)
        self.assertEqual(actual, expected)
        self.assertEqual(sum(cell["episodes"] for cell in actual["terrain_cells"]), 960)

    def test_exact_15_cell_rejects_partial_duplicate_extra_and_bad_provenance(self):
        import copy

        protocol, protocol_sha, resolved_sha, evaluator_sha, checkpoint, cells = self._fake_checkpoint_and_cells()
        invalid = {
            "partial_14": cells[:-1],
            "duplicate": cells[:-1] + [cells[0]],
            "extra": cells + [cells[0]],
        }
        mutations = {
            "wrong_hash": ("protocol_sha256", "x" * 64),
            "foreign_checkpoint": ("checkpoint", {**checkpoint, "name": "model_1.pt"}),
            "invalid_seed": ("seed", 4),
            "invalid_terrain": ("terrain", "foreign"),
            "misordered_identity": ("cell_index", 14),
            "nonfinite": ("command_velocity_rmse", float("nan")),
            "wrong_evaluator": ("child_provenance", {**cells[0]["child_provenance"], "evaluator_sha256": "f" * 64}),
            "wrong_argv": ("child_provenance", {**cells[0]["child_provenance"], "argv": ["child"]}),
        }
        for name, (key, value) in mutations.items():
            candidate = copy.deepcopy(cells)
            candidate[0][key] = value
            invalid[name] = candidate
        for name, candidate in invalid.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_and_order_cells(candidate, checkpoint, protocol, protocol_sha, resolved_sha, evaluator_sha)

    def test_cell_process_launches_15_shell_false_in_frozen_order(self):
        import inspect
        import legged_gym.scripts.evaluate_official_wim_sweep as sweep_module

        protocol = load_protocol(PROTOCOL)
        checkpoint = {"path": "/run/model_0.pt", "sha256": "c" * 64}
        args = SimpleNamespace(
            task=protocol["task"], resolved_config="/run/resolved.json", protocol=str(PROTOCOL),
            sim_device="cuda:0", rl_device="cuda:0",
        )
        calls = []

        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with mock.patch("legged_gym.scripts.evaluate_official_wim_sweep.subprocess.run", side_effect=fake_run):
            for index, terrain in enumerate(protocol["implementation_choice_agent_recommendation"]["terrain_cells"]):
                for offset, seed in enumerate(protocol["implementation_choice_agent_recommendation"]["seeds"]):
                    cell_index = index * 3 + offset
                    run_fresh_cell_process(args, checkpoint, terrain, seed, cell_index, Path("/tmp/cell.json"), "p" * 64, "r" * 64)
        self.assertEqual(len(calls), 15)
        self.assertEqual(
            [(command[command.index("--terrain") + 1], int(command[command.index("--seed") + 1])) for command, _ in calls],
            expected_cell_identities(protocol),
        )
        self.assertTrue(all(kwargs["shell"] is False for _, kwargs in calls))
        self.assertEqual(len({tuple(command) for command, _ in calls}), 15)
        self.assertNotIn("evaluate_checkpoint_cell", inspect.getsource(sweep_module.main))

    def test_cell_process_single_cell_cli_rejects_duplicate_request(self):
        argv = [
            "--checkpoint-path", "c", "--resolved-config", "r", "--protocol", "p",
            "--terrain", "slope", "--terrain", "rough_slope", "--seed", "1",
            "--cell-index", "0", "--expected-checkpoint-sha256", "c" * 64,
            "--expected-protocol-sha256", "p" * 64,
            "--expected-resolved-config-sha256", "r" * 64, "--output", "o",
        ]
        with self.assertRaisesRegex(ValueError, "exactly once"):
            checkpoint_main(argv)

    def test_child_failure_and_missing_output_are_fail_closed(self):
        protocol = load_protocol(PROTOCOL)
        terrain = protocol["implementation_choice_agent_recommendation"]["terrain_cells"][0]
        args = SimpleNamespace(task=protocol["task"], resolved_config="r", protocol="p", sim_device="cuda:0", rl_device="cuda:0")
        checkpoint = {"name": "model_0.pt", "path": "c", "sha256": "c" * 64}
        with mock.patch("legged_gym.scripts.evaluate_official_wim_sweep.subprocess.run", return_value=SimpleNamespace(returncode=7, stdout="out", stderr="err")):
            _, process = run_fresh_cell_process(args, checkpoint, terrain, 1, 0, Path("missing"), "p" * 64, "r" * 64)
        with self.assertRaises(RuntimeError):
            require_cell_process_result(process, Path("missing"), checkpoint, terrain, 1)
        with self.assertRaises(RuntimeError):
            require_cell_process_result(SimpleNamespace(returncode=0), Path("missing"), checkpoint, terrain, 1)

    def test_rigid_body_binding_and_refresh_contract(self):
        import torch

        class FakeGym:
            def __init__(self, descriptor, on_refresh=None):
                self.descriptor = descriptor
                self.on_refresh = on_refresh
                self.acquire_count = 0
                self.refresh_count = 0

            def acquire_rigid_body_state_tensor(self, sim):
                self.acquire_count += 1
                return self.descriptor

            def refresh_rigid_body_state_tensor(self, sim):
                self.refresh_count += 1
                if self.on_refresh is not None:
                    self.on_refresh(self.descriptor, self.refresh_count)

        class FakeGymtorch:
            @staticmethod
            def wrap_tensor(descriptor):
                return descriptor

        class FakeEnv:
            num_envs = 2
            num_bodies = 5
            sim = object()
            feet_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)

            def __init__(self, descriptor, on_refresh=None):
                self.gym = FakeGym(descriptor, on_refresh)

        descriptor = torch.arange(2 * 5 * 13, dtype=torch.float32)
        env = FakeEnv(descriptor)
        states = _acquire_rigid_body_states(env, FakeGymtorch, torch)
        self.assertEqual(tuple(states.shape), (2, 5, 13))
        self.assertEqual(env.gym.acquire_count, 1)
        self.assertEqual(env.gym.refresh_count, 1)

        for bad_size in (2 * 5 * 13 - 1, 2 * 5 * 13 + 1):
            with self.subTest(bad_size=bad_size), self.assertRaises(RuntimeError):
                _acquire_rigid_body_states(FakeEnv(torch.zeros(bad_size)), FakeGymtorch, torch)

        for feet in (torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2, 5])):
            with self.subTest(feet=feet.tolist()), self.assertRaises(RuntimeError):
                bad_env = FakeEnv(torch.zeros(2 * 5 * 13))
                bad_env.feet_indices = feet
                _acquire_rigid_body_states(bad_env, FakeGymtorch, torch)

        for nonfinite in (float("nan"), float("inf")):
            with self.subTest(initial_nonfinite=nonfinite), self.assertRaises(RuntimeError):
                bad = torch.zeros(2 * 5 * 13)
                bad[0] = nonfinite
                _acquire_rigid_body_states(FakeEnv(bad), FakeGymtorch, torch)

        refreshed = torch.zeros(2 * 5 * 13)
        refreshed.view(2, 5, 13)[:, :4, 7] = float("nan")
        refresh_env = FakeEnv(
            refreshed,
            lambda value, count: value.view(2, 5, 13)[:, :4, 7].fill_(1.0),
        )
        refreshed_states = _acquire_rigid_body_states(refresh_env, FakeGymtorch, torch)
        self.assertTrue(bool(torch.isfinite(refreshed_states).all()))
        self.assertEqual(refresh_env.gym.refresh_count, 1)

        for nonfinite in (float("nan"), float("inf")):
            def inject_nonfinite(value, count, marker=nonfinite):
                if count == 2:
                    value.view(2, 5, 13)[0, 0, 7] = marker

            step_env = FakeEnv(torch.zeros(2 * 5 * 13), inject_nonfinite)
            step_states = _acquire_rigid_body_states(step_env, FakeGymtorch, torch)
            with self.subTest(step_nonfinite=nonfinite), self.assertRaises(RuntimeError):
                _refresh_rigid_body_states(step_env, step_states, torch, "test step")

    def test_two_cell_fresh_process_smoke(self):
        run_dir = os.environ.get("OFFICIAL_WIM_SMOKE_RUN_DIR")
        if not run_dir:
            self.skipTest("set OFFICIAL_WIM_SMOKE_RUN_DIR to run the simulator smoke")
        child_source = r'''
import hashlib
import json
import os
import sys
from pathlib import Path

import isaacgym
import torch
import legged_gym.envs
from isaacgym import gymtorch
from rsl_rl.runners import OnPolicyRunner
from legged_gym.scripts.evaluate_official_wim_checkpoint import (
    _acquire_rigid_body_states,
    _configure_frozen_terrain,
    _refresh_rigid_body_states,
    apply_episode_constant_commands,
)
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import class_to_dict, get_args

run_dir = Path(sys.argv[1]).resolve()
repo = Path(sys.argv[2]).resolve()
protocol_path = repo / "legged_gym/evaluation/official_wim_a1_rough_selector_v1.json"
resolved_path = run_dir / "resolved_config.json"
checkpoint = run_dir / "model_0.pt"
inventories = [
    repo / "logs/evaluations/official-wim-a1-rough-control" / (run_dir.name + "_sweep_v1/inventory.json"),
    repo / "logs/evaluations/official-wim-a1-rough-control" / (run_dir.name + "_sweep_v2/inventory.json"),
    repo / "logs/evaluations/official-wim-a1-rough-control" / (run_dir.name + "_sweep_v3/inventory.json"),
]
immutable = sorted(run_dir.glob("model_*.pt")) + [protocol_path, resolved_path] + inventories
before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in immutable}
sweep_v3 = repo / "logs/evaluations/official-wim-a1-rough-control" / (run_dir.name + "_sweep_v3")
assert sorted(path.name for path in sweep_v3.iterdir()) == ["inventory.json"]
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
terrain = protocol["implementation_choice_agent_recommendation"]["terrain_cells"][0]
seed = int(sys.argv[3])

sys.argv = ["official-wim-smoke", "--task", protocol["task"], "--headless", "--num_envs", "64", "--seed", str(seed)]
args = get_args()
env_cfg, train_cfg = task_registry.get_cfgs(args.task)
_configure_frozen_terrain(env_cfg, terrain)
env_cfg.commands.heading_command = False
env_cfg.commands.resampling_time = 1000.0
env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
runner = OnPolicyRunner(env, class_to_dict(train_cfg), log_dir=None, device=args.rl_device)
runner.load(str(checkpoint), load_optimizer=False)
policy = runner.get_inference_policy(device=env.device)
observations = env.get_observations()
assert tuple(observations.shape)[0] == 64 and bool(torch.isfinite(observations).all())
rigid_body_states = _acquire_rigid_body_states(env, gymtorch, torch)
apply_episode_constant_commands(env)
with torch.inference_mode():
    raw_actions = policy(observations)
assert bool(torch.isfinite(raw_actions).all())
observations, _, _, _, _ = env.step(raw_actions)
foot_velocity = _refresh_rigid_body_states(env, rigid_body_states, torch, "one-cell one-step smoke")
contacts = env.contact_forces[:, env.feet_indices, 2] > 1.0
slip = (torch.norm(foot_velocity[:, :, :2], dim=2) * contacts.float()).mean(dim=1)
assert tuple(rigid_body_states.shape) == (64, env.num_bodies, 13)
assert tuple(foot_velocity.shape) == (64, 4, 3)
assert bool(torch.isfinite(observations).all())
assert bool(torch.isfinite(rigid_body_states).all())
assert bool(torch.isfinite(foot_velocity).all())
assert bool(torch.isfinite(slip).all())
after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in immutable}
assert after == before
assert sorted(path.name for path in sweep_v3.iterdir()) == ["inventory.json"]
assert "wandb" not in sys.modules
print(json.dumps({
    "pid": os.getpid(), "seed": seed, "requested_cells": 1, "simulators": 1,
    "checkpoint_sha256": before[str(checkpoint)],
    "protocol_sha256": before[str(protocol_path)],
    "resolved_config_sha256": before[str(resolved_path)],
    "rigid_body_shape": list(rigid_body_states.shape),
    "foot_velocity_shape": list(foot_velocity.shape), "finite": True,
}, sort_keys=True))
'''
        records = []
        for seed in (1, 2):
            process = subprocess.run(
                [sys.executable, "-c", child_source, str(Path(run_dir).resolve()), str(PROTOCOL.parents[2]), str(seed)],
                cwd=str(PROTOCOL.parents[2]), env=dict(os.environ), text=True, encoding="utf-8",
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            combined = process.stdout + process.stderr
            self.assertNotIn("Foundation object exists already", combined)
            self.assertEqual(process.returncode, 0, {"seed": seed, "stdout": process.stdout, "stderr": process.stderr})
            records.append(json.loads(process.stdout.strip().splitlines()[-1]))
        self.assertEqual([record["seed"] for record in records], [1, 2])
        self.assertEqual(len({record["pid"] for record in records}), 2)
        self.assertTrue(all(record["pid"] > 0 for record in records))
        self.assertTrue(all(record["requested_cells"] == record["simulators"] == 1 for record in records))
        self.assertEqual(len({record["checkpoint_sha256"] for record in records}), 1)
        self.assertEqual(len({record["protocol_sha256"] for record in records}), 1)
        self.assertEqual(len({record["resolved_config_sha256"] for record in records}), 1)
        self.assertTrue(all(record["rigid_body_shape"][0] == 64 for record in records))
        self.assertTrue(all(record["foot_velocity_shape"] == [64, 4, 3] for record in records))
        self.assertTrue(all(record["finite"] for record in records))

    def test_arbitrary_nonterminal_state_is_rejected(self):
        self.assertFalse(simulator_evaluation_state_allowed({"state": "RUNNING"}, self.SUCCESSFUL_EXIT))

    def test_paper_success_and_wilson_boundaries(self):
        self.assertTrue(crossed_finish_plane(2.0, 10.0))
        self.assertTrue(base_contact_failure([0.0, 0.0, 1.1]))
        self.assertLess(wilson_lower_bound(80, 100), 0.8)
        self.assertGreater(wilson_lower_bound(95, 100), 0.8)

    def test_frozen_eligibility_and_tie_break(self):
        protocol = load_protocol(PROTOCOL)
        cells = [{"terrain": terrain["name"], "seed": seed, "episodes": 64, "successes": 64, "success_rate": 1.0} for terrain in protocol["implementation_choice_agent_recommendation"]["terrain_cells"] for seed in [1, 2, 3]]
        row = {"finite": True, "terrain_cells": cells, "aggregate_wilson_lower_bound": 0.99, "base_contact_rate": 0.0, "median_progress_ratio": 1.0, "command_velocity_rmse": 0.1, "checkpoint": {"iteration": 2}}
        self.assertTrue(checkpoint_eligibility(row, protocol)["eligible"])
        earlier = dict(row); earlier["checkpoint"] = {"iteration": 1}
        self.assertLess(selection_key(earlier), selection_key(row))

    def test_inventory_rejects_malformed_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "model_1.pt"; path.write_bytes(b"checkpoint")
            before = path.read_bytes()
            self.assertEqual(inventory_checkpoints(root)[0]["iteration"], 1)
            self.assertEqual(path.read_bytes(), before)
            (Path(root) / "model_latest.pt").write_bytes(b"bad")
            with self.assertRaises(ValueError): inventory_checkpoints(root)


if __name__ == "__main__":
    unittest.main()
