"""Stdlib-only smoke tests for custom agents and managed-run control flow."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legged_gym.harness.cli import _monitor_event, smoke_loop
import legged_gym.harness.manager as harness_manager
from legged_gym.harness.manager import (
    HarnessError,
    RunState,
    RunStore,
    _completion_artifact_error,
    _inject_managed_identity,
    _materialize_managed_identity,
    _validate_production_command,
    begin_verification,
    collect_run,
    finish_modification,
    launch_run,
    reopen_analysis,
    resume_blocked_analysis,
    route_analysis,
    stop_run,
    validate_p0_gate,
)

class AgentConfigurationTest(unittest.TestCase):
    def test_custom_agent_files_and_codex_parser(self):
        expected = {"tracker", "analysis", "modifier", "supervisor", "notion_reporter"}
        names = set()
        for name in expected:
            path = REPO_ROOT / ".codex" / "agents" / (name + ".toml")
            self.assertTrue(path.is_file(), str(path))
            text = path.read_text(encoding="utf-8")
            match = re.search(r'^name\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
            self.assertIsNotNone(match, str(path))
            self.assertEqual(match.group(1), name)
            self.assertIn('description = "', text)
            self.assertIn('developer_instructions = """', text)
            names.add(match.group(1))
        self.assertEqual(names, expected)

        codex = shutil.which("codex")
        if codex is None:
            self.skipTest("codex CLI is unavailable")
        process = subprocess.run(
            [codex, "debug", "prompt-input", "agent-config-smoke"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        lowered = process.stderr.lower()
        self.assertNotIn("ignoring malformed agent role definition", lowered)
        self.assertNotIn("deserialization", lowered)


class TrainingHarnessTest(unittest.TestCase):
    def _production_command(self, entrypoint, task, *extra):
        return [sys.executable, entrypoint, "--task", task, "--headless", *extra]

    def test_production_command_exact_allowlist(self):
        allowed = (
            ("legged_gym/scripts/train.py", "a1_limping_base"),
            ("legged_gym/scripts/train.py", "a1_limping_base_v2"),
            ("legged_gym/scripts/train.py", "a1_limping_base_wim"),
            ("legged_gym/scripts/train.py", "a1_official_wim_teacher243_failure"),
            ("legged_gym/scripts/train_official_wim_managed.py", "a1_official_wim_rough"),
        )
        for entrypoint, task in allowed:
            with self.subTest(entrypoint=entrypoint, task=task):
                _validate_production_command(
                    self._production_command(entrypoint, task), REPO_ROOT
                )
        _validate_production_command(
            self._production_command(
                str(REPO_ROOT / "legged_gym/scripts/train.py"), "a1_limping_base"
            ),
            REPO_ROOT,
        )

        rejected = (
            self._production_command("legged_gym/scripts/train.py", "a1_official_wim_rough"),
            self._production_command(
                "legged_gym/scripts/train_official_wim_managed.py", "a1_limping_base"
            ),
            self._production_command("legged_gym/scripts/train.py", "a1_limping_fake"),
            self._production_command("other/legged_gym/scripts/train.py", "a1_limping_base"),
            [sys.executable, "-m", "legged_gym.scripts.train", "--task", "a1_limping_base", "--headless"],
            [sys.executable, "legged_gym/scripts/train.py", "--headless"],
            [sys.executable, "legged_gym/scripts/train.py", "--task", "a1_limping_base", "--task=a1_limping_base", "--headless"],
            [sys.executable, "legged_gym/scripts/train.py", "--task", "a1_limping_base"],
            ["bash", "legged_gym/scripts/train.py", "--task", "a1_limping_base", "--headless"],
            [sys.executable, "legged_gym/scripts/train.py", "--task", "a1_limping_base", "--headless", "&&"],
            [sys.executable, "legged_gym/scripts/train.py", "legged_gym/scripts/train.py", "--task", "a1_limping_base", "--headless"],
        )
        for command in rejected:
            with self.subTest(command=command), self.assertRaises(HarnessError):
                _validate_production_command(command, REPO_ROOT)

    def test_managed_identity_materialization_and_injection(self):
        run_dir = (REPO_ROOT / "logs" / "managed" / "fixture-run").resolve()
        run_id = "fixture-run"
        cases = (
            (["--output", "{run_dir}", "--id", "{run_id}", "--name", "{run_name}"], ["--output", str(run_dir), "--id", run_id, "--name", run_id]),
            (["--log_dir={run_dir}", "--run_name={run_name}"], ["--log_dir=" + str(run_dir), "--run_name=" + run_id]),
        )
        for command, expected in cases:
            with self.subTest(command=command):
                self.assertEqual(
                    _materialize_managed_identity(command, run_dir, run_id), expected
                )

        injected = _inject_managed_identity(["train"], run_dir, run_id)
        self.assertEqual(injected, ["train", "--log_dir", str(run_dir), "--run_name", run_id])
        self.assertEqual(
            _inject_managed_identity(
                ["train", "--log_dir=" + str(run_dir), "--run_name", run_id],
                run_dir,
                run_id,
            ),
            ["train", "--log_dir=" + str(run_dir), "--run_name", run_id],
        )

        rejected_materialization = (
            ["{unknown}"],
            ["prefix-{run_id}"],
            ["--other={run_dir}"],
            ["unmatched{"],
        )
        for command in rejected_materialization:
            with self.subTest(command=command), self.assertRaises(HarnessError):
                _materialize_managed_identity(command, run_dir, run_id)

        rejected_identity = (
            ["train", "--log_dir"],
            ["train", "--run_name="],
            ["train", "--log_dir", str(run_dir), "--log_dir=" + str(run_dir)],
            ["train", "--run_name", run_id, "--run_name=" + run_id],
            ["train", "--log_dir", str(run_dir) + "-other"],
            ["train", "--run_name", run_id + "-other"],
        )
        for command in rejected_identity:
            with self.subTest(command=command), self.assertRaises(HarnessError):
                _inject_managed_identity(command, run_dir, run_id)

    def test_official_manifest_persists_exact_final_argv_without_process(self):
        with tempfile.TemporaryDirectory(prefix="harness-official-manifest-") as temporary:
            output_root = Path(temporary) / "managed"
            process = mock.Mock(pid=os.getpid())
            command = self._production_command(
                "legged_gym/scripts/train_official_wim_managed.py",
                "a1_official_wim_rough",
                "--log_dir={run_dir}",
                "--run_name={run_id}",
            )
            with mock.patch("legged_gym.harness.manager.gpu_snapshot", return_value={"available": True, "gpus": []}), mock.patch(
                "legged_gym.harness.manager.disk_snapshot", return_value={"free_gib": 100.0}
            ), mock.patch(
                "legged_gym.harness.manager._repo_snapshot", return_value={"commit": "fixture"}
            ), mock.patch("legged_gym.harness.manager.subprocess.Popen", return_value=process):
                launched = launch_run(
                    repo_root=REPO_ROOT,
                    phase="fixture",
                    experiment="official",
                    label="identity",
                    command=command,
                    authority="supervisor",
                    output_root=output_root,
                    require_rtx4090=False,
                    mock=False,
                )
            manifest = json.loads(
                (Path(launched["run_dir"]) / "manifest.json").read_text(encoding="utf-8")
            )
            final = manifest["command"]
            self.assertEqual(sum(token.endswith("train_official_wim_managed.py") for token in final), 1)
            self.assertEqual(final.count("--task"), 1)
            self.assertEqual(final[final.index("--task") + 1], "a1_official_wim_rough")
            log_dir = next(token.split("=", 1)[1] for token in final if token.startswith("--log_dir="))
            run_name = next(token.split("=", 1)[1] for token in final if token.startswith("--run_name="))
            self.assertEqual(sum(token.startswith("--log_dir=") for token in final), 1)
            self.assertEqual(sum(token.startswith("--run_name=") for token in final), 1)
            self.assertEqual(log_dir, manifest["run_dir"])
            self.assertEqual(run_name, manifest["run_id"])
            self.assertEqual(Path(log_dir).name, run_name)
            self.assertFalse(any("{" in token or "}" in token for token in final))

    def _analysis_plan(self, marker="fixture"):
        return {
            "analysis_mode": "incident",
            "observed_outcome": marker,
            "primary_hypothesis": "fixture hypothesis",
            "exact_proposed_changes": ["fixture change"],
            "expected_metric_response": "fixture response",
            "disconfirming_result": "fixture disconfirmation",
            "validation_commands": ["python3 -m unittest"],
            "next_run_budget": {"iterations": 1},
        }

    def _write_ready_reopen_fixture(self, run_dir):
        store = RunStore(run_dir)
        store.initialize({"run_id": "reopen-smoke"})
        store.transition(
            RunState.RUNNING, reason="smoke", expected_states={RunState.PREFLIGHT.value}
        )
        store.transition(
            RunState.COMPLETED, reason="smoke", expected_states={RunState.RUNNING.value}
        )
        store.claim_analysis()
        prior_plan_path = run_dir / "analysis_plan.json"
        prior_plan_path.write_text(json.dumps(self._analysis_plan("prior")), encoding="utf-8")
        route_analysis(run_dir, disposition="ready", authority="supervisor")
        ready = store.state()
        prior_digest = ready["analysis_plan"]["sha256"]
        prior_plan_path.write_text(json.dumps(self._analysis_plan("replacement")), encoding="utf-8")
        return store, ready, prior_digest

    def _write_blocked_resume_fixture(self, run_dir):
        store = RunStore(run_dir)
        store.initialize({"run_id": "resume-blocked-smoke"})
        store.transition(
            RunState.ERROR,
            reason="fixture incident",
            expected_states={RunState.PREFLIGHT.value},
        )
        store.claim_analysis()
        prior_plan_path = run_dir / "analysis_plan.json"
        prior_plan_path.write_text(json.dumps(self._analysis_plan("prior")), encoding="utf-8")
        prior_digest = hashlib.sha256(prior_plan_path.read_bytes()).hexdigest()
        blocked = store.transition(
            RunState.BLOCKED,
            reason="third recurrence requires explicit user decision",
            expected_states={RunState.ANALYZING.value},
            blocked_reason="third recurrence requires explicit user decision",
        )
        return store, blocked, prior_digest, prior_plan_path.read_bytes()

    def _write_active_fixture(self, run_dir):
        store = RunStore(run_dir)
        manifest = {
            "run_id": "liveness-fixture",
            "repo_root": str(REPO_ROOT),
            "run_dir": str(run_dir),
            "command": ["fixture"],
            "mock": True,
            "supervisor_pid": 424242,
            "supervisor_pgid": 424242,
            "supervisor_start_ticks": 12345,
            "boot_id": None,
        }
        store.initialize(manifest)
        store.transition(RunState.RUNNING, expected_states={RunState.PREFLIGHT.value})
        return store, manifest

    def _write_heartbeat(self, run_dir, manifest, wall_time):
        heartbeat = {
            "schema_version": 1,
            "run_id": manifest["run_id"],
            "supervisor_pid": manifest["supervisor_pid"],
            "supervisor_start_ticks": manifest["supervisor_start_ticks"],
            "train_pid": 424243,
            "updated_at": "fixture",
            "wall_time_unix": wall_time,
            "child_return_code": None,
        }
        (run_dir / "supervisor_heartbeat.json").write_text(
            json.dumps(heartbeat), encoding="utf-8"
        )

    def _write_p0_fixture(self, run_dir, authority="supervisor"):
        run_dir.mkdir(parents=True)
        manifest = {
            "run_id": "p0-fixture",
            "phase": "p00-harness",
            "mock": False,
            "authority": authority,
            "gpu_preflight": {"available": True, "gpus": [{"name": "NVIDIA GeForce RTX 4090"}]},
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run_dir / "exit_status.json").write_text(
            json.dumps({"return_code": 0, "received_signal": None}), encoding="utf-8"
        )
        (run_dir / "console.log").write_text(
            "+++ Using GPU PhysX\nGPU Pipeline: enabled\n", encoding="utf-8"
        )
        config = {
            "task": "a1_limping_base_v2",
            "environment": {
                "seed": 1,
                "env": {
                    "task_name": "a1_limping_base_v2",
                    "num_envs": 64,
                    "num_observations": 45,
                    "num_privileged_obs": 45,
                },
                "terrain": {"mesh_type": "plane", "measure_heights": False},
                "rewards": {"only_positive_rewards": True},
            },
            "schema": {
                "actor_observation_dim": 45,
                "privileged_observation_dim": 45,
                "teacher_latent_dim": 8,
                "actor_input_dim": 53,
                "critic_input_dim": 53,
            },
            "training": {
                "seed": 1,
                "runner": {
                    "num_steps_per_env": 24,
                    "max_iterations": 2,
                    "save_interval": 1,
                },
                "algorithm": {
                    "num_learning_epochs": 5,
                    "num_mini_batches": 4,
                    "entropy_coef": 0.01,
                    "desired_kl": 0.01,
                },
            },
        }
        (run_dir / "resolved_config.json").write_text(json.dumps(config), encoding="utf-8")
        metrics = []
        for iteration, transitions in ((0, 1536), (1, 3072)):
            metrics.append(
                {
                    "iteration": iteration,
                    "total_transitions": transitions,
                    "PPO/completed_updates": 20,
                    "PPO/planned_updates": 20,
                    "PPO/nonfinite_update_skipped": 0,
                    "PPO/hard_kl_stopped": 0,
                    "PPO/mean_kl": 0.01,
                    "PPO/max_kl": 0.068,
                    "Rollout/command_vx_rmse": 0.5,
                }
            )
        (run_dir / "metrics.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in metrics), encoding="utf-8"
        )
        for iteration in (0, 1, 2):
            (run_dir / "model_{}.pt".format(iteration)).write_bytes(b"checkpoint")
        return manifest, config, metrics

    def test_launch_rejects_non_supervisor_before_initialization(self):
        with tempfile.TemporaryDirectory(prefix="harness-authority-") as temporary:
            output_root = Path(temporary) / "managed"
            with self.assertRaisesRegex(HarnessError, "authority=supervisor"):
                launch_run(
                    repo_root=REPO_ROOT,
                    phase="p00-harness",
                    experiment="fixture",
                    label="leader",
                    command=["unused"],
                    authority="leader",
                    output_root=output_root,
                    require_rtx4090=False,
                    mock=True,
                )
            self.assertFalse(output_root.exists())

    def test_production_completion_artifacts(self):
        with tempfile.TemporaryDirectory(prefix="harness-artifacts-") as temporary:
            run_dir = Path(temporary) / "run"
            self._write_p0_fixture(run_dir)
            self.assertIsNone(_completion_artifact_error(run_dir, mock=False))
            for name, expected in (
                ("resolved_config.json", "resolved_config.json"),
                ("model_2.pt", "stable checkpoint"),
            ):
                with self.subTest(name=name):
                    path = run_dir / name
                    content = path.read_bytes()
                    path.unlink()
                    if name == "model_2.pt":
                        (run_dir / "model_0.pt").unlink()
                        (run_dir / "model_1.pt").unlink()
                    self.assertIn(expected, _completion_artifact_error(run_dir, mock=False))
                    if name == "model_2.pt":
                        (run_dir / "model_0.pt").write_bytes(b"checkpoint")
                        (run_dir / "model_1.pt").write_bytes(b"checkpoint")
                    path.write_bytes(content)
            (run_dir / "console.log").unlink()
            self.assertEqual(validate_p0_gate(run_dir)["failures"][0]["code"], "console_missing")

    def test_p0_gate_rejects_each_contract_violation(self):
        mutations = {
            "missing_console": lambda run_dir, manifest, config, metrics: (run_dir / "console.log").unlink(),
            "missing_final_checkpoint": lambda run_dir, manifest, config, metrics: (run_dir / "model_2.pt").unlink(),
            "temporary_checkpoint": lambda run_dir, manifest, config, metrics: (run_dir / "model_2.pt.tmp-fixture").write_bytes(b"temporary"),
            "malformed_iteration": lambda run_dir, manifest, config, metrics: metrics[1].update(iteration=2),
            "wrong_transition_budget": lambda run_dir, manifest, config, metrics: metrics[1].update(total_transitions=3071),
            "wrong_environment_observation_dim": lambda run_dir, manifest, config, metrics: config["environment"]["env"].update(num_observations=49),
            "missing_environment_observation_dim": lambda run_dir, manifest, config, metrics: config["environment"]["env"].pop("num_observations"),
            "wrong_actor_observation_dim": lambda run_dir, manifest, config, metrics: config["schema"].update(actor_observation_dim=49),
            "missing_actor_observation_dim": lambda run_dir, manifest, config, metrics: config["schema"].pop("actor_observation_dim"),
            "wrong_teacher_latent_dim": lambda run_dir, manifest, config, metrics: config["schema"].update(teacher_latent_dim=9),
            "missing_teacher_latent_dim": lambda run_dir, manifest, config, metrics: config["schema"].pop("teacher_latent_dim"),
            "wrong_actor_input_dim": lambda run_dir, manifest, config, metrics: config["schema"].update(actor_input_dim=57),
            "missing_actor_input_dim": lambda run_dir, manifest, config, metrics: config["schema"].pop("actor_input_dim"),
            "wrong_epoch_count": lambda run_dir, manifest, config, metrics: config["training"]["algorithm"].update(num_learning_epochs=4),
            "missing_epoch_count": lambda run_dir, manifest, config, metrics: config["training"]["algorithm"].pop("num_learning_epochs"),
            "wrong_planned_updates": lambda run_dir, manifest, config, metrics: metrics[0].update({"PPO/planned_updates": 39}),
            "wrong_completed_updates": lambda run_dir, manifest, config, metrics: metrics[0].update({"PPO/completed_updates": 39}),
            "missing_mean_kl": lambda run_dir, manifest, config, metrics: metrics[0].pop("PPO/mean_kl"),
            "nonfinite_mean_kl": lambda run_dir, manifest, config, metrics: metrics[0].update({"PPO/mean_kl": float("nan")}),
            "negative_mean_kl": lambda run_dir, manifest, config, metrics: metrics[0].update({"PPO/mean_kl": -0.01}),
            "missing_max_kl": lambda run_dir, manifest, config, metrics: metrics[0].pop("PPO/max_kl"),
            "nonfinite_max_kl": lambda run_dir, manifest, config, metrics: metrics[0].update({"PPO/max_kl": float("inf")}),
            "negative_max_kl": lambda run_dir, manifest, config, metrics: metrics[0].update({"PPO/max_kl": -0.01}),
            "nonfinite_history": lambda run_dir, manifest, config, metrics: metrics[0].update(loss=float("nan")),
            "skipped_update": lambda run_dir, manifest, config, metrics: metrics[1].update({"PPO/nonfinite_update_skipped": 1}),
            "nonzero_exit": lambda run_dir, manifest, config, metrics: (run_dir / "exit_status.json").write_text(json.dumps({"return_code": 1, "received_signal": None}), encoding="utf-8"),
            "signal_exit": lambda run_dir, manifest, config, metrics: (run_dir / "exit_status.json").write_text(json.dumps({"return_code": 0, "received_signal": 2}), encoding="utf-8"),
            "missing_gpu_console": lambda run_dir, manifest, config, metrics: (run_dir / "console.log").write_text("CPU pipeline\n", encoding="utf-8"),
            "hard_error_console": lambda run_dir, manifest, config, metrics: (run_dir / "console.log").write_text("Using GPU PhysX\nGPU Pipeline: enabled\nTraceback (most recent call last):\n", encoding="utf-8"),
            "legacy_authority": lambda run_dir, manifest, config, metrics: manifest.update(authority="leader"),
        }
        expected_codes = {
            "missing_console": "console_missing",
            "missing_final_checkpoint": "checkpoints_missing",
            "temporary_checkpoint": "temporary_checkpoint_present",
            "malformed_iteration": "metric_iterations_mismatch",
            "wrong_transition_budget": "metric_transitions_mismatch",
            "wrong_environment_observation_dim": "config_mismatch",
            "missing_environment_observation_dim": "config_mismatch",
            "wrong_actor_observation_dim": "config_mismatch",
            "missing_actor_observation_dim": "config_mismatch",
            "wrong_teacher_latent_dim": "config_mismatch",
            "missing_teacher_latent_dim": "config_mismatch",
            "wrong_actor_input_dim": "config_mismatch",
            "missing_actor_input_dim": "config_mismatch",
            "wrong_epoch_count": "config_mismatch",
            "missing_epoch_count": "config_mismatch",
            "wrong_planned_updates": "ppo_updates_mismatch",
            "wrong_completed_updates": "ppo_updates_mismatch",
            "missing_mean_kl": "ppo_mean_kl_invalid",
            "nonfinite_mean_kl": "ppo_mean_kl_invalid",
            "negative_mean_kl": "ppo_mean_kl_invalid",
            "missing_max_kl": "ppo_max_kl_invalid",
            "nonfinite_max_kl": "ppo_max_kl_invalid",
            "negative_max_kl": "ppo_max_kl_invalid",
            "nonfinite_history": "metric_nonfinite",
            "skipped_update": "ppo_nonfinite_update_skipped",
            "nonzero_exit": "exit_code_nonzero",
            "signal_exit": "exit_signal_present",
            "missing_gpu_console": "console_gpu_evidence_missing",
            "hard_error_console": "console_hard_error",
            "legacy_authority": "manifest_authority_mismatch",
        }
        with tempfile.TemporaryDirectory(prefix="harness-p0-") as temporary:
            valid_dir = Path(temporary) / "valid"
            self._write_p0_fixture(valid_dir)
            self.assertTrue(validate_p0_gate(valid_dir)["valid"])
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    run_dir = Path(temporary) / name
                    manifest, config, metrics = self._write_p0_fixture(run_dir)
                    mutate(run_dir, manifest, config, metrics)
                    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                    (run_dir / "resolved_config.json").write_text(json.dumps(config), encoding="utf-8")
                    (run_dir / "metrics.jsonl").write_text(
                        "".join(json.dumps(record) + "\n" for record in metrics), encoding="utf-8"
                    )
                    report = validate_p0_gate(run_dir)
                    self.assertFalse(report["valid"])
                    self.assertIn(expected_codes[name], {item["code"] for item in report["failures"]})

    def test_p0_gate_accepts_current_wim_contract_with_hard_kl_rail(self):
        with tempfile.TemporaryDirectory(prefix="harness-p0-wim-") as temporary:
            run_dir = Path(temporary) / "valid"
            _, config, metrics = self._write_p0_fixture(run_dir)
            self.assertEqual(config["training"]["algorithm"]["num_mini_batches"], 4)
            self.assertGreater(metrics[0]["PPO/max_kl"], 0.05)
            report = validate_p0_gate(run_dir)
            self.assertTrue(report["valid"], report["failures"])
            self.assertEqual(report["summary"]["failure_count"], 0)

    def test_post_run_gpu_telemetry_failure_is_nonfatal(self):
        with tempfile.TemporaryDirectory(prefix="harness-gpu-degraded-") as temporary:
            run_dir = Path(temporary) / "run"
            manifest, _, _ = self._write_p0_fixture(run_dir)
            store = RunStore(run_dir)
            (run_dir / "manifest.json").unlink()
            store.initialize(manifest)
            store.transition(RunState.RUNNING, expected_states={RunState.PREFLIGHT.value})
            (run_dir / "exit_status.json").write_text(
                json.dumps({"return_code": 0, "received_signal": None}), encoding="utf-8"
            )
            degraded = {"available": False, "error": "nvidia-smi exit 9", "gpus": []}
            with mock.patch("legged_gym.harness.manager.gpu_snapshot", return_value=degraded):
                snapshot = collect_run(run_dir)
            self.assertEqual(snapshot["state"]["state"], RunState.COMPLETED.value)
            report = validate_p0_gate(run_dir)
            self.assertTrue(report["valid"])
            self.assertIn(
                "post_run_gpu_telemetry_unavailable",
                {item["code"] for item in report["warnings"]},
            )

    def test_completion_error_and_trend_stop(self):
        with tempfile.TemporaryDirectory(prefix="harness-unittest-") as temporary:
            results = smoke_loop(temporary)
        self.assertEqual(results["normal"]["state"], RunState.COMPLETED.value)
        self.assertEqual(results["normal"]["exit"]["return_code"], 0)
        self.assertTrue(results["normal"]["analysis_requested"])

        self.assertEqual(results["error"]["state"], RunState.ERROR.value)
        self.assertNotEqual(results["error"]["exit"]["return_code"], 0)
        self.assertTrue(results["error"]["analysis_requested"])

        self.assertEqual(results["trend"]["state"], RunState.STOPPED_TREND.value)
        self.assertTrue(results["trend"]["trend"]["candidate_stop"])
        self.assertTrue(results["trend"]["analysis_requested"])
        self.assertIn(RunState.STOPPING.value, results["trend"]["events"])

    def test_false_proc_with_fresh_heartbeat_stays_running(self):
        with tempfile.TemporaryDirectory(prefix="harness-heartbeat-") as temporary:
            run_dir = Path(temporary) / "run"
            _, manifest = self._write_active_fixture(run_dir)
            self._write_heartbeat(run_dir, manifest, wall_time=100.0)
            with mock.patch("legged_gym.harness.manager._process_alive", return_value=False), mock.patch(
                "legged_gym.harness.manager.time.time", return_value=105.0
            ), mock.patch("legged_gym.harness.manager.gpu_snapshot", return_value={}):
                snapshot = collect_run(run_dir)
            self.assertEqual(snapshot["state"]["state"], RunState.RUNNING.value)
            self.assertFalse(snapshot["process_visible"])
            self.assertEqual(snapshot["heartbeat_age_seconds"], 5.0)
            self.assertIn(
                "supervisor_pid_unobservable",
                {item["kind"] for item in snapshot["alerts"]},
            )
            self.assertFalse((run_dir / "liveness_probe.json").exists())

    def test_stale_heartbeat_requires_repeated_grace_miss(self):
        with tempfile.TemporaryDirectory(prefix="harness-liveness-grace-") as temporary:
            run_dir = Path(temporary) / "run"
            store, manifest = self._write_active_fixture(run_dir)
            self._write_heartbeat(run_dir, manifest, wall_time=0.0)
            with mock.patch("legged_gym.harness.manager._process_alive", return_value=False), mock.patch(
                "legged_gym.harness.manager.gpu_snapshot", return_value={}
            ):
                with mock.patch("legged_gym.harness.manager.time.time", return_value=100.0):
                    first = collect_run(run_dir)
                probe = json.loads((run_dir / "liveness_probe.json").read_text(encoding="utf-8"))
                self.assertEqual(first["state"]["state"], RunState.RUNNING.value)
                self.assertEqual(probe["consecutive_misses"], 1)
                with mock.patch("legged_gym.harness.manager.time.time", return_value=116.0):
                    second = collect_run(run_dir)
            self.assertEqual(second["state"]["state"], RunState.ERROR.value)
            self.assertEqual(store.state()["last_error"]["kind"], "missing_supervisor")

    def test_transient_local_handle_timeout_does_not_break_collection(self):
        with tempfile.TemporaryDirectory(prefix="harness-local-handle-") as temporary:
            run_dir = Path(temporary) / "run"
            self._write_active_fixture(run_dir)
            handle = mock.Mock()
            handle.wait.side_effect = subprocess.TimeoutExpired("supervisor", 1)
            with mock.patch.dict(
                harness_manager._LOCAL_SUPERVISOR_HANDLES, {424242: handle}, clear=True
            ), mock.patch(
                "legged_gym.harness.manager._process_alive", return_value=False
            ), mock.patch(
                "legged_gym.harness.manager.time.time", return_value=100.0
            ), mock.patch("legged_gym.harness.manager.gpu_snapshot", return_value={}):
                snapshot = collect_run(run_dir)
                self.assertIs(harness_manager._LOCAL_SUPERVISOR_HANDLES[424242], handle)
            self.assertEqual(snapshot["state"]["state"], RunState.RUNNING.value)

    def test_rc0_exit_completes_when_proc_is_invisible(self):
        with tempfile.TemporaryDirectory(prefix="harness-invisible-exit-") as temporary:
            run_dir = Path(temporary) / "run"
            self._write_active_fixture(run_dir)
            (run_dir / "metrics.jsonl").write_text('{"iteration": 0}\n', encoding="utf-8")
            (run_dir / "exit_status.json").write_text(
                json.dumps({"return_code": 0, "received_signal": None}), encoding="utf-8"
            )
            with mock.patch("legged_gym.harness.manager._process_alive", return_value=False), mock.patch(
                "legged_gym.harness.manager.gpu_snapshot", return_value={}
            ):
                snapshot = collect_run(run_dir)
        self.assertEqual(snapshot["state"]["state"], RunState.COMPLETED.value)

    def test_stop_fails_closed_on_fresh_heartbeat_with_invisible_pid(self):
        with tempfile.TemporaryDirectory(prefix="harness-stop-domain-") as temporary:
            run_dir = Path(temporary) / "run"
            store, manifest = self._write_active_fixture(run_dir)
            self._write_heartbeat(run_dir, manifest, wall_time=time.time())
            with mock.patch("legged_gym.harness.manager._process_alive", return_value=False):
                with self.assertRaisesRegex(HarnessError, "execution-domain mismatch"):
                    stop_run(
                        run_dir,
                        kind="operator",
                        reason="fixture",
                        authority="supervisor",
                    )
            self.assertEqual(store.state()["state"], RunState.RUNNING.value)
            self.assertFalse((run_dir / "tracker_decision.json").exists())

    def test_tracker_cannot_operator_stop_or_bypass_trend_burn_in(self):
        with tempfile.TemporaryDirectory(prefix="harness-stop-authority-") as temporary:
            run_dir = Path(temporary) / "run"
            store, _ = self._write_active_fixture(run_dir)
            checkpoint = run_dir / "model_100.pt"
            checkpoint.write_bytes(b"checkpoint")
            (run_dir / "metrics.jsonl").write_text(
                json.dumps({"iteration": 100}) + "\n", encoding="utf-8"
            )
            decision = {
                "evidence_window": "fixture",
                "alternatives_considered": ["continue"],
                "confidence": 0.9,
                "continuation_cost": "fixture",
                "last_usable_checkpoint": str(checkpoint),
            }
            before = store.state()
            with self.assertRaisesRegex(HarnessError, "authority=supervisor"):
                stop_run(
                    run_dir,
                    kind="operator",
                    reason="fixture",
                    authority="tracker",
                    decision=decision,
                )
            with self.assertRaisesRegex(HarnessError, "before iteration 250"):
                stop_run(
                    run_dir,
                    kind="trend",
                    reason="fixture",
                    authority="tracker",
                    decision=decision,
                )
            self.assertEqual(store.state(), before)
            self.assertFalse((run_dir / "tracker_decision.json").exists())

    def test_event_driven_monitor_is_compact_and_deduplicated(self):
        with tempfile.TemporaryDirectory(prefix="harness-monitor-event-") as temporary:
            run_dir = Path(temporary)
            checkpoint = run_dir / "model_100.pt"
            checkpoint.write_bytes(b"checkpoint")
            snapshot = {
                "collected_at": "2026-08-11T01:00:00+09:00",
                "run_id": "monitor-fixture",
                "state": {"state": RunState.RUNNING.value, "state_version": 2},
                "alerts": [],
                "trend": {"candidate_stop": False, "latest_iteration": 100},
                "checkpoints": [
                    {"iteration": 100, "path": str(checkpoint), "size": checkpoint.stat().st_size}
                ],
            }
            first = _monitor_event(run_dir, snapshot, checkpoint_interval=100)
            duplicate = _monitor_event(run_dir, snapshot, checkpoint_interval=100)
            self.assertEqual(first["event_type"], "checkpoint")
            self.assertIsNone(duplicate)
            self.assertNotIn("latest_metrics", first)
            events = (run_dir / "monitor_events.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(events), 1)

            snapshot["collected_at"] = "2026-08-11T01:01:00+09:00"
            snapshot["state"] = {
                "state": RunState.COMPLETED.value,
                "state_version": 3,
            }
            terminal = _monitor_event(run_dir, snapshot, checkpoint_interval=100)
            self.assertEqual(terminal["event_type"], "terminal")
            self.assertEqual(terminal["state"], RunState.COMPLETED.value)

    def test_launch_cli_parent_exit_preserves_supervisor_heartbeat(self):
        with tempfile.TemporaryDirectory(prefix="harness-cli-parent-") as temporary:
            output_root = Path(temporary) / "managed"
            command = [
                sys.executable,
                "-m",
                "legged_gym.harness.cli",
                "launch",
                "--authority",
                "supervisor",
                "--phase",
                "p00-smoke",
                "--experiment",
                "parent-lifetime",
                "--label",
                "parent-exit",
                "--output-root",
                str(output_root),
                "--repo-root",
                str(REPO_ROOT),
                "--mock",
                "--",
                sys.executable,
                "-m",
                "legged_gym.harness.mock_training",
                "--run-dir",
                "{run_dir}",
                "--mode",
                "normal",
                "--steps",
                "30",
                "--delay",
                "0.05",
            ]
            launched_process = subprocess.run(
                command, cwd=str(REPO_ROOT), text=True, capture_output=True, timeout=10
            )
            self.assertEqual(launched_process.returncode, 0, launched_process.stderr)
            launched = json.loads(launched_process.stdout)
            run_dir = Path(launched["run_dir"])
            deadline = time.time() + 5.0
            while time.time() < deadline and not (run_dir / "supervisor_heartbeat.json").exists():
                time.sleep(0.05)
            self.assertTrue((run_dir / "supervisor_heartbeat.json").is_file())

            collect_command = [
                sys.executable,
                "-m",
                "legged_gym.harness.cli",
                "collect",
                "--run-dir",
                str(run_dir),
            ]
            first_collect = subprocess.run(
                collect_command, cwd=str(REPO_ROOT), text=True, capture_output=True, timeout=10
            )
            self.assertEqual(first_collect.returncode, 0, first_collect.stderr)
            first_snapshot = json.loads(first_collect.stdout)
            self.assertNotEqual(first_snapshot["state"]["state"], RunState.ERROR.value)
            deadline = time.time() + 10.0
            terminal = first_snapshot
            while time.time() < deadline and terminal["state"]["state"] == RunState.RUNNING.value:
                time.sleep(0.1)
                collected = subprocess.run(
                    collect_command, cwd=str(REPO_ROOT), text=True, capture_output=True, timeout=10
                )
                self.assertEqual(collected.returncode, 0, collected.stderr)
                terminal = json.loads(collected.stdout)
            self.assertEqual(terminal["state"]["state"], RunState.COMPLETED.value)
            events = (run_dir / "events.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("missing_supervisor", events)

    def test_analysis_claim_is_exactly_once(self):
        with tempfile.TemporaryDirectory(prefix="harness-claim-") as temporary:
            run_dir = Path(temporary) / "run"
            store = RunStore(run_dir)
            store.initialize({"run_id": "claim-smoke"})
            store.transition(
                RunState.ERROR,
                reason="smoke",
                expected_states={RunState.PREFLIGHT.value},
            )
            first = store.claim_analysis()
            second = store.claim_analysis()
        self.assertTrue(first["claimed"])
        self.assertEqual(first["mode"], "incident")
        self.assertEqual(first["state"]["state"], RunState.ANALYZING.value)
        self.assertFalse(second["claimed"])

    def test_reopen_analysis_is_supervisor_only_and_fails_without_mutation(self):
        with tempfile.TemporaryDirectory(prefix="harness-reopen-auth-") as temporary:
            run_dir = Path(temporary) / "run"
            store, ready, prior_digest = self._write_ready_reopen_fixture(run_dir)
            before_events = store.events_path.read_bytes()
            with self.assertRaisesRegex(HarnessError, "authority=supervisor"):
                reopen_analysis(run_dir, "modifier", 5, prior_digest, "invalidated")
            self.assertEqual(store.state(), ready)
            self.assertEqual(store.events_path.read_bytes(), before_events)

    def test_reopen_analysis_rejects_wrong_state_version_digest_and_duplicate(self):
        cases = (
            ("state", lambda store, ready, digest: store.transition(RunState.BLOCKED)),
            ("version", lambda store, ready, digest: None),
            ("digest", lambda store, ready, digest: None),
        )
        for name, mutate in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix="harness-reopen-cas-") as temporary:
                run_dir = Path(temporary) / "run"
                store, ready, prior_digest = self._write_ready_reopen_fixture(run_dir)
                mutate(store, ready, prior_digest)
                before = store.state()
                before_events = store.events_path.read_bytes()
                kwargs = {
                    "expected_state_version": 4 if name == "version" else 5,
                    "expected_plan_sha256": "0" * 64 if name == "digest" else prior_digest,
                }
                with self.assertRaisesRegex(HarnessError, "mismatch"):
                    reopen_analysis(run_dir, "supervisor", reason="invalidated", **kwargs)
                self.assertEqual(store.state(), before)
                self.assertEqual(store.events_path.read_bytes(), before_events)

        with tempfile.TemporaryDirectory(prefix="harness-reopen-duplicate-") as temporary:
            run_dir = Path(temporary) / "run"
            store, _, prior_digest = self._write_ready_reopen_fixture(run_dir)
            reopen_analysis(run_dir, "supervisor", 5, prior_digest, "invalidated")
            after_first = store.state()
            events_after_first = store.events_path.read_bytes()
            with self.assertRaisesRegex(HarnessError, "expected READY"):
                reopen_analysis(run_dir, "supervisor", 5, prior_digest, "invalidated")
            self.assertEqual(store.state(), after_first)
            self.assertEqual(store.events_path.read_bytes(), events_after_first)

    def test_reopen_analysis_rejects_invalid_or_unchanged_replacement_plan(self):
        with tempfile.TemporaryDirectory(prefix="harness-reopen-plan-") as temporary:
            run_dir = Path(temporary) / "run"
            store, ready, prior_digest = self._write_ready_reopen_fixture(run_dir)
            replacement = run_dir / "analysis_plan.json"
            replacement.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(HarnessError, "Missing or invalid"):
                reopen_analysis(run_dir, "supervisor", 5, prior_digest, "invalidated")
            replacement.write_text(json.dumps(self._analysis_plan("prior")), encoding="utf-8")
            with self.assertRaisesRegex(HarnessError, "must differ"):
                reopen_analysis(run_dir, "supervisor", 5, prior_digest, "invalidated")
            self.assertEqual(store.state(), ready)

    def test_reopen_analysis_preserves_claim_fields_and_appends_one_monotonic_event(self):
        with tempfile.TemporaryDirectory(prefix="harness-reopen-preserve-") as temporary:
            run_dir = Path(temporary) / "run"
            store, ready, prior_digest = self._write_ready_reopen_fixture(run_dir)
            claim_fields = {
                key: ready[key]
                for key in ("analysis_claimed_at", "analysis_dispatched_version", "analysis_origin")
            }
            events_before = store.events_path.read_text(encoding="utf-8").splitlines()
            reopened = reopen_analysis(run_dir, "supervisor", 5, prior_digest, "normal gait failed")
            events_after = store.events_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(reopened["state"], RunState.ANALYZING.value)
            self.assertEqual(reopened["state_version"], 6)
            self.assertFalse(reopened["analysis_requested"])
            self.assertEqual({key: reopened[key] for key in claim_fields}, claim_fields)
            self.assertEqual(reopened["superseded_readiness"]["record"], ready)
            self.assertEqual(reopened["superseded_readiness"]["reason"], "normal gait failed")
            self.assertEqual(len(events_after), len(events_before) + 1)
            versions = [json.loads(line)["version"] for line in events_after]
            self.assertEqual(versions, list(range(1, 7)))

    def test_resume_blocked_analysis_preserves_plan_and_blocked_audit(self):
        with tempfile.TemporaryDirectory(prefix="harness-resume-blocked-") as temporary:
            run_dir = Path(temporary) / "run"
            store, blocked, prior_digest, prior_bytes = self._write_blocked_resume_fixture(run_dir)
            events_before = store.events_path.read_text(encoding="utf-8").splitlines()
            resumed = resume_blocked_analysis(
                run_dir,
                "supervisor",
                blocked["state_version"],
                prior_digest,
                "user approved a materially different P1 BaseEnv scope",
            )
            archive_path = run_dir / "analysis-history" / (prior_digest + ".json")
            events_after = store.events_path.read_text(encoding="utf-8").splitlines()

            self.assertEqual(resumed["state"], RunState.ANALYZING.value)
            self.assertEqual(resumed["state_version"], blocked["state_version"] + 1)
            self.assertFalse(resumed["analysis_requested"])
            self.assertEqual(archive_path.read_bytes(), prior_bytes)
            self.assertEqual(oct(archive_path.stat().st_mode & 0o777), "0o444")
            audit = resumed["blocked_analysis_resume"]
            self.assertEqual(audit["prior_state_record"], blocked)
            self.assertEqual(audit["prior_blocked_reason"], blocked["blocked_reason"])
            self.assertEqual(audit["prior_state"], RunState.BLOCKED.value)
            self.assertEqual(audit["prior_state_version"], blocked["state_version"])
            self.assertEqual(audit["prior_analysis_plan"]["sha256"], prior_digest)
            self.assertEqual(audit["prior_analysis_plan"]["archive_path"], str(archive_path))
            self.assertEqual(len(events_after), len(events_before) + 1)
            versions = [json.loads(line)["version"] for line in events_after]
            self.assertEqual(versions, list(range(1, resumed["state_version"] + 1)))

    def test_resume_blocked_analysis_invalid_attempts_are_byte_exact_noops(self):
        cases = (
            ("wrong-authority", "modifier", 0, None, "new scope", "authority=supervisor"),
            ("stale-version", "supervisor", -1, None, "new scope", "version mismatch"),
            ("wrong-digest", "supervisor", 0, "0" * 64, "new scope", "digest mismatch"),
            ("empty-scope", "supervisor", 0, None, "  ", "non-empty scope"),
        )
        for name, authority, version_delta, digest_override, scope, error in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory(
                prefix="harness-resume-blocked-noop-"
            ) as temporary:
                run_dir = Path(temporary) / "run"
                store, blocked, prior_digest, prior_bytes = self._write_blocked_resume_fixture(run_dir)
                before_state = store.state_path.read_bytes()
                before_events = store.events_path.read_bytes()
                with self.assertRaisesRegex(HarnessError, error):
                    resume_blocked_analysis(
                        run_dir,
                        authority,
                        blocked["state_version"] + version_delta,
                        digest_override or prior_digest,
                        scope,
                    )
                self.assertEqual(store.state_path.read_bytes(), before_state)
                self.assertEqual(store.events_path.read_bytes(), before_events)
                self.assertEqual((run_dir / "analysis_plan.json").read_bytes(), prior_bytes)
                self.assertFalse((run_dir / "analysis-history").exists())
                self.assertEqual(store.state(), blocked)

    def test_resume_blocked_analysis_replay_is_a_byte_exact_noop(self):
        with tempfile.TemporaryDirectory(prefix="harness-resume-blocked-replay-") as temporary:
            run_dir = Path(temporary) / "run"
            store, blocked, prior_digest, _ = self._write_blocked_resume_fixture(run_dir)
            resume_blocked_analysis(
                run_dir, "supervisor", blocked["state_version"], prior_digest, "new scope"
            )
            before_state = store.state_path.read_bytes()
            before_events = store.events_path.read_bytes()
            archive_path = run_dir / "analysis-history" / (prior_digest + ".json")
            before_archive = archive_path.read_bytes()
            with self.assertRaisesRegex(HarnessError, "expected BLOCKED"):
                resume_blocked_analysis(
                    run_dir, "supervisor", blocked["state_version"], prior_digest, "new scope"
                )
            self.assertEqual(store.state_path.read_bytes(), before_state)
            self.assertEqual(store.events_path.read_bytes(), before_events)
            self.assertEqual(archive_path.read_bytes(), before_archive)

    def test_resume_blocked_analysis_cli_success_and_required_parser_arguments(self):
        with tempfile.TemporaryDirectory(prefix="harness-resume-blocked-cli-") as temporary:
            run_dir = Path(temporary) / "run"
            store, blocked, prior_digest, _ = self._write_blocked_resume_fixture(run_dir)
            command = [
                sys.executable,
                "-m",
                "legged_gym.harness.cli",
                "resume-blocked-analysis",
                "--run-dir",
                str(run_dir),
                "--authority",
                "supervisor",
                "--expected-state-version",
                str(blocked["state_version"]),
                "--expected-plan-sha256",
                prior_digest,
                "--scope-decision",
                "user approved the replacement scope",
            ]
            process = subprocess.run(
                command, cwd=str(REPO_ROOT), text=True, capture_output=True, timeout=10
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(json.loads(process.stdout)["state"], RunState.ANALYZING.value)
            self.assertEqual(store.state()["state"], RunState.ANALYZING.value)

            help_process = subprocess.run(
                command[:4] + ["--help"],
                cwd=str(REPO_ROOT),
                text=True,
                capture_output=True,
                timeout=10,
            )
            self.assertEqual(help_process.returncode, 0, help_process.stderr)
            for option in (
                "--authority",
                "--expected-state-version",
                "--expected-plan-sha256",
                "--scope-decision",
            ):
                self.assertIn(option, help_process.stdout)

    def test_route_analysis_raises_on_state_or_version_mismatch(self):
        with tempfile.TemporaryDirectory(prefix="harness-route-strict-") as temporary:
            run_dir = Path(temporary) / "run"
            store, ready, _ = self._write_ready_reopen_fixture(run_dir)
            before_events = store.events_path.read_bytes()
            with self.assertRaisesRegex(HarnessError, "state mismatch"):
                route_analysis(run_dir, "modify", "supervisor")
            self.assertEqual(store.state(), ready)
            self.assertEqual(store.events_path.read_bytes(), before_events)

            store.transition(RunState.BLOCKED)

        with tempfile.TemporaryDirectory(prefix="harness-route-version-") as temporary:
            run_dir = Path(temporary) / "run"
            store = RunStore(run_dir)
            store.initialize({"run_id": "route-version"})
            store.transition(RunState.ERROR)
            store.claim_analysis()
            (run_dir / "analysis_plan.json").write_text(
                json.dumps(self._analysis_plan()), encoding="utf-8"
            )
            with self.assertRaisesRegex(HarnessError, "version mismatch"):
                route_analysis(run_dir, "modify", "supervisor", expected_state_version=4)
            self.assertEqual(store.state()["state"], RunState.ANALYZING.value)

    def test_route_analysis_cli_exits_nonzero_on_state_mismatch(self):
        with tempfile.TemporaryDirectory(prefix="harness-route-cli-") as temporary:
            run_dir = Path(temporary) / "run"
            store, ready, _ = self._write_ready_reopen_fixture(run_dir)
            before_events = store.events_path.read_bytes()
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "legged_gym.harness.cli",
                    "route-analysis",
                    "--run-dir",
                    str(run_dir),
                    "--authority",
                    "supervisor",
                    "--disposition",
                    "modify",
                ],
                cwd=str(REPO_ROOT),
                text=True,
                capture_output=True,
                timeout=10,
            )
            self.assertEqual(process.returncode, 1)
            self.assertIn("state mismatch", process.stderr)
            self.assertEqual(store.state(), ready)
            self.assertEqual(store.events_path.read_bytes(), before_events)

    def test_post_run_lifecycle_reaches_ready_and_rejects_illegal_transition(self):
        with tempfile.TemporaryDirectory(prefix="harness-lifecycle-") as temporary:
            run_dir = Path(temporary) / "run"
            store = RunStore(run_dir)
            store.initialize({"run_id": "lifecycle-smoke"})
            store.transition(
                RunState.ERROR,
                reason="smoke",
                expected_states={RunState.PREFLIGHT.value},
            )
            self.assertTrue(store.claim_analysis()["claimed"])
            plan = self._analysis_plan()
            (run_dir / "analysis_plan.json").write_text(json.dumps(plan), encoding="utf-8")
            routed = route_analysis(run_dir, disposition="modify", authority="supervisor")
            self.assertEqual(routed["state"], RunState.MODIFYING.value)
            verifying = begin_verification(run_dir, authority="modifier")
            self.assertEqual(verifying["state"], RunState.VERIFYING.value)
            report = {"production_launch_ready": True, "tests": ["fixture pass"]}
            (run_dir / "modification_report.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            ready = finish_modification(run_dir, authority="modifier")
            self.assertEqual(ready["state"], RunState.READY.value)
            with self.assertRaises(HarnessError):
                store.transition(RunState.RUNNING, reason="illegal terminal overwrite")


if __name__ == "__main__":
    unittest.main()
