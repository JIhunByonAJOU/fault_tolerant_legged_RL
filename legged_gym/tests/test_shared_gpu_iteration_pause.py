import inspect
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import isaacgym  # noqa: F401; Isaac Gym must precede torch
import torch

from legged_gym.harness.manager import (
    HarnessError,
    _shared_gpu_pacing_from_command,
)
from legged_gym.learning.comparison_runners import (
    ComparisonJointTeacherStudentRunner,
    SeparateStudentDistillationRunner,
)
from legged_gym.learning.joint_teacher_student_runner import (
    JointTeacherStudentRunner,
)
from legged_gym.learning.shared_gpu_pacing import (
    SharedGpuIterationPacer,
    validate_iteration_sleep_ms,
)
from legged_gym.learning.student_distillation import StudentDistillation
from legged_gym.learning.teacher_ppo import TeacherPPO
from legged_gym.learning.teacher_runner import TeacherOnPolicyRunner
from legged_gym.utils.helpers import update_cfg_from_args


class _DistillationActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.student_encoder = torch.nn.Linear(1, 1, bias=False)

    def adaptation_loss(self, observations, privileged_observations):
        del privileged_observations
        return self.student_encoder(observations).square().mean()

    def student_action_loss(self, observations, privileged_observations):
        del privileged_observations
        return self.student_encoder(observations).square().mean()


class _Storage:
    def mini_batch_generator(self, num_mini_batches, num_learning_epochs):
        for _ in range(num_mini_batches * num_learning_epochs):
            yield torch.ones(1, 1), torch.zeros(1, 1)

    def clear(self):
        pass


class _PpoActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.location = torch.nn.Parameter(torch.zeros(()))

    def update_distribution(self, observations, privileged_observations):
        del privileged_observations
        batch_size = observations.shape[0]
        self.action_mean = self.location.expand(batch_size, 1)
        self.action_std = (self.location * 0.0 + 1.0).expand(batch_size, 1)
        self.entropy = (self.location * 0.0).expand(batch_size, 1)

    def get_actions_log_prob(self, actions):
        return -0.5 * torch.square(actions - self.action_mean).sum(dim=-1)

    def evaluate(self, observations, privileged_observations):
        del privileged_observations
        return (self.location * 0.0).expand(observations.shape[0], 1)


class _PpoStorage:
    def mini_batch_generator(self, num_mini_batches, num_learning_epochs):
        for _ in range(num_mini_batches * num_learning_epochs):
            values = torch.zeros(4, 1)
            yield (
                values,
                values,
                values,
                values,
                values,
                values,
                torch.zeros(4),
                values,
                torch.ones(4, 1),
                None,
                None,
            )

    def clear(self):
        pass


class _RunnerActor:
    def __init__(self):
        self.action_mean = torch.zeros(1, 1)

    def train(self):
        pass


class _RunnerAlgorithm:
    def __init__(self):
        self.actor_critic = _RunnerActor()
        self.last_update_metrics = {}

    def act(self, observations, privileged_observations):
        del privileged_observations
        self.actor_critic.action_mean = torch.zeros_like(observations)
        return torch.zeros_like(observations)

    def process_env_step(self, rewards, dones, infos):
        del rewards, dones, infos

    def compute_returns(self, observations, privileged_observations):
        del observations, privileged_observations

    def update(self):
        self.last_update_metrics = {
            "completed_updates": 20,
            "planned_updates": 20,
            "minibatch_sync_count": 20,
            "minibatch_sync_seconds": 0.02,
        }
        return 0.0, 0.0


class _RunnerEnvironment:
    num_envs = 1
    base_lin_vel = torch.zeros(1, 3)

    def get_observations(self):
        return torch.zeros(1, 1)

    def get_privileged_observations(self):
        return torch.zeros(1, 1)

    def step(self, actions):
        del actions
        return (
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.zeros(1),
            torch.zeros(1),
            {},
        )


class _Writer:
    def flush(self):
        pass

    def close(self):
        pass


class SharedGpuIterationPauseTest(unittest.TestCase):
    def test_cli_default_validation_and_config_propagation(self):
        cfg = SimpleNamespace(
            runner=SimpleNamespace(), algorithm=SimpleNamespace()
        )
        args = SimpleNamespace(
            seed=None,
            max_iterations=None,
            num_steps_per_env=None,
            num_mini_batches=None,
            save_interval=None,
            shared_gpu_step_sleep_ms=100.0,
            shared_gpu_minibatch_sleep_ms=100.0,
            shared_gpu_iteration_sleep_ms=3000.0,
            comparison_pilot=False,
            resume=False,
            experiment_name=None,
            run_name=None,
            load_run=None,
            checkpoint=None,
        )
        _, updated = update_cfg_from_args(None, cfg, args)
        self.assertEqual(updated.runner.shared_gpu_iteration_sleep_ms, 3000.0)
        self.assertEqual(validate_iteration_sleep_ms(0.0), 0.0)
        for invalid in (-0.1, float("nan"), float("inf"), -float("inf")):
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                validate_iteration_sleep_ms(invalid)

        pacing = _shared_gpu_pacing_from_command(
            ["--shared_gpu_iteration_sleep_ms", "3000"]
        )
        self.assertEqual(
            pacing,
            {
                "step_sleep_ms": 0.0,
                "minibatch_sleep_ms": 0.0,
                "iteration_sleep_ms": 3000.0,
            },
        )
        for invalid in ("nan", "inf", "-1"):
            with self.assertRaises(HarnessError):
                _shared_gpu_pacing_from_command(
                    ["--shared_gpu_iteration_sleep_ms", invalid]
                )

    def test_minibatch_synchronize_then_sleep_exact_twenty_times(self):
        actor = _DistillationActor()
        algorithm = object.__new__(StudentDistillation)
        algorithm.actor_critic = actor
        algorithm.optimizer = torch.optim.Adam(actor.parameters(), lr=1.0e-5)
        algorithm.storage = _Storage()
        algorithm.num_mini_batches = 4
        algorithm.num_learning_epochs = 5
        algorithm.max_grad_norm = 1.0
        algorithm.shared_gpu_minibatch_sleep_ms = 1.0
        algorithm.device = "cuda:0"
        events = []
        with mock.patch(
            "legged_gym.learning.student_distillation.synchronize_active_cuda",
            side_effect=lambda device: events.append(("sync", device)) or 0.001,
        ), mock.patch(
            "legged_gym.learning.student_distillation.time.sleep",
            side_effect=lambda seconds: events.append(("sleep", seconds)),
            create=True,
        ):
            algorithm.update()
        self.assertEqual(algorithm.last_update_metrics["minibatch_sync_count"], 20)
        self.assertAlmostEqual(
            algorithm.last_update_metrics["minibatch_sync_seconds"], 0.020
        )
        self.assertEqual(
            events,
            [("sync", "cuda:0"), ("sleep", 0.001)] * 20,
        )

        teacher = TeacherPPO(
            _PpoActor(),
            num_learning_epochs=5,
            num_mini_batches=4,
            shared_gpu_minibatch_sleep_ms=1.0,
            schedule="adaptive",
            desired_kl=0.01,
            device="cpu",
        )
        teacher.storage = _PpoStorage()
        events = []
        with mock.patch(
            "legged_gym.learning.teacher_ppo.synchronize_active_cuda",
            side_effect=lambda device: events.append(("sync", device)) or 0.001,
        ), mock.patch(
            "legged_gym.learning.teacher_ppo.time.sleep",
            side_effect=lambda seconds: events.append(("sleep", seconds)),
        ):
            teacher.update()
        self.assertEqual(teacher.last_update_metrics["minibatch_sync_count"], 20)
        self.assertAlmostEqual(
            teacher.last_update_metrics["minibatch_sync_seconds"], 0.020
        )
        self.assertEqual(
            events,
            [("sync", "cpu"), ("sleep", 0.001)] * 20,
        )

    def test_pause_event_schema_flush_and_terminal_count(self):
        with tempfile.TemporaryDirectory(prefix="boundary-pacing-") as temporary:
            pacer = SharedGpuIterationPacer(10.0, "cuda:0", temporary)
            with mock.patch(
                "legged_gym.learning.shared_gpu_pacing.synchronize_active_cuda",
                return_value=0.002,
            ), mock.patch("os.fsync") as fsync:
                records = [
                    pacer.pause_after_completed_iteration(
                        completed_iteration=iteration,
                        next_iteration=iteration + 1,
                        active_iteration_seconds=0.1,
                    )
                    for iteration in range(99)
                ]
            rows = [
                json.loads(line)
                for line in (
                    Path(temporary) / "iteration_boundary_pacing.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 99)
            self.assertEqual(rows, records)
            self.assertEqual(fsync.call_count, 99)
            self.assertEqual(
                [row["completed_iteration"] for row in rows], list(range(99))
            )
            self.assertEqual([row["next_iteration"] for row in rows], list(range(1, 100)))
            required = {
                "completed_iteration",
                "cuda_sync_seconds",
                "requested_sleep_ms",
                "observed_sleep_seconds",
                "pause_start_monotonic_ns",
                "pause_end_monotonic_ns",
                "next_iteration",
                "Perf/iteration_cycle_time",
            }
            self.assertEqual(set(rows[0]), required)
            self.assertTrue(all(row["observed_sleep_seconds"] >= 0.01 for row in rows))
            self.assertFalse(any(row["completed_iteration"] == 99 for row in rows))

    def test_default_zero_noop_and_stop_response(self):
        with tempfile.TemporaryDirectory(prefix="boundary-stop-") as temporary:
            zero = SharedGpuIterationPacer(0.0, "cuda:0", temporary)
            with mock.patch(
                "legged_gym.learning.shared_gpu_pacing.synchronize_active_cuda"
            ) as synchronize:
                self.assertIsNone(
                    zero.pause_after_completed_iteration(0, 1, 0.1)
                )
            synchronize.assert_not_called()
            self.assertFalse(
                (Path(temporary) / "iteration_boundary_pacing.jsonl").exists()
            )

            state = {"stop": False}
            paced = SharedGpuIterationPacer(3000.0, "cuda:0", temporary)

            def interrupt(_seconds):
                state["stop"] = True

            started = time.monotonic()
            with mock.patch(
                "legged_gym.learning.shared_gpu_pacing.synchronize_active_cuda",
                return_value=0.0,
            ), mock.patch(
                "legged_gym.learning.shared_gpu_pacing.time.sleep",
                side_effect=interrupt,
            ):
                row = paced.pause_after_completed_iteration(
                    0, 1, 0.1, stop_requested=lambda: state["stop"]
                )
            self.assertLess(time.monotonic() - started, 0.25)
            self.assertLess(row["observed_sleep_seconds"], 3.0)

    def test_step_and_boundary_ordering_and_runner_parity(self):
        source = inspect.getsource(TeacherOnPolicyRunner.learn)
        process_index = source.index("self.alg.process_env_step")
        sync_index = source.index("synchronize_active_cuda(self.device)")
        sleep_index = source.index(
            "time.sleep(self.shared_gpu_step_sleep_ms / 1000.0)"
        )
        log_index = source.index("self.log(")
        checkpoint_index = source.index("self.save(", log_index)
        bookkeeping_index = source.index("completed_iterations += 1")
        pause_index = source.index("pause_after_completed_iteration")
        self.assertLess(process_index, sync_index)
        self.assertLess(sync_index, sleep_index)
        self.assertLess(log_index, checkpoint_index)
        self.assertLess(checkpoint_index, bookkeeping_index)
        self.assertLess(bookkeeping_index, pause_index)
        self.assertIn("iteration + 1 < total_iterations", source)
        self.assertIn("if self._stop_requested", source)

        self.assertTrue(
            issubclass(ComparisonJointTeacherStudentRunner, TeacherOnPolicyRunner)
        )
        self.assertTrue(
            issubclass(SeparateStudentDistillationRunner, TeacherOnPolicyRunner)
        )
        self.assertTrue(issubclass(JointTeacherStudentRunner, TeacherOnPolicyRunner))

    def test_two_iteration_cpu_smoke_has_24_20_1_counts_and_order(self):
        runner = object.__new__(TeacherOnPolicyRunner)
        runner.log_dir = "/tmp/mock-shared-gpu-runner"
        runner.writer = _Writer()
        runner.device = "cpu"
        runner.env = _RunnerEnvironment()
        runner.alg = _RunnerAlgorithm()
        runner.num_steps_per_env = 24
        runner.save_interval = 1
        runner.shared_gpu_step_sleep_ms = 1.0
        runner.current_learning_iteration = 0
        runner._stop_requested = False
        runner._stop_reason = None
        events = []
        metric_locs = []

        class _Pacer:
            def pause_after_completed_iteration(self, **kwargs):
                events.append(("pause", kwargs["completed_iteration"]))

        runner._shared_gpu_pacer = _Pacer()
        runner.log = lambda locs: (
            metric_locs.append(locs), events.append(("metric", locs["it"]))
        )
        runner.save = lambda path, **kwargs: events.append(
            ("checkpoint", kwargs.get("iteration"), Path(path).name)
        )

        with mock.patch(
            "legged_gym.learning.teacher_runner.synchronize_active_cuda",
            side_effect=lambda device: events.append(("sync", device)) or 0.001,
        ), mock.patch(
            "legged_gym.learning.teacher_runner.time.sleep",
            side_effect=lambda seconds: events.append(("sleep", seconds)),
        ):
            runner.learn(2)

        pacing_events = [event for event in events if event[0] in {"sync", "sleep"}]
        self.assertEqual(
            pacing_events,
            [("sync", "cpu"), ("sleep", 0.001)] * 48,
        )
        self.assertEqual(
            [locs["step_sync_count"] for locs in metric_locs], [24, 24]
        )
        self.assertEqual(
            [locs["ppo_metrics"]["minibatch_sync_count"] for locs in metric_locs],
            [20, 20],
        )
        first_metric = events.index(("metric", 0))
        first_checkpoint = events.index(("checkpoint", 0, "model_0.pt"))
        boundary = events.index(("pause", 0))
        second_metric = events.index(("metric", 1))
        self.assertLess(first_metric, first_checkpoint)
        self.assertLess(first_checkpoint, boundary)
        self.assertLess(boundary, second_metric)
        self.assertEqual([event for event in events if event[0] == "pause"], [("pause", 0)])


if __name__ == "__main__":
    unittest.main()
