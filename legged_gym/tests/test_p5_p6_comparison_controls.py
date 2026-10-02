"""Focused CPU unit contracts for the P5/P6 comparison controls."""

import copy
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import isaacgym  # noqa: F401
import torch

from legged_gym.envs.a1_official_wim_teacher import (
    A1OfficialWimCurrentRepeatOnset,
    A1OfficialWimCurrentRepeatOnsetCfg,
    A1OfficialWimCurrentRepeatOnsetCfgPPO,
    A1OfficialWimJointFailureOnset,
    A1OfficialWimJointFailureOnsetCfg,
    A1OfficialWimJointFailureOnsetCfgPPO,
    A1OfficialWimSeparateStudentOnsetCfgPPO,
)
from legged_gym.envs.a1_official_wim_teacher.joint_schema import (
    CURRENT_OBSERVATION_DIM,
    HISTORY_FRAME_DIM,
    HISTORY_LENGTH,
    JOINT_OBSERVATION_DIM,
)
from legged_gym.learning.comparison_runners import (
    COMPARISON_ITERATIONS,
    COMPARISON_ORIGIN,
    SELECTED_CHECKPOINT_BATCHES,
    COMPARISON_TARGET_NEXT_ITERATION,
    ComparisonJointTeacherStudentRunner,
    SeparateStudentDistillationRunner,
    _sha256,
    post_update_batches,
)
from legged_gym.learning.joint_teacher_student_actor_critic import (
    JointTeacherStudentActorCritic,
    SeparateStudentActorCritic,
)
from legged_gym.learning.joint_teacher_student_runner import JointTeacherStudentRunner
from legged_gym.learning.official_wim_teacher_actor_critic import (
    OfficialWimTeacherActorCritic,
)
from legged_gym.learning.student_distillation import StudentDistillation
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import class_to_dict


def _state(module):
    return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}


class _SyntheticStorage:
    def __init__(self, obs, privileged, irrelevant_fill):
        self.obs = obs
        self.privileged = privileged
        self.irrelevant_fill = irrelevant_fill
        self.cleared = False

    def mini_batch_generator(self, num_mini_batches, num_learning_epochs):
        for _ in range(num_mini_batches * num_learning_epochs):
            irrelevant = torch.full((self.obs.shape[0], 1), self.irrelevant_fill)
            yield (self.obs, self.privileged) + (irrelevant,) * 9

    def clear(self):
        self.cleared = True


class ComparisonControlUnitTest(unittest.TestCase):
    def _comparison_runner(
        self, runner_class, model, max_iterations=COMPARISON_ITERATIONS,
        comparison_pilot=False,
    ):
        runner = object.__new__(runner_class)
        runner.cfg = {
            "expected_source_path": "/tmp/tf/model_43000.pt",
            "expected_source_sha256": "tf-sha",
            "profile_id": runner_class.comparison_profile_id,
            "max_iterations": max_iterations,
            "comparison_pilot": comparison_pilot,
        }
        runner.env = SimpleNamespace(
            cfg=SimpleNamespace(seed=1), num_envs=4096
        )
        runner.num_steps_per_env = 24
        runner.device = "cpu"
        runner.log_dir = None
        runner.current_learning_iteration = COMPARISON_ORIGIN
        runner._comparison_config_sha256 = "config-sha"
        runner._comparison_code_sha256 = "code-sha"
        runner._configure_run_mode()
        runner.alg = SimpleNamespace(
            actor_critic=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=1.0e-5),
        )
        runner._comparison_lineage = runner._new_lineage()
        return runner

    def test_b1_repeats_current_frame_on_step_and_reset_semantics(self):
        env = object.__new__(A1OfficialWimCurrentRepeatOnset)
        env.observation_history = torch.empty(2, HISTORY_LENGTH, HISTORY_FRAME_DIM)

        def parent_compute(instance):
            instance.obs_buf = torch.cat(
                (instance.test_current, torch.randn(2, HISTORY_LENGTH * HISTORY_FRAME_DIM)),
                dim=-1,
            )

        with mock.patch.object(
            A1OfficialWimJointFailureOnset, "compute_observations", new=parent_compute
        ):
            for value in (1.0, -3.0):
                env.test_current = torch.full((2, CURRENT_OBSERVATION_DIM), value)
                env.compute_observations()
                history = env.obs_buf[:, CURRENT_OBSERVATION_DIM:].reshape(
                    2, HISTORY_LENGTH, HISTORY_FRAME_DIM
                )
                expected = env.test_current[:, :HISTORY_FRAME_DIM].unsqueeze(1).expand_as(history)
                self.assertTrue(torch.equal(history, expected))

    def test_b1_config_diff_is_the_approved_control_only(self):
        canonical_env = class_to_dict(A1OfficialWimJointFailureOnsetCfg())
        control_env = class_to_dict(A1OfficialWimCurrentRepeatOnsetCfg())
        control_env["env"]["task_name"] = canonical_env["env"]["task_name"]
        self.assertEqual(control_env, canonical_env)

        canonical = class_to_dict(A1OfficialWimJointFailureOnsetCfgPPO())
        control = class_to_dict(A1OfficialWimCurrentRepeatOnsetCfgPPO())
        self.assertEqual(control["policy"]["student_embedding_dim"], 64)
        self.assertEqual(control["algorithm"]["learning_rate"], 1.0e-5)
        self.assertEqual(control["runner"]["max_iterations"], 34501)
        self.assertEqual(control["runner"]["save_interval"], 500)
        self.assertEqual(control["runner"]["comparison_profile"],
                         "current-repeat / temporal-history control")
        for key in ("student_embedding_dim",):
            control["policy"][key] = canonical["policy"][key]
        control["algorithm"]["learning_rate"] = canonical["algorithm"]["learning_rate"]
        control["runner_class_name"] = canonical["runner_class_name"]
        for key in (
            "experiment_name", "max_iterations", "save_interval", "resume",
            "load_run", "checkpoint",
        ):
            control["runner"][key] = canonical["runner"][key]
        for key in ("expected_source_path", "expected_source_sha256", "profile_id",
                    "comparison_profile"):
            control["runner"].pop(key)
        self.assertEqual(control, canonical)

    def test_b2_only_student_is_optimized_with_unsquared_l2(self):
        torch.manual_seed(17)
        model = SeparateStudentActorCritic(
            JOINT_OBSERVATION_DIM, 45, 12, student_embedding_dim=64
        )
        algorithm = StudentDistillation(
            model,
            learning_rate=1.0e-5,
            schedule="fixed",
            desired_kl=None,
            max_grad_norm=1.0,
            num_learning_epochs=5,
            num_mini_batches=4,
            device="cpu",
        )
        optimized = {id(parameter) for group in algorithm.optimizer.param_groups
                     for parameter in group["params"]}
        self.assertEqual(optimized, {id(p) for p in model.student_encoder.parameters()})
        self.assertTrue(all(p.requires_grad for p in model.student_encoder.parameters()))
        for frozen in (model.teacher_encoder, model.actor, model.critic):
            self.assertFalse(any(p.requires_grad for p in frozen.parameters()))
        self.assertFalse(model.std.requires_grad)

        obs = torch.randn(8, JOINT_OBSERVATION_DIM)
        privileged = torch.randn(8, 45)
        expected = torch.linalg.vector_norm(
            model.encode_history(obs) - model.encode_privileged(privileged).detach(), dim=-1
        ).mean()
        self.assertEqual(model.adaptation_loss(obs, privileged).item(), expected.item())
        frozen_before = {
            "teacher": _state(model.teacher_encoder),
            "actor": _state(model.actor),
            "critic": _state(model.critic),
            "std": model.std.detach().clone(),
        }
        algorithm.storage = _SyntheticStorage(obs, privileged, irrelevant_fill=1234.0)
        algorithm.update()
        self.assertEqual(algorithm.last_update_metrics["planned_supervised_updates"], 20)
        self.assertEqual(algorithm.last_update_metrics["completed_supervised_updates"], 20)
        self.assertEqual(algorithm.last_update_metrics["ppo_updates"], 0)
        self.assertEqual(algorithm.last_update_metrics["action_mse_weight"], 0.0)
        self.assertGreater(
            algorithm.last_update_metrics["student_parameter_step_l2"], 0.0
        )
        for name, module in (("teacher", model.teacher_encoder), ("actor", model.actor),
                             ("critic", model.critic)):
            for key, value in _state(module).items():
                self.assertTrue(torch.equal(value, frozen_before[name][key]))
        self.assertTrue(torch.equal(model.std.detach(), frozen_before["std"]))

    def test_b2_update_is_independent_of_ppo_batch_fields(self):
        torch.manual_seed(23)
        first = SeparateStudentActorCritic(JOINT_OBSERVATION_DIM, 45, 12, student_embedding_dim=64)
        second = SeparateStudentActorCritic(JOINT_OBSERVATION_DIM, 45, 12, student_embedding_dim=64)
        second.load_state_dict(copy.deepcopy(first.state_dict()))
        kwargs = dict(learning_rate=1.0e-5, schedule="fixed", desired_kl=None,
                      max_grad_norm=1.0, num_learning_epochs=5,
                      num_mini_batches=4, device="cpu")
        alg_a, alg_b = StudentDistillation(first, **kwargs), StudentDistillation(second, **kwargs)
        obs, privileged = torch.randn(8, JOINT_OBSERVATION_DIM), torch.randn(8, 45)
        alg_a.storage = _SyntheticStorage(obs, privileged, irrelevant_fill=-1.0e9)
        alg_b.storage = _SyntheticStorage(obs, privileged, irrelevant_fill=1.0e9)
        alg_a.update()
        alg_b.update()
        for key, value in first.student_encoder.state_dict().items():
            self.assertTrue(torch.equal(value, second.student_encoder.state_dict()[key]))

    def test_common_tf_state_and_fresh_student_initialization_are_equal(self):
        torch.manual_seed(31)
        b1 = JointTeacherStudentActorCritic(
            JOINT_OBSERVATION_DIM, 45, 12, student_embedding_dim=64
        )
        torch.manual_seed(31)
        b2 = SeparateStudentActorCritic(
            JOINT_OBSERVATION_DIM, 45, 12, student_embedding_dim=64
        )
        for key, value in b1.student_encoder.state_dict().items():
            self.assertTrue(torch.equal(value, b2.student_encoder.state_dict()[key]))
        teacher = OfficialWimTeacherActorCritic(235, 45, 12)
        for model in (b1, b2):
            incompatible = model.load_state_dict(teacher.state_dict(), strict=False)
            self.assertEqual(incompatible.unexpected_keys, [])
            self.assertTrue(all(k.startswith("student_encoder.") for k in incompatible.missing_keys))
        for component in ("teacher_encoder", "actor", "critic"):
            first_state = getattr(b1, component).state_dict()
            second_state = getattr(b2, component).state_dict()
            for key, value in first_state.items():
                self.assertTrue(torch.equal(value, second_state[key]))
        self.assertTrue(torch.equal(b1.std, b2.std))

    def test_registry_budget_and_global_checkpoint_labels(self):
        for task in (
            "a1_official_wim_jt_history_free_onset",
            "a1_official_wim_separate_student_onset",
        ):
            env_cfg, train_cfg = task_registry.get_cfgs(task)
            self.assertEqual(env_cfg.env.num_envs, 4096)
            self.assertEqual(train_cfg.runner.num_steps_per_env, 24)
            self.assertEqual(train_cfg.runner.max_iterations, COMPARISON_ITERATIONS)
            self.assertEqual(train_cfg.runner.checkpoint, COMPARISON_ORIGIN)
            self.assertTrue(train_cfg.runner.resume)
        for label, expected in SELECTED_CHECKPOINT_BATCHES.items():
            self.assertEqual(post_update_batches(label), expected)
            self.assertEqual(label % 500, 0)
        self.assertEqual(COMPARISON_ORIGIN + COMPARISON_ITERATIONS, 77501)

    def test_periodic_checkpoint_resume_restores_optimizer_without_duplicate_batch(self):
        torch.manual_seed(47)
        model = JointTeacherStudentActorCritic(2635, 45, 12, student_embedding_dim=64)
        runner = self._comparison_runner(ComparisonJointTeacherStudentRunner, model)
        runner.alg.optimizer.param_groups[0]["lr"] = 7.0e-6
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model_53000.pt"
            runner.save(str(path), iteration=53000)
            saved = torch.load(path, map_location="cpu")
            self.assertEqual(saved["iter"], 53001)
            self.assertEqual(saved["infos"]["next_iteration"], 53001)
            self.assertEqual(saved["infos"]["completed_new_batches"], 10001)

            torch.manual_seed(99)
            restored_model = JointTeacherStudentActorCritic(
                2635, 45, 12, student_embedding_dim=64
            )
            restored = self._comparison_runner(
                ComparisonJointTeacherStudentRunner, restored_model,
                max_iterations=24500,
            )
            restored.load(str(path), load_optimizer=True)
            self.assertEqual(restored.current_learning_iteration, 53001)
            self.assertEqual(restored.remaining_iterations(), 77501 - 53001)
            self.assertEqual(restored.alg.optimizer.param_groups[0]["lr"], 7.0e-6)
            self.assertEqual(
                restored.current_learning_iteration + restored.remaining_iterations(),
                COMPARISON_TARGET_NEXT_ITERATION,
            )
            with mock.patch.object(
                JointTeacherStudentRunner, "learn", return_value="resumed"
            ) as learn:
                self.assertEqual(restored.learn(24500), "resumed")
            learn.assert_called_once_with(24500, init_at_random_ep_len=False)

            wrong_budget = self._comparison_runner(
                ComparisonJointTeacherStudentRunner,
                JointTeacherStudentActorCritic(
                    2635, 45, 12, student_embedding_dim=64
                ),
                max_iterations=100,
            )
            with self.assertRaisesRegex(ValueError, "does not equal remaining"):
                wrong_budget.load(str(path), load_optimizer=True)

    def test_terminal_checkpoint_duplicates_selected_state_without_extra_batch(self):
        model = JointTeacherStudentActorCritic(2635, 45, 12, student_embedding_dim=64)
        runner = self._comparison_runner(ComparisonJointTeacherStudentRunner, model)
        runner.current_learning_iteration = COMPARISON_TARGET_NEXT_ITERATION
        with tempfile.TemporaryDirectory() as tmp:
            selected = Path(tmp) / "model_77500.pt"
            terminal = Path(tmp) / "model_77501.pt"
            runner.save(str(selected), iteration=77500)
            runner.save(str(terminal))
            first = torch.load(selected, map_location="cpu")
            last = torch.load(terminal, map_location="cpu")
            self.assertEqual(first["infos"]["next_iteration"], 77501)
            self.assertEqual(last["infos"]["next_iteration"], 77501)
            self.assertEqual(first["infos"]["completed_new_batches"], 34501)
            self.assertTrue(last["infos"]["terminal_duplicate_state"])
            for key, value in first["model_state_dict"].items():
                self.assertTrue(torch.equal(value, last["model_state_dict"][key]))

    def test_lineage_rejects_cross_method_canonical_and_wrong_tf(self):
        b1_model = JointTeacherStudentActorCritic(2635, 45, 12, student_embedding_dim=64)
        b1 = self._comparison_runner(ComparisonJointTeacherStudentRunner, b1_model)
        with tempfile.TemporaryDirectory() as tmp:
            own = Path(tmp) / "model_53000.pt"
            b1.save(str(own), iteration=53000)
            loaded = torch.load(own, map_location="cpu")

            b2_model = SeparateStudentActorCritic(2635, 45, 12, student_embedding_dim=64)
            b2 = self._comparison_runner(SeparateStudentDistillationRunner, b2_model)
            with self.assertRaisesRegex(RuntimeError, "lineage mismatch"):
                b2._validate_own_checkpoint(loaded)

            canonical = dict(loaded)
            canonical["infos"] = {"joint_schedule_origin": 43000}
            with self.assertRaisesRegex(RuntimeError, "not a lineage-tagged"):
                b1._validate_own_checkpoint(canonical)

            tf = Path(tmp) / "model_43000.pt"
            tf.write_bytes(b"wrong source")
            b1.cfg["expected_source_path"] = str(tf)
            b1.cfg["expected_source_sha256"] = _sha256(tf)
            b1._verify_source(str(tf))
            tf.write_bytes(b"mutated")
            with self.assertRaisesRegex(RuntimeError, "source hash changed"):
                b1._verify_source(str(tf))

    def test_b2_frozen_hash_metadata_detects_mutation(self):
        model = SeparateStudentActorCritic(2635, 45, 12, student_embedding_dim=64)
        runner = self._comparison_runner(SeparateStudentDistillationRunner, model)
        metadata = runner._checkpoint_extra_metadata()
        self.assertEqual(
            metadata["frozen_component_sha256"], runner._frozen_hashes()
        )
        with torch.no_grad():
            next(model.actor.parameters()).add_(1.0)
        with self.assertRaisesRegex(RuntimeError, "frozen B2 component changed"):
            runner._checkpoint_extra_metadata()

    def test_pacing_changes_sleep_only_for_supervised_update(self):
        torch.manual_seed(53)
        base = SeparateStudentActorCritic(2635, 45, 12, student_embedding_dim=64)
        paced = SeparateStudentActorCritic(2635, 45, 12, student_embedding_dim=64)
        paced.load_state_dict(copy.deepcopy(base.state_dict()))
        kwargs = dict(learning_rate=1.0e-5, schedule="fixed", desired_kl=None,
                      max_grad_norm=1.0, num_learning_epochs=5,
                      num_mini_batches=4, device="cpu")
        plain_alg = StudentDistillation(base, shared_gpu_minibatch_sleep_ms=0.0, **kwargs)
        paced_alg = StudentDistillation(paced, shared_gpu_minibatch_sleep_ms=2.0, **kwargs)
        obs, privileged = torch.randn(8, 2635), torch.randn(8, 45)
        plain_alg.storage = _SyntheticStorage(obs, privileged, 0.0)
        paced_alg.storage = _SyntheticStorage(obs, privileged, 0.0)
        plain_alg.update()
        with mock.patch("time.sleep") as sleeper:
            paced_alg.update()
        self.assertEqual(sleeper.call_count, 20)
        self.assertTrue(all(call.args == (0.002,) for call in sleeper.call_args_list))
        for key, value in base.student_encoder.state_dict().items():
            self.assertTrue(torch.equal(value, paced.student_encoder.state_dict()[key]))
        self.assertEqual(
            plain_alg.last_update_metrics["student_parameter_step_l2"],
            paced_alg.last_update_metrics["student_parameter_step_l2"],
        )

    def test_pilot_requires_opt_in_fresh_tf_and_is_not_production_compatible(self):
        production_short = self._comparison_runner(
            ComparisonJointTeacherStudentRunner,
            JointTeacherStudentActorCritic(2635, 45, 12, student_embedding_dim=64),
            max_iterations=2,
        )
        with self.assertRaisesRegex(ValueError, "require --comparison_pilot"):
            production_short._validate_fresh_source_invocation()

        pilot = self._comparison_runner(
            ComparisonJointTeacherStudentRunner,
            JointTeacherStudentActorCritic(2635, 45, 12, student_embedding_dim=64),
            max_iterations=2,
            comparison_pilot=True,
        )
        pilot._validate_fresh_source_invocation()
        self.assertEqual(pilot._comparison_run_class, "pilot")
        self.assertEqual(pilot.remaining_iterations(), 2)
        self.assertEqual(pilot._comparison_lineage["checkpoint_class"],
                         "pilot_only_not_production")
        with mock.patch.object(
            JointTeacherStudentRunner, "learn", return_value="pilot"
        ) as learn:
            self.assertEqual(pilot.learn(2), "pilot")
        learn.assert_called_once_with(2, init_at_random_ep_len=False)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "pilot_model_43001.pt"
            pilot.save(str(checkpoint), iteration=43001)
            loaded = torch.load(checkpoint, map_location="cpu")

            production = self._comparison_runner(
                ComparisonJointTeacherStudentRunner,
                JointTeacherStudentActorCritic(
                    2635, 45, 12, student_embedding_dim=64
                ),
            )
            with self.assertRaisesRegex(RuntimeError, "lineage mismatch"):
                production._validate_own_checkpoint(loaded)
            with self.assertRaisesRegex(RuntimeError, "must start fresh"):
                pilot._restore_own_checkpoint(loaded, load_optimizer=True)

        with self.assertRaisesRegex(ValueError, "\[1, 100\]"):
            self._comparison_runner(
                ComparisonJointTeacherStudentRunner,
                JointTeacherStudentActorCritic(
                    2635, 45, 12, student_embedding_dim=64
                ),
                max_iterations=101,
                comparison_pilot=True,
            )


if __name__ == "__main__":
    unittest.main()
