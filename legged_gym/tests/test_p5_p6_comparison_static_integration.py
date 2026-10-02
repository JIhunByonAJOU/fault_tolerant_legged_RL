"""Independent static/state-dict conformance pass for comparison tasks."""

import inspect
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

import isaacgym  # noqa: F401
import torch

from legged_gym.envs.a1_official_wim_teacher import (
    A1OfficialWimCurrentRepeatOnset,
    A1OfficialWimJointFailureOnset,
    A1OfficialWimJointFailureOnsetCfgPPO,
)
from legged_gym.learning.joint_teacher_student_actor_critic import (
    JointTeacherStudentActorCritic,
    SeparateStudentActorCritic,
)
from legged_gym.learning.comparison_runners import (
    COMPARISON_ORIGIN,
    ComparisonJointTeacherStudentRunner,
    SEED1_WIDTH64_REFERENCE_SHA256,
    SeparateStudentDistillationRunner,
    _model_hashes,
)
from legged_gym.learning.teacher_ppo import TeacherPPO
from legged_gym.learning.teacher_runner import TeacherOnPolicyRunner
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import class_to_dict, update_cfg_from_args


class ComparisonStaticIntegrationTest(unittest.TestCase):
    def test_canonical_task_objects_and_schedule_remain_canonical(self):
        env_cfg, train_cfg = task_registry.get_cfgs(
            "a1_official_wim_jt_failure_fullrange_onset"
        )
        expected = class_to_dict(A1OfficialWimJointFailureOnsetCfgPPO())
        self.assertIs(task_registry.get_task_class(
            "a1_official_wim_jt_failure_fullrange_onset"), A1OfficialWimJointFailureOnset)
        self.assertEqual(class_to_dict(train_cfg), expected)
        model = JointTeacherStudentActorCritic(2635, 45, 12)
        model.set_schedule_origin(43000)
        for iteration, alpha, beta in (
            (43000, 0.0, 1.0), (48000, 0.5, 0.5), (53000, 1.0, 0.0)
        ):
            model.set_training_iteration(iteration)
            self.assertEqual((model.adaptation_alpha, model.adaptation_beta), (alpha, beta))

    def test_registered_controls_have_distinct_environment_behavior(self):
        self.assertIs(task_registry.get_task_class(
            "a1_official_wim_jt_history_free_onset"), A1OfficialWimCurrentRepeatOnset)
        self.assertIs(task_registry.get_task_class(
            "a1_official_wim_separate_student_onset"), A1OfficialWimJointFailureOnset)
        source = inspect.getsource(A1OfficialWimCurrentRepeatOnset.compute_observations)
        self.assertIn("expand(-1, HISTORY_LENGTH, -1)", source)
        self.assertNotIn("[:-1]", source)

    def test_separate_rollout_uses_student_and_ignores_privileged_action_input(self):
        torch.manual_seed(41)
        model = SeparateStudentActorCritic(2635, 45, 12, student_embedding_dim=64)
        observations = torch.randn(4, 2635)
        privileged_a = torch.randn(4, 45)
        privileged_b = torch.randn(4, 45) * 100.0
        with torch.inference_mode():
            action_a = model.act_inference(observations, privileged_a)
            action_b = model.act_inference(observations, privileged_b)
            action_student = model.act_inference_student(observations)
        self.assertTrue(torch.equal(action_a, action_b))
        self.assertTrue(torch.equal(action_a, action_student))
        self.assertEqual(model.adaptation_alpha, 1.0)
        self.assertEqual(model.adaptation_beta, 0.0)

    def test_default_pacing_is_zero_and_canonical_config_values_are_preserved(self):
        cfg = A1OfficialWimJointFailureOnsetCfgPPO()
        before = class_to_dict(cfg)
        args = SimpleNamespace(
            seed=None, max_iterations=None, num_steps_per_env=None,
            num_mini_batches=None, save_interval=None, resume=False,
            experiment_name=None, run_name=None, load_run=None, checkpoint=None,
            shared_gpu_step_sleep_ms=0.0,
            shared_gpu_minibatch_sleep_ms=0.0,
            comparison_pilot=False,
        )
        _, updated = update_cfg_from_args(None, cfg, args)
        self.assertEqual(updated.runner.shared_gpu_step_sleep_ms, 0.0)
        self.assertEqual(updated.algorithm.shared_gpu_minibatch_sleep_ms, 0.0)
        self.assertFalse(updated.runner.comparison_pilot)
        after = class_to_dict(updated)
        after["runner"].pop("shared_gpu_step_sleep_ms")
        after["algorithm"].pop("shared_gpu_minibatch_sleep_ms")
        after["runner"].pop("comparison_pilot")
        self.assertEqual(after, before)
        model = JointTeacherStudentActorCritic(2635, 45, 12)
        ppo = TeacherPPO(model)
        self.assertEqual(ppo.shared_gpu_minibatch_sleep_ms, 0.0)
        with self.assertRaises(ValueError):
            TeacherPPO(model, shared_gpu_minibatch_sleep_ms=-0.1)

    def test_collection_pacing_occurs_after_each_control_step_only_when_positive(self):
        source = inspect.getsource(TeacherOnPolicyRunner.learn)
        process_index = source.index("self.alg.process_env_step")
        condition_index = source.index("if self.shared_gpu_step_sleep_ms > 0.0")
        sleep_index = source.index("time.sleep(self.shared_gpu_step_sleep_ms / 1000.0)")
        self.assertLess(process_index, condition_index)
        self.assertLess(condition_index, sleep_index)

    def test_real_tf43000_load_is_strict_and_b2_frozen_hashes_match(self):
        source = Path(
            "logs/official_wim_teacher243_failure_fullrange_fromscratch/"
            "Aug13_03-44-07_seed1-50000iter-fromscratch-fullrange/model_43000.pt"
        ).resolve()
        model = SeparateStudentActorCritic(2635, 45, 12, student_embedding_dim=64)
        runner = object.__new__(SeparateStudentDistillationRunner)
        runner.cfg = {
            "expected_source_path": str(source),
            "expected_source_sha256": (
                "944a697abb30dfc8023e15544d0909acfcdaa4d8c4c0f930656847398f150635"
            ),
            "profile_id": "b2_separate_student_v1",
            "max_iterations": 34501,
            "comparison_pilot": False,
        }
        runner.env = SimpleNamespace(cfg=SimpleNamespace(seed=1), num_envs=4096)
        runner.num_steps_per_env = 24
        runner.device = "cpu"
        runner.log_dir = None
        runner.current_learning_iteration = 0
        runner._comparison_config_sha256 = "synthetic-config"
        runner._comparison_code_sha256 = "current-source"
        runner._configure_run_mode()
        runner.alg = SimpleNamespace(
            actor_critic=model,
            optimizer=torch.optim.Adam(model.student_encoder.parameters(), lr=1.0e-5),
        )
        runner.load(str(source), load_optimizer=False)
        self.assertEqual(runner.current_learning_iteration, COMPARISON_ORIGIN)
        hashes = _model_hashes(model)
        self.assertEqual(
            hashes["teacher_encoder"],
            SEED1_WIDTH64_REFERENCE_SHA256["teacher_encoder"],
        )
        self.assertEqual(hashes["actor"], SEED1_WIDTH64_REFERENCE_SHA256["actor"])
        self.assertEqual(
            runner._comparison_lineage["original_tf_source_sha256"],
            runner.cfg["expected_source_sha256"],
        )
        self.assertEqual(
            runner._checkpoint_extra_metadata()["frozen_component_sha256"],
            runner._frozen_hashes(),
        )

    def test_b2_resolved_config_contains_full_environment_and_update_contract(self):
        env_cfg, train_cfg = task_registry.get_cfgs(
            "a1_official_wim_separate_student_onset"
        )
        training = class_to_dict(train_cfg)
        runner = object.__new__(SeparateStudentDistillationRunner)
        runner.cfg = training["runner"]
        runner.alg_cfg = training["algorithm"]
        runner.policy_cfg = training["policy"]
        runner.env = SimpleNamespace(
            cfg=env_cfg, num_envs=env_cfg.env.num_envs
        )
        runner.num_steps_per_env = train_cfg.runner.num_steps_per_env
        runner.shared_gpu_step_sleep_ms = 0.0
        runner.alg = SimpleNamespace(shared_gpu_minibatch_sleep_ms=0.0)
        runner._comparison_contract(training)
        resolved = runner._resolved_config(training)
        self.assertEqual(resolved["environment"], class_to_dict(env_cfg))
        self.assertEqual(resolved["seed"], env_cfg.seed)
        self.assertFalse(resolved["exact_rma_reproduction"])
        self.assertFalse(resolved["single_variable_ablation"])
        self.assertFalse(resolved["ppo_updates"])
        self.assertEqual(
            resolved["optimizer_and_update_contract"]["supervised_updates_per_rollout"],
            20,
        )
        for section in ("commands", "domain_rand", "terrain", "rewards", "noise"):
            self.assertIn(section, resolved["environment"])
        self.assertIn("curriculum", resolved["environment"]["terrain"])
        self.assertEqual(
            resolved["rma_reference_differences"]["original_rma_learning_rate"],
            5.0e-4,
        )
        self.assertEqual(
            resolved["rma_reference_differences"]["original_rma_training_steps"],
            80000000,
        )

    def test_production_contract_normalizes_resume_invocation_budget(self):
        env_cfg, train_cfg = task_registry.get_cfgs(
            "a1_official_wim_jt_history_free_onset"
        )
        initial = class_to_dict(train_cfg)
        resumed = copy.deepcopy(initial)
        resumed["runner"]["max_iterations"] = 24500
        resumed["runner"]["load_run"] = "/tmp/b1-run"
        resumed["runner"]["checkpoint"] = 53000

        def contract(training):
            runner = object.__new__(ComparisonJointTeacherStudentRunner)
            runner.cfg = training["runner"]
            runner.env = SimpleNamespace(cfg=env_cfg)
            result = runner._comparison_contract(training)
            return result, runner._comparison_config_sha256

        initial_contract, initial_sha = contract(initial)
        resumed_contract, resumed_sha = contract(resumed)
        self.assertEqual(initial_sha, resumed_sha)
        self.assertEqual(initial_contract, resumed_contract)
        self.assertEqual(
            resumed_contract["training"]["runner"]["max_iterations"], 34501
        )


if __name__ == "__main__":
    unittest.main()
