"""Strict warm-start runners for the P5/P6 internal comparison tasks."""

import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from .joint_teacher_student_actor_critic import SeparateStudentActorCritic
from .joint_teacher_student_runner import JointTeacherStudentRunner
from .student_distillation import StudentDistillation
from .teacher_runner import TeacherOnPolicyRunner
from legged_gym.utils.helpers import class_to_dict


COMPARISON_ORIGIN = 43000
COMPARISON_ITERATIONS = 34501
SELECTED_CHECKPOINT_BATCHES = {53000: 10001, 63000: 20001, 77500: 34501}
COMPARISON_TARGET_NEXT_ITERATION = COMPARISON_ORIGIN + COMPARISON_ITERATIONS
SEED1_WIDTH64_REFERENCE_SHA256 = {
    "student_encoder": "e2c04719ee51192cedff56ebb96a1ab359bad45ee05e4a8dab6b69fc1f75342d",
    "teacher_encoder": "f92d81eadd3b55e9b3f5a79df7fd166262e85844efb6c4f89ce3185c25923c31",
    "actor": "293febb19611bba914cb9b0d72206ec47939b41b7693f115c458b5804b062380",
}
_SOURCE_FILES = (
    "legged_gym/envs/a1_official_wim_teacher/a1_official_wim_joint.py",
    "legged_gym/envs/a1_official_wim_teacher/a1_official_wim_joint_config.py",
    "legged_gym/learning/joint_teacher_student_actor_critic.py",
    "legged_gym/learning/comparison_runners.py",
    "legged_gym/learning/student_distillation.py",
    "legged_gym/learning/teacher_ppo.py",
    "legged_gym/learning/teacher_runner.py",
    "legged_gym/utils/helpers.py",
    "legged_gym/utils/task_registry.py",
)


def post_update_batches(checkpoint_label, origin=COMPARISON_ORIGIN):
    """Translate the runner's post-update periodic label into batch exposure."""
    return int(checkpoint_label) - int(origin) + 1


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_sha256(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _tensor_sha256(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _json_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _code_sha256():
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for relative in _SOURCE_FILES:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _model_hashes(model):
    return {
        "student_encoder": _module_sha256(model.student_encoder),
        "teacher_encoder": _module_sha256(model.teacher_encoder),
        "actor": _module_sha256(model.actor),
        "critic": _module_sha256(model.critic),
        "action_std": _tensor_sha256(model.std),
    }


class _StrictTF43000Mixin:
    comparison_profile_id = None
    comparison_profile_name = None

    def _configure_run_mode(self):
        self._comparison_pilot = bool(self.cfg.get("comparison_pilot", False))
        self._invocation_max_iterations = int(self.cfg["max_iterations"])
        if self._comparison_pilot:
            if not 1 <= self._invocation_max_iterations <= 100:
                raise ValueError("comparison pilot max_iterations must be in [1, 100]")
            self._comparison_run_class = "pilot"
            self._comparison_max_new_batches = self._invocation_max_iterations
        else:
            if not 1 <= self._invocation_max_iterations <= COMPARISON_ITERATIONS:
                raise ValueError(
                    "comparison production invocation must be in [1, 34501]"
                )
            self._comparison_run_class = "production"
            self._comparison_max_new_batches = COMPARISON_ITERATIONS
        self._comparison_target_next_iteration = (
            COMPARISON_ORIGIN + self._comparison_max_new_batches
        )

    def _comparison_contract(self, train_cfg):
        if self.cfg.get("profile_id") != self.comparison_profile_id:
            raise RuntimeError("comparison profile id/config mismatch")
        self._configure_run_mode()
        training = json.loads(json.dumps(train_cfg, sort_keys=True))
        runner = training["runner"]
        # Resume location is operational state, not a method/config change.
        runner["resume"] = True
        runner["load_run"] = self.cfg["expected_source_path"].rsplit("/", 1)[0]
        runner["checkpoint"] = COMPARISON_ORIGIN
        runner["run_name"] = ""
        runner["max_iterations"] = self._comparison_max_new_batches
        runner["comparison_pilot"] = self._comparison_pilot
        contract = {
            "environment": class_to_dict(self.env.cfg),
            "training": training,
            "profile_id": self.comparison_profile_id,
            "run_class": self._comparison_run_class,
            "scientific_target_new_batches": self._comparison_max_new_batches,
        }
        self._comparison_config_sha256 = _json_sha256(contract)
        self._comparison_code_sha256 = _code_sha256()
        self._comparison_contract_value = contract
        return contract

    def _verify_source(self, path):
        expected_path = os.path.realpath(self.cfg["expected_source_path"])
        actual_path = os.path.realpath(path)
        if actual_path != expected_path:
            raise RuntimeError(
                "comparison warmstart path mismatch: {} != {}".format(
                    actual_path, expected_path
                )
            )
        actual_sha = _sha256(actual_path)
        if actual_sha != self.cfg["expected_source_sha256"]:
            raise RuntimeError("TF43000 source hash changed: {}".format(actual_sha))

    def _validate_fresh_source_invocation(self):
        if (
            self._comparison_run_class == "production"
            and self._invocation_max_iterations != COMPARISON_ITERATIONS
        ):
            raise ValueError(
                "short comparison runs require --comparison_pilot; "
                "fresh production requires 34501 iterations"
            )

    def _new_lineage(self):
        model = self.alg.actor_critic
        return {
            "schema_version": 1,
            "profile_id": self.comparison_profile_id,
            "comparison_profile": self.comparison_profile_name,
            "run_class": self._comparison_run_class,
            "checkpoint_class": (
                "pilot_only_not_production"
                if self._comparison_pilot
                else "production_comparison"
            ),
            "original_tf_source_path": os.path.realpath(
                self.cfg["expected_source_path"]
            ),
            "original_tf_source_sha256": self.cfg["expected_source_sha256"],
            "seed": int(self.env.cfg.seed),
            "student_width": model.student_encoder.frame_encoder[0].out_features,
            "origin_iteration": COMPARISON_ORIGIN,
            "target_next_iteration": self._comparison_target_next_iteration,
            "max_new_batches": self._comparison_max_new_batches,
            "invocation_max_iterations": self._invocation_max_iterations,
            "rollout_exposure": {
                "num_envs": int(self.env.num_envs),
                "steps_per_env": int(self.num_steps_per_env),
            },
            "schedule": self._schedule_contract(),
            "configuration_sha256": self._comparison_config_sha256,
            "code_source_sha256": self._comparison_code_sha256,
            "optimizer_initialized_fresh_from_tf": True,
            "initial_module_sha256": _model_hashes(model),
            "seed1_width64_reference_sha256": SEED1_WIDTH64_REFERENCE_SHA256,
            "initialization_distinction": (
                "B1/B2 use fresh width64 Students from TF43000; canonical JT provenance "
                "and its selected JT77500 seed1 checkpoint remain separate"
            ),
        }

    def _write_launch_artifacts(self, resumed_from=None):
        manifest = dict(self._comparison_lineage)
        manifest["resume"] = resumed_from is not None
        manifest["resumed_from"] = (
            os.path.realpath(resumed_from) if resumed_from is not None else None
        )
        manifest["optimizer_state"] = (
            "restored_from_own_profile" if resumed_from is not None else "fresh"
        )
        manifest["actual_invocation_max_iterations"] = self._invocation_max_iterations
        manifest["remaining_batches_at_launch"] = self.remaining_iterations()
        if self.log_dir is not None:
            with open(
                os.path.join(self.log_dir, "comparison_manifest.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True)
            resolved_path = os.path.join(self.log_dir, "resolved_config.json")
            with open(resolved_path, encoding="utf-8") as handle:
                resolved = json.load(handle)
            resolved["initialization"] = manifest
            with open(resolved_path, "w", encoding="utf-8") as handle:
                json.dump(resolved, handle, indent=2, sort_keys=True)
        return manifest

    def _validate_own_checkpoint(self, loaded):
        infos = loaded.get("infos")
        if not isinstance(infos, dict) or not isinstance(
            infos.get("comparison_lineage"), dict
        ):
            raise RuntimeError("checkpoint is not a lineage-tagged comparison checkpoint")
        lineage = infos["comparison_lineage"]
        expected = {
            "profile_id": self.comparison_profile_id,
            "run_class": self._comparison_run_class,
            "original_tf_source_path": os.path.realpath(
                self.cfg["expected_source_path"]
            ),
            "original_tf_source_sha256": self.cfg["expected_source_sha256"],
            "seed": int(self.env.cfg.seed),
            "student_width": self.alg.actor_critic.student_encoder.frame_encoder[0].out_features,
            "origin_iteration": COMPARISON_ORIGIN,
            "target_next_iteration": self._comparison_target_next_iteration,
            "max_new_batches": self._comparison_max_new_batches,
            "schedule": self._schedule_contract(),
            "configuration_sha256": self._comparison_config_sha256,
            "code_source_sha256": self._comparison_code_sha256,
        }
        mismatches = {
            key: (lineage.get(key), value)
            for key, value in expected.items()
            if lineage.get(key) != value
        }
        if mismatches:
            raise RuntimeError("comparison checkpoint lineage mismatch: {}".format(mismatches))
        if "optimizer_state_dict" not in loaded:
            raise RuntimeError("comparison resume checkpoint lacks optimizer state")
        next_iteration = infos.get("next_iteration")
        completed = infos.get("completed_new_batches")
        if not isinstance(next_iteration, int) or not isinstance(completed, int):
            raise RuntimeError("comparison checkpoint lacks exact iteration metadata")
        if completed != next_iteration - COMPARISON_ORIGIN:
            raise RuntimeError("comparison checkpoint batch ledger is inconsistent")
        if not COMPARISON_ORIGIN <= next_iteration <= self._comparison_target_next_iteration:
            raise RuntimeError("comparison checkpoint next_iteration is outside budget")
        return lineage, next_iteration

    def _restore_own_checkpoint(self, loaded, load_optimizer):
        if self._comparison_pilot:
            raise RuntimeError("comparison pilot must start fresh from TF43000")
        lineage, next_iteration = self._validate_own_checkpoint(loaded)
        expected_remaining = self._comparison_target_next_iteration - next_iteration
        if load_optimizer and self._invocation_max_iterations != expected_remaining:
            raise ValueError(
                "resume max_iterations {} does not equal remaining {}".format(
                    self._invocation_max_iterations, expected_remaining
                )
            )
        self.alg.actor_critic.load_state_dict(loaded["model_state_dict"], strict=True)
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded["optimizer_state_dict"])
        self.current_learning_iteration = next_iteration
        self.alg.actor_critic.set_schedule_origin(COMPARISON_ORIGIN)
        self._comparison_lineage = lineage
        self._post_resume_validation(loaded)
        self._write_launch_artifacts(resumed_from=loaded.get("_checkpoint_path"))
        return loaded.get("infos")

    def _post_resume_validation(self, loaded):
        del loaded

    def _save_comparison(self, save_method, path, infos, iteration):
        next_iteration = (
            self.current_learning_iteration if iteration is None else int(iteration) + 1
        )
        completed = next_iteration - COMPARISON_ORIGIN
        info = dict(infos or {})
        info.update(
            {
                "comparison_lineage": self._comparison_lineage,
                "profile_id": self.comparison_profile_id,
                "origin_iteration": COMPARISON_ORIGIN,
                "next_iteration": next_iteration,
                "completed_new_batches": completed,
                "terminal_duplicate_state": iteration is None
                and next_iteration == self._comparison_target_next_iteration,
            }
        )
        info.update(self._checkpoint_extra_metadata())
        return save_method(path, infos=info, iteration=next_iteration)

    def _checkpoint_extra_metadata(self):
        return {}

    def remaining_iterations(self):
        return self._comparison_target_next_iteration - self.current_learning_iteration

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        remaining = self.remaining_iterations()
        if remaining <= 0:
            raise RuntimeError("comparison budget is already complete")
        if int(num_learning_iterations) != remaining:
            raise ValueError(
                "comparison learn budget must equal the exact remaining batches"
            )
        return super().learn(remaining, init_at_random_ep_len=init_at_random_ep_len)


class ComparisonJointTeacherStudentRunner(_StrictTF43000Mixin, JointTeacherStudentRunner):
    comparison_profile_id = "b1_current_repeat_v1"
    comparison_profile_name = "current-repeat / temporal-history control"

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self._comparison_train_cfg = train_cfg
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        if not hasattr(self, "_comparison_contract_value"):
            self._comparison_contract(train_cfg)

    def _schedule_contract(self):
        return {
            "optimizer": "PPO plus mean unsquared latent L2",
            "alpha": "0->1 linear",
            "beta": "1->0 linear",
            "schedule_batches": 10000,
            "learning_rate_initial": 1.0e-5,
        }

    def _resolved_config(self, train_cfg):
        resolved = super()._resolved_config(train_cfg)
        contract = getattr(self, "_comparison_contract_value", None)
        if contract is None:
            contract = self._comparison_contract(train_cfg)
        resolved.update(
            {
                "profile_id": self.comparison_profile_id,
                "comparison_profile": "current-repeat / temporal-history control",
                "history_intervention": (
                    "repeat the current 48-D frame 50 times at every step and reset; "
                    "current observation and previous action remain available"
                ),
                "source_checkpoint": {
                    "path": self.cfg["expected_source_path"],
                    "sha256": self.cfg["expected_source_sha256"],
                    "iteration": 43000,
                },
                "fresh_optimizer": True,
                "comparison_contract": contract,
                "configuration_sha256": self._comparison_config_sha256,
                "code_source_sha256": self._comparison_code_sha256,
                "shared_gpu_pacing": {
                    "step_sleep_ms": float(self.cfg.get("shared_gpu_step_sleep_ms", 0.0)),
                    "minibatch_sleep_ms": float(
                        self.alg_cfg.get("shared_gpu_minibatch_sleep_ms", 0.0)
                    ),
                },
                "run_class": self._comparison_run_class,
                "actual_invocation_max_iterations": self._invocation_max_iterations,
                "scientific_target_new_batches": self._comparison_max_new_batches,
            }
        )
        return resolved

    def load(self, path, load_optimizer=True):
        loaded = torch.load(path, map_location=self.device)
        loaded["_checkpoint_path"] = path
        state = loaded["model_state_dict"]
        if any(key.startswith("student_encoder.") for key in state):
            return self._restore_own_checkpoint(loaded, load_optimizer)
        self._validate_fresh_source_invocation()
        self._verify_source(path)
        result = super().load(path, load_optimizer=False)
        self._comparison_lineage = self._new_lineage()
        self._write_launch_artifacts()
        return result

    def save(self, path, infos=None, iteration=None):
        return self._save_comparison(super().save, path, infos, iteration)


class SeparateStudentDistillationRunner(_StrictTF43000Mixin, TeacherOnPolicyRunner):
    """Student-driven rollout runner with supervised-only Student updates."""

    comparison_profile_id = "b2_separate_student_v1"
    comparison_profile_name = "RMA-style internal two-stage control"

    def _schedule_contract(self):
        return {
            "optimizer": "Adam student_encoder only",
            "learning_rate": 1.0e-5,
            "learning_rate_schedule": "fixed",
            "gradient_clip": 1.0,
            "epochs": 5,
            "minibatches": 4,
            "supervised_updates_per_rollout": 20,
            "ppo_updates": 0,
            "objective": "mean unsquared L2",
            "action_mse_weight": 0.0,
        }

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self._comparison_train_cfg = train_cfg
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is None:
            raise ValueError("separate Student requires privileged labels")
        if self.cfg["policy_class_name"] != "SeparateStudentActorCritic":
            raise ValueError("separate Student policy class mismatch")
        if self.cfg["algorithm_class_name"] != "StudentDistillation":
            raise ValueError("separate Student algorithm class mismatch")
        actor_critic = SeparateStudentActorCritic(
            self.env.num_obs,
            self.env.num_privileged_obs,
            self.env.num_actions,
            **self.policy_cfg
        ).to(self.device)
        self.alg = StudentDistillation(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.shared_gpu_step_sleep_ms = float(
            self.cfg.get("shared_gpu_step_sleep_ms", 0.0)
        )
        if self.shared_gpu_step_sleep_ms < 0.0:
            raise ValueError("shared_gpu_step_sleep_ms must be nonnegative")
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
        )
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self._stop_requested = False
        self._stop_reason = None
        self._comparison_contract(train_cfg)
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(os.path.join(self.log_dir, "resolved_config.json"), "w") as handle:
                json.dump(self._resolved_config(train_cfg), handle, indent=2, sort_keys=True)
        self.env.reset()

    def _resolved_config(self, train_cfg):
        contract = self._comparison_contract_value
        return {
            "task": getattr(self.env.cfg.env, "task_name", "unknown"),
            "profile_id": self.comparison_profile_id,
            "mode": "adaptation_only",
            "comparison_profile": "RMA-style internal two-stage control",
            "exact_rma_reproduction": False,
            "single_variable_ablation": False,
            "rollout_policy": "student latent drives frozen original TF actor",
            "supervision": "no_grad TF teacher latent label",
            "objective": "mean unsquared L2 norm",
            "action_mse": {"role": "diagnostic", "weight": 0.0},
            "ppo_updates": False,
            "planned_supervised_updates_per_rollout": 20,
            "control_frequency_hz": 50,
            "environment_difference": "random-onset actuator degradation in WIM onset environment",
            "source_checkpoint": {
                "path": self.cfg["expected_source_path"],
                "sha256": self.cfg["expected_source_sha256"],
                "iteration": 43000,
            },
            "seed": int(self.env.cfg.seed),
            "student_width": int(self.policy_cfg["student_embedding_dim"]),
            "environment": class_to_dict(self.env.cfg),
            "training": train_cfg,
            "optimizer_and_update_contract": self._schedule_contract(),
            "exposure_contract": {
                "num_envs": int(self.env.num_envs),
                "steps_per_env": int(self.num_steps_per_env),
                "new_rollout_batches": self._comparison_max_new_batches,
            },
            "comparison_contract": contract,
            "configuration_sha256": self._comparison_config_sha256,
            "code_source_sha256": self._comparison_code_sha256,
            "shared_gpu_pacing": {
                "step_sleep_ms": self.shared_gpu_step_sleep_ms,
                "minibatch_sleep_ms": self.alg.shared_gpu_minibatch_sleep_ms,
            },
            "run_class": self._comparison_run_class,
            "actual_invocation_max_iterations": self._invocation_max_iterations,
            "scientific_target_new_batches": self._comparison_max_new_batches,
            "rma_reference_differences": {
                "canonical_jt_learning_rate_schedule": "adaptive_KL",
                "original_rma_learning_rate": 5.0e-4,
                "original_rma_objective": "MSE",
                "original_rma_training_steps": 80000000,
                "this_control_learning_rate": 1.0e-5,
                "this_control_objective": "mean unsquared L2",
                "claim": "matched-transition internal pipeline; no optimal-RMA claim",
            },
        }

    def load(self, path, load_optimizer=True):
        loaded = torch.load(path, map_location=self.device)
        loaded["_checkpoint_path"] = path
        state = loaded["model_state_dict"]
        source_is_teacher = not any(key.startswith("student_encoder.") for key in state)
        if source_is_teacher:
            self._validate_fresh_source_invocation()
            self._verify_source(path)
            incompatible = self.alg.actor_critic.load_state_dict(state, strict=False)
            allowed_missing = {
                key for key in self.alg.actor_critic.state_dict()
                if key.startswith("student_encoder.")
            }
            if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "TF->separate Student mismatch: missing={} unexpected={}".format(
                        incompatible.missing_keys, incompatible.unexpected_keys
                    )
                )
            self.current_learning_iteration = COMPARISON_ORIGIN
            self.alg.actor_critic.set_schedule_origin(self.current_learning_iteration)
            self._comparison_lineage = self._new_lineage()
            self._write_launch_artifacts()
            return loaded.get("infos")
        return self._restore_own_checkpoint(loaded, load_optimizer)

    def _checkpoint_extra_metadata(self):
        frozen = self._frozen_hashes()
        initial = self._comparison_lineage["initial_module_sha256"]
        for name, value in frozen.items():
            if value != initial[name]:
                raise RuntimeError("frozen B2 component changed: {}".format(name))
        return {"frozen_component_sha256": frozen}

    def _frozen_hashes(self):
        hashes = _model_hashes(self.alg.actor_critic)
        hashes.pop("student_encoder")
        return hashes

    def _post_resume_validation(self, loaded):
        metadata = loaded["infos"].get("frozen_component_sha256")
        actual = self._frozen_hashes()
        if metadata != actual:
            raise RuntimeError("B2 frozen component hashes do not match checkpoint metadata")
        initial = self._comparison_lineage["initial_module_sha256"]
        if any(actual[name] != initial[name] for name in actual):
            raise RuntimeError("B2 frozen components diverged from original TF initialization")

    def save(self, path, infos=None, iteration=None):
        info = dict(infos or {})
        info.update(
            {
                "training_mode": "adaptation_only",
                "global_iteration_origin": 43000,
                "student_evaluation_mode": "act_inference_student",
            }
        )
        return self._save_comparison(super().save, path, info, iteration)

    def log(self, locs, width=80, pad=35):
        del width, pad
        iteration_time = locs["collection_time"] + locs["learn_time"]
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += iteration_time
        update = locs["ppo_metrics"]
        metrics = {
            "iteration": locs["it"],
            "wall_time_unix": time.time(),
            "total_transitions": self.tot_timesteps,
            "Adaptation/mode": update["mode"],
            "Adaptation/l2_unsquared": update["adaptation_l2_unsquared"],
            "Adaptation/action_mse_diagnostic": update["action_mse_diagnostic"],
            "Adaptation/action_mse_weight": update["action_mse_weight"],
            "Adaptation/planned_supervised_updates": update["planned_supervised_updates"],
            "Adaptation/completed_supervised_updates": update["completed_supervised_updates"],
            "Adaptation/student_gradient_norm": update["mean_student_gradient_norm"],
            "Adaptation/student_parameter_step_l2": update[
                "student_parameter_step_l2"
            ],
            "Adaptation/nonfinite_update_skipped": update["nonfinite_update_skipped"],
            "Optimization/learning_rate": self.alg.learning_rate,
            "Pacing/step_sleep_ms": self.shared_gpu_step_sleep_ms,
            "Pacing/minibatch_sleep_ms": self.alg.shared_gpu_minibatch_sleep_ms,
            "Rollout/student_driven": True,
            "Rollout/mean_forward_velocity": locs["rollout_forward_velocity"],
            "Rollout/reset_rate_per_step": locs["rollout_reset_rate"],
            "Perf/total_fps": int(
                self.num_steps_per_env * self.env.num_envs / max(iteration_time, 1.0e-9)
            ),
            "Perf/collection_time": locs["collection_time"],
            "Perf/learning_time": locs["learn_time"],
        }
        if len(locs["rewbuffer"]) > 0:
            metrics["Train/mean_reward"] = statistics.mean(locs["rewbuffer"])
            metrics["Train/mean_episode_length"] = statistics.mean(locs["lenbuffer"])
        for key, value in metrics.items():
            if self.writer is not None and isinstance(value, (int, float, bool)):
                self.writer.add_scalar(key, value, locs["it"])
        with open(os.path.join(self.log_dir, "metrics.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
            handle.flush()
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is not None:
            wandb.log(metrics, step=locs["it"])
