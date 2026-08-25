"""Runner that warm-starts joint training from the selected TF checkpoint."""

import os
import re

import torch

from .teacher_runner import TeacherOnPolicyRunner


class JointTeacherStudentRunner(TeacherOnPolicyRunner):
    def load(self, path, load_optimizer=True):
        loaded = torch.load(path, map_location=self.device)
        state = loaded["model_state_dict"]
        source_is_teacher = not any(
            key.startswith("student_encoder.") for key in state
        )
        if not source_is_teacher:
            result = super().load(path, load_optimizer=load_optimizer)
            self.alg.actor_critic.set_schedule_origin(
                int(loaded.get("infos", {}).get("joint_schedule_origin", 0))
                if isinstance(loaded.get("infos"), dict)
                else 0
            )
            return result

        incompatible = self.alg.actor_critic.load_state_dict(state, strict=False)
        allowed_missing = {
            key
            for key in self.alg.actor_critic.state_dict()
            if key.startswith("student_encoder.")
        }
        if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "TF->JT checkpoint mismatch: missing={} unexpected={}".format(
                    incompatible.missing_keys, incompatible.unexpected_keys
                )
            )
        match = re.fullmatch(r"model_(\d+)\.pt", os.path.basename(path))
        self.current_learning_iteration = int(
            match.group(1) if match else loaded.get("iter", 0)
        )
        self.alg.actor_critic.set_schedule_origin(self.current_learning_iteration)
        print(
            "Initialized JT from TF at iteration {}; optimizer reset for new student parameters".format(
                self.current_learning_iteration
            ),
            flush=True,
        )
        return loaded.get("infos")

    def save(self, path, infos=None, iteration=None):
        info = dict(infos or {})
        info["joint_schedule_origin"] = self.alg.actor_critic.schedule_origin_iteration
        return super().save(path, infos=info, iteration=iteration)
