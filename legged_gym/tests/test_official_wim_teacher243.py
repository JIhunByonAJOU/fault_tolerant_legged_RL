import tempfile
import unittest
from pathlib import Path

import isaacgym  # noqa: F401; Isaac Gym must precede torch
import torch

from legged_gym.envs.a1_official_wim_teacher import (
    A1OfficialWimTeacher243Cfg,
    A1OfficialWimTeacher243CfgPPO,
)
from legged_gym.envs.a1_official_wim_teacher.schema import schema_manifest
from legged_gym.learning.official_wim_teacher_actor_critic import (
    OfficialWimTeacherActorCritic,
)
from legged_gym.scripts.convert_official_wim_to_teacher243 import convert
from legged_gym.utils.helpers import class_to_dict


class OfficialWimTeacher243Test(unittest.TestCase):
    def test_schema_and_config(self):
        schema = schema_manifest()
        self.assertEqual(schema["actor_observation_dim"], 235)
        self.assertEqual(schema["privileged_observation_dim"], 45)
        self.assertEqual(schema["teacher_latent_dim"], 8)
        self.assertEqual(schema["actor_input_dim"], 243)
        env = class_to_dict(A1OfficialWimTeacher243Cfg())
        train = class_to_dict(A1OfficialWimTeacher243CfgPPO())
        self.assertEqual(env["env"]["num_observations"], 235)
        self.assertEqual(env["env"]["num_privileged_obs"], 45)
        self.assertTrue(env["terrain"]["measure_heights"])
        self.assertEqual(len(env["terrain"]["measured_points_x"]) * len(env["terrain"]["measured_points_y"]), 187)
        self.assertEqual(train["policy"]["teacher_latent_dim"], 8)
        self.assertEqual(train["runner"]["policy_class_name"], "OfficialWimTeacherActorCritic")

    def test_architecture_gradients_and_interventions(self):
        torch.manual_seed(3)
        model = OfficialWimTeacherActorCritic(235, 45, 12)
        obs = torch.randn(16, 235)
        privileged = torch.randn(16, 45)
        latent = model.encode_privileged(privileged)
        self.assertEqual(tuple(latent.shape), (16, 8))
        self.assertEqual(model.actor[0].in_features, 243)
        actions = model.act_inference(obs, privileged)
        zero_actions = model.act_inference_with_latent(obs, torch.zeros_like(latent))
        self.assertEqual(tuple(actions.shape), (16, 12))
        self.assertGreater((actions - zero_actions).abs().max().item(), 0.0)
        loss = actions.square().mean() + model.evaluate(obs, privileged).square().mean()
        loss.backward()
        encoder_grad = sum(
            parameter.grad.abs().sum().item()
            for parameter in model.teacher_encoder.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(encoder_grad, 0.0)

    def test_warm_start_is_exact_and_refuses_overwrite(self):
        torch.manual_seed(5)
        source_model = OfficialWimTeacherActorCritic(235, 45, 12)
        source = {
            key: value.clone()
            for key, value in source_model.state_dict().items()
            if not key.startswith("teacher_encoder.")
        }
        source["actor.0.weight"] = source["actor.0.weight"][:, :235].clone()
        source["critic.0.weight"] = source["critic.0.weight"][:, :235].clone()
        with tempfile.TemporaryDirectory(prefix="teacher243-test-") as temporary:
            root = Path(temporary)
            checkpoint = root / "model_1500.pt"
            source_optimizer = torch.optim.Adam(
                [torch.nn.Parameter(value.clone()) for value in source.values()],
                lr=1.1390625e-4,
            )
            for parameter in source_optimizer.param_groups[0]["params"]:
                parameter.grad = torch.ones_like(parameter)
            source_optimizer.step()
            torch.save(
                {
                    "model_state_dict": source,
                    "optimizer_state_dict": source_optimizer.state_dict(),
                    "iter": 1500,
                },
                checkpoint,
            )
            output_dir = root / "warm"
            record = convert(checkpoint, output_dir)
            self.assertEqual(record["action_max_abs_error"], 0.0)
            self.assertEqual(record["value_max_abs_error"], 0.0)
            converted = torch.load(output_dir / "model_0.pt", map_location="cpu")
            self.assertTrue(converted["warm_start_from_official_wim"])
            self.assertIn("optimizer_state_dict", converted)
            self.assertAlmostEqual(
                converted["optimizer_state_dict"]["param_groups"][0]["lr"],
                1.1390625e-4,
            )
            self.assertTrue(torch.equal(
                converted["model_state_dict"]["actor.0.weight"][:, :235],
                source["actor.0.weight"],
            ))
            self.assertEqual(
                converted["model_state_dict"]["actor.0.weight"][:, 235:].abs().max().item(),
                0.0,
            )
            with self.assertRaises(FileExistsError):
                convert(checkpoint, output_dir)


if __name__ == "__main__":
    unittest.main()
