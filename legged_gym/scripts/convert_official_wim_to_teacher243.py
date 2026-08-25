"""Create a behavior-preserving teacher243 warm-start checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import isaacgym  # noqa: F401; package imports require Isaac Gym before torch
import torch

from legged_gym.learning.official_wim_teacher_actor_critic import (
    OfficialWimTeacherActorCritic,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(source, output_dir):
    source = Path(source).resolve()
    output_dir = Path(output_dir).resolve()
    output = output_dir / "model_0.pt"
    manifest = output_dir / "warm_start_manifest.json"
    if not source.is_file() or source.name != "model_1500.pt":
        raise ValueError("--source must be the preserved official model_1500.pt")
    if output.exists() or manifest.exists():
        raise FileExistsError("warm-start output already exists")
    output_dir.mkdir(parents=True, exist_ok=True)

    original = torch.load(str(source), map_location="cpu")
    source_state = original["model_state_dict"]
    model = OfficialWimTeacherActorCritic(235, 45, 12)
    target_state = model.state_dict()

    for key in target_state:
        if key.startswith("teacher_encoder."):
            continue
        if key in ("actor.0.weight", "critic.0.weight"):
            source_weight = source_state[key]
            if source_weight.shape[1] != 235 or target_state[key].shape[1] != 243:
                raise RuntimeError("unexpected first-layer checkpoint shape")
            target_state[key].zero_()
            target_state[key][:, :235].copy_(source_weight)
            continue
        if key not in source_state or source_state[key].shape != target_state[key].shape:
            raise RuntimeError("official checkpoint mismatch at {}".format(key))
        target_state[key].copy_(source_state[key])
    model.load_state_dict(target_state)
    optimizer_state = _expand_optimizer_state(
        original["optimizer_state_dict"], model
    )

    torch.manual_seed(17)
    obs = torch.randn(32, 235)
    privileged = torch.randn(32, 45)
    with torch.inference_mode():
        source_actions = _sequential(source_state, "actor", obs)
        source_values = _sequential(source_state, "critic", obs)
        target_actions = model.act_inference(obs, privileged)
        target_values = model.evaluate(obs, privileged)
    action_error = (source_actions - target_actions).abs().max().item()
    value_error = (source_values - target_values).abs().max().item()
    if action_error != 0.0 or value_error != 0.0:
        raise RuntimeError("warm start did not preserve official behavior exactly")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer_state,
            "iter": 0,
            "infos": {
                "source_checkpoint": str(source),
                "source_sha256": sha256(source),
                "action_max_abs_error": action_error,
                "value_max_abs_error": value_error,
            },
            "warm_start_from_official_wim": True,
        },
        str(output),
    )
    record = {
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "actor_input": "235+8=243",
        "latent_columns_initialized_to_zero": True,
        "optimizer_state_expanded": True,
        "continued_learning_rate": optimizer_state["param_groups"][0]["lr"],
        "action_max_abs_error": action_error,
        "value_max_abs_error": value_error,
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def _expand_optimizer_state(source_optimizer, model):
    """Preserve Adam history while inserting encoder and latent parameters."""
    source_names = ["std"]
    source_names += [
        "actor.{}.{}".format(index, field)
        for index in (0, 2, 4, 6)
        for field in ("weight", "bias")
    ]
    source_names += [
        "critic.{}.{}".format(index, field)
        for index in (0, 2, 4, 6)
        for field in ("weight", "bias")
    ]
    source_ids = dict(zip(source_names, source_optimizer["param_groups"][0]["params"]))
    target_named = list(model.named_parameters())
    target_state = {}
    for target_id, (name, parameter) in enumerate(target_named):
        if name.startswith("teacher_encoder."):
            continue
        source_id = source_ids[name]
        state = source_optimizer["state"][source_id]
        copied = {"step": state["step"].clone()}
        for field in ("exp_avg", "exp_avg_sq"):
            source_tensor = state[field]
            if source_tensor.shape == parameter.shape:
                copied[field] = source_tensor.clone()
            elif name in ("actor.0.weight", "critic.0.weight"):
                expanded = torch.zeros_like(parameter)
                expanded[:, :235].copy_(source_tensor)
                copied[field] = expanded
            else:
                raise RuntimeError("optimizer shape mismatch at {}".format(name))
        target_state[target_id] = copied
    group = dict(source_optimizer["param_groups"][0])
    group["params"] = list(range(len(target_named)))
    return {"state": target_state, "param_groups": [group]}


def _sequential(state, prefix, inputs):
    value = inputs
    for index in (0, 2, 4, 6):
        value = torch.nn.functional.linear(
            value, state["{}.{}.weight".format(prefix, index)], state["{}.{}.bias".format(prefix, index)]
        )
        if index != 6:
            value = torch.nn.functional.elu(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
