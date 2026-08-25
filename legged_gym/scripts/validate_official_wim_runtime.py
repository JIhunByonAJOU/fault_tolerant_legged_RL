"""Non-training 64-environment x 1000-step GPU validation."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


def _atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_runtime(args):
    import torch
    from legged_gym.envs import task_registry

    if args.task != "a1_official_wim_rough" or args.num_envs != 64 or args.steps != 1000:
        raise ValueError("runtime contract is exactly a1_official_wim_rough 64x1000")
    if not args.headless or not str(args.sim_device).startswith("cuda") or not str(args.rl_device).startswith("cuda"):
        raise ValueError("runtime contract requires headless CUDA simulation and RL devices")
    env, cfg = task_registry.make_env(name=args.task, args=args)
    observations = env.get_observations()
    if tuple(observations.shape) != (64, 235):
        raise RuntimeError("unexpected observation shape {}".format(tuple(observations.shape)))
    checked = ("observations", "actions", "rewards", "root_states", "contact_forces")
    reset_count = 0
    for step in range(1000):
        actions = torch.zeros((64, 12), dtype=torch.float, device=env.device)
        observations, _, rewards, dones, _ = env.step(actions)
        tensors = {"observations": observations, "actions": actions, "rewards": rewards, "root_states": env.root_states, "contact_forces": env.contact_forces}
        for name, tensor in tensors.items():
            if not torch.isfinite(tensor).all():
                raise RuntimeError("non-finite {} at step {}".format(name, step))
        reset_count += int(dones.sum().item())
    source = Path(__file__).resolve()
    return {
        "valid": True,
        "task": args.task,
        "num_envs": 64,
        "steps": 1000,
        "seed": args.seed,
        "headless": True,
        "sim_device": args.sim_device,
        "rl_device": args.rl_device,
        "observation_shape": [64, 235],
        "checked_tensors": checked,
        "finite_every_step": True,
        "episode_resets": reset_count,
        "validator_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", required=True)
    known, remaining = parser.parse_known_args(argv)
    aliases = {"--num-envs": "--num_envs", "--sim-device": "--sim_device", "--rl-device": "--rl_device"}
    remaining = [aliases.get(token, token) for token in remaining]
    saved = sys.argv
    try:
        sys.argv = [saved[0]] + remaining
        import isaacgym  # must precede helpers, which imports torch
        import legged_gym.envs
        from legged_gym.utils.helpers import get_args
        args = get_args()
    finally:
        sys.argv = saved
    args.steps = known.steps
    result = validate_runtime(args)
    _atomic(known.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
