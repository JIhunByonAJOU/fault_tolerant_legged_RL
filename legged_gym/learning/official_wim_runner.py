"""Logging-only adapter around the unmodified rsl_rl v1.0.2 runner."""

import json
import math
import os
import statistics
import tempfile
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner


_PRE_ENV_CONTEXT = None


def set_pre_environment_context(context):
    global _PRE_ENV_CONTEXT
    _PRE_ENV_CONTEXT = context


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class OfficialWimOnPolicyRunner(OnPolicyRunner):
    """Official runner with post-hoc, scalar-only durable logging."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        context = _PRE_ENV_CONTEXT
        wandb_run = None if context is None else context.get("wandb_run")
        if wandb_run is None:
            raise RuntimeError("Official WIM runner requires one active pre-environment W&B run")
        if log_dir is None or Path(log_dir).resolve() != Path(context["log_dir"]).resolve():
            raise RuntimeError("Official WIM runner log directory differs from managed preflight")
        self._wandb_run = wandb_run
        self._resolved_config = dict(context["resolved_config"])
        self._resolved_config["training"] = train_cfg
        self._resolved_config["environment"] = context["environment"]
        self._resolved_config["schema"] = {
            "actor_observation_dim": env.num_obs,
            "critic_observation_dim": env.num_privileged_obs if env.num_privileged_obs is not None else env.num_obs,
            "num_actions": env.num_actions,
        }
        resolved_path = Path(log_dir) / "resolved_config.json"
        if resolved_path.exists():
            raise RuntimeError("resolved_config.json already exists")
        _atomic_json(resolved_path, self._resolved_config)

    def _append_metric_row(self, row):
        if not all(
            math.isfinite(float(value))
            for value in row.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ):
            raise RuntimeError("refusing to serialize a non-finite official WIM metric")
        row["finite"] = True
        encoded = json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
        path = Path(self.log_dir) / "metrics.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        self._wandb_run.log(dict(row), step=row["iteration"])
        mirror_path = Path(self.log_dir) / "wandb_metrics.jsonl"
        with mirror_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)
        row = {
            "iteration": int(locs["it"]),
            "total_transitions": int(self.tot_timesteps),
            "timing/collection": float(locs["collection_time"]),
            "timing/learning": float(locs["learn_time"]),
            "loss/value_function": float(locs["mean_value_loss"]),
            "loss/surrogate": float(locs["mean_surrogate_loss"]),
            "learning_rate": float(self.alg.learning_rate),
            "policy/mean_noise_std": float(self.alg.actor_critic.std.mean().item()),
        }
        if len(locs["rewbuffer"]) > 0:
            row["train/mean_reward"] = float(statistics.mean(locs["rewbuffer"]))
            row["train/mean_episode_length"] = float(statistics.mean(locs["lenbuffer"]))
        for key in sorted(locs["ep_infos"][0]) if locs["ep_infos"] else ():
            values = []
            for info in locs["ep_infos"]:
                value = info[key]
                values.append(float(value.detach().float().mean().cpu()) if isinstance(value, torch.Tensor) else float(value))
            row["episode/" + key] = float(statistics.mean(values))
        self._append_metric_row(row)
