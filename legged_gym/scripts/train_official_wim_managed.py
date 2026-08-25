"""Isolated managed launcher that activates W&B before constructing Isaac Gym."""

import json
import os
import signal
import sys
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXPECTED_PACKAGE_INIT = (_REPO_ROOT / "legged_gym" / "__init__.py").resolve()


def _prepend_unique_path(path, entries):
    path = Path(path).resolve()
    entries[:] = [entry for entry in entries if Path(entry or os.curdir).resolve() != path]
    entries.insert(0, str(path))


def _bootstrap_managed_imports():
    """Pin imports and build-tool lookup to this checkout and interpreter."""
    _prepend_unique_path(_REPO_ROOT, sys.path)
    path_entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    _prepend_unique_path(Path(sys.executable).resolve().parent, path_entries)
    os.environ["PATH"] = os.pathsep.join(path_entries)


_bootstrap_managed_imports()

import legged_gym

if Path(legged_gym.__file__).resolve() != _EXPECTED_PACKAGE_INIT:
    raise ImportError(
        "managed official WIM entrypoint resolved legged_gym outside this repository: {}".format(
            legged_gym.__file__
        )
    )

from legged_gym.official_wim.conformance import build_provenance, pre_environment_conformance


TASK = "a1_official_wim_rough"
_TRAINING_ARTIFACT_NAMES = {"resolved_config.json", "metrics.jsonl"}


class ManagedSignal(RuntimeError):
    pass


def _incident(log_dir, stage, exc):
    path = Path(log_dir) / "official_wim_preflight_incident.json"
    payload = {"stage": stage, "error_type": type(exc).__name__, "error": str(exc)}
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def run_pure_preflight(args):
    if args.task != TASK:
        raise ValueError("managed official WIM entrypoint requires task={}".format(TASK))
    if not args.wandb:
        raise ValueError("managed official WIM entrypoint requires --wandb")
    if args.resume or args.load_run is not None or args.checkpoint is not None:
        raise ValueError("managed official WIM runs must start fresh with no resume source")
    if not args.log_dir or not args.run_name:
        raise ValueError("explicit --log_dir and --run_name are required")
    log_dir = Path(args.log_dir).resolve()
    if log_dir.name != args.run_name:
        raise ValueError("run_name must exactly equal the log_dir basename")
    log_dir.mkdir(parents=True, exist_ok=True)
    forbidden = list(_TRAINING_ARTIFACT_NAMES)
    forbidden.extend(path.name for path in log_dir.glob("model_*.pt"))
    forbidden.extend(path.name for path in log_dir.glob("events.out.tfevents.*"))
    existing = sorted(name for name in set(forbidden) if (log_dir / name).exists())
    if existing:
        raise ValueError("managed output directory contains training artifacts: {}".format(existing))
    conformance = pre_environment_conformance()
    return {"log_dir": str(log_dir), "conformance": conformance}


def initialize_wandb_before_environment(args):
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B is unavailable") from exc
    run = wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        dir=str(Path(args.log_dir).resolve()),
        config={
            "task": TASK,
            "metric_schema": [
                "iteration", "total_transitions", "timing/collection", "timing/learning",
                "loss/value_function", "loss/surrogate", "learning_rate",
                "policy/mean_noise_std", "finite",
            ],
        },
    )
    if run is None:
        raise RuntimeError("wandb.init returned no active run")
    if getattr(wandb, "run", None) is not run:
        run.finish()
        raise RuntimeError("wandb.init did not activate the returned run")
    return run


def finish_wandb_and_restore_signals(wandb_run, previous_handlers):
    for signum, previous in previous_handlers.items():
        signal.signal(signum, previous)
    if wandb_run is not None:
        wandb_run.finish()


def train(args=None):
    if args is None:
        import isaacgym  # must precede helpers, which imports torch
        import legged_gym.envs  # initialize the stock registry without constructing an environment
        from legged_gym.utils.helpers import get_args
        args = get_args()
    wandb_run = None
    previous_handlers = {}
    stage = "pure_preflight"
    try:
        preflight = run_pure_preflight(args)
        stage = "wandb_init"
        wandb_run = initialize_wandb_before_environment(args)

        stage = "environment_construction"
        from legged_gym.envs import task_registry
        from legged_gym.learning.official_wim_runner import set_pre_environment_context
        from legged_gym.utils.helpers import class_to_dict

        env, env_cfg = task_registry.make_env(name=args.task, args=args)
        provenance = build_provenance(args, preflight["conformance"])
        context = {
            "wandb_run": wandb_run,
            "log_dir": preflight["log_dir"],
            "environment": class_to_dict(env_cfg),
            "resolved_config": provenance,
        }
        set_pre_environment_context(context)
        stage = "runner_construction"
        ppo_runner, train_cfg = task_registry.make_alg_runner(
            env=env, name=args.task, args=args, log_root=args.log_dir
        )
        wandb_run.config.update(ppo_runner._resolved_config, allow_val_change=True)

        def _signal_handler(signum, _frame):
            raise ManagedSignal("received signal {}".format(signum))

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _signal_handler)
        stage = "learn"
        ppo_runner.learn(
            num_learning_iterations=train_cfg.runner.max_iterations,
            init_at_random_ep_len=True,
        )
        return ppo_runner
    except BaseException as exc:
        if args is not None and getattr(args, "log_dir", None) and stage != "learn":
            try:
                _incident(args.log_dir, stage, exc)
            except OSError:
                pass
        raise
    finally:
        finish_wandb_and_restore_signals(wandb_run, previous_handlers)


def main():
    train()


if __name__ == "__main__":
    main()
