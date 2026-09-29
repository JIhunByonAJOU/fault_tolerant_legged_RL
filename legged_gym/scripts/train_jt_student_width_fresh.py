"""Fresh JT training from the selected TF43000 with a configurable student width."""

import argparse
import hashlib
import json
import math
import signal
import sys
from pathlib import Path

import isaacgym  # noqa: F401
import torch

from legged_gym.envs import task_registry
from legged_gym.utils.helpers import class_to_dict, get_args


TASK = 'a1_official_wim_jt_failure_fullrange_onset'
TF = Path('logs/official_wim_teacher243_failure_fullrange_fromscratch/'
          'Aug13_03-44-07_seed1-50000iter-fromscratch-fullrange/model_43000.pt')
EXPECTED_TF_SHA256 = '944a697abb30dfc8023e15544d0909acfcdaa4d8c4c0f930656847398f150635'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def module_digest(module):
    value = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--student-width', type=int, choices=(32, 64), required=True)
    parser.add_argument('--updates', type=int, required=True)
    parser.add_argument('--initial-learning-rate', type=float, default=None,
                        help='Explicit fresh-JT starting LR; subsequent adaptive schedule is unchanged.')
    own, rest = parser.parse_known_args()
    if own.updates <= 0:
        raise ValueError('updates must be positive')
    if own.initial_learning_rate is not None and (
            not math.isfinite(own.initial_learning_rate) or own.initial_learning_rate <= 0):
        raise ValueError('initial learning rate must be finite and positive')
    sys.argv = [sys.argv[0]] + rest
    args = get_args()
    args.task = TASK
    if not args.headless or args.sim_device != 'cuda:0' or args.rl_device != 'cuda:0':
        raise ValueError('fresh JT requires headless CUDA device 0')
    if args.resume or args.load_run is not None or args.checkpoint is not None:
        raise ValueError('fresh JT cannot resume a checkpoint or optimizer')
    if args.num_envs is None or args.seed is None or not args.log_dir:
        raise ValueError('explicit num_envs, seed, and log_dir required')
    output = Path(args.log_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    source = TF.resolve()
    if digest(source) != EXPECTED_TF_SHA256:
        raise RuntimeError('TF43000 source hash changed')
    output.mkdir(parents=True)
    _, train_cfg = task_registry.get_cfgs(TASK)
    train_cfg.policy.student_embedding_dim = own.student_width
    train_cfg.runner.max_iterations = own.updates
    if own.initial_learning_rate is not None:
        train_cfg.algorithm.learning_rate = own.initial_learning_rate
    env, env_cfg = task_registry.make_env(name=TASK, args=args)
    runner, train_cfg = task_registry.make_alg_runner(env=env, train_cfg=train_cfg, args=args)
    runner.load(str(source), load_optimizer=False)
    model = runner.alg.actor_critic
    if runner.current_learning_iteration != 43000 or model.schedule_origin_iteration != 43000:
        raise RuntimeError('fresh JT origin is not TF43000')
    if model.student_encoder.frame_encoder[0].out_features != own.student_width:
        raise RuntimeError('student width mismatch')
    expected_lr = float(train_cfg.algorithm.learning_rate)
    initial_group_lrs = [float(group['lr']) for group in runner.alg.optimizer.param_groups]
    if runner.alg.learning_rate != expected_lr or any(lr != expected_lr for lr in initial_group_lrs):
        raise RuntimeError('fresh JT starting learning rate was not applied to algorithm and optimizer')
    manifest = {
        'source': str(source), 'source_sha256': EXPECTED_TF_SHA256,
        'source_iteration': 43000, 'optimizer_loaded': False,
        'initial_learning_rate': expected_lr,
        'initial_optimizer_group_lrs': initial_group_lrs,
        'student_width': own.student_width, 'new_iterations': own.updates,
        'seed': args.seed, 'num_envs': args.num_envs,
        'rollout_steps': runner.num_steps_per_env,
        'initial_teacher_sha256': module_digest(model.teacher_encoder),
        'initial_student_sha256': module_digest(model.student_encoder),
        'initial_actor_sha256': module_digest(model.actor),
        'environment': class_to_dict(env_cfg),
        'training': class_to_dict(train_cfg),
    }
    manifest_path = output / 'fresh_jt_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.run_name or output.name, dir=str(output),
            config=manifest, sync_tensorboard=False,
        )
        wandb.define_metric('iteration')
        wandb.define_metric('*', step_metric='iteration')
        manifest['wandb_run_id'] = wandb_run.id
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    def stop(signum, _frame):
        runner.request_stop('signal %d' % signum)

    previous = {number: signal.signal(number, stop) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        runner.learn(num_learning_iterations=own.updates, init_at_random_ep_len=True)
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
        if wandb_run is not None:
            wandb_run.finish()
        manifest['completed_new_iterations'] = runner.current_learning_iteration - 43000
        manifest['final_teacher_sha256'] = module_digest(model.teacher_encoder)
        manifest['final_student_sha256'] = module_digest(model.student_encoder)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print('COMPLETE', manifest['completed_new_iterations'], flush=True)


if __name__ == '__main__':
    main()
