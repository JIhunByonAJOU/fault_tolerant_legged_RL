"""Controlled JT pilots from an explicit TF or JT checkpoint.

The baseline, teacher-freeze, and student-action arms share the canonical JT
environment and PPO settings. Each invocation writes a new run directory; a
fresh TF43000 source starts JT with a new optimizer and student encoder.
"""

import argparse
import hashlib
import json
import signal
from pathlib import Path

import isaacgym  # noqa: F401
import torch
from legged_gym.envs import task_registry
from legged_gym.utils.helpers import class_to_dict, get_args


TASK = 'a1_official_wim_jt_failure_fullrange_onset'
TF = Path('logs/official_wim_teacher243_failure_fullrange_fromscratch/'
          'Aug13_03-44-07_seed1-50000iter-fromscratch-fullrange/model_43000.pt')


def sha_file(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def sha_parameters(module):
    value = hashlib.sha256()
    for name, parameter in sorted(module.state_dict().items()):
        value.update(name.encode())
        value.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--condition', required=True, choices=['baseline', 'freeze_teacher_encoder', 'student_action_aux', 'beta_floor', 'wider_student'])
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--updates', type=int, required=True)
    parser.add_argument('--source-checkpoint', default=str(TF))
    parser.add_argument('--student-action-beta', type=float, default=10.0)
    parser.add_argument('--adaptation-beta-floor', type=float, default=0.1)
    parser.add_argument('--student-width', type=int, choices=(32, 64), default=64)
    own, rest = parser.parse_known_args()
    if own.updates <= 0:
        raise ValueError('updates must be positive')
    if own.student_action_beta <= 0.0:
        raise ValueError('student-action-beta must be positive')
    if not 0.0 < own.adaptation_beta_floor < 1.0:
        raise ValueError('adaptation-beta-floor must be in (0, 1)')
    import sys
    sys.argv = [sys.argv[0]] + rest
    args = get_args()
    args.task = TASK
    if not args.headless or args.sim_device != 'cuda:0' or args.rl_device != 'cuda:0':
        raise ValueError('Ablation requires headless CUDA on device 0')
    if args.num_envs is None or args.seed is None:
        raise ValueError('Explicit num_envs and seed required')
    if args.wandb or args.resume or args.load_run is not None or args.checkpoint is not None:
        raise ValueError('This launcher loads only its fixed TF source')
    output = Path(own.output_dir).resolve()
    source = Path(own.source_checkpoint).resolve()
    if not source.is_file() or not source.stem.startswith('model_'):
        raise ValueError('source checkpoint must be an existing model_*.pt')
    source_iteration = int(source.stem.split('_')[-1])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    args.log_dir = str(output)
    env, env_cfg = task_registry.make_env(name=TASK, args=args)
    _, train_cfg = task_registry.get_cfgs(TASK)
    if own.condition == 'student_action_aux':
        train_cfg.policy.student_action_beta = own.student_action_beta
    if own.condition == 'beta_floor':
        train_cfg.policy.adaptation_beta_floor = own.adaptation_beta_floor
    if own.condition == 'wider_student':
        train_cfg.policy.student_embedding_dim = own.student_width
    runner, train_cfg = task_registry.make_alg_runner(env=env, train_cfg=train_cfg, args=args)
    source_is_teacher = source == TF.resolve()
    runner.load(str(source), load_optimizer=not source_is_teacher)
    if not source_is_teacher:
        runner.alg.learning_rate = float(runner.alg.optimizer.param_groups[0]['lr'])
    model = runner.alg.actor_critic
    if model.schedule_origin_iteration != 43000 or runner.current_learning_iteration != source_iteration:
        raise RuntimeError('JT schedule or source iteration mismatch')
    if own.condition == 'freeze_teacher_encoder':
        model.teacher_encoder.requires_grad_(False)
    trainable_teacher = sum(p.numel() for p in model.teacher_encoder.parameters() if p.requires_grad)
    manifest = {
        'condition': own.condition,
        'teacher_encoder_trainable_parameters': trainable_teacher,
        'student_action_beta': model.student_action_beta,
        'student_width': model.student_encoder.frame_encoder[0].out_features,
        'adaptation_beta_floor': model.adaptation_beta_floor,
        'source_checkpoint': str(source),
        'source_checkpoint_sha256': sha_file(source),
        'source_is_teacher': source_is_teacher,
        'source_iteration': source_iteration,
        'optimizer_learning_rate': float(runner.alg.optimizer.param_groups[0]['lr']),
        'initial_teacher_sha256': sha_parameters(model.teacher_encoder),
        'initial_student_sha256': sha_parameters(model.student_encoder),
        'initial_actor_sha256': sha_parameters(model.actor),
        'seed': args.seed,
        'num_envs': args.num_envs,
        'rollout_steps': runner.num_steps_per_env,
        'planned_new_iterations': own.updates,
        'resolved_environment': class_to_dict(env_cfg),
        'resolved_training': class_to_dict(train_cfg),
    }
    (output / 'ablation_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True))

    def stop(signum, _frame):
        runner.request_stop('signal %d' % signum)

    previous = {number: signal.signal(number, stop) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        runner.learn(num_learning_iterations=own.updates, init_at_random_ep_len=True)
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    manifest['completed_new_iterations'] = runner.current_learning_iteration - source_iteration
    manifest['final_teacher_sha256'] = sha_parameters(model.teacher_encoder)
    manifest['final_student_sha256'] = sha_parameters(model.student_encoder)
    (output / 'ablation_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print('COMPLETE', own.condition, manifest['completed_new_iterations'], flush=True)


if __name__ == '__main__':
    main()
