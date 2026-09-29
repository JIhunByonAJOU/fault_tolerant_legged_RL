"""Versioned diagnostic evaluator: common environment, locked command, fixed horizon.

No training is performed. Policy checkpoints are loaded strictly, without optimizers.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import isaacgym  # noqa: F401 -- must precede torch
import torch
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry
from legged_gym.learning.official_wim_teacher_actor_critic import OfficialWimTeacherActorCritic
from legged_gym.learning.joint_teacher_student_actor_critic import JointTeacherStudentActorCritic
from legged_gym.evaluation.onset_protocol_v2 import make_specs
from legged_gym.scripts.evaluate_teacher243_failure_onset import atomic_jsonl, sha256_file


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--checkpoint-path', required=True)
    parser.add_argument('--policy-mode', choices=['teacher', 'student', 'jt_teacher', 'jt_mixed'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--corpus-output')
    parser.add_argument('--replicates', type=int, default=48)
    parser.add_argument('--rates', type=float, nargs='+', default=[0., .2, .4, .6, .8, 1.])
    parser.add_argument('--post-seconds', type=float, default=20.)
    parser.add_argument('--command-x', type=float, default=.5)
    parser.add_argument('--terrain', choices=['mixed', 'plane'], default='mixed')
    own, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = get_args()
    if not args.headless or args.sim_device != 'cuda:0' or args.rl_device != 'cuda:0':
        raise ValueError('Scientific evaluation requires headless CUDA devices')
    if own.command_x <= .2 or own.post_seconds <= 0:
        raise ValueError('This protocol requires vx > .2 m/s and positive horizon')
    output = Path(own.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    start = time.monotonic()
    task = 'a1_official_wim_jt_failure_fullrange_onset'
    args.task = task
    cfg, _ = task_registry.get_cfgs(task)
    dt = cfg.sim.dt * cfg.control.decimation
    specs = make_specs(args.seed, own.replicates, own.rates, dt)
    n = len(specs)
    args.num_envs = n
    cfg.env.num_envs = n
    cfg.env.episode_length_s = 10 + own.post_seconds + 2
    cfg.terrain.curriculum = False
    if own.terrain == 'plane':
        cfg.terrain.mesh_type = 'plane'
    cfg.commands.curriculum = False
    cfg.commands.heading_command = False
    cfg.commands.ranges.lin_vel_x = [own.command_x, own.command_x]
    cfg.commands.ranges.lin_vel_y = [0., 0.]
    cfg.commands.ranges.ang_vel_yaw = [0., 0.]
    cfg.noise.add_noise = False
    cfg.domain_rand.push_robots = False
    cfg.domain_rand.failure_onset_time_range_s = [1000000., 1000000.]
    env, _ = task_registry.make_env(task, args, env_cfg=cfg)
    obs, privileged = env.reset()
    # One neutral initialization step is excluded from the evaluation clock.
    # This is identical for the two policies and is recorded in the metadata.
    initial = {name: tensor_hash(getattr(env, name)) for name in (
        'root_states', 'dof_pos', 'dof_vel', 'applied_friction', 'added_base_mass',
        'motor_strength_gt', 'kp_scale_gt', 'kd_scale_gt', 'obs_buf')}
    if hasattr(env, 'terrain'):
        initial['terrain'] = hashlib.sha256(env.terrain.height_field_raw.tobytes()).hexdigest()
    initial_levels = getattr(env, 'terrain_levels', torch.full((n,), -1)).cpu().tolist()
    initial_columns = getattr(env, 'terrain_types', torch.full((n,), -1)).cpu().tolist()
    rng_cpu = torch.get_rng_state()
    rng_gpu = torch.cuda.get_rng_state_all()
    checkpoint = Path(own.checkpoint_path).resolve()
    state = torch.load(str(checkpoint), map_location='cpu')
    if own.policy_mode == 'teacher':
        model = OfficialWimTeacherActorCritic(235, 45, 12)
    else:
        width = state['model_state_dict']['student_encoder.frame_encoder.0.weight'].shape[0]
        model = JointTeacherStudentActorCritic(2635, 45, 12, student_embedding_dim=width)
    model.load_state_dict(state['model_state_dict'], strict=True)
    if own.policy_mode != 'teacher':
        model.set_schedule_origin(int(state.get('infos', {}).get('joint_schedule_origin', 43000)))
        model.set_training_iteration(int(checkpoint.stem.split('_')[-1]))
    model.to(env.device).eval()
    torch.set_rng_state(rng_cpu)
    torch.cuda.set_rng_state_all(rng_gpu)
    device = env.device
    joint = torch.tensor([s[0] for s in specs], device=device)
    rate = torch.tensor([s[1] for s in specs], device=device)
    onset = torch.tensor([s[2] for s in specs], device=device)
    horizon = round(own.post_seconds / dt)
    end = onset + horizon
    ids = torch.arange(n, device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    first_done = torch.full((n,), -1, device=device, dtype=torch.long)
    count = torch.zeros(n, device=device)
    sums = torch.zeros((n, 3), device=device)
    actual_vx = torch.zeros(n, device=device)
    stable_count = torch.zeros(n, device=device)
    strict_count = torch.zeros(n, device=device)
    loose_count = torch.zeros(n, device=device)
    final_count = torch.zeros(n, device=device)
    trace = []
    corpus = []
    sample_ids = torch.tensor([rep * 12 * len(own.rates) + (rep % 12) * len(own.rates) + j
                               for rep in range(own.replicates) for j in range(len(own.rates))], device=device)
    window = round(1. / dt)
    final_steps = min(horizon, round(5. / dt))
    expected = torch.tensor([own.command_x, 0., 0.], device=device)
    expected_obs = expected * env.commands_scale
    with torch.inference_mode():
        for step in range(int(end.max())):
            due = (onset == step) & alive
            due_ids = ids[due]
            if due_ids.numel():
                env.actuator_degradation[due_ids, joint[due_ids]] = rate[due_ids]
                env.motor_strength_gt[due_ids] = env.nominal_motor_strength_gt[due_ids]
                env.motor_strength_gt[due_ids, joint[due_ids]] *= 1. - rate[due_ids]
                env.degraded_joint_index[due_ids] = joint[due_ids]
                # Refresh only the GT channels; do not append a duplicate history frame.
                privileged[:, 2:14] = env.motor_strength_gt
                privileged[:, -1] = (env.actuator_degradation.max(dim=1).values > 0).float()
            if not torch.allclose(env.commands[:, :3], expected.expand(n, -1)):
                raise RuntimeError('Command drift')
            if not torch.allclose(obs[:, 9:12], expected_obs.expand(n, -1)):
                raise RuntimeError('Observed command differs from evaluation target')
            history_commands = obs[:, 235:].reshape(n, 50, 48)[:, :, 9:12]
            history_ok = ((history_commands - expected_obs).abs() < 1e-6).all(dim=-1)
            padding = (history_commands == 0).all(dim=-1)
            if not (history_ok | padding).all():
                raise RuntimeError('Unexpected command in student history')
            if own.policy_mode == 'teacher':
                action = model.act_inference(obs[:, :235], privileged)
            elif own.policy_mode == 'jt_teacher':
                action = model.act_inference_with_latent(obs, model.encode_privileged(privileged))
            elif own.policy_mode == 'jt_mixed':
                action = model.act_inference(obs, privileged)
            else:
                action = model.act_inference_student(obs)
            if not torch.isfinite(action).all():
                raise RuntimeError('Nonfinite action')
            # Sample state at the beginning of each commanded control interval.
            post = alive & (step >= onset) & (step < end)
            if own.corpus_output and step % 25 == 0:
                keep = sample_ids[alive[sample_ids] & (step < end[sample_ids])]
                corpus.append(dict(obs=obs[keep].cpu().clone(), privileged=privileged[keep].cpu().clone(),
                                   env_id=keep.cpu(), rate=rate[keep].cpu(),
                                   elapsed_since_onset=(step-onset[keep]).cpu()*dt))
            error = torch.stack((env.base_lin_vel[:, 0] - own.command_x,
                                 env.base_lin_vel[:, 1], env.base_ang_vel[:, 2]), dim=1)
            if not torch.isfinite(error[alive]).all():
                raise RuntimeError('Nonfinite velocity')
            count += post.float()
            sums += error.square() * post[:, None]
            actual_vx += env.base_lin_vel[:, 0] * post
            # Occupancy includes lateral tracking, unlike the legacy recovery event.
            stable = post & (error[:, 0].abs() <= .1) & (error[:, 1].abs() <= .1) & (error[:, 2].abs() <= .2)
            stable_count += stable.float()
            strict_count += (post & (error[:, :2].abs() <= .05).all(dim=1) & (error[:, 2].abs() <= .1)).float()
            loose_count += (post & (error[:, :2].abs() <= .2).all(dim=1) & (error[:, 2].abs() <= .4)).float()
            final_count += (stable & (step >= end - final_steps)).float()
            if step % 5 == 0:
                for i in range(min(6, n)):
                    trace.append({'step': step, 'env_id': i, 'alive': bool(alive[i]),
                                  'vx': float(env.base_lin_vel[i, 0]), 'vy': float(env.base_lin_vel[i, 1]),
                                  'yaw_rate': float(env.base_ang_vel[i, 2]), 'onset_step': int(onset[i])})
            obs, privileged, _, dones, _ = env.step(action)
            newly_done = alive & (dones > 0) & (step < end)
            first_done[newly_done] = step + 1
            alive[newly_done] = False
            if step % 250 == 0:
                print('step {}/{} elapsed {:.1f}s'.format(step, int(end.max()), time.monotonic()-start), flush=True)
    rows = []
    for i, (j, d, onset_step, rep) in enumerate(specs):
        valid = int(count[i])
        done = int(first_done[i])
        survived = done < 0  # Contact on the final transition also counts as failure.
        rmse = [math.sqrt(float(x) / valid) if valid else None for x in sums[i]]
        rows.append(dict(env_id=i, seed=args.seed, replicate=rep, joint_index=j,
                         joint_name=env.dof_names[j], degradation_rate=d,
                         onset_seconds=onset_step*dt, post_horizon_seconds=own.post_seconds,
                         terrain_level=initial_levels[i], terrain_column=initial_columns[i],
                         failed_before_onset=(done >= 0 and done <= onset_step),
                         survived_post_failure_horizon=survived, first_done_step=done,
                         valid_post_steps=valid, post_command_vx_rmse=rmse[0],
                         post_command_vy_rmse=rmse[1], post_yaw_rate_rmse=rmse[2],
                         post_mean_vx=float(actual_vx[i])/valid if valid else None,
                         tracking_time_fraction=float(stable_count[i])/horizon,
                         tracking_time_fraction_strict=float(strict_count[i])/horizon,
                         tracking_time_fraction_loose=float(loose_count[i])/horizon,
                         final_5s_tracking_fraction=float(final_count[i])/final_steps,
                         ))
    metadata = dict(schema_version=2, evaluation='paired_random_onset_diagnostic_v2',
                    task=task, checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
                    evaluator_sha256=sha256_file(Path(__file__)),
                    protocol_sha256=sha256_file(Path(__file__).parents[1] / 'evaluation/onset_protocol_v2.py'),
                    policy_mode=own.policy_mode, seed=args.seed, num_envs=n, dt=dt,
                    command=[own.command_x, 0., 0.], onset_range_seconds=[2.,10.],
                    post_seconds=own.post_seconds, rates=own.rates, replicates=own.replicates,
                    initial_state_hashes=initial, specs_sha256=hashlib.sha256(json.dumps(specs).encode()).hexdigest(),
                    terrain=own.terrain, noise=False, pushes=False, initialization_neutral_steps=1,
                    metric_sampling='pre-action state; absorbing failure; terminal final transition is failure',
                    tracking_thresholds=[.1,.1,.2], strict_thresholds=[.05,.05,.1], loose_thresholds=[.2,.2,.4],
                    tracking_denominator='full post horizon, including time after failure',
                    elapsed_seconds=time.monotonic()-start)
    if own.corpus_output:
        cp = Path(own.corpus_output)
        if cp.exists():
            raise FileExistsError(cp)
        torch.save({key: torch.cat([x[key] for x in corpus]) for key in corpus[0]}, str(cp))
        metadata['corpus_path'] = str(cp.resolve())
        metadata['corpus_sha256'] = sha256_file(cp)
    metadata['evaluation'] = 'jt_branch_diagnostic_v1'
    metadata['inference_alpha'] = getattr(model, 'adaptation_alpha', None)
    atomic_jsonl(output, metadata, rows)
    atomic_jsonl(output.with_suffix('.trace.jsonl'), metadata, trace, row_record_type='step')
    print(json.dumps({'output': str(output), 'elapsed_seconds': metadata['elapsed_seconds']}), flush=True)
    env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
