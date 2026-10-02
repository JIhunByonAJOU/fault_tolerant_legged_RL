"""Versioned diagnostic evaluator: common environment, locked command, fixed horizon.

No training is performed. Policy checkpoints are loaded strictly, without optimizers.
"""
import argparse
import hashlib
import json
import math
import sys
import time
import subprocess
from pathlib import Path

import isaacgym  # noqa: F401 -- must precede torch
import torch
from isaacgym.torch_utils import quat_apply
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
    parser.add_argument('--policy-mode', choices=['teacher', 'student', 'oracle', 'history_repeat'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--replicates', type=int, default=48)
    parser.add_argument('--rates', type=float, nargs='+', default=[0., .2, .4, .6, .8, 1.])
    parser.add_argument('--post-seconds', type=float, default=20.)
    parser.add_argument('--pre-seconds', type=float, default=2.)
    parser.add_argument('--command-x', type=float, default=.5)
    parser.add_argument('--terrain', choices=['mixed', 'plane'], default='mixed')
    parser.add_argument('--step-sleep-ms', type=float, default=0.)
    parser.add_argument('--shared-gpu-monitor', action='store_true')
    parser.add_argument('--allow-cpu-smoke', action='store_true')
    own, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = get_args()
    cuda_scientific = args.sim_device == 'cuda:0' and args.rl_device == 'cuda:0'
    cpu_smoke = own.allow_cpu_smoke and args.sim_device == 'cpu' and args.rl_device == 'cpu'
    if not args.headless or not (cuda_scientific or cpu_smoke):
        raise ValueError('Scientific evaluation requires headless CUDA; CPU is smoke-only')
    if own.command_x <= .2 or own.post_seconds <= 0 or own.pre_seconds <= 0:
        raise ValueError('This protocol requires vx > .2 m/s and positive windows')
    output = Path(own.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    start = time.monotonic()
    evaluator_sha_at_start = sha256_file(Path(__file__))
    latent_sq_sum = 0.
    latent_sample_count = 0
    action_sq_sum = 0.
    teacher_latent_sq_sum = 0.
    student_latent_sq_sum = 0.
    latent_cosine_sum = 0.
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
    if own.policy_mode != 'teacher':
        width = int(state['model_state_dict']['student_encoder.frame_encoder.0.weight'].shape[0])
        model = JointTeacherStudentActorCritic(2635, 45, 12, student_embedding_dim=width)
    else:
        model = OfficialWimTeacherActorCritic(235, 45, 12)
    model.load_state_dict(state['model_state_dict'], strict=True)
    model.to(env.device).eval()
    torch.set_rng_state(rng_cpu)
    torch.cuda.set_rng_state_all(rng_gpu)
    device = env.device
    joint = torch.tensor([s[0] for s in specs], device=device)
    rate = torch.tensor([s[1] for s in specs], device=device)
    onset = torch.tensor([s[2] for s in specs], device=device)
    horizon = round(own.post_seconds / dt)
    pre_horizon = round(own.pre_seconds / dt)
    end = onset + horizon
    ids = torch.arange(n, device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    first_done = torch.full((n,), -1, device=device, dtype=torch.long)
    count = torch.zeros(n, device=device)
    pre_count = torch.zeros(n, device=device)
    pre_vx_sum = torch.zeros(n, device=device)
    sums = torch.zeros((n, 3), device=device)
    actual_vx = torch.zeros(n, device=device)
    stationary_count = torch.zeros(n, device=device)
    moving_count = torch.zeros(n, device=device)
    backward_count = torch.zeros(n, device=device)
    low_forward_count = torch.zeros(n, device=device)
    onset_origin_xy = torch.zeros((n, 2), device=device)
    last_alive_xy = torch.zeros((n, 2), device=device)
    onset_forward_world = torch.zeros((n, 2), device=device)
    world_x = torch.zeros((n, 3), device=device)
    world_x[:, 0] = 1.
    stable_count = torch.zeros(n, device=device)
    strict_count = torch.zeros(n, device=device)
    loose_count = torch.zeros(n, device=device)
    stable_run = torch.zeros(n, dtype=torch.long, device=device)
    recovered_at = torch.full((n,), -1, dtype=torch.long, device=device)
    final_count = torch.zeros(n, device=device)
    trace = []
    overutil_checks = 0
    resource_checks = []
    window = round(1. / dt)
    final_steps = min(horizon, round(5. / dt))
    expected = torch.tensor([own.command_x, 0., 0.], device=device)
    expected_obs = expected * env.commands_scale
    with torch.inference_mode():
        for step in range(int(end.max())):
            due = (onset == step) & alive
            due_ids = ids[due]
            if due_ids.numel():
                onset_origin_xy[due_ids] = env.root_states[due_ids, :2]
                last_alive_xy[due_ids] = env.root_states[due_ids, :2]
                onset_forward_world[due_ids] = quat_apply(
                    env.root_states[due_ids, 3:7], world_x[due_ids]
                )[:, :2]
                heading_norm = torch.linalg.vector_norm(onset_forward_world[due_ids], dim=-1, keepdim=True)
                onset_forward_world[due_ids] /= heading_norm.clamp_min(1e-8)
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
            else:
                policy_obs = obs
                if own.policy_mode == 'history_repeat':
                    policy_obs = obs.clone()
                    policy_obs[:, 235:] = obs[:, :48].repeat(1, 50)
                z_student = model.encode_history(policy_obs)
                z_teacher = model.encode_privileged(privileged)
                student_action = model.act_inference_with_latent(policy_obs, z_student)
                teacher_action = model.act_inference_with_latent(policy_obs, z_teacher)
                action = teacher_action if own.policy_mode == 'oracle' else student_action
                comparison_mask = alive & (step >= onset) & (step < end)
                if comparison_mask.any():
                    latent_sq_sum += float((z_student[comparison_mask] - z_teacher[comparison_mask]).square().sum())
                    action_sq_sum += float((student_action[comparison_mask] - teacher_action[comparison_mask]).square().sum())
                    teacher_latent_sq_sum += float(z_teacher[comparison_mask].square().sum())
                    student_latent_sq_sum += float(z_student[comparison_mask].square().sum())
                    latent_cosine_sum += float(torch.nn.functional.cosine_similarity(z_student[comparison_mask], z_teacher[comparison_mask], dim=-1).sum())
                    latent_sample_count += int(comparison_mask.sum())
            if not torch.isfinite(action).all():
                raise RuntimeError('Nonfinite action')
            # Sample state at the beginning of each commanded control interval.
            pre = alive & (step < onset) & (step >= onset - pre_horizon)
            post = alive & (step >= onset) & (step < end)
            error = torch.stack((env.base_lin_vel[:, 0] - own.command_x,
                                 env.base_lin_vel[:, 1], env.base_ang_vel[:, 2]), dim=1)
            if not torch.isfinite(error[alive]).all():
                raise RuntimeError('Nonfinite velocity')
            pre_count += pre.float()
            pre_vx_sum += env.base_lin_vel[:, 0] * pre
            count += post.float()
            sums += error.square() * post[:, None]
            actual_vx += env.base_lin_vel[:, 0] * post
            planar_speed = torch.linalg.vector_norm(env.base_lin_vel[:, :2], dim=1)
            stationary_count += (post & (planar_speed < .1)).float()
            moving_count += (post & (planar_speed >= .1)).float()
            backward_count += (post & (env.base_lin_vel[:, 0] < 0.)).float()
            low_forward_count += (post & (env.base_lin_vel[:, 0] < .1)).float()
            last_alive_xy[post] = env.root_states[post, :2]
            # Occupancy includes lateral tracking, unlike the legacy recovery event.
            stable = post & (error[:, 0].abs() <= .1) & (error[:, 1].abs() <= .1) & (error[:, 2].abs() <= .2)
            stable_count += stable.float()
            strict_count += (post & (error[:, :2].abs() <= .05).all(dim=1) & (error[:, 2].abs() <= .1)).float()
            loose_count += (post & (error[:, :2].abs() <= .2).all(dim=1) & (error[:, 2].abs() <= .4)).float()
            legacy_stable = post & (error[:, 0].abs() <= .1) & (error[:, 2].abs() <= .2)
            stable_run = torch.where(legacy_stable, stable_run + 1, torch.zeros_like(stable_run))
            new_recovery = (recovered_at < 0) & (stable_run >= window)
            recovered_at[new_recovery] = step + 1 - window
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
            if own.step_sleep_ms:
                time.sleep(own.step_sleep_ms / 1000.)
            if own.shared_gpu_monitor and step % 50 == 0:
                usage = subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.free','--format=csv,noheader,nounits'], text=True).strip().split(',')
                util, free = [int(v.strip()) for v in usage]
                resource_checks.append({'step':step,'utilization_pct':util,'free_mib':free})
                overutil_checks = overutil_checks + 1 if util > 90 else 0
                if free < 2048 or overutil_checks >= 6:
                    raise RuntimeError('Shared GPU headroom exhausted; stopping this evaluation only')
            if step % 250 == 0:
                print('step {}/{} elapsed {:.1f}s'.format(step, int(end.max()), time.monotonic()-start), flush=True)
    rows = []
    for i, (j, d, onset_step, rep) in enumerate(specs):
        valid = int(count[i])
        valid_pre = int(pre_count[i])
        done = int(first_done[i])
        survived = done < 0  # Contact on the final transition also counts as failure.
        rmse = [math.sqrt(float(x) / valid) if valid else None for x in sums[i]]
        delta_xy = last_alive_xy[i] - onset_origin_xy[i]
        forward_axis = onset_forward_world[i]
        lateral_axis = torch.stack((-forward_axis[1], forward_axis[0]))
        world_forward_distance = float(torch.dot(delta_xy, forward_axis)) if valid else 0.
        world_lateral_displacement = float(torch.dot(delta_xy, lateral_axis)) if valid else 0.
        body_forward_distance = float(actual_vx[i]) * dt if valid else 0.
        rows.append(dict(env_id=i, seed=args.seed, replicate=rep, joint_index=j,
                         joint_name=env.dof_names[j], degradation_rate=d,
                         onset_seconds=onset_step*dt, post_horizon_seconds=own.post_seconds,
                         terrain_level=initial_levels[i], terrain_column=initial_columns[i],
                         failed_before_onset=(done >= 0 and done <= onset_step),
                         survived_post_failure_horizon=survived, first_done_step=done,
                         valid_pre_steps=valid_pre,
                         pre_mean_vx=float(pre_vx_sum[i])/valid_pre if valid_pre else None,
                         valid_post_steps=valid,
                         alive_time_fraction=valid/horizon,
                         post_command_vx_rmse=rmse[0],
                         post_command_vy_rmse=rmse[1], post_yaw_rate_rmse=rmse[2],
                         post_mean_vx=float(actual_vx[i])/valid if valid else None,
                         body_forward_distance_m=body_forward_distance,
                         body_progress_ratio=body_forward_distance/(own.command_x*own.post_seconds),
                         world_forward_distance_m=world_forward_distance,
                         world_progress_ratio=world_forward_distance/(own.command_x*own.post_seconds),
                         world_lateral_displacement_m=world_lateral_displacement,
                         stationary_time_fraction_alive=float(stationary_count[i])/valid if valid else None,
                         stationary_time_fraction_horizon=float(stationary_count[i])/horizon,
                         moving_time_fraction_horizon=float(moving_count[i])/horizon,
                         backward_time_fraction_alive=float(backward_count[i])/valid if valid else None,
                         low_forward_time_fraction_alive=float(low_forward_count[i])/valid if valid else None,
                         tracking_time_fraction=float(stable_count[i])/horizon,
                         tracking_time_fraction_strict=float(strict_count[i])/horizon,
                         tracking_time_fraction_loose=float(loose_count[i])/horizon,
                         final_5s_tracking_fraction=float(final_count[i])/final_steps,
                         recovered_stable_window=bool(recovered_at[i] >= 0),
                         recovery_window_start_seconds=(int(recovered_at[i])-onset_step)*dt if recovered_at[i]>=0 else None))
    metadata = dict(schema_version=5, evaluation='paper_paired_random_onset_mobility_v2',
                    task=task, checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
                    evaluator_sha256=evaluator_sha_at_start,
                    protocol_sha256=sha256_file(Path(__file__).resolve().parents[1] / 'evaluation/onset_protocol_v2.py'),
                    policy_mode=own.policy_mode, seed=args.seed, num_envs=n, dt=dt,
                    command=[own.command_x, 0., 0.], onset_range_seconds=[2.,10.],
                    pre_seconds=own.pre_seconds, post_seconds=own.post_seconds,
                    rates=own.rates, replicates=own.replicates,
                    initial_state_hashes=initial, specs_sha256=hashlib.sha256(json.dumps(specs).encode()).hexdigest(),
                    terrain=own.terrain, noise=False, pushes=False, initialization_neutral_steps=1,
                    scientific_result=cuda_scientific and bool(env.sim_params.use_gpu_pipeline) and bool(env.sim_params.physx.use_gpu),
                    runtime_gpu_pipeline=bool(env.sim_params.use_gpu_pipeline),
                    runtime_physx_gpu=bool(env.sim_params.physx.use_gpu),
                    metric_sampling='pre-action state; absorbing failure; terminal final transition is failure',
                    tracking_thresholds=[.1,.1,.2], strict_thresholds=[.05,.05,.1], loose_thresholds=[.2,.2,.4],
                    tracking_denominator='full post horizon, including time after failure',
                    mobility_definitions={
                        'alive_time_fraction': 'valid post-fault control intervals / full post horizon',
                        'body_forward_distance_m': 'integral of alive body-frame vx over post-fault horizon',
                        'world_forward_distance_m': 'last-alive pre-action world displacement projected onto unit-normalized planar heading at fault onset; final transition displacement omitted',
                        'world_lateral_displacement_m': 'last-alive world displacement orthogonal to heading at fault onset',
                        'stationary': 'alive and planar body speed < 0.1 m/s',
                        'backward': 'alive and body-frame vx < 0 m/s',
                        'low_forward': 'alive and body-frame vx < 0.1 m/s',
                    },
                    recovery_definition='legacy: abs(vx error)<=0.1, abs(yaw rate)<=0.2 for 1 second; not sustained recovery',
                    latent_imitation={
                        'post_alive_samples': latent_sample_count,
                        'latent_coordinate_rmse': math.sqrt(latent_sq_sum/(8*latent_sample_count)) if latent_sample_count else None,
                        'teacher_latent_coordinate_rms': math.sqrt(teacher_latent_sq_sum/(8*latent_sample_count)) if latent_sample_count else None,
                        'student_latent_coordinate_rms': math.sqrt(student_latent_sq_sum/(8*latent_sample_count)) if latent_sample_count else None,
                        'relative_latent_rmse': math.sqrt(latent_sq_sum/teacher_latent_sq_sum) if teacher_latent_sq_sum else None,
                        'mean_latent_vector_cosine': latent_cosine_sum/latent_sample_count if latent_sample_count else None,
                        'joint_target_coordinate_rmse_rad': cfg.control.action_scale * math.sqrt(action_sq_sum/(12*latent_sample_count)) if latent_sample_count else None,
                        'action_coordinate_rmse': math.sqrt(action_sq_sum/(12*latent_sample_count)) if latent_sample_count else None,
                        'note': 'same state oracle/student branch disagreement; not independent trajectories or semantic identification'},
                    shared_gpu_resource_checks=resource_checks,
                    step_sleep_ms=own.step_sleep_ms,
                    history_ablation='repeat newest frame 50 times' if own.policy_mode == 'history_repeat' else None,
                    elapsed_seconds=time.monotonic()-start)
    atomic_jsonl(output, metadata, rows)
    atomic_jsonl(output.with_suffix('.trace.jsonl'), metadata, trace, row_record_type='step')
    print(json.dumps({'output': str(output), 'elapsed_seconds': metadata['elapsed_seconds']}), flush=True)
    env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
