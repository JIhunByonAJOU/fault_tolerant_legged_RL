"""One frozen paired-course job. No training; complete traces for every slot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import MethodType

import isaacgym
from isaacgym import gymtorch
import numpy as np
import torch
from isaacgym.torch_utils import quat_rotate_inverse

from legged_gym.envs import *  # noqa
from legged_gym.utils import get_args, task_registry
from legged_gym.evaluation.course_campaign import ROOT, atomic_json, sha, specs, stable_seed
from legged_gym.evaluation.course_controller import CourseController
from legged_gym.learning.official_wim_teacher_actor_critic import OfficialWimTeacherActorCritic
from legged_gym.learning.joint_teacher_student_actor_critic import JointTeacherStudentActorCritic


FIELDS=['world_x','world_y','body_yaw','body_vx','body_vy','body_wz','cmd_vx','cmd_vy','cmd_wz',
        'ref_x','ref_y','path_s','furthest_s','cross_track','world_vx','world_vy']


def tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def save_chunk(path, values, flags, mask, start):
    steps, env_ids=np.nonzero(mask)
    path=Path(path); tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, values=values[steps,env_ids], flags=flags[steps,env_ids],
                            env_id=env_ids.astype(np.uint16), step=(steps+start).astype(np.int32))
    tmp.replace(path)
    return dict(path=path.name, sha256=sha(path), rows=len(steps), first_step=start,
                last_step=start+len(mask)-1)


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--pilot-seconds',type=float)
    parser.add_argument('--plane-pilot',action='store_true')
    own,rest=parser.parse_known_args(); sys.argv=[sys.argv[0]]+rest
    args=get_args()
    if not args.headless or args.sim_device!='cuda:0' or args.rl_device!='cuda:0':
        raise ValueError('Reported evaluation requires headless GPU PhysX')
    p=json.loads((own.campaign/'protocol.json').read_text()); job=json.loads(own.job.read_text())
    rows=specs(p,job['stage'],job['seed'],job['conditions'],job['family_repeats'],job['column_counts'])
    output=own.campaign/'jobs'/job['id']
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'job.json',job)
    policy=p['policies'][job['policy']]; checkpoint=ROOT/policy['checkpoint']
    if sha(checkpoint)!=policy['sha256']: raise RuntimeError('Checkpoint hash changed')
    n=len(rows); args.num_envs=n
    args.task='a1_official_wim_jt_failure_fullrange_onset'
    args.seed=stable_seed('terrain',job['seed'])
    cfg,_=task_registry.get_cfgs(args.task)
    cfg.env.num_envs=n; cfg.env.episode_length_s=1000.
    cfg.terrain.num_rows=5; cfg.terrain.num_cols=12; cfg.terrain.curriculum=True
    cfg.terrain.selected=False; cfg.terrain.max_init_terrain_level=0
    cfg.terrain.terrain_proportions=[.1,.1,.35,.25,.2]
    if own.plane_pilot: cfg.terrain.mesh_type='plane'
    cfg.commands.curriculum=False; cfg.commands.heading_command=False
    cfg.commands.resampling_time=1000000.
    cfg.commands.ranges.lin_vel_x=[.5,.5]; cfg.commands.ranges.lin_vel_y=[0.,0.]
    cfg.commands.ranges.ang_vel_yaw=[0.,0.]
    cfg.noise.add_noise=False; cfg.domain_rand.push_robots=False
    cfg.domain_rand.randomize_friction=False; cfg.domain_rand.randomize_base_mass=False
    cfg.domain_rand.failure_onset_time_range_s=[1000000.,1000000.]
    # Apply static rigid properties at actor construction, before prepare_sim.
    # Updating articulated mass after GPU preparation can invalidate cached state.
    physical=[]
    for row in rows:
        rng=np.random.RandomState(row['physics_seed'])
        physical.append((float(rng.uniform(.5,1.25)),float(rng.uniform(0,2)),
                         rng.uniform(.9,1.1,12),rng.uniform(.95,1.05,12),rng.uniform(.9,1.1,12)))
    base_class=task_registry.get_task_class(args.task)
    class ManifestEnvironment(base_class):
        def _process_rigid_shape_props(self,props,env_id):
            if env_id==0:
                self.friction_coeffs=torch.tensor([x[0] for x in physical]).reshape(n,1)
            for shape in props: shape.friction=physical[env_id][0]
            return props
        def _process_rigid_body_props(self,props,env_id):
            if env_id==0:
                self.added_base_mass=torch.zeros((n,1),device=self.device)
                self.original_base_masses=[]
            self.original_base_masses.append(float(props[0].mass))
            props[0].mass+=physical[env_id][1]
            self.added_base_mass[env_id]=physical[env_id][1]
            return props
    task_registry.task_classes[args.task]=ManifestEnvironment
    started=time.monotonic()
    env,_=task_registry.make_env(args.task,args,env_cfg=cfg)
    executor=ThreadPoolExecutor(max_workers=1)
    futures=[]; chunk_files=[]
    try:
        cfg.terrain.curriculum=False
        if not env.sim_params.use_gpu_pipeline or not env.sim_params.physx.use_gpu:
            raise RuntimeError('GPU pipeline/PhysX inactive')
        # Initialize once, without env.reset()'s neutral step/queued reset setters.
        # PhysX permits one root-state setter per simulate interval.
        torch.cuda.synchronize()
        # No automatic resets: terminal states stay available for logging.
        env.reset_idx=MethodType(lambda self,ids:None,env)
        device=env.device
        root=np.zeros((n,13),dtype=np.float32)
        dof=np.zeros((n,12),dtype=np.float32)
        nominal=np.zeros((n,12),dtype=np.float32); kp=nominal.copy(); kd=nominal.copy()
        actual_props=[]
        for i,row in enumerate(rows):
            friction,mass,nominal[i],kp[i],kd[i]=physical[i]
            original_mass=env.original_base_masses[i]
            got_shapes=env.gym.get_actor_rigid_shape_properties(env.envs[i],env.actor_handles[i])
            got_bodies=env.gym.get_actor_rigid_body_properties(env.envs[i],env.actor_handles[i])
            if abs(got_shapes[0].friction-friction)>1e-5 or abs(got_bodies[0].mass-original_mass-mass)>1e-5:
                raise RuntimeError('Applied physics differs from specification')
            env.applied_friction[i]=got_shapes[0].friction
            env.added_base_mass[i]=got_bodies[0].mass-original_mass
            actual_props.append([float(s.friction) for s in got_shapes]+[float(b.mass) for b in got_bodies])
            rng=np.random.RandomState(row['initial_seed'])
            root[i,0]=p['start_x']+rng.uniform(-.05,.05)
            root[i,1]=(row['terrain_column_id']-.5)*8+rng.uniform(-.05,.05)
            root[i,2]=float(env.base_init_state[2])
            yaw=row['profile']['yaw']
            if yaw is None: yaw=math.atan(row['profile']['amplitude']*2*math.pi/p['wavelength'])
            yaw+=rng.uniform(-.03,.03)
            root[i,5]=math.sin(yaw/2); root[i,6]=math.cos(yaw/2)
            dof[i]=env.default_dof_pos[0].cpu().numpy()*rng.uniform(.95,1.05,12)
        env.root_states.copy_(torch.as_tensor(root,device=device))
        env.dof_pos.copy_(torch.as_tensor(dof,device=device)); env.dof_vel.zero_()
        env.nominal_motor_strength_gt.copy_(torch.as_tensor(nominal,device=device))
        env.motor_strength_gt.copy_(env.nominal_motor_strength_gt)
        env.kp_scale_gt.copy_(torch.as_tensor(kp,device=device)); env.kd_scale_gt.copy_(torch.as_tensor(kd,device=device))
        env.actuator_degradation.zero_(); env.degraded_joint_index.fill_(-1)
        env.actions.zero_(); env.last_actions.zero_(); env.last_dof_vel.zero_()
        env.episode_length_buf.zero_(); env.observation_history.zero_(); env.reset_buf.zero_()
        ids=torch.arange(n,device=device,dtype=torch.int32)
        torch.cuda.synchronize()
        if not env.gym.set_actor_root_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.root_states),gymtorch.unwrap_tensor(ids),n):
            raise RuntimeError('Root state setter rejected')
        if not env.gym.set_dof_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.dof_state),gymtorch.unwrap_tensor(ids),n):
            raise RuntimeError('DOF state setter rejected')
        torch.cuda.synchronize()
        env.base_quat.copy_(env.root_states[:,3:7])
        env.base_lin_vel.zero_(); env.base_ang_vel.zero_()
        env.projected_gravity[:]=quat_rotate_inverse(env.base_quat,env.gravity_vec)
        env.measured_heights=env._get_heights()
        controller=CourseController(p,rows,device)
        original_callback=env._post_physics_step_callback
        def callback(self):
            original_callback()
            self.commands[:,:3]=controller.evaluate(self.root_states)['command']
        env._post_physics_step_callback=MethodType(callback,env)
        env.commands[:,:3]=controller.evaluate(env.root_states)['command']
        env.compute_observations()
        obs=env.obs_buf; privileged=env.privileged_obs_buf
        if own.pilot_seconds:
            print('PILOT_INITIAL',json.dumps(dict(root=env.root_states[:2].cpu().tolist(),dof=env.dof_pos[:2].cpu().tolist(),
                  initial_z_range=[float(env.root_states[:,2].min()),float(env.root_states[:,2].max())])),flush=True)
        initial_hashes={key:tensor_hash(getattr(env,key)) for key in [
            'root_states','dof_pos','dof_vel','applied_friction','added_base_mass',
            'nominal_motor_strength_gt','kp_scale_gt','kd_scale_gt','obs_buf']}
        initial_hashes['rigid_properties']=hashlib.sha256(np.asarray(actual_props,dtype=np.float32).tobytes()).hexdigest()
        initial_hashes['terrain']=hashlib.sha256(env.terrain.height_field_raw.tobytes()).hexdigest() if hasattr(env,'terrain') else 'plane'
        with (output/'initial_applied_state.npz').open('wb') as f:
            np.savez_compressed(f,root=root,dof=dof,nominal_motor_strength=nominal,kp_scale=kp,kd_scale=kd,
                                actual_friction=env.applied_friction.cpu().numpy(),added_mass=env.added_base_mass.cpu().numpy())
        state=torch.load(str(checkpoint),map_location='cpu')['model_state_dict']
        if job['policy']=='Normal': model=OfficialWimTeacherActorCritic(235,45,12)
        else:
            width=state['student_encoder.frame_encoder.0.weight'].shape[0]
            model=JointTeacherStudentActorCritic(2635,45,12,student_embedding_dim=int(width))
        model.load_state_dict(state,strict=True); model.to(device).eval()
        onset=round(p['onset_s']/p['dt']); window=round(p['stall_window_s']/p['dt'])
        caps=torch.tensor([math.ceil((r['post_cap_s']+p['onset_s'])/p['dt']) for r in rows],device=device)
        if own.pilot_seconds: caps.fill_(round(own.pilot_seconds/p['dt']))
        active=torch.ones(n,dtype=torch.bool,device=device)
        first_end=torch.full((n,),-1,dtype=torch.long,device=device)
        reason=torch.zeros(n,dtype=torch.long,device=device)
        terminal_xy=torch.zeros((n,2),device=device)
        terminal_root=torch.zeros((n,13),device=device)
        counts=torch.zeros(n,dtype=torch.long,device=device)
        squares=torch.zeros((n,4),dtype=torch.float64,device=device)
        pre_counts=torch.zeros(n,device=device); pre_squares=torch.zeros((n,3),device=device)
        frontier=controller.last['s'].clone()
        history=torch.zeros((window+1,n),device=device)
        stalls=torch.zeros(n,dtype=torch.long,device=device)
        max_stall=torch.zeros(n,dtype=torch.long,device=device); streak=torch.zeros_like(stalls)
        row_arrival=torch.full((n,6),-1,dtype=torch.long,device=device)
        chunk_size=250
        values=torch.empty((chunk_size,n,16),device=device)
        flags=torch.empty((chunk_size,n),device=device,dtype=torch.uint8)
        masks=torch.empty((chunk_size,n),device=device,dtype=torch.bool)
        chunk_start=0; chunk_index=0; used=0
        rates=torch.tensor([r['condition']['loss'] for r in rows],device=device)
        joints=torch.tensor([r['condition']['joint'] or 0 for r in rows],device=device)
        def flush(used,start,index):
            vals=values[:used].cpu().numpy(); fl=flags[:used].cpu().numpy(); valid=masks[:used].cpu().numpy()
            futures.append(executor.submit(save_chunk,output/('trace_%04d.npz'%index),vals,fl,valid,start))
            if len(futures)>=2: chunk_files.append(futures.pop(0).result())
        metadata=dict(job=job, protocol_sha256=sha(own.campaign/'protocol.json'), checkpoint_sha256=sha(checkpoint),
            initial_hashes=initial_hashes, fields=FIELDS, flag_bits={'alive':1,'post_fault':2,'stall':4,'goal':8},
            dt=p['dt'], no_stall_termination=True, no_speed_termination=True, pilot=bool(own.pilot_seconds),
            plane=own.plane_pilot, gpu_physx=True, num_envs=n, binary_survival=True)
        atomic_json(output/'metadata.json',metadata)
        atomic_json(output/'specs.json',rows)
        with torch.inference_mode():
            for step in range(int(caps.max())):
                cur=controller.last
                if step==onset:
                    ix=torch.arange(n,device=device)
                    env.actuator_degradation[ix,joints]=rates
                    env.motor_strength_gt.copy_(env.nominal_motor_strength_gt)
                    env.motor_strength_gt[ix,joints]*=1-rates
                    env.degraded_joint_index[:]=torch.where(rates>0,joints,torch.full_like(joints,-1))
                    privileged[:,2:14]=env.motor_strength_gt; privileged[:,-1]=(rates>0).float()
                    frontier=cur['s'].clone()
                    expected=env.nominal_motor_strength_gt.clone()
                    expected[ix,joints]*=1-rates
                    if not torch.equal(expected,env.motor_strength_gt): raise RuntimeError('Fault multiplier mismatch')
                    with (output/'fault_applied.npz').open('wb') as f:
                        np.savez_compressed(f,step=step,rate=rates.cpu().numpy(),joint=joints.cpu().numpy(),
                                            nominal=env.nominal_motor_strength_gt.cpu().numpy(),
                                            applied=env.motor_strength_gt.cpu().numpy())
                if not torch.allclose(obs[:,9:12],env.commands[:,:3]*env.commands_scale,atol=1e-6):
                    raise RuntimeError('Observed command differs from applied command')
                if not torch.allclose(obs[:,235:].reshape(n,50,48)[:,-1,9:12],obs[:,9:12],atol=1e-6):
                    raise RuntimeError('Newest history command mismatch')
                if job['policy']=='Normal': action=model.act_inference(obs[:,:235],privileged)
                else: action=model.act_inference_student(obs)
                if not torch.isfinite(action).all(): raise RuntimeError('Nonfinite policy output')
                action[~active]=0
                frontier=torch.maximum(frontier,cur['s'])
                stalled=torch.zeros_like(active)
                if step>=onset+window:
                    gain=frontier-history[(step-window)%(window+1)]
                    stalled=active & (gain<p['stall_progress_m'])
                history[step%(window+1)]=frontier
                post=active & (step>=onset)
                error=torch.stack((env.base_lin_vel[:,0]-env.commands[:,0],env.base_lin_vel[:,1]-env.commands[:,1],
                                   env.base_ang_vel[:,2]-env.commands[:,2],cur['cross']),dim=1)
                if not torch.isfinite(error[active]).all(): raise RuntimeError('Nonfinite valid state')
                squares+=torch.where(post[:,None],error.double().square(),torch.zeros_like(squares)); counts+=post.long()
                pre=active & (step<onset); pre_counts+=pre.float(); pre_squares+=error[:,:3].square()*pre[:,None]
                stalls+=stalled.long(); streak=torch.where(stalled,streak+1,torch.zeros_like(streak))
                max_stall=torch.maximum(max_stall,streak)
                for boundary in range(6):
                    reached=active & (env.root_states[:,0]>=8*boundary) & (row_arrival[:,boundary]<0)
                    row_arrival[reached,boundary]=step
                values[used]=torch.cat((env.root_states[:,:2],cur['yaw'][:,None],env.base_lin_vel[:,:2],
                    env.base_ang_vel[:,2:3],env.commands[:,:3],cur['ref'],cur['s'][:,None],frontier[:,None],
                    cur['cross'][:,None],env.root_states[:,7:9]),dim=1)
                flags[used]=active.to(torch.uint8)+2*post.to(torch.uint8)+4*stalled.to(torch.uint8)
                masks[used]=active; used+=1
                obs,privileged,_,done,_=env.step(action)
                if step==0:
                    displacement=torch.linalg.vector_norm(env.root_states[:,:2]-torch.as_tensor(root[:,:2],device=device),dim=1)
                    if displacement.max()>.2 or env.root_states[:,2].min()<.25:
                        raise RuntimeError('Initial state not actually applied to simulation')
                if own.pilot_seconds and step<2:
                    print('PILOT_STEP',step,json.dumps(dict(root=env.root_states[:2].cpu().tolist(),
                          z_range=[float(env.root_states[:,2].min()),float(env.root_states[:,2].max())],
                          done_ids=done.nonzero().flatten().cpu().tolist())),flush=True)
                fall=active & done.bool()
                goal=active & ~fall & controller.last['goal']
                cap=active & ~fall & ~goal & (step+1>=caps)
                ended=fall|goal|cap
                first_end[ended]=step+1; terminal_xy[ended]=env.root_states[ended,:2]
                terminal_root[ended]=env.root_states[ended]
                reason[fall]=1; reason[goal]=2; reason[cap]=3
                active[ended]=False
                if used==chunk_size or not active.any() or step+1==int(caps.max()):
                    flush(used,chunk_start,chunk_index); chunk_index+=1; chunk_start=step+1; used=0
                if step%250==0:
                    status=dict(step=step, simulation_s=(step+1)*p['dt'], active=int(active.sum()),
                                ended=int((~active).sum()), elapsed_s=time.monotonic()-started)
                    atomic_json(output/'progress.json',status)
                    print(json.dumps(status),flush=True)
                    if shutil.disk_usage(output).free<10*1024**3: raise RuntimeError('Disk reserve exhausted')
                    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                    if free<2048: raise RuntimeError('GPU reserve exhausted; stopping this collector only')
                if not active.any(): break
        for future in futures: chunk_files.append(future.result())
        executor.shutdown(wait=True)
        end=first_end.cpu().numpy(); reasons=reason.cpu().numpy(); sq=squares.cpu().numpy(); cnt=counts.cpu().numpy()
        out_rows=[]
        for i,row in enumerate(rows):
            code=int(reasons[i]); valid=int(cnt[i]); ended_s=float(end[i]*p['dt'])
            if code==0: raise RuntimeError('Incomplete episode')
            rmse=np.sqrt(sq[i]/valid).tolist() if valid else [None]*4
            out_rows.append(dict(row, policy=job['policy'], env_id=i,
                trial_id=row['paired_trial_id']+'_'+job['policy'],
                termination={1:'fall',2:'goal',3:'administrative_cap'}[code], survived=int(code!=1),
                end_time_s=ended_s, observed_post_s=max(0.,ended_s-p['onset_s']),
                fall_post_s=max(0.,ended_s-p['onset_s']) if code==1 else None,
                failed_before_onset=bool(code==1 and end[i]<=onset), valid_post_samples=valid,
                command_vx_rmse=rmse[0], command_vy_rmse=rmse[1], command_wz_rmse=rmse[2], cross_track_rmse=rmse[3],
                error_sum_sq=sq[i].tolist(), stall_seconds=float(stalls[i]*p['dt']),
                longest_stall_flag_run_s=float(max_stall[i]*p['dt']),
                terminal_world_xy=terminal_xy[i].cpu().tolist(),
                terminal_root_state=terminal_root[i].cpu().tolist(),
                row_arrival_s=[float(x*p['dt']) if x>=0 else None for x in row_arrival[i].cpu().tolist()]))
        atomic_json(output/'results.json',out_rows)
        atomic_json(output/'trace_manifest.json',dict(chunks=chunk_files, fields=FIELDS,
                    valid_rows=sum(c['rows'] for c in chunk_files), terminal='results.json contains exact terminal time/state'))
        atomic_json(output/'COMPLETE.json',dict(job_id=job['id'], episodes=n, result_sha256=sha(output/'results.json'),
                    trace_manifest_sha256=sha(output/'trace_manifest.json'), elapsed_s=time.monotonic()-started,
                    pilot=bool(own.pilot_seconds)))
    finally:
        executor.shutdown(wait=True)
        env.gym.destroy_sim(env.sim)


if __name__=='__main__': main()
