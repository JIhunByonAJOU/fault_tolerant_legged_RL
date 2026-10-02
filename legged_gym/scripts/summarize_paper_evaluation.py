"""Publish compact audited evaluation summaries; no simulator or GPU access."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
LABELS = {'tb22500': 'Normal Base (privileged)', 'jt77500': 'Final Student (history)', 'oracle77500': 'Final Privileged Reference', 'history_repeat77500': 'Repeated-frame Ablation'}
DEFAULT_RAW = ROOT / 'logs/evaluations/joint_severity_heatmaps_20261001/raw'
JOINTS = ['FL hip', 'FL thigh', 'FL calf', 'FR hip', 'FR thigh', 'FR calf', 'RL hip', 'RL thigh', 'RL calf', 'RR hip', 'RR thigh', 'RR calf']

def read(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    return records[0], [r for r in records[1:] if r['record_type'] == 'robot']

def summarize(rows, dt):
    n = np.array([r['valid_post_steps'] for r in rows], dtype=float)
    vx = np.array([r['post_mean_vx'] or 0 for r in rows])
    h = np.array([r['post_horizon_seconds'] for r in rows])
    result = dict(n=len(rows), S20=np.mean([r['survived_post_failure_horizon'] for r in rows]),
                  alive_time_s=np.mean(n*dt), body_D_m=np.mean(vx*n*dt), body_P_cmd=np.mean(vx*n*dt/(.5*h)),
                  vx_alive_mean_m_s=np.sum(vx*n)/n.sum(),
                  Q_standard=np.mean([r['tracking_time_fraction'] for r in rows]),
                  Q_strict=np.mean([r['tracking_time_fraction_strict'] for r in rows]))
    for axis,key in [('vx','post_command_vx_rmse'), ('vy','post_command_vy_rmse'), ('yaw','post_yaw_rate_rmse')]:
        rmse=np.array([r[key] or 0 for r in rows])
        result[f'{axis}_alive_pooled_rmse']=np.sqrt(np.sum(rmse**2*n)/n.sum())
    for key in ['world_forward_distance_m','world_progress_ratio','stationary_time_fraction_alive','backward_time_fraction_alive']:
        if key in rows[0]:
            vals=np.array([r[key] or 0 for r in rows])
            result[key]=np.sum(vals*n)/n.sum() if key.endswith('_alive') else np.mean(vals)
    if 'stationary_time_fraction_horizon' in rows[0]:
        result['stall_time_s']=np.mean([r['stationary_time_fraction_horizon']*r['post_horizon_seconds'] for r in rows])
        result['backward_time_s']=np.mean([(r['backward_time_fraction_alive'] or 0)*r['valid_post_steps']*dt for r in rows])
        result['moving_time_fraction_horizon']=np.mean([r['moving_time_fraction_horizon'] for r in rows])
    return result

def write_csv(path, records):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--raw-dir',type=Path,default=DEFAULT_RAW); p.add_argument('--output-dir',type=Path,default=ROOT/'docs/submission/ieie2026/results'); p.add_argument('--seeds',type=int,nargs='+',default=[131,132,133]); p.add_argument('--models',nargs='+',default=['tb22500','jt77500']); args=p.parse_args()
    out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    data={}; audit=[]; ref={}
    for model in args.models:
        data[model]=[]
        for seed in args.seeds:
            path=args.raw_dir/f'{model}_seed{seed}.jsonl'; meta, rows=read(path)
            log=path.with_suffix('.log')
            gpu=meta.get('scientific_result',False) or (log.exists() and '+++ Using GPU PhysX' in log.read_text() and 'GPU Pipeline: enabled' in log.read_text())
            if not gpu: raise ValueError(f'Not proven GPU PhysX: {path}')
            if len(rows)!=meta['num_envs']: raise ValueError('Incomplete result')
            if meta['command'] != [.5, 0., 0.] or meta['dt'] != .02 or meta['post_seconds'] != 20.:
                raise ValueError('This report is fixed to command .5/0/0, dt .02, horizon 20')
            keys=['initial_state_hashes','specs_sha256','protocol_sha256','terrain','command','num_envs','dt']
            if seed not in ref: ref[seed]=meta
            if any(meta[k]!=ref[seed][k] for k in keys): raise ValueError(f'Unpaired initial state: {path}')
            data[model]+=rows
            audit.append(dict(model=model,seed=seed,rows=len(rows),gpu_physx_verified=gpu,paired_hashes_verified=True,checkpoint_sha256=meta['checkpoint_sha256'],raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),metadata=meta))
    cells=[]; rates=[]
    for model,rows in data.items():
        for d in [0.,.2,.4,.6,.8,1.]:
            selected=[r for r in rows if r['degradation_rate']==d]
            rates.append(dict(model=model,degradation_rate=d,**summarize(selected,ref[args.seeds[0]]['dt'])))
            for j in range(12): cells.append(dict(model=model,joint=JOINTS[j],joint_index=j,degradation_rate=d,**summarize([r for r in selected if r['joint_index']==j],.02)))
    write_csv(out/'metrics_by_severity.csv',rates); write_csv(out/'metrics_by_joint_severity.csv',cells)
    # Paired episode bootstrap is conditional on these 3 seeds, not training-seed uncertainty.
    rng=np.random.default_rng(20261002); diffs=[]
    if len(args.models)>=2:
        base,student=args.models[:2]
        for d in [0.,.2,.4,.6,.8,1.]:
            a=sorted([r for r in data[base] if r['degradation_rate']==d],key=lambda r:(r['seed'],r['env_id']))
            b=sorted([r for r in data[student] if r['degradation_rate']==d],key=lambda r:(r['seed'],r['env_id']))
            pair_keys=['seed','env_id','joint_index','degradation_rate','onset_seconds','replicate']
            if len(a)!=len(b) or any(any(x[k]!=y[k] for k in pair_keys) for x,y in zip(a,b)):
                raise ValueError('Episode pairing keys differ')
            strata=[np.array([i for i,r in enumerate(a) if r['seed']==seed and r['joint_index']==j]) for seed in args.seeds for j in range(12)]
            if len({len(ix) for ix in strata})!=1: raise ValueError('Unbalanced strata')
            metric_functions=[('S20',lambda r:float(r['survived_post_failure_horizon'])),('body_P_cmd',lambda r:(r['post_mean_vx'] or 0)*r['valid_post_steps']*.02/10),('Q_strict',lambda r:r['tracking_time_fraction_strict']),('body_D_m',lambda r:(r['post_mean_vx'] or 0)*r['valid_post_steps']*.02)]
            if 'world_forward_distance_m' in a[0]:
                metric_functions += [('world_forward_distance_m',lambda r:r['world_forward_distance_m']),('world_progress_ratio',lambda r:r['world_progress_ratio']),('stall_time_s',lambda r:r['stationary_time_fraction_horizon']*20),('backward_time_s',lambda r:(r['backward_time_fraction_alive'] or 0)*r['valid_post_steps']*.02)]
            for metric,fn in metric_functions:
                delta=np.array([fn(y)-fn(x) for x,y in zip(a,b)])
                values=np.stack([delta[ix] for ix in strata])
                draw=rng.integers(0,values.shape[1],(2000,*values.shape))
                boot=values[np.arange(len(strata))[None,:,None],draw].mean(axis=(1,2))
                diffs.append(dict(degradation_rate=d,metric=metric,student_minus_base=delta.mean(),conditional_episode_bootstrap_low=np.quantile(boot,.025),conditional_episode_bootstrap_high=np.quantile(boot,.975),n_pairs=len(delta),bootstrap_design='paired episodes within seed-joint strata; 2000 draws'))
        write_csv(out/'paired_differences.csv',diffs)
    (out/'evidence_manifest.json').write_text(json.dumps(dict(status='independent_GPU_validation' if (set(args.seeds).issubset({201,202,203}) or set(args.seeds).issubset({301,302,303})) else 'existing_selection_associated_GPU_evidence',independent_validation=(set(args.seeds).issubset({201,202,203}) or set(args.seeds).issubset({301,302,303})),training_seeds=1,audit=audit),indent=2))
    fig,axes=plt.subplots(len(args.models),4,figsize=(18,4*len(args.models)),squeeze=False,layout='constrained')
    specs=[('S20','20 s survival (%)',100,'YlGnBu',0,100),('body_P_cmd','Body progress / commanded (%)',100,'YlGnBu',0,100),('vx_alive_pooled_rmse','Forward RMSE while alive (m/s)',1,'YlOrRd',0,.5),('Q_strict','Strict tracking occupancy (%)',100,'magma',0,100)]
    for i,model in enumerate(args.models):
        for k,(metric,title,scale,cmap,lo,hi) in enumerate(specs):
            mat=np.array([[next(r[metric] for r in cells if r['model']==model and r['joint_index']==j and r['degradation_rate']==d)*scale for d in [0.,.2,.4,.6,.8,1.]] for j in range(12)])
            ax=axes[i,k]; im=ax.imshow(mat,cmap=cmap,vmin=lo,vmax=hi,aspect='auto'); ax.set_title(f'{LABELS.get(model, model)}: {title}'); ax.set_xticks(range(6),['0','.2','.4','.6','.8','1']); ax.set_yticks(range(12),JOINTS); ax.set_xlabel('PD motor-strength loss d')
            for y in range(12):
                for x in range(6): ax.text(x,y,f'{mat[y,x]:.1f}' if scale==100 else f'{mat[y,x]:.2f}',ha='center',va='center',fontsize=7,color='white' if (cmap=='magma' and (mat[y,x]-lo)/(hi-lo)<.65) or (cmap!='magma' and (mat[y,x]-lo)/(hi-lo)>.65) else 'black')
            fig.colorbar(im,ax=ax,shrink=.7)
    fig.savefig(out/'base_vs_final_heatmaps.png',dpi=160); fig.savefig(out/'base_vs_final_heatmaps.pdf'); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for ax,key,title in zip(axes,['S20','body_P_cmd','Q_strict'],['Survival','Body progress ratio','Strict tracking occupancy']):
        for model in args.models:
            rr=[r for r in rates if r['model']==model]; ax.plot([r['degradation_rate'] for r in rr],[100*r[key] for r in rr],marker='o',label=LABELS.get(model,model))
        ax.set_title(title); ax.set_xlabel('Loss d'); ax.set_ylabel('%'); ax.set_ylim(0,110 if key=='body_P_cmd' else 100); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(out/'base_vs_final_severity.png',dpi=180); fig.savefig(out/'base_vs_final_severity.pdf')
    print(json.dumps(rates,indent=2))
if __name__=='__main__': main()
