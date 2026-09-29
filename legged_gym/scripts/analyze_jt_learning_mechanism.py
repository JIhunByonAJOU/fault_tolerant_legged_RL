"""Read-only CPU audit of TF warm start, PPO logs, and latent-fusion transfer.

No optimizer, simulation, or parameter learning. Parameter interpolation below
is a diagnostic of stored updates, not a replay of optimization at another LR.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from diagnose_teacher_encoder_offline import (
    torch, ROOT, TF, JT, CORPUS, mlp, Student, call, masks, get_selection, rms, sha,
)

NEW=ROOT/'logs/jt_wim243_failure_fullrange_onset_width64_fresh_20260929/run_seed1_35000'


def read_metrics(path):
    data=path.read_bytes()
    if not data.endswith(b'\n'):data=data[:data.rfind(b'\n')+1]
    rows=[json.loads(x) for x in data.splitlines() if x]
    return rows,hashlib.sha256(data).hexdigest()


def kl(old_mu,old_std,new_mu,new_std):
    return (torch.log(new_std/old_std)+(old_std.square()+(old_mu-new_mu).square())/(2*new_std.square())-.5).sum(-1)


def describe_actions(a,base,std,base_std,groups):
    k=kl(base,base_std,a,std).clamp(min=0)
    result={}
    for name,m in groups.items():
        error=(a[m]-base[m]).abs().mean(-1)
        result[name]={'n':int(m.sum()),'action_mae':float(error.mean()),
                      'joint_target_mae_rad':float(error.mean())*.25,
                      'kl_mean':float(k[m].mean()),'kl_p95':float(k[m].quantile(.95)),
                      'fraction_kl_above_0p02':float((k[m]>.02).float().mean())}
    return result


def interpolate(base,new,scale):
    return {k:base[k]+scale*(new[k]-base[k]) for k in base}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    report={'protocol':{'timestamp_utc':datetime.now(timezone.utc).isoformat(),
                       'device':'cpu','threads':1,'no_optimization':True,'no_new_rollout':True,
                       'script_sha256':sha(__file__),'checkpoints':{},'logs':{}},'training':{},'corpora':{}}
    fields=['Loss/learning_rate','PPO/mean_kl','PPO/max_kl','PPO/mean_clip_fraction',
            'PPO/mean_gradient_norm','Adaptation/loss','Adaptation/alpha','Adaptation/beta',
            'Adaptation/student_teacher_action_delta','Latent/mean_std','Latent/mean_abs',
            'Loss/value_function','Loss/surrogate','Train/mean_reward','Train/mean_episode_length',
            'Policy/mean_noise_std','Episode/rew_tracking_lin_vel','Episode/rew_tracking_ang_vel',
            'PPO/completed_updates','PPO/hard_kl_stopped']
    for label,run in [('original32',JT),('fresh64',NEW)]:
        path=run/'metrics.jsonl';rows,digest=read_metrics(path)
        report['protocol']['logs'][label]={'path':str(path),'snapshot_sha256':digest,
                                         'last_iteration':rows[-1]['iteration'],'rows':len(rows)}
        windows=[]
        edges=[43000,43001,43010,43500,44000,45000,47000,49000,51000,52000,53000,60000,66000,71500,73000,78000]
        for lo,hi in zip(edges[:-1],edges[1:]):
            selected=[r for r in rows if lo<=r['iteration']<hi]
            if not selected:continue
            win={'start':lo,'end_exclusive':min(hi,rows[-1]['iteration']+1),'rows':len(selected)}
            win.update({f:sum(r[f] for r in selected if isinstance(r.get(f),(int,float)))/sum(isinstance(r.get(f),(int,float)) for r in selected) for f in fields if any(isinstance(r.get(f),(int,float)) for r in selected)})
            win['iterations_at_min_lr_fraction']=sum(r['Loss/learning_rate']<=1.00001e-5 for r in selected)/len(selected)
            windows.append(win)
        report['training'][label]={'first_20':[{k:r[k] for k in ['iteration']+fields if k in r} for r in rows[:20]],'windows':windows}
    raw=torch.load(TF,map_location='cpu');tf=raw['model_state_dict']
    report['protocol']['tf_optimizer_lr']=raw['optimizer_state_dict']['param_groups'][0]['lr']
    report['protocol']['tf_optimizer_steps']=[float(v['step']) for v in raw['optimizer_state_dict']['state'].values()][:3]
    report['protocol']['tf_sha256']=sha(TF)
    paths={}
    for label,run in [('original32',JT),('fresh64',NEW)]:
        for it in [43000,43500,44000,45000,47000,49000,50000,51000,52000,52500,53000,60000,66000,71500,73000]:
            p=run/f'model_{it}.pt'
            # Only complete atomic .pt saves are used, never .tmp or future files.
            if p.is_file():paths[f'{label}/{it}']=p
    states={}
    for name,path in paths.items():
        checkpoint=torch.load(path,map_location='cpu');states[name]=checkpoint['model_state_dict']
        report['protocol']['checkpoints'][name]={'path':str(path),'sha256':sha(path),
            'stored_iteration':checkpoint['iter'],'optimizer_lr':checkpoint['optimizer_state_dict']['param_groups'][0]['lr'],
            'optimizer_first_step_count':float(next(iter(checkpoint['optimizer_state_dict']['state'].values()))['step'])}
    with torch.inference_mode():
        for filename in ['common_teacher_inputs.pt','common_student_inputs.pt']:
            c=torch.load(CORPUS/filename,map_location='cpu');sel=get_selection(c);c={k:v[sel] for k,v in c.items()}
            obs,p=c['obs'],c['privileged'];cur=obs[:,:235];groups=masks(c)
            ztf=call(mlp(tf,'teacher_encoder'),p);atf=call(mlp(tf,'actor'),torch.cat((cur,ztf),1))
            result={'n':len(p),'corpus_sha256':sha(CORPUS/filename),'startup':{},'checkpoints':{}}
            for label in ['original32','fresh64']:
                state=states[label+'/43000'];z=call(mlp(state,'teacher_encoder'),p)
                a=call(mlp(state,'actor'),torch.cat((cur,z),1))
                enc_restore=call(mlp(state,'actor'),torch.cat((cur,ztf),1))
                act_restore=call(mlp(tf,'actor'),torch.cat((cur,z),1))
                diag={'first_saved_update':describe_actions(a,atf,state['std'],tf['std'],groups),
                      'restore_tf_encoder_only':describe_actions(enc_restore,atf,state['std'],tf['std'],groups),
                      'restore_tf_actor_only':describe_actions(act_restore,atf,tf['std'],tf['std'],groups),
                      'stored_update_interpolation':{}}
                for scale in [0.,.005,.01,.025,.05,.1,.25,.5,1.]:
                    s=interpolate(tf,state,scale);zz=call(mlp(s,'teacher_encoder'),p)
                    aa=call(mlp(s,'actor'),torch.cat((cur,zz),1))
                    diag['stored_update_interpolation'][str(scale)]=describe_actions(aa,atf,s['std'],tf['std'],groups)
                result['startup'][label]=diag
            for name,state in states.items():
                iteration=int(name.split('/')[1]);alpha=min((iteration-43000)/10000,1.)
                zt=call(mlp(state,'teacher_encoder'),p);zs=call(Student(state),obs)
                actor=mlp(state,'actor');fused=(1-alpha)*zt+alpha*zs
                af=call(actor,torch.cat((cur,fused),1));ass=call(actor,torch.cat((cur,zs),1))
                at=call(actor,torch.cat((cur,zt),1))
                row={'alpha':alpha,'student_vs_fused':describe_actions(ass,af,state['std'],state['std'],groups),
                     'teacher_vs_tf':describe_actions(at,atf,state['std'],tf['std'],groups),
                     'student_vs_current_teacher':describe_actions(ass,at,state['std'],state['std'],groups),
                     'latent':{}}
                for g,m in groups.items():
                    row['latent'][g]={'n':int(m.sum()),'teacher_rms':rms(zt[m]),'student_rms':rms(zs[m]),
                                     'relative_imitation_rms':rms(zs[m]-zt[m])/max(rms(zt[m]),1e-9),
                                     'weighted_teacher_rms':rms((1-alpha)*zt[m]),
                                     'weighted_student_rms':rms(alpha*zs[m]),
                                     'fused_rms':rms(fused[m]),'fused_to_student_rms':rms(fused[m]-zs[m])}
                # One schedule step, without weight changes, versus forced student-only.
                next_alpha=min(alpha+.0001,1.)
                nextz=(1-next_alpha)*zt+next_alpha*zs
                nexta=call(actor,torch.cat((cur,nextz),1))
                row['one_iteration_schedule_only']=describe_actions(nexta,af,state['std'],state['std'],groups)
                result['checkpoints'][name]=row
                print(filename,name,'alpha',alpha,'student/fused KL',round(row['student_vs_fused']['post_fault_1s_plus']['kl_mean'],5),flush=True)
            report['corpora'][filename]=result
    report['protocol']['cuda_initialized']=torch.cuda.is_initialized()
    assert not torch.cuda.is_initialized()
    assert sha(TF)==report['protocol']['tf_sha256']
    assert all(sha(v['path'])==v['sha256'] for v in report['protocol']['checkpoints'].values())
    (out/'results.json').write_text(json.dumps(report,indent=2))
    print(out/'results.json',flush=True)


if __name__=='__main__':main()
