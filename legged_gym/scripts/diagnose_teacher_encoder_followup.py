"""Read-only CPU follow-up: separate within-episode velocity variation and lag."""
import argparse
import json
from pathlib import Path

from diagnose_teacher_encoder_offline import (
    torch, TF, JT, CORPUS, mlp, Student, call, masks, differences, rms, sha,
)


def partition(z,env):
    residual=z.clone()
    for e in env.unique():
        m=env==e
        residual[m]-=z[m].mean(0)
    total=(z-z.mean(0)).square().sum()
    return {'within_episode_centered_energy_fraction':float(residual.square().sum()/total),
            'within_episode_rms':rms(residual),'total_centered_rms':rms(z-z.mean(0))}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    names={'TF43000':TF,**{f'JT{i}':JT/f'model_{i}.pt' for i in [45000,49000,51000,53000,71500]}}
    result={'protocol':{'device':'cpu','threads':1,'no_optimizer_or_simulator':True,'script_sha256':sha(__file__),
                       'warning':'Oracle context and time-lag interventions are only recorded-input diagnostics. They do not estimate a new policy Q.'},'corpora':{}}
    with torch.inference_mode():
        for filename in ('common_teacher_inputs.pt','common_student_inputs.pt'):
            c=torch.load(CORPUS/filename,map_location='cpu'); p=c['privileged']; obs=c['obs'];cur=obs[:,:235]
            history=obs[:,235:].reshape(-1,50,48)
            assert torch.allclose(history[:,-1],cur[:,:48],atol=1e-6)
            post=(c['rate']>0)&(c['elapsed_since_onset']>=1)
            per_env=[]
            for env in c['env_id'][post].unique():
                rows=p[post&(c['env_id']==env)]
                constant=torch.cat((rows[:,:38],rows[:,44:]),1)
                per_env.append(float((constant-constant[0]).abs().max()))
            assert max(per_env)<1e-6
            velocity=torch.cat((cur[:,:3]/2,cur[:,3:6]/.25),1)
            report={'latest_history_matches_current':True,
                    'clean_vs_current_velocity_max_abs':float((p[:,38:44]-velocity).abs().max()),
                    'postfault_nonvelocity_privileged_max_within_episode_change':max(per_env),
                    'models':{}}
            for name,path in names.items():
                state=torch.load(path,map_location='cpu')['model_state_dict']
                encoder,actor=mlp(state,'teacher_encoder'),mlp(state,'actor')
                z=call(encoder,p);a=call(actor,torch.cat((cur,z),1));ms=masks(c)
                item={'postfault_variance_partition':partition(z[post],c['env_id'][post]),'velocity_lag':{}}
                if name.startswith('JT'):
                    zs=call(Student(state),obs);ass=call(actor,torch.cat((cur,zs),1))
                    item['student_current_teacher_gap']=differences(zs,ass,z,a,ms)
                for frame in [49,48,47,44,39]:
                    q=p.clone();q[:,38:41]=history[:,frame,:3]/2;q[:,41:44]=history[:,frame,3:6]/.25
                    zl=call(encoder,q);al=call(actor,torch.cat((cur,zl),1))
                    entry={'lag_seconds':(49-frame)*.02,'teacher_lag_effect':differences(zl,al,z,a,ms)}
                    if name.startswith('JT'):
                        entry['student_vs_lagged_teacher']=differences(zs,ass,zl,al,ms)
                    item['velocity_lag'][str(frame)]=entry
                report['models'][name]=item
                print(filename,name,'within fraction',round(item['postfault_variance_partition']['within_episode_centered_energy_fraction'],3),
                      '40ms action',round(item['velocity_lag']['47']['teacher_lag_effect']['post_fault_1s_plus']['action_mae'],4),flush=True)
            result['corpora'][filename]=report
    result['protocol']['cuda_initialized']=torch.cuda.is_initialized()
    assert not torch.cuda.is_initialized()
    output.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
