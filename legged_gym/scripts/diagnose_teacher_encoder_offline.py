"""CPU-only checkpoint/input diagnostics. No simulator, optimizer or training imports.

Interventions are counterfactual network-input probes, not new physical rollouts.
Affine fits only align latent coordinates analytically; no network is updated.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]
TF = ROOT / 'logs/official_wim_teacher243_failure_fullrange_fromscratch/Aug13_03-44-07_seed1-50000iter-fromscratch-fullrange/model_43000.pt'
JT = ROOT / 'logs/jt_wim243_failure_fullrange_onset/Aug14_00-37-37_'
CORPUS = ROOT / 'logs/evaluations/jt-transition-audit-20260928'
GROUPS = {'friction': [0], 'payload': [1], 'motor_strength': list(range(2,14)),
          'kp': list(range(14,26)), 'kd': list(range(26,38)),
          'linear_velocity': list(range(38,41)), 'angular_velocity': list(range(41,44)),
          'failure_flag': [44]}
JOINTS = ['FL_hip','FL_thigh','FL_calf','FR_hip','FR_thigh','FR_calf',
          'RL_hip','RL_thigh','RL_calf','RR_hip','RR_thigh','RR_calf']
DIM_NAMES = (['friction','payload'] + [f'motor_{j}' for j in JOINTS]
             + [f'kp_{j}' for j in JOINTS] + [f'kd_{j}' for j in JOINTS]
             + ['vx','vy','vz','wx','wy','wz','failure_flag'])


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def mlp(state, prefix):
    layers = []
    keys = sorted([k for k in state if k.startswith(prefix+'.') and k.endswith('.weight')],
                  key=lambda k:int(k.split('.')[1]))
    for i,k in enumerate(keys):
        w = state[k]
        layers.append(nn.Linear(w.shape[1],w.shape[0]))
        if i < len(keys)-1:
            layers.append(nn.ELU())
    model = nn.Sequential(*layers).eval()
    model.load_state_dict({k[len(prefix)+1:]:v for k,v in state.items() if k.startswith(prefix+'.')}, strict=True)
    model.requires_grad_(False)
    return model


class Student(nn.Module):
    def __init__(self,state):
        super().__init__()
        w=state['student_encoder.frame_encoder.0.weight'].shape[0]
        self.frame_encoder=nn.Sequential(nn.Linear(48,w),nn.ELU())
        self.temporal=nn.Sequential(nn.Conv1d(w,w,8,4),nn.ELU(),nn.Conv1d(w,w,5),nn.ELU(),nn.Conv1d(w,w,5),nn.ELU())
        self.projection=nn.Linear(w*3,8)
        self.load_state_dict({k[len('student_encoder.'):]:v for k,v in state.items() if k.startswith('student_encoder.')},strict=True)
        self.requires_grad_(False)
        self.eval()

    def forward(self,x):
        x=x[:,235:].reshape(-1,50,48)
        return self.projection(self.temporal(self.frame_encoder(x).transpose(1,2)).flatten(1))


def call(model, x):
    return torch.cat([model(b) for b in x.split(512)])


def rms(x):
    return float(x.square().mean().sqrt())


def masks(c):
    t,r=c['elapsed_since_onset'],c['rate']
    return {'all':torch.ones(len(r),dtype=torch.bool), 'pre_fault':(t<0)&(r>0),
            'post_fault_1s_plus':(t>=1)&(r>0), 'd1_1s_plus':(t>=1)&(r==1),
            'post_fault_5s_plus':(t>=5)&(r>0)}


def differences(z,a,z0,a0,ms):
    out={}
    for name,m in ms.items():
        dz,da=z[m]-z0[m],a[m]-a0[m]
        per=da.abs().mean(1)
        out[name]={'n':int(m.sum()),'latent_l2':float(dz.norm(dim=1).mean()),
                   'latent_relative_rms':rms(dz)/max(rms(z0[m]),1e-9),
                   'action_mae':float(per.mean()), 'joint_target_mae_rad':float(per.mean())*.25,
                   'action_mae_p95_state':float(per.quantile(.95))}
    return out


def get_selection(c):
    # Cover every recorded episode and retain pre/early/late fault observations.
    selected=[]
    for env in c['env_id'].unique(sorted=True):
        ids=torch.where(c['env_id']==env)[0]
        t=c['elapsed_since_onset'][ids]
        for mask,cap in [(t<0,3),((t>=0)&(t<1),2),(t>=1,8)]:
            candidates=ids[mask]
            if len(candidates):
                selected.extend(candidates[torch.linspace(0,len(candidates)-1,min(cap,len(candidates))).round().long()].tolist())
    return torch.tensor(sorted(set(selected)),dtype=torch.long)


def nominal(c):
    p=c['privileged']
    out=p.clone()
    out[:,0]=.875;out[:,1]=1.;out[:,14:38]=1.
    out[:,38:44]=p[:,38:44].median(dim=0).values
    out[:,44]=0.
    for env in c['env_id'].unique():
        ids=torch.where(c['env_id']==env)[0]
        pre=ids[c['elapsed_since_onset'][ids]<0]
        assert len(pre), 'Every episode must have a pre-onset motor reference'
        motor=p[pre[0],2:14]
        assert torch.allclose(p[pre,2:14],motor.expand(len(pre),12),atol=1e-6)
        out[ids,2:14]=motor
    return out


def permutation(c):
    # Match severity and coarse onset phase; replace all coordinates in a group together.
    t=c['elapsed_since_onset']
    phase=torch.bucketize(t,torch.tensor([0.,1.,3.,10.]))
    strata=(c['rate']*5).round().long()*10+phase
    out=torch.arange(len(t));g=torch.Generator().manual_seed(20260929)
    for s in strata.unique():
        ids=torch.where(strata==s)[0]
        out[ids]=ids[torch.randperm(len(ids),generator=g)]
    return out


def geometry(z):
    centered=z-z.mean(0)
    eig=torch.linalg.eigvalsh(centered.T@centered/max(len(z)-1,1)).clamp(min=0)
    prob=eig/eig.sum().clamp(min=1e-12)
    return {'rms':rms(z),'centered_rms':rms(centered),'mean_norm':float(z.mean(0).norm()),
            'per_dim_mean':z.mean(0).tolist(),'per_dim_std':z.std(0).tolist(),
            'effective_rank':float(torch.exp(-(prob*prob.clamp(min=1e-12).log()).sum())),
            'eigenvalues':eig.tolist()}


def aligned(x,y,train):
    # Least-squares coordinate alignment, fit on 36/48 replicate blocks, held out on 12/48.
    x=x.double(); y=y.double()
    xm,ym=x[train].mean(0),y[train].mean(0)
    xc,yc=x[train]-xm,y[train]-ym
    scalar=(xc*yc).sum()/xc.square().sum().clamp(min=1e-12)
    pred_scalar=(x-xm)*scalar+ym
    design=torch.cat((x,torch.ones(len(x),1,dtype=x.dtype)),dim=1)
    coef=torch.linalg.lstsq(design[train],y[train]).solution
    return pred_scalar.float(),(design@coef).float(),float(scalar)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20260929)
    started=time.time()
    paths={'TF43000':TF,**{f'JT{i}':JT/f'model_{i}.pt' for i in [43000,43500,45000,49000,51000,53000,71500]}}
    states={name:torch.load(path,map_location='cpu')['model_state_dict'] for name,path in paths.items()}
    report={'protocol':{'device':'cpu','threads':1,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
                       'no_optimizer_or_simulator':True,'checkpoint_sha256':{k:sha(v) for k,v in paths.items()},
                       'checkpoints':{k:str(v) for k,v in paths.items()},'script_sha256':sha(__file__),
                       'groups':GROUPS,'dimensions':DIM_NAMES,
                       'warning':'Input interventions can be inconsistent with recorded dynamics. They quantify network sensitivity, not Q, physical causality or achievable student information.',
                       'alignment_split':'(env_id // 72 // 12) % 4 == 3 held out; all 12 joints in both splits',
                       'cohorts':'two recorded visitation distributions, one simulation seed; not independent training replications'},'corpora':{}}
    with torch.inference_mode():
        for filename in ['common_teacher_inputs.pt','common_student_inputs.pt']:
            c=torch.load(CORPUS/filename,map_location='cpu')
            assert all(v.device.type=='cpu' for v in c.values())
            assert len(c['privileged'][0])==45 and c['obs'].shape[1]==2635
            sel=get_selection(c); cs={k:v[sel] for k,v in c.items()}
            p=cs['privileged'];obs=cs['obs'][:,:235];ms=masks(cs)
            nom=nominal(c)[sel]; perm=permutation(cs)
            result={'sha256':sha(CORPUS/filename),'n_full':len(c['obs']),'n_probed':len(sel),
                    'episode_count':int(c['env_id'].unique().numel()),'group_interventions':{},
                    'dimension_interventions':{},'latent_drift':{},'student_latent_repairs':{},
                    'selection_indices':sel.tolist(),'group_ranges':{k:{'min':p[:,v].min(0).values.tolist(),'max':p[:,v].max(0).values.tolist()} for k,v in GROUPS.items()}}
            ref=mlp(states['TF43000'],'teacher_encoder')
            zref=call(ref,c['privileged'])
            holdout=((c['env_id']//72)//12)%4==3
            train=~holdout
            for split in (holdout,train):
                assert len(((c['env_id'][split]%72)//6).unique())==12
            for name,state in states.items():
                enc,actor=mlp(state,'teacher_encoder'),mlp(state,'actor')
                z=call(enc,p);a=call(actor,torch.cat((obs,z),1))
                full_z=call(enc,c['privileged'])
                zp_scalar,zp_affine,scale=aligned(zref,full_z,train)
                drift={'scalar_fit':scale,'groups':{}}
                full_ms=masks(c)
                for mname,m in full_ms.items():
                    h=m&holdout;den=max(rms(full_z[h]-full_z[h].mean(0)),1e-9)
                    drift['groups'][mname]={'n':int(h.sum()),'geometry':geometry(full_z[h]),
                        'relative_drift_from_tf':rms(full_z[h]-zref[h])/max(rms(zref[h]),1e-9),
                        'scalar_alignment_centered_nrmse':rms(zp_scalar[h]-full_z[h])/den,
                        'affine_alignment_centered_nrmse':rms(zp_affine[h]-full_z[h])/den}
                affine_action=call(actor,torch.cat((obs,zp_affine[sel]),1))
                drift['affine_transport_action_error']=differences(zp_affine[sel],affine_action,z,a,{k:v&holdout[sel] for k,v in ms.items()})
                # Mean/std change of latent coordinates, compensated exactly in actor's first layer.
                mu=full_z[train].mean(0);sd=full_z[train].std(0).clamp(min=1e-6)
                adjusted=mlp(state,'actor')
                adjusted[0].bias.add_(adjusted[0].weight[:,235:]@mu)
                adjusted[0].weight[:,235:].mul_(sd[None,:])
                normalized_a=call(adjusted,torch.cat((obs,(z-mu)/sd),1))
                drift['normalization_reparameterization_max_action_diff']=float((normalized_a-a).abs().max())
                # Relative perturbation: +/- 5% of each latent coordinate's recorded std.
                noises=[]
                for dim in range(8):
                    pert=z.clone();pert[:,dim]+=.05*sd[dim]
                    aa=call(actor,torch.cat((obs,pert),1))
                    noises.append(differences(pert,aa,z,a,ms))
                drift['latent_5pct_std_sensitivity_per_dim']=noises
                result['latent_drift'][name]=drift
                probes={}
                for g,idx in GROUPS.items():
                    for mode in ['nominal','matched_permutation']:
                        q=p.clone();q[:,idx]=(nom if mode=='nominal' else p[perm])[:,idx]
                        zz=call(enc,q);aa=call(actor,torch.cat((obs,zz),1))
                        probes[f'{g}/{mode}']=differences(zz,aa,z,a,ms)
                for pname,idx,vals in [
                    ('motor_and_flag_restore',list(range(2,14))+[44],nom),
                    ('velocities_from_current_observation',list(range(38,44)),None),
                    ('linear_velocity_from_current_observation',list(range(38,41)),None),
                    ('angular_velocity_from_current_observation',list(range(41,44)),None)]:
                    q=p.clone()
                    if vals is not None:q[:,idx]=vals[:,idx]
                    else:
                        observed=torch.cat((obs[:,:3]/2.,obs[:,3:6]/.25),1)
                        for j in idx:q[:,j]=observed[:,j-38]
                    zz=call(enc,q);aa=call(actor,torch.cat((obs,zz),1))
                    probes[pname]=differences(zz,aa,z,a,ms)
                result['group_interventions'][name]=probes
                if name in ('TF43000','JT53000'):
                    dims={}
                    for j,dname in enumerate(DIM_NAMES):
                        q=p.clone();q[:,j]=nom[:,j]
                        zz=call(enc,q);aa=call(actor,torch.cat((obs,zz),1))
                        dims[dname]=differences(zz,aa,z,a,ms)
                    result['dimension_interventions'][name]=dims
                if name.startswith('JT'):
                    stu=Student(state);zs=call(stu,cs['obs']);aa=call(actor,torch.cat((obs,zs),1))
                    repairs={'base':differences(zs,aa,z,a,ms),'one_dimension_teacher_replacement':[]}
                    for dim in range(8):
                        repaired=zs.clone();repaired[:,dim]=z[:,dim]
                        ra=call(actor,torch.cat((obs,repaired),1))
                        repairs['one_dimension_teacher_replacement'].append(differences(repaired,ra,z,a,ms))
                    result['student_latent_repairs'][name]=repairs
                print(filename,name,'post RMS',round(drift['groups']['post_fault_1s_plus']['geometry']['rms'],3),
                      'affine residual',round(drift['groups']['post_fault_1s_plus']['affine_alignment_centered_nrmse'],3),flush=True)
            report['corpora'][filename]=result
            (out/'results.json').write_text(json.dumps(report,indent=2))
    report['protocol']['elapsed_seconds']=time.time()-started
    report['protocol']['cuda_initialized']=torch.cuda.is_initialized()
    assert not torch.cuda.is_initialized()
    assert all(sha(paths[k])==v for k,v in report['protocol']['checkpoint_sha256'].items())
    (out/'results.json').write_text(json.dumps(report,indent=2))
    print('complete',round(time.time()-started,1),'seconds',flush=True)


if __name__=='__main__':
    main()
