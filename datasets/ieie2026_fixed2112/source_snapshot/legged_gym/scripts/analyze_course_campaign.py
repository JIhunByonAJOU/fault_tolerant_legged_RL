"""Complete-grid audit, paired summaries and bounded exploratory mixture search."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from legged_gym.evaluation.course_campaign import (
    BASE_COUNTS, CLASSES, FAMILIES, atomic_json, make_jobs, specs, sha,
)

METRICS=['survived','command_vx_rmse','command_vy_rmse','command_wz_rmse','cross_track_rmse']


def mean(values):
    a=[v for v in values if v is not None]
    return float(np.mean(a)) if a else None


def csv_write(path,rows):
    if not rows: return
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def analyze(campaign,stage,selection=None):
    campaign=Path(campaign); p=json.loads((campaign/'protocol.json').read_text())
    repeats=selection['family_units'] if stage=='confirmation' else None
    columns=selection['terrain_counts'] if stage=='confirmation' else None
    jobs=make_jobs(p,stage,repeats,columns)
    data=[]; seen=set(); pairs={}
    for job in jobs:
        folder=campaign/'jobs'/job['id']
        complete=json.loads((folder/'COMPLETE.json').read_text())
        if complete['pilot'] or sha(folder/'results.json')!=complete['result_sha256']:
            raise ValueError('Incomplete or changed results '+job['id'])
        metadata=json.loads((folder/'metadata.json').read_text())
        if metadata['protocol_sha256']!=sha(campaign/'protocol.json'): raise ValueError('Protocol changed')
        expected={r['paired_trial_id'] for r in specs(p,stage,job['seed'],job['conditions'],repeats,columns)}
        results=json.loads((folder/'results.json').read_text())
        if {r['paired_trial_id'] for r in results}!=expected: raise ValueError('Grid coverage mismatch')
        for row in results:
            if row['trial_id'] in seen: raise ValueError('Duplicated trial')
            seen.add(row['trial_id']); row['job_id']=job['id']; data.append(row)
            pairs.setdefault(row['paired_trial_id'],{})[row['policy']]=row
    if any(set(pair)!=set(p['policies']) for pair in pairs.values()): raise ValueError('Missing policy partner')
    out=campaign/'analysis'/stage; out.mkdir(parents=True,exist_ok=True)
    flat=[]
    for r in data:
        flat.append(dict(trial_id=r['trial_id'],paired_trial_id=r['paired_trial_id'],job_id=r['job_id'],
            policy=r['policy'],seed=r['seed'],profile=r['profile']['id'],family=r['profile']['family'],
            terrain_class=r['terrain_class'],terrain_column=r['terrain_column_id'],allocation_slot=r['allocation_slot'],
            joint=r['condition']['joint'],loss=r['condition']['loss'],termination=r['termination'],
            failed_before_onset=r['failed_before_onset'],observed_post_s=r['observed_post_s'],
            valid_post_samples=r['valid_post_samples'],stall_seconds=r['stall_seconds'],**{m:r[m] for m in METRICS}))
    csv_write(out/'all_trials.csv',flat)
    def summarize(keys,name):
        groups={}
        for row in flat:
            key=tuple(row[k] for k in keys); groups.setdefault(key,[]).append(row)
        summary=[]
        for key,rows in sorted(groups.items(),key=lambda x:str(x[0])):
            result=dict(zip(keys,key)); result['episodes']=len(rows)
            result['mean_valid_post_s']=mean([r['observed_post_s'] for r in rows])
            for metric in METRICS:
                result[metric]=mean([r[metric] for r in rows])
                result[metric+'_valid_n']=sum(r[metric] is not None for r in rows)
            result['goal_rate']=mean([float(r['termination']=='goal') for r in rows])
            result['stall_flag_rate']=mean([float(r['stall_seconds']>0) for r in rows])
            summary.append(result)
        csv_write(out/name,summary)
        return summary
    summarize(['policy','profile','terrain_column','joint','loss'],'by_every_condition.csv')
    summarize(['policy','family','terrain_class','loss'],'by_family_terrain_loss.csv')
    summarize(['policy','family','loss'],'by_family_loss.csv')
    summarize(['policy','terrain_class','loss'],'by_terrain_loss.csv')
    summarize(['policy','seed','loss'],'by_seed_loss.csv')
    overall=summarize(['policy'],'overall.csv')
    paired=[]; gains=[[[] for _ in CLASSES] for _ in FAMILIES]
    for pair_id,pair in pairs.items():
        normal=pair['Normal']; student=pair['Student']
        row=dict(paired_trial_id=pair_id,profile=normal['profile']['id'],family=normal['profile']['family'],
                 terrain_class=normal['terrain_class'],terrain_column=normal['terrain_column_id'],
                 joint=normal['condition']['joint'],loss=normal['condition']['loss'],seed=normal['seed'],
                 normal_job=normal['job_id'],student_job=student['job_id'],
                 normal_env=normal['env_id'],student_env=student['env_id'])
        gain=[]
        for index,metric in enumerate(METRICS):
            a=normal[metric]; b=student[metric]
            diff=None if a is None or b is None else float(b-a if index==0 else a-b)
            row[metric+'_student_gain']=diff; gain.append(diff)
        paired.append(row)
        if row['loss']>0:
            gains[FAMILIES.index(row['family'])][CLASSES.index(row['terrain_class'])].append(gain)
    csv_write(out/'paired_differences.csv',paired)
    top=sorted(paired,key=lambda r:(r['survived_student_gain'],r['cross_track_rmse_student_gain'] or 0),reverse=True)
    csv_write(out/'largest_differences.csv',top[:200])
    missing={m:sum(r[m] is None for r in flat) for m in METRICS}
    audit=dict(stage=stage,complete=True,jobs=len(jobs),robot_episodes=len(data),paired_robot_trials=len(pairs),
               policy_seed_runs=len(data)//12,missing_metric_counts=missing,primary_survival='mean of binary non-fall outcomes',
               error_definition='mean per-trial RMSE on valid post-onset pre-fall/goal/cap samples; zero samples -> NA',
               short_survival_errors='conditional on observed alive interval; interpret jointly with survival',
               healthy_failures=sum(r['loss']==0 and r['survived']==0 for r in flat))
    atomic_json(out/'coverage.json',audit)
    if stage=='exploration':
        matrix=np.zeros((4,6,5))
        for f in range(4):
            for t in range(6):
                for m in range(5):
                    v=mean([g[m] for g in gains[f][t]])
                    matrix[f,t,m]=v if v is not None else 0.
        base=np.einsum('f,t,ftm->m',np.ones(4)/4,np.array(BASE_COUNTS)/12,matrix)
        candidates=[]
        for units in itertools.product(range(1,4),repeat=4):
            if sum(units)!=8: continue
            for counts in itertools.product(*[range(max(1,c-1),c+2) for c in BASE_COUNTS]):
                if sum(counts)!=12: continue
                score=np.einsum('f,t,ftm->m',np.array(units)/8,np.array(counts)/12,matrix)
                feasible=bool(np.all(score[1:]>=base[1:]-1e-10))
                candidates.append(dict(family_units=list(units),terrain_counts=list(counts),gains=score.tolist(),
                                       feasible=feasible,extra_survival_gain=float(score[0]-base[0])))
        feasible=[c for c in candidates if c['feasible']]
        chosen=max(feasible,key=lambda c:(round(c['gains'][0],6),round(c['gains'][4],6),
                    round(c['gains'][1],6),round(c['gains'][2],6),round(c['gains'][3],6),
                    -sum(abs(x-2) for x in c['family_units'])-sum(abs(x-y) for x,y in zip(c['terrain_counts'],BASE_COUNTS))))
        selection=dict(chosen, family_names=FAMILIES,terrain_names=CLASSES,
                       baseline_family_units=[2]*4,baseline_terrain_counts=BASE_COUNTS,
                       baseline_gains=base.tolist(),exploratory=True,
                       constraints='family 1..3 units/8; terrain each original count +/-1 with min1; all four tracking gains >= baseline',
                       objective='largest binary survival gain with tracking constraints; no mixed-unit scalar score',
                       selected_from_stage='exploration',confirmation_seeds=p['confirmation_seeds'],
                       confirmation_profile_repeats='8 total motion slots; family_units allocated with counterbalanced subprofiles across conditions and seeds',
                       selection_data_sha256=sha(out/'paired_differences.csv'))
        atomic_json(campaign/'selection.json',selection)
        rows_candidates=[]
        for c in sorted(candidates,key=lambda x:x['gains'][0],reverse=True):
            rows_candidates.append(dict(family_units=json.dumps(c['family_units']),terrain_counts=json.dumps(c['terrain_counts']),
                feasible=c['feasible'],**{METRICS[i]+'_gain':c['gains'][i] for i in range(5)}))
        csv_write(out/'all_mixture_candidates.csv',rows_candidates)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        profile_ids=[r['id'] for r in p['profiles']]
        for metric in [name+'_student_gain' for name in METRICS]:
            matrix=np.full((8,12),np.nan)
            for i,profile in enumerate(profile_ids):
                for j in range(12):
                    vals=[r[metric] for r in paired if r['profile']==profile and r['terrain_column']==j+1 and r['loss']>0]
                    value=mean(vals)
                    if value is not None: matrix[i,j]=value
            fig,ax=plt.subplots(figsize=(10,5)); im=ax.imshow(matrix,cmap='RdBu',aspect='auto')
            ax.set(yticks=range(8),yticklabels=profile_ids,xticks=range(12),xticklabels=range(1,13),
                   xlabel='Original terrain column',title=metric+' (fault conditions)')
            unit='fraction' if metric.startswith('survived') else ('rad/s' if 'wz_' in metric else ('m' if 'cross_track' in metric else 'm/s'))
            fig.colorbar(im,ax=ax,label=unit); fig.tight_layout(); fig.savefig(out/(metric+'.png'),dpi=160); plt.close(fig)
    except ImportError:
        pass
    report=['# '+stage+' evaluation','',
            'Survival is the fraction without a fall through goal or administrative cap. Stall does not terminate a trial.',
            '', 'Complete robot records: %d; policy/seed runs: %d.'%(len(data),len(data)//12),'',
            'Errors are conditional on valid observed motion before termination; compare with survival and valid duration.',
            '', 'Overall policy summaries:','', '```json',json.dumps(overall,indent=2),'```','']
    if selection:
        report+=['Mixture selected on exploration; confirmation uses unused seeds and the same frozen mixture.',
                 '', '```json',json.dumps(selection,indent=2),'```','']
    (out/'report.md').write_text('\n'.join(report))
    return selection


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--stage',choices=['exploration','confirmation'],required=True)
    args=parser.parse_args()
    selection=json.loads((args.campaign/'selection.json').read_text()) if args.stage=='confirmation' else None
    analyze(args.campaign,args.stage,selection)
