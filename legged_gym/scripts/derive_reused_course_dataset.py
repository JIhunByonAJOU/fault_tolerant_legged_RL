"""Reuse complete original paired records for the fixed 2:1:1:2 design.

No simulator, policy inference, outcome-dependent filtering, or synthetic run
completion markers. Every selected record retains its original trial/job/env ID.
"""
import collections
import json
from pathlib import Path
import shutil
import time

import numpy as np
from legged_gym.evaluation.course_campaign import ROOT, conditions, atomic_json, sha, BASE_COUNTS
from legged_gym.scripts.analyze_course_campaign import csv_write, METRICS, mean
from legged_gym.scripts.assemble_fixed_course_delivery import build

SOURCE=ROOT/'logs/evaluation/ieie2026_eight_20261016'
OUT=ROOT/'logs/evaluation/ieie2026_reused2112_20261016'
SEEDS=[1101,1102,1103]


def main():
    OUT.mkdir(exist_ok=True);a=OUT/'analysis/fixed2112';a.mkdir(parents=True,exist_ok=True)
    atomic_json(OUT/'state.json',dict(status='ANALYZING',stage='reuse_audit',started_at=time.time(),additional_simulations=0))
    p=json.loads((SOURCE/'protocol.json').read_text())
    p.pop('selection',None);p['protocol_id']='ieie_reused_fixed2112_v1'
    p['data_origin']='derived subset of original complete exploration records'
    atomic_json(OUT/'protocol.json',p)
    design=dict(family_units=[2,1,1,2],terrain_counts=BASE_COUNTS,evaluation_seeds=SEEDS,policy_seed_runs=2196,
        robot_episodes=26352,additional_simulations=0,data_origin='기존 첫 전수 수집에서 조건 식별자로 추출; 추가 시뮬레이션 없음',
        analysis_subdir='fixed2112',independent_confirmation=False,source_campaign=str(SOURCE),
        selection_rule='F01,F02,S20,S15 always; diagonal=(condition_index+seed_index+1)%2; lateral=(condition_index+seed_index+2)%2; all 12 columns; both policies',
        outcome_dependent_selection=False)
    atomic_json(OUT/'fixed_design.json',design)
    wanted=set()
    for si,seed in enumerate(SEEDS):
        for ci,cond in enumerate(conditions()):
            profiles=['F01','F02',['DL','DR'][(ci+si+1)%2],['L','R'][(ci+si+2)%2],'S20','S15']
            for profile in profiles:
                for col in range(1,13):wanted.add((seed,cond['id'],profile,col))
    data=[];manifest=[];seen=set()
    for job in json.loads((SOURCE/'exploration_jobs.json').read_text()):
        folder=SOURCE/'jobs'/job['id'];marker=json.loads((folder/'COMPLETE.json').read_text())
        assert not marker['pilot'] and sha(folder/'results.json')==marker['result_sha256']
        meta=json.loads((folder/'metadata.json').read_text())
        assert meta['protocol_sha256']==sha(SOURCE/'protocol.json')
        trace=json.loads((folder/'trace_manifest.json').read_text())
        for chunk in trace['chunks']:assert sha(folder/chunk['path'])==chunk['sha256']
        if job['policy']=='Student':
            normal=SOURCE/'jobs'/(job['pair_job']+'_Normal')/'metadata.json'
            assert meta['initial_hashes']==json.loads(normal.read_text())['initial_hashes']
        count=0
        for r in json.loads((folder/'results.json').read_text()):
            key=(r['seed'],r['condition']['id'],r['profile']['id'],r['terrain_column_id'])
            if key not in wanted:continue
            assert (key,r['policy']) not in seen;seen.add((key,r['policy']))
            assert r['survived']==int(r['termination']!='fall')
            assert r['valid_post_samples']>=0
            data.append(dict(r,source_job=job['id'],source_folder=str(folder),source_result_sha256=marker['result_sha256']))
            count+=1
        manifest.append(dict(job=job['id'],selected_episodes=count,source_episodes=marker['episodes'],
            results=str(folder/'results.json'),results_sha256=marker['result_sha256'],
            trace_manifest_sha256=sha(folder/'trace_manifest.json'),initial_hashes=meta['initial_hashes']))
    assert len(data)==26352 and len(seen)==len(wanted)*2
    assert all((k,policy) in seen for k in wanted for policy in ['Normal','Student'])
    pairs={}
    for r in data:pairs.setdefault(r['paired_trial_id'],{})[r['policy']]=r
    assert len(pairs)==13176 and all(set(v)=={'Normal','Student'} for v in pairs.values())
    counts=collections.Counter(r['profile']['family'] for r in data)
    assert [counts[k] for k in ['forward','diagonal','lateral','sine']]==[8784,4392,4392,8784]
    atomic_json(OUT/'derived_records.json',data)
    refs=[];flat=[]
    for r in data:
        refs.append(dict(trial_id=r['trial_id'],paired_trial_id=r['paired_trial_id'],source_job=r['source_job'],source_env_id=r['env_id'],
                         source_results=str(Path(r['source_folder'])/'results.json'),source_result_sha256=r['source_result_sha256']))
        flat.append(dict(trial_id=r['trial_id'],paired_trial_id=r['paired_trial_id'],source_job=r['source_job'],source_env_id=r['env_id'],
            policy=r['policy'],seed=r['seed'],profile=r['profile']['id'],family=r['profile']['family'],terrain_class=r['terrain_class'],
            terrain_column=r['terrain_column_id'],joint=r['condition']['joint'],loss=r['condition']['loss'],termination=r['termination'],
            failed_before_onset=r['failed_before_onset'],observed_post_s=r['observed_post_s'],valid_post_samples=r['valid_post_samples'],
            stall_seconds=r['stall_seconds'],**{m:r[m] for m in METRICS}))
    csv_write(a/'all_trials.csv',flat);csv_write(a/'source_record_index.csv',refs)
    def summarize(keys,name):
        groups={}
        for r in flat:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
        out=[]
        for key,rs in sorted(groups.items(),key=lambda t:str(t[0])):
            row=dict(zip(keys,key),episodes=len(rs),mean_valid_post_s=mean([r['observed_post_s'] for r in rs]))
            for m in METRICS:
                row[m]=mean([r[m] for r in rs]);row[m+'_valid_n']=sum(r[m] is not None for r in rs)
            row['goal_rate']=mean([float(r['termination']=='goal') for r in rs])
            row['stall_flag_rate']=mean([float(r['stall_seconds']>0) for r in rs]);out.append(row)
        csv_write(a/name,out);return out
    summarize(['policy'],'overall.csv')
    summarize(['policy','profile','terrain_column','joint','loss'],'by_every_condition.csv')
    summarize(['policy','family','terrain_class','loss'],'by_family_terrain_loss.csv')
    summarize(['policy','family','loss'],'by_family_loss.csv')
    summarize(['policy','terrain_class','loss'],'by_terrain_loss.csv')
    summarize(['policy','seed','loss'],'by_seed_loss.csv')
    differences=[]
    for pairid,pair in pairs.items():
        n,s=pair['Normal'],pair['Student']
        row=dict(paired_trial_id=pairid,seed=n['seed'],profile=n['profile']['id'],family=n['profile']['family'],
                 terrain_class=n['terrain_class'],terrain_column=n['terrain_column_id'],joint=n['condition']['joint'],loss=n['condition']['loss'],
                 normal_job=n['source_job'],student_job=s['source_job'],normal_env=n['env_id'],student_env=s['env_id'])
        for i,m in enumerate(METRICS):row[m+'_student_gain']=None if n[m] is None or s[m] is None else (s[m]-n[m] if i==0 else n[m]-s[m])
        differences.append(row)
    csv_write(a/'paired_differences.csv',differences)
    order=lambda r:(r['survived_student_gain'],r['cross_track_rmse_student_gain'] or 0)
    csv_write(a/'largest_differences.csv',sorted(differences,key=order,reverse=True)[:200])
    csv_write(a/'largest_student_losses.csv',sorted(differences,key=order)[:200])
    audit=dict(complete=True,source_completed_jobs=18,source_robot_episodes=35136,selected_robot_episodes=len(data),
        policy_seed_runs=len(data)//12,paired_robot_trials=len(pairs),missing_required_trials=0,duplicate_required_trials=0,
        family_episode_counts=dict(counts),additional_simulations=0,independent_confirmation=False,
        metric_missing_counts={m:sum(r[m] is None for r in flat) for m in METRICS})
    atomic_json(a/'coverage.json',audit)
    atomic_json(OUT/'derivation_manifest.json',dict(design=design,sources=manifest,source_protocol_sha256=sha(SOURCE/'protocol.json'),
        derived_records_sha256=sha(OUT/'derived_records.json'),source_record_index_sha256=sha(a/'source_record_index.csv'),audit=audit))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    profiles=['F01','F02','DL','DR','L','R','S20','S15']
    for metric in METRICS:
        values=np.full((8,12),np.nan)
        for i,profile in enumerate(profiles):
            for j in range(12):
                v=mean([r[metric+'_student_gain'] for r in differences if r['profile']==profile and r['terrain_column']==j+1 and r['loss']>0])
                if v is not None:values[i,j]=v*(100 if metric=='survived' else 1)
        bound=max(1e-8,float(np.nanmax(np.abs(values))))
        fig,ax=plt.subplots(figsize=(10,5));im=ax.imshow(values,cmap='RdBu',norm=Normalize(-bound,bound),aspect='auto')
        ax.set(xticks=range(12),xticklabels=range(1,13),yticks=range(8),yticklabels=profiles,
               xlabel='Original terrain column',title=metric+' Student gain; fixed ratio 2:1:1:2')
        unit='percentage points' if metric=='survived' else ('rad/s' if 'wz' in metric else ('m' if metric=='cross_track_rmse' else 'm/s'))
        fig.colorbar(im,ax=ax,label=unit);fig.tight_layout();fig.savefig(a/(metric+'_student_gain.png'),dpi=160);plt.close(fig)
    files=['legged_gym/scripts/derive_reused_course_dataset.py','legged_gym/scripts/assemble_fixed_course_delivery.py','legged_gym/scripts/assemble_course_delivery.py']
    atomic_json(OUT/'source_manifest.json',dict(files={f:sha(ROOT/f) for f in files},source_campaign=str(SOURCE)))
    atomic_json(OUT/'FINAL.json',dict(status='COMPLETE',kind='derived_existing_records',additional_simulations=0,
        fixed_design=design,coverage=audit,completed_at=time.time()))
    build(OUT)
    atomic_json(OUT/'state.json',dict(status='COMPLETE',stage='reused_fixed2112',additional_simulations=0,source_robot_episodes=35136,
        selected_robot_episodes=26352,policy_seed_runs=2196,task_ratio=[2,1,1,2],evaluation_seeds=SEEDS,completed_at=time.time()))
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
