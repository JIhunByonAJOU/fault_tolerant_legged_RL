#!/usr/bin/env python3
"""Portable NumPy/Matplotlib analysis. Runs on Windows without Isaac Gym/PyTorch."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

BASE=Path(__file__).resolve().parent
JOINTS=[f'{leg} {part}' for leg in ['FL','FR','RL','RR'] for part in ['hip','thigh','calf']]
RATES=[.2,.4,.6,.8,1.]
FIELDS=['values','flags','env_id','step']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def payload(arrays):
    h=hashlib.sha256()
    for k in FIELDS:
        a=arrays[k];h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def mean(values):
    v=[x for x in values if x is not None]
    return float(np.mean(v)) if v else None


def vxy(r):
    n=r['valid_post_samples']
    return float(np.sqrt((r['error_sum_sq'][0]+r['error_sum_sq'][1])/n)) if n else None


def write(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def verify(rows,raw=False):
    m=json.loads((BASE/'manifest.json').read_text(encoding='utf-8'))
    for i,(name,info) in enumerate(m['files'].items(),1):
        p=BASE/name
        assert p.exists(),name
        assert p.stat().st_size==info['bytes'],name
        assert sha(p)==info['sha256'],name
        if i%100==0:print('File checksums:',i,'/',len(m['files']),flush=True)
    assert len(rows)==m['robot_episodes']==26352
    paired=collections.defaultdict(set)
    for r in rows:paired[r['paired_trial_id']].add(r['policy'])
    assert len(paired)==13176 and all(v=={'Normal','Student'} for v in paired.values())
    total=0
    if raw:
        for job in m['jobs']:
            folder=BASE/'raw'/job['job'];trace=json.loads((folder/'trace_manifest.json').read_text(encoding='utf-8'))
            source=json.loads((folder/'source_complete.json').read_text(encoding='utf-8'))
            count=np.zeros(source['episodes'],dtype=np.int64);post=np.zeros_like(count)
            for part in trace['chunks']:
                with np.load(folder/part['path'],allow_pickle=False) as z:
                    arrays={k:z[k] for k in FIELDS}
                assert payload(arrays)==part['payload_sha256'],part['path']
                assert len(arrays['env_id'])==part['rows']
                count+=np.bincount(arrays['env_id'],minlength=len(count))
                post+=np.bincount(arrays['env_id'][(arrays['flags']&2)>0],minlength=len(count))
            selected=[r for r in rows if r['source_job']==job['job']]
            for r in selected:
                assert count[r['env_id']]==round(r['end_time_s']/.02)
                assert post[r['env_id']]==r['valid_post_samples']
            total+=int(count.sum());print('Raw row check:',job['job'],int(count.sum()),flush=True)
        assert total==m['raw_rows']
    print('VERIFIED: 26,352 robot episodes; 13,176 policy pairs;',m['raw_rows'],'raw 50 Hz rows',flush=True)


def summaries(rows,out):
    result=[]
    for j in range(12):
        for d in RATES:
            for pol in ['Normal','Student']:
                rs=[r for r in rows if r['policy']==pol and r['condition']['joint']==j and r['condition']['loss']==d]
                alive=[r for r in rs if r['survived']]
                stalled=[r for r in alive if r['stall_seconds']>0]
                capstalled=[r for r in alive if r['termination']=='administrative_cap' and r['stall_seconds']>0]
                result.append(dict(joint_index=j,joint=JOINTS[j],degradation=d,policy=pol,episodes=len(rs),
                    alive_n=len(alive),survival_percent=100*len(alive)/len(rs),vxy_rmse=mean([vxy(r) for r in rs]),
                    stall_experience_alive_n=len(stalled),stall_among_survivors_percent=100*len(stalled)/len(alive) if alive else None,
                    capped_stall_alive_n=len(capstalled),capped_stall_among_survivors_percent=100*len(capstalled)/len(alive) if alive else None,
                    goal_n=sum(r['termination']=='goal' for r in rs),valid_rmse_n=sum(r['valid_post_samples']>0 for r in rs)))
    write(out/'joint_vxy_and_stagnation.csv',result)
    for r in result:
        if r['joint_index']==8 and r['degradation']==1.:print('RL calf 100%:',r)
    return result


def plots(summary,out,font=None):
    from paper_figures import main as render_paper_figures
    args=['--output',str(out)]
    if font:args+=['--font',str(font)]
    render_paper_figures(args)


def inspect(rows,trial,out):
    r=next((r for r in rows if r['trial_id']==trial),None)
    if r is None:raise ValueError('Unknown trial ID; see records.json or analysis/all_trials.csv')
    folder=BASE/r['source_folder'];m=json.loads((folder/'trace_manifest.json').read_text(encoding='utf-8'))
    parts=[];steps=[];flags=[]
    for info in m['chunks']:
        with np.load(folder/info['path'],allow_pickle=False) as z:
            keep=z['env_id']==r['env_id']
            if keep.any():parts.append(z['values'][keep]);steps.append(z['step'][keep]);flags.append(z['flags'][keep])
    v=np.concatenate(parts);s=np.concatenate(steps);f=np.concatenate(flags)
    assert len(v)==round(r['end_time_s']/.02)
    with (out/(trial+'.csv')).open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream);writer.writerow(['step','time_s','time_after_fault_s','flags']+m['fields'])
        writer.writerows([int(step),float(step*.02),float(step*.02-2),int(flag)]+list(map(float,row)) for step,flag,row in zip(s,f,v))
    print('Full 50 Hz trial CSV:',out/(trial+'.csv'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');p.add_argument('--verify-raw',action='store_true')
    p.add_argument('--verify-only',action='store_true');p.add_argument('--trial');p.add_argument('--output',type=Path,default=BASE/'generated')
    p.add_argument('--font',type=Path,help='Optional local font file for paper figures')
    a=p.parse_args();rows=json.loads((BASE/'records.json').read_text(encoding='utf-8'));a.output.mkdir(parents=True,exist_ok=True)
    if a.verify or a.verify_raw:verify(rows,a.verify_raw)
    if a.verify_only:return
    if a.trial:inspect(rows,a.trial,a.output)
    else:plots(summaries(rows,a.output),a.output,a.font)


if __name__=='__main__':main()
