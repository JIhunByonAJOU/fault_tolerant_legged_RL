"""User-fixed 2:1:1:2 collection, paired auditing, analysis, manuscript comparison."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from legged_gym.evaluation.course_campaign import ROOT, BASE_COUNTS, make_jobs, atomic_json, sha
from legged_gym.scripts.run_course_campaign import SOURCES
from legged_gym.scripts.analyze_course_campaign import analyze
from legged_gym.scripts.assemble_fixed_course_delivery import build


def run(campaign,previous):
    campaign=Path(campaign).resolve();previous=Path(previous).resolve();campaign.mkdir(exist_ok=True)
    p=json.loads((previous/'protocol.json').read_text())
    p.pop('selection',None);p['protocol_id']='ieie_fixed2112_course_v1'
    p['exploration_seeds']=[];p['confirmation_seeds']=[3101,3102,3103]
    fixed=dict(family_units=[2,1,1,2],terrain_counts=BASE_COUNTS,source='explicit user instruction',
               outcome_driven_selection=False,policy_seed_runs=2196,robot_episodes=26352,
               subprofiles='counterbalanced left/right across conditions and three seeds',
               previous_campaign=str(previous))
    p['fixed_design']=fixed
    atomic_json(campaign/'protocol.json',p);atomic_json(campaign/'fixed_design.json',fixed)
    names=SOURCES+['legged_gym/scripts/run_fixed_course_campaign.py','legged_gym/scripts/assemble_fixed_course_delivery.py',
                   'legged_gym/scripts/assemble_course_delivery.py']
    source=dict(files={s:sha(ROOT/s) for s in names},protocol_sha256=sha(campaign/'protocol.json'))
    for name in names:
        target=campaign/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    atomic_json(campaign/'source_manifest.json',source)
    state=dict(status='RUNNING',stage='fixed_confirmation',manager_pid=os.getpid(),started_at=time.time(),completed_jobs=0,task_ratio=fixed['family_units'],terrain_counts=BASE_COUNTS)
    def save(**kw):state.update(kw);state['updated_at']=time.time();atomic_json(campaign/'state.json',state)
    jobs=make_jobs(p,'confirmation',fixed['family_units'],BASE_COUNTS)
    assert sum(j['num_envs'] for j in jobs)==26352
    atomic_json(campaign/'confirmation_jobs.json',jobs)
    (campaign/'logs').mkdir(exist_ok=True);(campaign/'job_specs').mkdir(exist_ok=True)
    save(total_jobs=len(jobs))
    try:
        old=json.loads((previous/'state.json').read_text());pid=old.get('draining_child_pid')
        if pid:
            proc=Path('/proc')/str(pid)
            while True:
                try: running=b'collect_course_batch' in (proc/'cmdline').read_bytes()
                except FileNotFoundError: running=False
                if not running: break
                save(stage='waiting_for_previous_collector',draining_child_pid=pid);time.sleep(10)
        save(stage='fixed_confirmation',draining_child_pid=None)
        for index,job in enumerate(jobs):
            assert sha(campaign/'protocol.json')==source['protocol_sha256']
            for name,expected in source['files'].items():
                if sha(ROOT/name)!=expected:raise RuntimeError('Frozen source changed: '+name)
            spec=campaign/'job_specs'/(job['id']+'.json');atomic_json(spec,job)
            folder=campaign/'jobs'/job['id']
            if folder.exists():raise RuntimeError('Preserve existing job; refuse overwrite: '+str(folder))
            command=[sys.executable,'-u','-m','legged_gym.scripts.collect_course_batch','--campaign',str(campaign),
                     '--job',str(spec),'--headless','--sim_device','cuda:0','--rl_device','cuda:0']
            with (campaign/'logs'/(job['id']+'.log')).open('w') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                save(current_job=job['id'],child_pid=child.pid,current_num_envs=job['num_envs'],current_progress=None)
                while child.poll() is None:
                    progress=folder/'progress.json'
                    if progress.exists():
                        try:save(current_progress=json.loads(progress.read_text()))
                        except json.JSONDecodeError:pass
                    time.sleep(10)
                if child.returncode:raise RuntimeError('Collector failed: '+job['id'])
            marker=json.loads((folder/'COMPLETE.json').read_text());rows=json.loads((folder/'results.json').read_text())
            assert not marker['pilot'] and sha(folder/'results.json')==marker['result_sha256']
            meta=json.loads((folder/'metadata.json').read_text());assert meta['protocol_sha256']==source['protocol_sha256']
            manifest=json.loads((folder/'trace_manifest.json').read_text())
            assert manifest['valid_rows']==sum(round(r['end_time_s']/p['dt']) for r in rows)
            for chunk in manifest['chunks']:assert sha(folder/chunk['path'])==chunk['sha256']
            if job['policy']=='Student':
                other=campaign/'jobs'/(job['pair_job']+'_Normal')/'metadata.json'
                assert json.loads(other.read_text())['initial_hashes']==meta['initial_hashes']
                atomic_json(folder/'PAIR_PASS.json',dict(paired=True,normal_job=job['pair_job']+'_Normal'))
            save(completed_jobs=index+1,child_pid=None)
        save(status='ANALYZING')
        analyze(campaign,'confirmation',fixed)
        # The reusable analyzer's old narrative describes selection; replace that
        # narrative for this user-fixed design without changing its numeric audit.
        (campaign/'analysis/confirmation/report.md').write_text('# 고정 비율 2:1:1:2 평가\n\n사용자가 고정한 과제군 배정으로 수집했다. 비율 탐색·비율별 우열 비교는 수행하지 않았다. 전체 수치는 같은 폴더의CSV와coverage.json에 보존했다.\n')
        atomic_json(campaign/'FINAL.json',dict(status='COMPLETE',fixed_design=fixed,completed_at=time.time(),
            coverage=json.loads((campaign/'analysis/confirmation/coverage.json').read_text())))
        build(campaign)
        save(status='COMPLETE',stage='complete',report=str(campaign/'delivery/REPORT.md'))
    except BaseException as e:
        save(status='FAILED',error=str(e),traceback=traceback.format_exc());raise


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--campaign',type=Path,required=True);a.add_argument('--previous',type=Path,required=True)
    v=a.parse_args();run(v.campaign,v.previous)
