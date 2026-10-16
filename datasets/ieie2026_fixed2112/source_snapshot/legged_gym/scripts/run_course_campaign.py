"""Durable evaluation -> analysis -> bounded-mixture confirmation pipeline."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from legged_gym.evaluation.course_campaign import ROOT, atomic_json, make_jobs, sha
from legged_gym.scripts.analyze_course_campaign import analyze


SOURCES=[
    'legged_gym/evaluation/course_campaign.py','legged_gym/evaluation/course_controller.py',
    'legged_gym/scripts/collect_course_batch.py','legged_gym/scripts/analyze_course_campaign.py',
    'legged_gym/scripts/run_course_campaign.py','legged_gym/envs/base/legged_robot.py',
    'legged_gym/envs/a1_official_wim_teacher/a1_official_wim_joint.py',
    'legged_gym/envs/a1_official_wim_teacher/a1_official_wim_teacher.py',
    'legged_gym/learning/official_wim_teacher_actor_critic.py',
    'legged_gym/learning/joint_teacher_student_actor_critic.py','legged_gym/utils/terrain.py',
    'resources/robots/a1/urdf/a1.urdf',
]


def run(campaign):
    campaign=Path(campaign).resolve(); p=json.loads((campaign/'protocol.json').read_text())
    preflight=json.loads((campaign/'PREFLIGHT_PASS.json').read_text())
    if not preflight['large_pair_hashes_equal'] or not all(preflight[name]['binary_survival_and_trace_pass'] for name in ['Normal','Student']):
        raise RuntimeError('Preflight checks have not passed')
    freeze=campaign/'source_manifest.json'
    if not freeze.exists():
        manifest={s:sha(ROOT/s) for s in SOURCES}
        for s in SOURCES:
            target=campaign/'source_snapshot'/s; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/s,target)
        atomic_json(freeze,dict(files=manifest,protocol_sha256=sha(campaign/'protocol.json')))
    source=json.loads(freeze.read_text())
    state=dict(status='RUNNING',manager_pid=os.getpid(),started_at=time.time(),stage='exploration',completed_jobs=0)
    def save(**kwargs):
        state.update(kwargs); state['updated_at']=time.time(); atomic_json(campaign/'state.json',state)
    save()
    try:
        for stage in ['exploration','confirmation']:
            selection=json.loads((campaign/'selection.json').read_text()) if stage=='confirmation' else None
            repeats=selection['family_units'] if selection else None
            columns=selection['terrain_counts'] if selection else None
            jobs=make_jobs(p,stage,repeats,columns)
            atomic_json(campaign/(stage+'_jobs.json'),jobs)
            save(stage=stage,total_jobs=len(jobs),completed_jobs=0,child_pid=None)
            (campaign/'logs').mkdir(exist_ok=True)
            (campaign/'job_specs').mkdir(exist_ok=True)
            for index,job in enumerate(jobs):
                if stage=='confirmation':
                    frozen=json.loads((campaign/'confirmation_selection_frozen.json').read_text())
                    if sha(campaign/'selection.json')!=frozen['sha256']:
                        raise RuntimeError('Selected mixture changed during confirmation')
                if sha(campaign/'protocol.json')!=source['protocol_sha256']:
                    raise RuntimeError('Frozen protocol changed')
                for name,expected in source['files'].items():
                    if sha(ROOT/name)!=expected: raise RuntimeError('Frozen source changed: '+name)
                spec_path=campaign/'job_specs'/(job['id']+'.json')
                atomic_json(spec_path,job)
                folder=campaign/'jobs'/job['id']
                if not (folder/'COMPLETE.json').exists():
                    if folder.exists():
                        raise RuntimeError('Incomplete job exists; preserve and inspect: '+job['id'])
                    cmd=[sys.executable,'-u','-m','legged_gym.scripts.collect_course_batch',
                         '--campaign',str(campaign),'--job',str(spec_path),
                         '--headless','--sim_device','cuda:0','--rl_device','cuda:0']
                    with (campaign/'logs'/(job['id']+'.log')).open('w') as log:
                        child=subprocess.Popen(cmd,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT)
                        save(current_job=job['id'],current_num_envs=job['num_envs'],child_pid=child.pid)
                        while child.poll() is None:
                            progress=folder/'progress.json'
                            if progress.exists():
                                try: save(current_progress=json.loads(progress.read_text()))
                                except json.JSONDecodeError: pass
                            time.sleep(10)
                        if child.returncode:
                            raise RuntimeError('Collector failed (%s): %s'%(child.returncode,job['id']))
                complete=json.loads((folder/'COMPLETE.json').read_text())
                if complete['pilot'] or sha(folder/'results.json')!=complete['result_sha256']:
                    raise RuntimeError('Invalid completion marker')
                metadata=json.loads((folder/'metadata.json').read_text())
                if metadata['protocol_sha256']!=source['protocol_sha256']:
                    raise RuntimeError('Job used a different protocol')
                manifest=json.loads((folder/'trace_manifest.json').read_text())
                result_rows=json.loads((folder/'results.json').read_text())
                expected_rows=sum(round(r['end_time_s']/p['dt']) for r in result_rows)
                if manifest['valid_rows']!=expected_rows:
                    raise RuntimeError('Trace coverage is not complete')
                for chunk in manifest['chunks']:
                    if sha(folder/chunk['path'])!=chunk['sha256']: raise RuntimeError('Trace hash mismatch')
                if job['policy']=='Student':
                    other=campaign/'jobs'/(job['pair_job']+'_Normal')/'metadata.json'
                    normal=json.loads(other.read_text())
                    if normal['initial_hashes']!=metadata['initial_hashes']:
                        atomic_json(folder/'PAIR_MISMATCH.json',dict(normal=normal['initial_hashes'],student=metadata['initial_hashes']))
                        raise RuntimeError('Applied initial/physics pairing mismatch')
                    atomic_json(folder/'PAIR_PASS.json',dict(paired=True,normal_job=job['pair_job']+'_Normal'))
                save(completed_jobs=index+1,child_pid=None)
            save(status='ANALYZING',stage=stage)
            analyze(campaign,stage,selection)
            if stage=='exploration':
                selected=json.loads((campaign/'selection.json').read_text())
                atomic_json(campaign/'confirmation_selection_frozen.json',dict(
                    sha256=sha(campaign/'selection.json'),selection=selected,created_at=time.time()))
                save(status='RUNNING',selected_task_ratio=selected['family_units'],selected_terrain_ratio=selected['terrain_counts'])
        final=dict(status='COMPLETE',campaign=str(campaign),completed_at=time.time(),
                   exploration='analysis/exploration/report.md',confirmation='analysis/confirmation/report.md',
                   selection=json.loads((campaign/'selection.json').read_text()),
                   exploration_coverage=json.loads((campaign/'analysis/exploration/coverage.json').read_text()),
                   confirmation_coverage=json.loads((campaign/'analysis/confirmation/coverage.json').read_text()))
        atomic_json(campaign/'FINAL.json',final)
        lines=['# 평가 및 비율 조정 재평가 완료','',
               '생존율: 완주 또는 시간 상한까지 비낙상인 시행의 비율. 정체/저속으로 조기 종료하지 않았다.','',
               '- 첫 수집: [분석](analysis/exploration/report.md)',
               '- 조정 비율·선정 근거: [selection.json](selection.json)',
               '- 새 시드 재평가: [분석](analysis/confirmation/report.md)',
               '- 각 단계 all_trials.csv, by_every_condition.csv 및 paired_differences.csv에 전체 결과를 보존했다.','',
               '선정 비율은 탐색 데이터에서 선택했으며 재평가 데이터로 다시 선택하지 않았다. 재평가 결과가 개선되지 않아도 그대로 기록했다.','']
        (campaign/'REPORT.md').write_text('\n'.join(lines))
        save(status='COMPLETE',stage='complete',child_pid=None,report=str(campaign/'REPORT.md'))
    except BaseException as error:
        save(status='FAILED',error=str(error),traceback=traceback.format_exc())
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--campaign',type=Path,required=True)
    args=parser.parse_args(); run(args.campaign)
