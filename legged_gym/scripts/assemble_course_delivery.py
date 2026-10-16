"""Evidence-only delivery sidecar; never changes frozen collection or selection."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
import zipfile

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'logs/evaluations/ieie2026_independent_confirmation_v1'
OLD_TABLE=ROOT/'docs/submission/ieie2026/results/independent_confirmation/metrics_by_severity.csv'
NAMES=['command_vx_rmse','command_vy_rmse','command_wz_rmse','cross_track_rmse']
FAMILIES=['forward','diagonal','lateral','sine']
CLASSES=['smooth_low','smooth_high','rough_slope','stairs_low','stairs_high','obstacles']
BASE=[1,1,1,4,3,2]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


def write_csv(path,rows):
    if not rows: return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def avg(xs):
    xs=[x for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else None


def summarize(rows,keys):
    groups={}
    for r in rows: groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    out=[]
    for key,rs in sorted(groups.items(),key=lambda x:str(x[0])):
        count=sum(r['valid_post_samples'] for r in rs)
        a=dict(zip(keys,key),episodes=len(rs),survival=avg([r['survived'] for r in rs]),
               mean_valid_post_s=avg([r['valid_post_samples']*.02 for r in rs]),valid_samples=count)
        for i,m in enumerate(NAMES):
            a[m+'_trial_mean']=avg([r[m] for r in rs])
            a[m+'_pooled']=math.sqrt(sum(r['error_sum_sq'][i] for r in rs)/count) if count else None
            a[m+'_valid_n']=sum(r[m] is not None for r in rs)
        out.append(a)
    return out


def old_records(out,protocol):
    rows=[]; source=[]
    for policy,model in [('Normal','tb22500'),('Student','jt77500')]:
        for seed in [301,302,303]:
            path=OLD/('%s_seed%s.jsonl'%(model,seed))
            with path.open() as f: records=[json.loads(line) for line in f if line.strip()]
            meta=records[0]; robots=[r for r in records if r.get('record_type')=='robot']
            if len(robots)!=1152 or meta['checkpoint_sha256']!=protocol['policies'][policy]['sha256']:
                raise ValueError('Old source/checkpoint does not match manuscript lineage')
            source.append(dict(path=str(path),sha256=digest(path),policy=policy,seed=seed,episodes=len(robots)))
            for r in robots:
                vals=[r['post_command_vx_rmse'],r['post_command_vy_rmse'],r['post_yaw_rate_rmse'],None]
                n=r['valid_post_steps']
                rows.append(dict(policy=policy,seed=seed,loss=r['degradation_rate'],joint=r['joint_index'],
                    survived=int(r['survived_post_failure_horizon']),valid_post_samples=n,
                    error_sum_sq=[v*v*n if v is not None else 0 for v in vals],**dict(zip(NAMES,vals))))
    result=summarize(rows,['policy','loss'])
    with OLD_TABLE.open() as f: table=list(csv.DictReader(f))
    for r in result:
        ref=next(x for x in table if x['model']==('tb22500' if r['policy']=='Normal' else 'jt77500') and float(x['degradation_rate'])==r['loss'])
        for key,old in [('survival','S20'),('command_vx_rmse_pooled','vx_alive_pooled_rmse'),
                        ('command_vy_rmse_pooled','vy_alive_pooled_rmse'),('command_wz_rmse_pooled','yaw_alive_pooled_rmse')]:
            if abs(r[key]-float(ref[old]))>1e-8: raise ValueError('Old raw summary differs from manuscript CSV: '+key)
    # CTE was not recorded/defined in this cohort. Never replace it by zero.
    for r in result: r['cross_track_rmse_pooled']=None
    write_csv(out/'manuscript_recomputed.csv',result)
    return source,result


def standardize(rows,units,counts):
    pairs={}
    for r in rows:
        if r['condition']['loss']<=0: continue
        pairs.setdefault(r['paired_trial_id'],{})[r['policy']]=r
    cells={}
    for pair in pairs.values():
        a,b=pair['Normal'],pair['Student']
        key=(a['profile']['family'],a['terrain_class'])
        diffs=[b['survived']-a['survived']]+[None if a[m] is None or b[m] is None else a[m]-b[m] for m in NAMES]
        cells.setdefault(key,[]).append(diffs)
    means={k:[avg([d[i] for d in ds]) for i in range(5)] for k,ds in cells.items()}
    answer=[]
    for i in range(5):
        value=0.
        for f,family in enumerate(FAMILIES):
            for t,terrain in enumerate(CLASSES):
                cell=means[(family,terrain)][i]
                if cell is None: value=None; break
                value+=units[f]/sum(units)*counts[t]/sum(counts)*cell
            if value is None: break
        answer.append(value)
    return dict(zip(['survival_gain']+[m+'_gain' for m in NAMES],answer))


def build(campaign):
    out=campaign/'delivery'; out.mkdir(exist_ok=True)
    protocol=json.loads((campaign/'protocol.json').read_text())
    sources,old=old_records(out,protocol)
    selection=json.loads((campaign/'selection.json').read_text())
    all_summaries=[]; standardized=[]; seed_gains=[]
    for stage in ['exploration','confirmation']:
        coverage=json.loads((campaign/'analysis'/stage/'coverage.json').read_text())
        if not coverage['complete'] or coverage['robot_episodes']!=35136: raise ValueError('Incomplete coverage')
        jobs=json.loads((campaign/(stage+'_jobs.json')).read_text()); rows=[]
        for job in jobs:
            folder=campaign/'jobs'/job['id']; marker=json.loads((folder/'COMPLETE.json').read_text())
            if digest(folder/'results.json')!=marker['result_sha256']: raise ValueError('Changed result')
            rows+=json.loads((folder/'results.json').read_text())
        # Keep healthy separately: old protocol repeats healthy once for every named joint,
        # new protocol treats healthy as one condition; aggregate comparisons never hide this.
        summaries=summarize([dict(r,loss=r['condition']['loss'],profile_id=r['profile']['id'],
                            family=r['profile']['family']) for r in rows],['policy','loss'])
        for r in summaries: r['stage']=stage
        write_csv(out/(stage+'_by_loss_with_pooled_rmse.csv'),summaries)
        all_summaries+=summaries
        for name,units,counts in [('original',[2]*4,BASE),('selected',selection['family_units'],selection['terrain_counts'])]:
            standardized.append(dict(stage=stage,target_distribution=name,**standardize(rows,units,counts)))
            for seed in sorted({r['seed'] for r in rows}):
                seed_gains.append(dict(stage=stage,seed=seed,target_distribution=name,
                    **standardize([r for r in rows if r['seed']==seed],units,counts)))
        family=summarize([dict(r,family=r['profile']['family'],loss=r['condition']['loss']) for r in rows],['policy','family','loss'])
        write_csv(out/(stage+'_family_by_loss_with_pooled_rmse.csv'),family)
    write_csv(out/'mixture_comparison_same_target.csv',standardized)
    write_csv(out/'mixture_comparison_by_evaluation_seed.csv',seed_gains)
    contrast=[]
    for new in all_summaries:
        previous=next(r for r in old if r['policy']==new['policy'] and r['loss']==new['loss'])
        contrast.append(dict(stage=new['stage'],policy=new['policy'],loss=new['loss'],
            old_episodes=previous['episodes'],new_episodes=new['episodes'],old_survival=previous['survival'],
            new_survival=new['survival'],survival_difference_pp=100*(new['survival']-previous['survival']),
            old_vx_pooled=previous['command_vx_rmse_pooled'],new_vx_pooled=new['command_vx_rmse_pooled'],
            old_vy_pooled=previous['command_vy_rmse_pooled'],new_vy_pooled=new['command_vy_rmse_pooled'],
            old_wz_pooled=previous['command_wz_rmse_pooled'],new_wz_pooled=new['command_wz_rmse_pooled'],
            new_cross_track_trial_mean=new['cross_track_rmse_trial_mean'],
            interpretation='Changed task/terrain/controller/exposure; descriptive difference, not policy improvement'))
    write_csv(out/'manuscript_vs_new_by_loss.csv',contrast)
    lines=['# 8주행 전체 평가·비율 확인·원고 비교','',
        '두 단계 모두 전체 격자 검증을 통과했다. 생존율은 낙상 여부의 평균이다. 정체·저속으로 조기 종료하지 않았다.','',
        '## 실행과 표본','',
        '각 단계 2,928 정책·시드 런, 35,136 로봇 시행. 탐색 시드1101–1103, 확인 시드2101–2103. 독립 학습 반복이 아니다.','',
        '## 비율과 새 시드 결과','',
        '과제군 순서: 전진/대각/횡/사인파. 선정 '+str(selection['family_units'])+' / 8.',
        '지형군 순서: smooth-low/smooth-high/rough/stairs-low/stairs-high/obstacles. 선정 '+str(selection['terrain_counts'])+' / 12.','',
        '확인 데이터로 비율을 다시 선택하지 않았다. 아래 gain은 생존 Student−Normal, 오차 Normal−Student이며 양수가 Student에 유리하다.',
        '원래/선정 분포 모두 같은 family×terrain 셀 평균을 해당 목표 비율로 표준화한다. 새 시드에서 관찰된 차이를 선택에 쓴 탐색 차이와 구분한다.','',
        '|단계|목표 비율|생존 차이(%p)|vx RMSE 차이(m/s)|vy 차이(m/s)|wz 차이(rad/s)|횡오차 차이(m)|',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in standardized:
        nums=[r['survival_gain']*100]+[r[m+'_gain'] for m in NAMES]
        lines.append('|'+r['stage']+'|'+r['target_distribution']+'|'+'|'.join('NA' if x is None else '%.5f'%x for x in nums)+'|')
    lines+=['','## 기존 원고 대비','',
        '원자료 재집계가 원고의 기존 생존율과 통합 RMSE CSV에 1e-8 이내로 일치함을 확인했다. 동일 Normal22500/Student77500 checkpoint hash를 확인했다.',
        '원고: 직진 하나, random mixed, onset2–10초, 고장 후20초, 시드301–303×16회×12관절×6손실률.',
        '새 평가: 8주행,5행 점진적 mixed, 고정 onset2초, 공간 경로 제어기, 완주/낙상/긴 행정 상한까지 관찰. 정상 조건을12관절마다 복제하지 않는다.',
        '관찰 시간과 환경·명령·집계가 달라 새 수치의 증감을 정책 개선량으로 해석할 수 없다. 원고는 생존 표본 통합 RMSE, 새 주 지표는 시행별 RMSE 평균이다. 비교 CSV에 통합 RMSE를 별도로 보존했다.',
        '기존 횡오차는 정의·기록되지 않아 NA다. 세계 횡변위나 ADE로 대체하지 않는다. 기존 정지 지표(몸체속도<0.1)는 새30초/0.5m 정체 flag와 다르다.','',
        '|단계|정책|d|기존 생존%|새 생존%|기존 vx pooled|새 vx pooled|',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in contrast:
        if r['loss'] not in [0.,.8,1.]: continue
        lines.append('|%s|%s|%.1f|%.3f|%.3f|%.5f|%.5f|'%(r['stage'],r['policy'],r['loss'],r['old_survival']*100,r['new_survival']*100,r['old_vx_pooled'],r['new_vx_pooled']))
    lines+=['','## 原자료와 한계','',
        '각 단계 analysis/all_trials.csv에 모든 시행을 기록하고 by_every_condition.csv에 원래 열·관절·열화·과제·정책별 요약을 보존했다. paired_differences.csv는 두 정책의 동일 초기·물리 시행 차이다.',
        'mean_valid_post_s 및 metric_valid_n을 반드시 함께 본다. 일찍 낙상한 정책의 짧은 구간 오차가 작을 수 있다. 세 평가 시드의 차이는 별도 CSV에 남기며 독립 학습 분산이나 보편적 신뢰구간으로 해석하지 않는다.',
        '새 시드에서 선택 비율의 이점이 유지되지 않거나 역전되어도 그 결과를 그대로 보고한다. 기준 분포 결과를 숨기지 않는다.','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,stage in zip(axes,['exploration','confirmation']):
        for policy in ['Normal','Student']:
            records=sorted([r for r in all_summaries if r['stage']==stage and r['policy']==policy],key=lambda r:r['loss'])
            ax.plot([r['loss'] for r in records],[100*r['survival'] for r in records],marker='o',label=policy)
        ax.set(xlabel='Torque loss',ylabel='Non-fall outcomes (%)',ylim=(-2,102),title=stage); ax.legend(); ax.grid(alpha=.25)
    fig.suptitle('Full-course binary survival; different mixtures across stages')
    fig.tight_layout(); fig.savefig(out/'survival_by_loss.png',dpi=180); plt.close(fig)
    snapshot=out/'manuscript_body_snapshot.tex'
    snapshot.write_bytes((ROOT/'docs/submission/ieie2026/private/latex/body.tex').read_bytes())
    manifest=dict(status='COMPLETE',campaign=str(campaign),created_at=time.time(),old_sources=sources,
                  old_table=str(OLD_TABLE),old_table_sha256=digest(OLD_TABLE),manuscript_snapshot_sha256=digest(snapshot),
                  files={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.suffix in ['.csv','.md','.png','.tex']})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(out/'evaluation_delivery.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.name!='evaluation_delivery.zip' and p.is_file(): z.write(p,'delivery/'+p.name)
        for stage in ['exploration','confirmation']:
            for p in (campaign/'analysis'/stage).iterdir():
                if p.is_file(): z.write(p,'analysis/'+stage+'/'+p.name)
        for name in ['protocol.json','selection.json','confirmation_selection_frozen.json','FINAL.json','source_manifest.json','PREFLIGHT_PASS.json']:
            z.write(campaign/name,name)
    (campaign/'DELIVERY_COMPLETE.json').write_text(json.dumps(dict(report=str(out/'REPORT.md'),archive=str(out/'evaluation_delivery.zip'),
           archive_sha256=digest(out/'evaluation_delivery.zip'),completed_at=time.time()),indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--campaign',type=Path,required=True); parser.add_argument('--wait',action='store_true')
    args=parser.parse_args(); campaign=args.campaign.resolve()
    try:
        while not (campaign/'FINAL.json').exists():
            state=json.loads((campaign/'state.json').read_text())
            if state['status']=='FAILED': raise RuntimeError('Collection failed; no completed scientific delivery')
            if not args.wait: raise RuntimeError('Evaluation not finished')
            os.kill(state['manager_pid'],0)
            time.sleep(20)
        build(campaign)
    except BaseException:
        (campaign/'DELIVERY_FAILED.json').write_text(json.dumps(dict(error=traceback.format_exc(),time=time.time()),indent=2))
        raise


if __name__=='__main__': main()
