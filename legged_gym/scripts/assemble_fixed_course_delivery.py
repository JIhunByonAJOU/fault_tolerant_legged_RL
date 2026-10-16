"""Final fixed-design summaries; no alternative-mixture ranking or comparison."""
import csv
import json
import time
import zipfile
from pathlib import Path

from legged_gym.scripts.assemble_course_delivery import old_records, summarize, write_csv, digest, NAMES


def build(campaign):
    campaign=Path(campaign);out=campaign/'delivery';out.mkdir(exist_ok=True)
    protocol=json.loads((campaign/'protocol.json').read_text())
    sources,old=old_records(out,protocol)
    design=json.loads((campaign/'fixed_design.json').read_text())
    data=[]
    if (campaign/'derived_records.json').exists():
        originals=json.loads((campaign/'derived_records.json').read_text())
        for r in originals:
            data.append(dict(r,loss=r['condition']['loss'],joint=r['condition']['joint'],family=r['profile']['family'],profile_id=r['profile']['id'],column=r['terrain_column_id']))
    for job in ([] if data else json.loads((campaign/'confirmation_jobs.json').read_text())):
        folder=campaign/'jobs'/job['id'];marker=json.loads((folder/'COMPLETE.json').read_text())
        assert digest(folder/'results.json')==marker['result_sha256']
        for r in json.loads((folder/'results.json').read_text()):
            data.append(dict(r,loss=r['condition']['loss'],joint=r['condition']['joint'],family=r['profile']['family'],
                             profile_id=r['profile']['id'],column=r['terrain_column_id']))
    assert len(data)==26352
    summary=summarize(data,['policy','loss'])
    write_csv(out/'by_loss_with_pooled_rmse.csv',summary)
    write_csv(out/'by_family_loss_with_pooled_rmse.csv',summarize(data,['policy','family','loss']))
    write_csv(out/'by_profile_loss_with_pooled_rmse.csv',summarize(data,['policy','profile_id','loss']))
    write_csv(out/'by_seed_loss_with_pooled_rmse.csv',summarize(data,['policy','seed','loss']))
    differences=[]
    for d in sorted({r['loss'] for r in summary}):
        n=next(r for r in summary if r['policy']=='Normal' and r['loss']==d)
        s=next(r for r in summary if r['policy']=='Student' and r['loss']==d)
        a=dict(loss=d,normal_survival=n['survival'],student_survival=s['survival'],survival_gain_pp=100*(s['survival']-n['survival']))
        for m in NAMES:
            a['normal_'+m]=n[m+'_trial_mean'];a['student_'+m]=s[m+'_trial_mean']
            a[m+'_gain']=None if a['normal_'+m] is None or a['student_'+m] is None else a['normal_'+m]-a['student_'+m]
        differences.append(a)
    write_csv(out/'policy_differences_by_loss.csv',differences)
    contrast=[]
    for r in summary:
        prev=next(x for x in old if x['policy']==r['policy'] and x['loss']==r['loss'])
        a=dict(policy=r['policy'],loss=r['loss'],old_episodes=prev['episodes'],new_episodes=r['episodes'],
               old_survival=prev['survival'],new_survival=r['survival'])
        for m in NAMES[:3]:
            a['old_'+m+'_pooled']=prev[m+'_pooled'];a['new_'+m+'_pooled']=r[m+'_pooled']
        a['new_cross_track_rmse_trial_mean']=r['cross_track_rmse_trial_mean']
        contrast.append(a)
    write_csv(out/'manuscript_vs_new_by_loss.csv',contrast)
    lines=['# 고정 과제군 비율 2:1:1:2 평가 결과','',
        '사용자 결정에 따라 전진/대각/횡이동/사인파 비율을2:1:1:2로 고정했다. 다른 비율과의 우열 비교나 최적 비율 탐색을 최종 결과에 포함하지 않는다.','',
        '## 실행 조건','',
        '- 고정 비율 실제 수집:6주행 슬롯 ×61고장조건 ×2정책 ×3평가시드 =2,196 정책·시드 런.',
        '- 각 런12개 원래 지형 열. 총26,352 로봇 시행. 평가 시드'+str(design.get('evaluation_seeds',[3101,3102,3103]))+'.',
        '- 자료 출처: '+design.get('data_origin','고정 구성의 실제 수집')+'.',
        '- 전진2슬롯, 대각1슬롯, 횡1슬롯, 사인파2슬롯. 대각·횡의 좌우는 고장 조건과 시드 사이에서 번갈아 배정해 세 시드를 합치면 모두 포함한다.',
        '- 사인파는A2/λ16과A1.5/λ16. 기존12열 지형군1/1/1/4/3/2 유지. 결과에 따른 지형 배정 변경을 사용하지 않는다.',
        '- Normal22500의 특권 인코더 경로와 Student77500 학생 경로. 추가 학습 없이 동일 상태·물리·지형·고장 조건에서 GPU PhysX로 평가.',
        '- 생존율:낙상0, 완주 또는 행정 상한까지 비낙상1. 정체·저속으로 조기 종료하지 않았다.',
        '- 오차:고장 후 낙상/완주/상한 전 유효 표본의 시행별 RMSE 평균. 유효 구간0이면NA, 낙상 이후를0으로 채우지 않는다.','',
        '## Normal과 Student의 성능','',
        '|토크 저하율|Normal 생존%|Student 생존%|차이(%p)|vx RMSE 개선(m/s)|vy 개선(m/s)|wz 개선(rad/s)|횡오차 개선(m)|',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in differences:
        values=[r['loss'],100*r['normal_survival'],100*r['student_survival'],r['survival_gain_pp']]+[r[m+'_gain'] for m in NAMES]
        lines.append('|'+ '|'.join('NA' if x is None else '%.5f'%x for x in values)+'|')
    lines+=['','오차 개선은Normal−Student, 생존율 차이는Student−Normal이다. 원래 절대값은CSV에 모두 보존했다. 건강 조건d=0은 고장 조건과 구분해 해석한다.','',
        '## 과제·지형·관절별 결과','',
        'analysis/confirmation/all_trials.csv에 모든 시행, by_every_condition.csv에 과제·원래 열·관절·저하율별 결과, paired_differences.csv에 대응 시행 차이를 저장했다. delivery/by_profile_loss_with_pooled_rmse.csv에서 좌우·두 사인파를 구분한다. 차이가 큰 조건은largest_differences.csv와heatmap에서 확인한다.','',
        '## 현재 원고와 비교','',
        '기존 시드301–303의6개 원자료 파일을 다시 집계해 원고의 생존율·통합RMSE와1e-8 이내 일치함을 확인했다. 동일 checkpoint hash를 사용한다.',
        '원고는20초 직진·random mixed·onset2–10초·생존 표본 통합RMSE다. 새 데이터는다방향/곡선·점진적5행 코스·공간 경로 제어기·onset2초·완주/낙상/긴 상한까지 관찰한다. 수치 변화는 정책 자체의 재학습 개선량이 아니다.',
        '원고와 같은 통합RMSE도 별도로 계산했다. 기존에 경로 횡오차는 정의·기록되지 않아 비교 값은NA다.','',
        '|정책|저하율|기존 생존%|새 생존%|기존 vx pooled|새 vx pooled|',
        '|---|---:|---:|---:|---:|---:|']
    for r in contrast:
        if r['loss'] in [0.,.8,1.]:
            lines.append('|%s|%.1f|%.4f|%.4f|%.5f|%.5f|'%(r['policy'],r['loss'],100*r['old_survival'],100*r['new_survival'],r['old_command_vx_rmse_pooled'],r['new_command_vx_rmse_pooled']))
    lines+=['','## 집계 해석','',
        '생존 구간이 짧으면 오차가 작게 나타날 수 있어 유효 시행 수와 관찰 길이를 함께 본다. 평가 시드3개는 독립 학습3회가 아니다. 높은 생존율과 큰 명령 오차가 함께 나오면 비낙상과 요구 이동 수행을 구분한다.',
        '이전 전수 수집과 중단된3:1:1:3 수집은 원래 폴더에 보존하되 이 고정 평가와 합쳐 하나의 동일 분포 데이터로 제시하지 않는다.','']
    if design.get('data_origin')=='기존 첫 전수 수집에서 조건 식별자로 추출; 추가 시뮬레이션 없음':
        lines=[line.replace('최종 실제 수집','최종 조건별 추출').replace('고정 비율 실제 수집','기존 수집에서 고정 조건별 추출').replace('이 고정 평가와 합쳐 하나의 동일 분포 데이터로 제시하지 않는다.','최종 추출에 사용한 첫 전수 자료와 중복 합산하지 않는다.') for line in lines]
        lines.insert(2,'이 결과는 완료된 첫 전수 수집의 부분집합이다. 새로운 독립 시드 재평가나 별도 수집으로 부르지 않는다. 선택은 조건 식별자와 고정 좌우 교대 규칙만 사용하며 성능 결과로 시행을 선택하지 않는다.')
    lines=[line.replace('analysis/confirmation/','analysis/'+design.get('analysis_subdir','confirmation')+'/') for line in lines]
    (out/'REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,4))
    for policy in ['Normal','Student']:
        rs=sorted([r for r in summary if r['policy']==policy],key=lambda r:r['loss'])
        ax.plot([r['loss'] for r in rs],[100*r['survival'] for r in rs],marker='o',label=policy)
    ax.set(xlabel='Torque degradation',ylabel='Non-fall outcomes (%)',ylim=(-2,102),title='Fixed task-family ratio 2:1:1:2')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'survival_by_loss.png',dpi=180);plt.close(fig)
    manifest=dict(status='COMPLETE',task_ratio=[2,1,1,2],terrain_counts=[1,1,1,4,3,2],
        old_sources=sources,files={p.name:digest(p) for p in out.iterdir() if p.is_file()},completed_at=time.time())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(out/'evaluation_delivery.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p.suffix!='.zip':z.write(p,'delivery/'+p.name)
        for p in (campaign/'analysis'/design.get('analysis_subdir','confirmation')).iterdir():
            if p.is_file():z.write(p,'analysis/'+design.get('analysis_subdir','confirmation')+'/'+p.name)
        for n in ['protocol.json','fixed_design.json','source_manifest.json','FINAL.json','derivation_manifest.json']:
            if (campaign/n).exists():z.write(campaign/n,n)
    (campaign/'DELIVERY_COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',report=str(out/'REPORT.md'),archive_sha256=digest(out/'evaluation_delivery.zip'),completed_at=time.time()),indent=2)+'\n')
