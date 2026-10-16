"""Joint/torque-degradation heatmaps and raw-trace survival/tracking diagnostics."""
import collections
import csv
import json
from pathlib import Path
import zipfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT=Path(__file__).resolve().parents[2]
C=ROOT/'logs/evaluation/ieie2026_reused2112_20261016'
OUT=C/'explanatory';OUT.mkdir(exist_ok=True)
R=json.loads((C/'derived_records.json').read_text())
METRICS=['survived','command_vx_rmse','command_vy_rmse','command_wz_rmse','cross_track_rmse']
JOINTS=[f'{leg} {part}' for leg in ['FL','FR','RL','RR'] for part in ['hip','thigh','calf']]
LOSSES=[.2,.4,.6,.8,1.]
LABELS=['Survival gain (pp)','vx RMSE reduction (m/s)','vy RMSE reduction (m/s)','Yaw-rate RMSE reduction (rad/s)','Cross-track RMSE reduction (m)']

def write(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def avg(rs,key):return float(np.mean([r[key] for r in rs]))

def summary(rs):
    n=len(rs);alive=[r for r in rs if r['survived']]
    return dict(episodes=n,survived_n=len(alive),goal_n=sum(r['termination']=='goal' for r in rs),
        alive_at_cap_n=sum(r['termination']=='administrative_cap' for r in rs),
        alive_with_stall_flag_n=sum(r['stall_seconds']>0 for r in alive),
        alive_at_cap_with_stall_n=sum(r['termination']=='administrative_cap' and r['stall_seconds']>0 for r in rs),
        mean_observed_post_s=avg(rs,'observed_post_s'),**{m:avg(rs,m) for m in METRICS})

byloss=[];joint=[];jointdiff=[]
for d in [0.]+LOSSES:
    for policy in ['Normal','Student']:
        rs=[r for r in R if r['policy']==policy and r['condition']['loss']==d]
        byloss.append(dict(policy=policy,loss=d,**summary(rs)))
for j in range(12):
    for d in LOSSES:
        samples={p:[r for r in R if r['policy']==p and r['condition']['joint']==j and r['condition']['loss']==d] for p in ['Normal','Student']}
        assert all(len(rs)==216 for rs in samples.values())
        means={p:summary(rs) for p,rs in samples.items()}
        for p,a in means.items():joint.append(dict(joint_index=j,joint=JOINTS[j],policy=p,loss=d,**a))
        diff=dict(joint_index=j,joint=JOINTS[j],loss=d,episodes_per_policy=216)
        for i,m in enumerate(METRICS):diff[m+'_gain']=(means['Student'][m]-means['Normal'][m])*(100 if i==0 else -1)
        jointdiff.append(diff)
write('survival_tracking_by_loss.csv',byloss)
write('joint_loss_absolute.csv',joint)
write('joint_loss_gains.csv',jointdiff)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
fig,axes=plt.subplots(2,3,figsize=(12.8,8.2),layout='constrained')
for ax,m,label in zip(axes.flat,METRICS,LABELS):
    vals=np.array([[next(r[m+'_gain'] for r in jointdiff if r['joint_index']==j and r['loss']==d) for d in LOSSES] for j in range(12)])
    bound=max(1e-8,float(np.abs(vals).max()));im=ax.imshow(vals,aspect='auto',cmap='RdBu',norm=Normalize(-bound,bound))
    ax.set(xticks=range(5),xticklabels=[20,40,60,80,100],yticks=range(12),yticklabels=JOINTS,xlabel='Torque degradation (%)',title=label)
    for y in range(12):
        for x in range(5):ax.text(x,y,('%.1f' if m=='survived' else '%.3f')%(0. if abs(vals[y,x])<(0.05 if m=='survived' else .0005) else vals[y,x]),ha='center',va='center',fontsize=6.5,color='white' if abs(vals[y,x])>.65*bound else 'black')
    fig.colorbar(im,ax=ax,shrink=.8)
axes.flat[-1].axis('off');axes.flat[-1].text(.02,.95,'Positive / blue: Student better\nNegative / red: Normal better\n\nEach cell: 216 trials per policy\nFixed task-family ratio 2:1:1:2\nOriginal 12 terrain columns\nThree evaluation seeds\n\nErrors: mean per-trial RMSE\non valid post-fault alive samples.\nHealthy condition shown separately.',va='top',transform=axes.flat[-1].transAxes)
fig.suptitle('Joint and torque-degradation effects: paired-condition policy comparison')
for ext in ['png','pdf']:fig.savefig(OUT/('joint_loss_five_metric_gains.'+ext),dpi=180)
plt.close(fig)
# Survival alone, with identical absolute color limits, as a simpler figure option.
fig,axes=plt.subplots(1,3,figsize=(12,5.5),layout='constrained')
for ax,pol in zip(axes[:2],['Normal','Student']):
    a=np.array([[100*next(r['survived'] for r in joint if r['joint_index']==j and r['loss']==d and r['policy']==pol) for d in LOSSES] for j in range(12)])
    im=ax.imshow(a,aspect='auto',cmap='viridis',vmin=0,vmax=100);fig.colorbar(im,ax=ax,label='Non-fall outcomes (%)');ax.set_title(pol)
vals=np.array([[next(r['survived_gain'] for r in jointdiff if r['joint_index']==j and r['loss']==d) for d in LOSSES] for j in range(12)])
im=axes[2].imshow(vals,aspect='auto',cmap='RdBu',norm=Normalize(-100,100));fig.colorbar(im,ax=axes[2],label='Student - Normal (pp)');axes[2].set_title('Survival gain')
for ax in axes:ax.set(xticks=range(5),xticklabels=[20,40,60,80,100],yticks=range(12),yticklabels=JOINTS,xlabel='Torque degradation (%)')
fig.savefig(OUT/'joint_loss_survival_absolute.png',dpi=180);plt.close(fig)
# Overall absolute values; seed ranges are descriptive, not training-seed CIs.
fig,axes=plt.subplots(2,3,figsize=(12,7.2),layout='constrained')
for ax,m,label in zip(axes.flat,METRICS,['Survival (%)','vx RMSE (m/s)','vy RMSE (m/s)','Yaw-rate RMSE (rad/s)','Cross-track RMSE (m)']):
    for pol,color in [('Normal','#777777'),('Student','#007fa3')]:
        scale=100 if m=='survived' else 1
        ys=[next(r[m] for r in byloss if r['policy']==pol and r['loss']==d)*scale for d in [0.]+LOSSES]
        seedmeans=np.array([[avg([r for r in R if r['policy']==pol and r['seed']==s and r['condition']['loss']==d],m)*scale for d in [0.]+LOSSES] for s in [1101,1102,1103]])
        ax.plot(np.array([0.]+LOSSES)*100,ys,'-o',color=color,label=pol);ax.fill_between(np.array([0.]+LOSSES)*100,seedmeans.min(0),seedmeans.max(0),color=color,alpha=.12)
    ax.set(xlabel='Torque degradation (%)',ylabel=label);ax.grid(alpha=.2)
    if m=='survived':ax.set_ylim(0,102);ax.legend()
axes.flat[-1].axis('off');axes.flat[-1].text(.05,.95,'Shading: range of three\nevaluation-seed means.\nNot independent training variation.\n\nHealthy d=0 is separate\nfrom the 60 fault conditions.',va='top',transform=axes.flat[-1].transAxes)
fig.savefig(OUT/'five_metrics_by_loss.png',dpi=180);plt.close(fig)

casekeys=['exploration_s1102_S15_r0_slot08_j08_d100','exploration_s1101_S20_r0_slot10_j01_d100']
case_rows=[]
for ci,key in enumerate(casekeys,1):
    pairs={p:next(r for r in R if r['paired_trial_id']==key and r['policy']==p) for p in ['Normal','Student']}
    traces={}
    for pol,r in pairs.items():
        folder=Path(r['source_folder']);manifest=json.loads((folder/'trace_manifest.json').read_text());parts=[];steps=[];flags=[]
        for chunk in manifest['chunks']:
            with np.load(folder/chunk['path']) as z:
                keep=z['env_id']==r['env_id']
                if keep.any():parts.append(z['values'][keep]);steps.append(z['step'][keep]);flags.append(z['flags'][keep])
        v=np.concatenate(parts);s=np.concatenate(steps);f=np.concatenate(flags);post=(f&2)>0;v=v[post];s=s[post];f=f[post]
        assert len(v)==r['valid_post_samples']
        err=np.column_stack((v[:,3]-v[:,6],v[:,4]-v[:,7],v[:,5]-v[:,8],v[:,13])).astype('float64')
        assert np.allclose((err*err).sum(0),r['error_sum_sq'],rtol=2e-5,atol=.002)
        t=s*.02-2.;last=t>=t[-1]-30
        traces[pol]=(v,t,f)
        result=dict(case=ci,policy=pol,trial_id=r['trial_id'],joint=JOINTS[r['condition']['joint']],profile=r['profile']['id'],terrain_column=r['terrain_column_id'],
            termination=r['termination'],survived=r['survived'],valid_post_s=r['observed_post_s'],stall_flag_s=r['stall_seconds'],
            diagnostic_tail_duration_s=float(t[-1]-t[last][0]),diagnostic_tail_full_30s=bool(t[-1]-t[0]>=30),last30_furthest_progress_gain_m=float(v[-1,12]-v[last,12][0]),last30_median_world_planar_speed=float(np.median(np.linalg.norm(v[last,14:16],axis=1))),
            last30_median_body_planar_speed=float(np.median(np.linalg.norm(v[last,3:5],axis=1))),
            **{m:r[m] for m in METRICS[1:]})
        case_rows.append(result)
        np.savez_compressed(OUT/('case%d_%s_trace.npz'%(ci,pol)),values=v,time_post=t,flags=f,fields=np.array(manifest['fields']))
    fig,axes=plt.subplots(2,3,figsize=(13,7.6),layout='constrained');r=pairs['Student'];yc=(r['terrain_column_id']-.5)*8
    x=np.linspace(-1,40.5,1200);ref=r['profile']['amplitude']*np.sin(2*np.pi*(x+1)/16)
    axes[0,0].plot(x,ref,'k--',lw=1,label='Reference path')
    for pol,color in [('Normal','#777777'),('Student','#007fa3')]:
        v,t,f=traces[pol];stride=10
        axes[0,0].plot(v[:,0],v[:,1]-yc,color=color,label=pol);axes[0,0].scatter(v[-1,0],v[-1,1]-yc,c=color,marker='x',s=50)
        for ax,actual,cmd,m in [(axes[0,1],3,6,'command_vx_rmse'),(axes[0,2],4,7,'command_vy_rmse'),(axes[1,0],5,8,'command_wz_rmse')]:
            ax.plot(t[::stride],v[::stride,actual],color=color,lw=.7,label=pol+' actual');ax.plot(t[::stride],v[::stride,cmd],color=color,lw=.8,ls='--',label=pol+' command')
            ax.text(.02,.97-(.1 if pol=='Student' else 0),pol+' RMSE=%.3f'%pairs[pol][m],transform=ax.transAxes,va='top',color=color,fontsize=8)
        axes[1,1].plot(t[::stride],v[::stride,13],color=color,label=pol+' RMSE=%.3f'%pairs[pol]['cross_track_rmse'])
        axes[1,2].plot(t[::stride],100*v[::stride,12]/pairs[pol]['arc_length_m'],color=color,label=pol+' '+pairs[pol]['termination'])
    axes[0,0].set(xlabel='World x (m)',ylabel='Lateral position from column center (m)',title='Top view');axes[0,0].legend(fontsize=7)
    for ax,label in [(axes[0,1],'vx (m/s)'),(axes[0,2],'vy (m/s)'),(axes[1,0],'Yaw rate (rad/s)'),(axes[1,1],'Cross-track error (m)'),(axes[1,2],'Furthest route progress (%)')]:ax.set(xlabel='Time after fault (s)',ylabel=label);ax.grid(alpha=.2)
    axes[1,1].legend(fontsize=7);axes[1,2].legend(fontsize=7)
    fig.suptitle('Case %d: %s, 100%% torque degradation, %s, column %d'%(ci,JOINTS[r['condition']['joint']],r['profile']['id'],r['terrain_column_id']))
    fig.savefig(OUT/('case%d_survival_vs_tracking.png'%ci),dpi=170);plt.close(fig)
write('raw_verified_cases.csv',case_rows)

lines=['# 생존·명령 추종·경로 추종 연계 분석','',
'기본 수집·집계·Notion 게시 완료 후, 동일26,352개 시행에서 수행한 추가 해석이다. 추가 시뮬레이션은 없다.','',
'## 핵심 판정','',
'평균적으로60–100% 토크 저하에서 Student의 생존율과 네 추종 오차가 함께 개선되었다. 정상·낮은 저하의vx/yaw-rate 오차는 항상 개선되지 않았다. 모든 조건의 우월성으로 쓰지 않는다.','',
'## 100% 토크 저하 전체','',
'|정책|시행|생존|완주|비낙상 시간상한|생존 중 정체flag|vx RMSE|vy RMSE|yaw RMSE|횡오차 RMSE|',
'|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in byloss:
    if r['loss']==1.:
        lines.append('|%s|%d|%d|%d|%d|%d|%.5f|%.5f|%.5f|%.5f|'%(r['policy'],r['episodes'],r['survived_n'],r['goal_n'],r['alive_at_cap_n'],r['alive_with_stall_flag_n'],*[r[m] for m in METRICS[1:]]))
lines+=['','정체flag는30초 동안 새 경로 진행0.5m 미만이다. 물리적으로 완전히 정지했다는 판정과 같지 않으며, 일시 정체 후 완주한 시행도 포함한다. 시간상한 생존 시행은 두 정책 모두 정체flag가 있었다. 완주/상한은 주 생존율을 대체하는 새로운 성공 지표가 아니라 그 값을 해석하는 보조 기록이다.','',
'## 관절 분석','',
'히트맵은12관절×5비영 토크 저하율이다. 각 셀 정책당216시행(6과제슬롯×12열×3평가시드). 정상은 특정 고장 관절이 없으므로12행에 복제하지 않는다.',
'다섯 패널의 양수는 모두Student에 유리하다: 생존은Student−Normal(%p), 오차는Normal−Student(각 물리단위). 서로 다른 지표를 임의 점수로 합치지 않는다.','',
'RL calf100%에서Normal 생존188/216=87.04%이나 완주는3/216=1.39%다. Student 생존198/216=91.67%, 완주195/216=90.28%다. 생존 차이4.63%p만으로는 이 조건의 수행 차이를 설명하지 못한다.',
'FR thigh100%에서는Student 생존155/216=71.76%로12관절 중 가장 낮다. Normal은33/216=15.28%다. Student의 어려운 조건도 함께 남긴다.','',
'## 원시 기록 검증 사례','',
'사례1은RL calf100%,S15,9열의 동일 초기조건 쌍이다. Normal은479.88초까지 비낙상이지만449.70초의 정체flag가 기록되었고, Student는122.78초에 완주했다.',
'사례2는FL thigh100%,S20,11열의 동일 초기조건 쌍이다. Normal은17.60초 후 낙상했다. Student는504.62초 비낙상이지만456.56초의 정체flag, vx RMSE0.488m/s, 횡오차0.031m를 보였다. 낮은 횡오차와 높은 생존율이 명령 수행을 뜻하지 않는 실제 사례다.',
'두 사례의모든50Hz 유효 표본을 원래trace에서 다시 읽어 표본 수와 오차제곱합을JSON 집계와 대조했다. raw_verified_cases.csv에 마지막30초 진행/속도도 남겼다. 사례는 메커니즘 설명용으로 선정했으며 대표 빈도는 전체 집계표로 제시한다.','',
'## 표현안','',
'- 본문 핵심 문장: 고장 대응Student는60–100% 토크 저하의 과제 혼합 평균에서 낙상을 줄이고 명령·경로 추종 오차를 함께 낮췄다. 비낙상 정체도Normal보다 적게 관찰되었다.',
'- 표: 저하율별생존율/vx/vy/yaw/횡오차의 두 정책 절대값과 차이. 정상 조건 별도.',
'- 그림: 저하율–5지표 곡선,12관절×5저하율의5패널 차이 히트맵, 생존과 수행이 달라지는 대응 시행 그림.',
'- 보조: 과제군·좌우·두 사인파·지형별 표, 유효 관찰 길이, 정체flag와완주 기록, 세 평가시드의 기술적 변동.',
'- Student가더 오래 살았다고 단정하지 않는다. 여기 생존율은이진결과이며Student의짧은 관찰시간은 빠른완주 때문일 수 있다.','']
(OUT/'REPORT.md').write_text('\n'.join(lines))
(OUT/'manifest.json').write_text(json.dumps(dict(dataset=str(C),source_records=26352,joint_cell_episodes_per_policy=216,additional_simulations=0,raw_verified_cases=case_rows),indent=2)+'\n')
with zipfile.ZipFile(OUT/'joint_tracking_diagnostics.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.iterdir():
        if p.is_file() and p.suffix!='.zip':z.write(p,p.name)
print(json.dumps(dict(status='COMPLETE',output=str(OUT),cases=case_rows),indent=2))
