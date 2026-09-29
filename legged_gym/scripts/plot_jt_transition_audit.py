"""Plots for the checkpoint diagnostic report (no legacy recovery metric)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=Path('logs/evaluations/jt-transition-audit-20260928')
o=Path('docs/results/jt-transition-audit-20260928');o.mkdir(parents=True,exist_ok=True)
s=json.loads((p/'summary.json').read_text())['checkpoints']
its=sorted(map(int,s));xs=[(it-43000)/1000 for it in its]
ref=json.loads(Path('logs/evaluations/onset-v2-diagnostic-20260928/summary.json').read_text())
# First-seed screening is kept separate from the selected multi-seed confirmation.
fig,axes=plt.subplots(1,3,figsize=(13,3.6),constrained_layout=True)
for ax,key,label in zip(axes,['survived_post_failure_horizon','tracking_time_fraction','post_yaw_rate_rmse'],['Survival (%)','Tracking time (%)','Yaw rate RMSE (rad/s)']):
 scale=100 if key!='post_yaw_rate_rmse' else 1
 y=[s[str(it)]['seeds']['101']['damaged'][key]*scale for it in its]
 ax.plot(xs,y,'-o',color='#0072B2',markersize=4,label='Student-only, eval seed101')
 ax.axhline(ref['teacher']['damaged_by_seed']['101'][key]*scale,color='#555555',linestyle='--',label='TF43000 reference, same seed')
 ax.axvline(10,color='#AA3377',linestyle=':',label='End of fusion schedule')
 ax.set(xlabel='Iterations since JT start (thousands)',ylabel=label);ax.grid(alpha=.15)
 if scale==100:ax.set_ylim(0,100)
axes[1].legend(fontsize=7,loc='lower right');fig.savefig(o/'student_only_progress.png',dpi=180);plt.close(fig)
# Show severe condition separately.
fig,ax=plt.subplots(figsize=(7,3.7),constrained_layout=True)
for d in ['0.0','0.4','0.8','1.0']:
 ax.plot(xs,[100*s[str(it)]['seeds']['101']['by_severity'][d]['tracking_time_fraction'] for it in its],'-o',label='d='+d,markersize=3)
ax.axvline(10,color='gray',linestyle=':');ax.set(xlabel='Iterations since JT start (thousands)',ylabel='Tracking time (%)',ylim=(0,100));ax.legend();ax.grid(alpha=.15);fig.savefig(o/'severity_progress.png',dpi=180);plt.close(fig)
f=p/'latent_corpus_audit.json'
if f.exists():
 rows=json.loads(f.read_text())['rows'];xx=[(r['iteration']-43000)/1000 for r in rows];group='post_fault_1s_plus'
 fig,axes=plt.subplots(1,3,figsize=(13,3.6),constrained_layout=True)
 for k,label in [('teacher_latent_rms','Current teacher'),('student_latent_rms','Student')]:axes[0].plot(xx,[r['groups'][group][k] for r in rows],label=label)
 axes[0].set_ylabel('Latent RMS');axes[0].legend(fontsize=8)
 axes[1].plot(xx,[r['groups'][group]['latent_relative_rms'] for r in rows]);axes[1].set_ylabel('Relative latent RMS error')
 for k,label in [('actor_joint_target_mae_rad','Student vs current teacher branch'),('student_vs_tf_action_mae','Student vs frozen TF43000')]:
  mul=1 if k=='actor_joint_target_mae_rad' else .25
  axes[2].plot(xx,[mul*r['groups'][group][k] for r in rows],label=label)
 axes[2].set_ylabel('Joint target MAE (rad)');axes[2].legend(fontsize=7)
 for ax in axes:ax.axvline(10,color='gray',linestyle=':');ax.set_xlabel('Iterations since JT start (thousands)');ax.grid(alpha=.15)
 fig.suptitle('Identical TF-generated inputs, at least 1 s after fault (offline diagnostic)',fontsize=11)
 fig.savefig(o/'common_input_latent_audit.png',dpi=180);plt.close(fig)
print(o)
