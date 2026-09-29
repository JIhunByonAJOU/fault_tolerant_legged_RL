"""Make publication-style CPU plots/CSV from completed offline diagnostics."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'logs/evaluations/teacher_encoder_offline_20260929'
OUT=ROOT/'docs/results/teacher_encoder_offline_20260929'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    results=json.loads((DATA/'results.json').read_text())
    follow=json.loads((DATA/'followup.json').read_text())
    corpus_names=list(results['corpora'])
    colors=['#2864a0','#d48626'];labels=['TF-visited states','JT71500-visited states']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    keys=['linear_velocity','motor_strength','angular_velocity','failure_flag','payload','kp','kd','friction']
    ypos=np.arange(len(keys))
    for i,c in enumerate(corpus_names):
        d=results['corpora'][c]
        val=[d['group_interventions']['TF43000'][k+'/nominal']['post_fault_1s_plus']['joint_target_mae_rad'] for k in keys]
        axes[0,0].barh(ypos+(i-.5)*.34,val,height=.34,color=colors[i],label=labels[i])
    axes[0,0].set(yticks=ypos,yticklabels=[k.replace('_',' ') for k in keys],xlabel='Mean absolute joint-target change (rad)',title='A. TF43000: nominal-input intervention')
    axes[0,0].invert_yaxis();axes[0,0].legend(fontsize=8)
    stages=['TF43000','JT45000','JT49000','JT51000','JT53000','JT71500']
    xpos=np.arange(len(stages))
    for i,c in enumerate(corpus_names):
        vals=[results['corpora'][c]['latent_drift'][n]['groups']['post_fault_1s_plus']['geometry']['rms'] for n in stages]
        axes[0,1].plot(xpos,vals,'o-',color=colors[i],label=labels[i])
        vals=[results['corpora'][c]['latent_drift'][n]['groups']['post_fault_1s_plus']['affine_alignment_centered_nrmse'] for n in stages]
        axes[1,0].plot(xpos,vals,'o-',color=colors[i])
        vals=[100*follow['corpora'][c]['models'][n]['postfault_variance_partition']['within_episode_centered_energy_fraction'] for n in stages]
        axes[1,1].plot(xpos,vals,'o-',color=colors[i])
    for ax in (axes[0,1],axes[1,0],axes[1,1]):
        ax.set_xticks(xpos,stages,rotation=25,ha='right');ax.grid(axis='y',alpha=.2)
    axes[0,1].set(ylabel='Teacher latent RMS (dimensionless)',title='B. Same recorded inputs, different checkpoints')
    axes[1,0].set(ylabel='Held-out centered NRMSE',title='C. Residual after best affine coordinate alignment')
    axes[1,1].set(ylabel='Within-episode / total centered energy (%)',ylim=(0,100),title='D. Within-episode variation: only velocity inputs vary')
    fig.suptitle('Read-only teacher-encoder diagnostics: d > 0, at least 1 s after onset',fontsize=13)
    fig.savefig(OUT/'teacher_encoder_diagnostics.png',dpi=180)
    fig.savefig(OUT/'teacher_encoder_diagnostics.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for i,c in enumerate(corpus_names):
        dims=results['corpora'][c]['dimension_interventions']['TF43000']
        names=sorted(dims,key=lambda n:results['corpora'][corpus_names[0]]['dimension_interventions']['TF43000'][n]['post_fault_1s_plus']['action_mae'],reverse=True)[:12]
        values=[dims[n]['post_fault_1s_plus']['joint_target_mae_rad'] for n in names]
        axes[0].barh(np.arange(len(names))+(i-.5)*.34,values,height=.34,color=colors[i],label=labels[i])
        gains=[]
        for n in stages[1:]:
            m=follow['corpora'][c]['models'][n]
            base=m['student_current_teacher_gap']['post_fault_1s_plus']['action_mae']
            lag=m['velocity_lag']['47']['student_vs_lagged_teacher']['post_fault_1s_plus']['action_mae']
            gains.append((base-lag)/base*100)
        axes[1].plot(np.arange(len(gains)),gains,'o-',color=colors[i],label=labels[i])
    axes[0].set(yticks=np.arange(len(names)),yticklabels=names,xlabel='Mean absolute joint-target change (rad)',title='A. Largest one-coordinate nominal probes')
    axes[0].invert_yaxis()
    axes[1].set(xticks=np.arange(len(stages)-1),xticklabels=stages[1:],ylabel='Decrease in student/teacher action gap (%)',title='B. Teacher velocity shifted back by 40 ms')
    axes[1].tick_params(axis='x',rotation=25);axes[1].grid(axis='y',alpha=.2);axes[1].legend(fontsize=8)
    fig.savefig(OUT/'teacher_encoder_details.png',dpi=180);plt.close(fig)
    for category in ['group_interventions','dimension_interventions']:
        rows=[]
        for corpus,d in results['corpora'].items():
            for checkpoint,interventions in d[category].items():
                for name,cohorts in interventions.items():
                    for cohort,metrics in cohorts.items():
                        rows.append(dict(corpus=corpus,checkpoint=checkpoint,intervention=name,cohort=cohort,**metrics))
        with (DATA/(category+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(OUT)


if __name__=='__main__':main()
