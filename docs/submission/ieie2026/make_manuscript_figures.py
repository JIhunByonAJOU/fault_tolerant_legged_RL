"""Rebuild manuscript schematic and learning figure from archived curve points."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from matplotlib.text import Text
BASE=Path(__file__).resolve().parent
OUT=BASE/'paper_assets'

AUDIT={}
def save_checked(fig,name):
 fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[];sizes=set()
 for ax in fig.axes:assert not any(ax.get_title(loc=loc) for loc in ['left','center','right'])
 assert fig._suptitle is None
 for text in fig.findobj(Text):
  if not text.get_visible() or not text.get_text():continue
  sizes.add(float(text.get_fontsize()));box=text.get_window_extent(renderer)
  if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1:outside.append(text.get_text())
 assert not outside,(name,outside)
 assert sizes=={9.0},(name,sizes)
 AUDIT[name]={'font_size_pt':9,'width_mm':fig.get_figwidth()*25.4,'height_mm':fig.get_figheight()*25.4,'internal_title':False,'outside_text':outside}
 for ext in ['pdf','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=300)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--extract-local-logs',action='store_true');args=parser.parse_args()
 OUT.mkdir(exist_ok=True)
 for f in (Path.home()/'.local/share/fonts').rglob('*.ttf'):
  try:
   if font_manager.FontProperties(fname=f).get_name()=='Times New Roman':font_manager.fontManager.addfont(f)
  except (RuntimeError,ValueError):pass
 font=font_manager.findfont(font_manager.FontProperties(family='Times New Roman'),fallback_to_default=False)
 plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.labelsize':9,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'pdf.fonttype':42,'mathtext.fontset':'custom','mathtext.rm':'Times New Roman','mathtext.it':'Times New Roman:italic','mathtext.bf':'Times New Roman:bold'})
 if args.extract_local_logs:
  root=BASE.parents[2]
  paths=[root/'logs/official_wim_teacher243_failure_fullrange_fromscratch/Aug13_03-44-07_seed1-50000iter-fromscratch-fullrange/metrics.jsonl',root/'logs/jt_wim243_width64_lr1e5_fresh_20260929/run_seed1_35000/metrics.jsonl']
  rows=[];sources=[]
  for path,begin,end in zip(paths,[0,43000],[43000,77500]):
   raw=[json.loads(l) for l in path.open()];raw=[r for r in raw if begin<=r['iteration']<=end]
   iterations=np.array([r['iteration'] for r in raw]);reward=np.array([r['Train/mean_reward'] for r in raw]);cs=np.r_[0,np.cumsum(reward)];moving=(cs[500:]-cs[:-500])/500
   for k in range(499,len(raw)):
    rows.append({'iteration':int(iterations[k]),'mean_reward_trailing500':float(moving[k-499]),'alpha':float(raw[k].get('Adaptation/alpha',0)),'phase':'teacher' if begin==0 else 'joint_student'})
   sources.append({'path':str(path.relative_to(root)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'iteration_begin':begin,'iteration_end':end})
  with (OUT/'learning_curve.csv').open('w',newline='') as stream:
   w=csv.DictWriter(stream,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
  (OUT/'learning_sources.json').write_text(json.dumps({'sources':sources,'metric':'Train/mean_reward','smoothing':'Separate complete trailing 500-iteration windows in each training run','x_axis':'Cumulative iteration identifier including pretraining','training_seed':1,'font_family':'Times New Roman','font_size_pt':9},indent=2)+'\n')
 # Architecture: latent mixing is a training operation; inference uses student.
 fig,ax=plt.subplots(figsize=(80/25.4,40/25.4));fig.subplots_adjust(left=.02,right=.98,top=.99,bottom=.01);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
 def box(x,y,w,h,label,color='#f2f2f2'):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012,rounding_size=.015',facecolor=color,edgecolor='#444',linewidth=.65));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9)
 def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',linewidth=.8,color='#444',shrinkA=2,shrinkB=2))
 box(.02,.75,.43,.22,'Teacher MLP\n45 → 8');box(.55,.75,.43,.22,'Student CNN\n50 × 48 → 8','#e7f0f7')
 box(.55,.43,.43,.19,'Mixed feature\n8');box(.02,.43,.43,.19,'Observation\n235')
 box(.19,.04,.62,.23,'Actor / Critic\n235 + 8 = 243')
 arrow((.235,.75),(.67,.62));arrow((.765,.75),(.765,.62));arrow((.235,.43),(.35,.27));arrow((.765,.43),(.65,.27))
 save_checked(fig,'paper_architecture')
 plt.close(fig)
 rows=list(csv.DictReader((OUT/'learning_curve.csv').open()))
 fig,(ax,mix)=plt.subplots(2,1,figsize=(80/25.4,63/25.4),sharex=True,
                           gridspec_kw={'height_ratios':[2.6,1]},constrained_layout=True)
 for phase,color in [('teacher','#586477'),('joint_student','#007fa3')]:
  rr=[r for r in rows if r['phase']==phase]
  ax.plot([int(r['iteration'])/1000 for r in rr],[float(r['mean_reward_trailing500']) for r in rr],color=color,lw=1.1)
 rr=[r for r in rows if r['phase']=='joint_student']
 assert np.allclose([float(r['alpha']) for r in rr],np.clip([(int(r['iteration'])-43000)/10000 for r in rr],0,1))
 for panel in [ax,mix]:
  for left,right,color in [(0,43,'#eef0f3'),(43,53,'#f4cadb'),(53,77.5,'#fff2f7')]:
   panel.axvspan(left,right,color=color,zorder=0)
  for at in [43,53]:panel.axvline(at,color='#777',ls='--',lw=.6)
  for side in ['top','right']:panel.spines[side].set_visible(False)
  panel.tick_params(length=2,pad=2);panel.grid(alpha=.15);panel.set_xlim(0,77.5)
 ax.set(ylabel='Mean reward',ylim=(0,26),yticks=[0,10,20])
 for at,label in [(21.5,'Teacher'),(48,'Joint'),(65.25,'Student only')]:
  ax.text(at,24.0,label,ha='center',va='center',fontsize=9)
 weight_x=np.array([0,43,53,77.5]);alpha=np.array([0,0,1,1])
 mix.plot(weight_x,1-alpha,color='#586477',lw=1,ls='--',label=r'Teacher: $1-\alpha$')
 mix.plot(weight_x,alpha,color='#007fa3',lw=1,label=r'Student: $\alpha$')
 mix.set(ylabel='Weight',ylim=(-.08,1.15),yticks=[0,1],xlabel=r'Iteration ($\times10^3$)',
         xticks=[0,43,53,77.5],xticklabels=['0','43','53','77.5'])
 mix.legend(loc='center left',frameon=False,handlelength=1.2,handletextpad=.3,borderaxespad=.15,labelspacing=.15)
 save_checked(fig,'paper_learning')
 AUDIT['paper_learning']['restored_design']='Reward and Teacher/Joint/Student-only phases with pink transition band; lower teacher/student weights'
 AUDIT['paper_learning']['weight_schedule_verified_against_archived_alpha']=True
 plt.close(fig)
 (OUT/'figure_audit.json').write_text(json.dumps(AUDIT,indent=2)+'\n')
 print('Saved title-free architecture and learning figures in',OUT)
if __name__=='__main__':main()
