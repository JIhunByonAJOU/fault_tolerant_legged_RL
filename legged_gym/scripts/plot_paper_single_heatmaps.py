"""Column-width vector heatmaps of audited cell summaries."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
LABELS={'tb22500':'Normal Base (privileged)','jt77500':'Final Student (history)'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--results-dir',required=True,type=Path);a=p.parse_args()
    rows=list(csv.DictReader((a.results_dir/'metrics_by_joint_severity.csv').open()))
    plt.rcParams.update({'font.size':7.5,'axes.titlesize':8,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'pdf.fonttype':42,'ps.fonttype':42})
    manifest=json.loads((a.results_dir/'evidence_manifest.json').read_text()); unit_axis=all(x['metadata']['schema_version']>=5 for x in manifest['audit'])
    world_title='Initial-heading progress (%)' if unit_axis else 'Planar body-axis projection (%)'
    specs=[('S20','20 s survival (%)',100,0,100,'YlGnBu'),('world_progress_ratio',world_title,100,-25,100,'RdBu'),('vx_alive_pooled_rmse','Forward RMSE (m/s)',1,0,.5,'YlOrRd'),('Q_strict','Strict tracking occupancy (%)',100,0,100,'magma')]
    for model in ['tb22500','jt77500']:
        rr=[r for r in rows if r['model']==model]
        for key,title,scale,lo,hi,cmap in specs:
            if key not in rr[0]:continue
            matrix=np.array([[float(next(r[key] for r in rr if int(r['joint_index'])==j and float(r['degradation_rate'])==d))*scale for d in [0,.2,.4,.6,.8,1]] for j in range(12)])
            fig,ax=plt.subplots(figsize=(3.1,4.0),layout='constrained')
            im=ax.imshow(matrix,aspect='auto',cmap=cmap,norm=TwoSlopeNorm(vmin=lo,vcenter=0,vmax=hi)) if key=='world_progress_ratio' else ax.imshow(matrix,aspect='auto',cmap=cmap,vmin=lo,vmax=hi)
            ax.set_title(LABELS[model]+'\n'+title,pad=5);ax.set_xticks(range(6),['0','.2','.4','.6','.8','1']);ax.set_yticks(range(12),[next(r['joint'] for r in rr if int(r['joint_index'])==j) for j in range(12)]);ax.set_xlabel('Motor-strength loss d')
            for j in range(12):
                for k in range(6):
                    value=matrix[j,k];rgba=im.cmap(im.norm(value)); luminance=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                    ax.text(k,j,f'{value:.0f}' if scale==100 else f'{value:.2f}',ha='center',va='center',fontsize=7.5,color='black' if luminance>.45 else 'white')
            cb=fig.colorbar(im,ax=ax,fraction=.035,pad=.03);cb.ax.tick_params(labelsize=7)
            fig.savefig(a.results_dir/f'{model}_{key}_heatmap.pdf');fig.savefig(a.results_dir/f'{model}_{key}_heatmap.png',dpi=300);plt.close(fig)
if __name__=='__main__':main()
