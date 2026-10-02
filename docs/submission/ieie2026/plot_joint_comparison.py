#!/usr/bin/env python3
"""Comparable joint-by-loss panels; both policies share a colour scale."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=list(csv.DictReader(a.summary.open()))
    rates=sorted({float(r['degradation_rate']) for r in rows})
    joints=sorted({(int(r['joint_index']),r['joint']) for r in rows})
    labels=[name for _,name in joints]
    a.output.mkdir(parents=True,exist_ok=True)
    for key,name,label in [('S20','survival','20-s survival (%)'),
                           ('world_progress_ratio','progress','Onset-heading forward displacement / 10 m (%)'),
                           ('Q_strict','strict','Strict tracking occupancy (%)'),
                           ('vx_alive_pooled_rmse','rmse','Alive-sample longitudinal RMSE (m/s)')]:
        scale=1 if key=='vx_alive_pooled_rmse' else 100
        matrices=[]
        for model in ['tb22500','jt77500']:
            lookup={(int(r['joint_index']),float(r['degradation_rate'])):float(r[key])*scale for r in rows if r['model']==model}
            matrices.append(np.array([[lookup[index,d] for d in rates] for index,_ in joints]))
        if key in ['S20','Q_strict']:kwargs=dict(vmin=0,vmax=100,cmap='viridis')
        elif key=='vx_alive_pooled_rmse':kwargs=dict(vmin=0,vmax=max(m.max() for m in matrices),cmap='viridis_r')
        else:
            lo=min(-25,float(min(m.min() for m in matrices)))
            hi=max(100,float(max(m.max() for m in matrices)))
            kwargs=dict(norm=TwoSlopeNorm(vmin=lo,vcenter=0,vmax=hi),cmap='RdBu')
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
        for ax,m,title in zip(axes,matrices,['Normal Base (privileged)','Final Student (history)']):
            im=ax.imshow(m,aspect='auto',**kwargs)
            ax.set(xticks=range(len(rates)),xticklabels=[f'{x:g}' for x in rates],
                   yticks=range(len(labels)),yticklabels=labels,xlabel='Torque-multiplier loss d')
            ax.set_title(title,fontsize=14)
            ax.tick_params(labelsize=11)
            for i in range(len(labels)):
                for j in range(len(rates)):
                    colour=im.cmap(im.norm(m[i,j]));lum=.2126*colour[0]+.7152*colour[1]+.0722*colour[2]
                    value=f'{m[i,j]:.3f}' if scale==1 else f'{m[i,j]:.1f}'
                    ax.text(j,i,value,ha='center',va='center',fontsize=10,color='white' if lum<.48 else 'black')
        fig.colorbar(im,ax=axes,shrink=.92,label=label)
        for ext in ['png','pdf']:fig.savefig(a.output/f'paper_joint_{name}_compare.{ext}',dpi=220)
        plt.close(fig)


if __name__=='__main__':main()
