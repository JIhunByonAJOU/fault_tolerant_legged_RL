"""Pool same-state branch disagreement using recorded alive samples."""
import argparse
import json
from pathlib import Path
import math

def main():
    p=argparse.ArgumentParser();p.add_argument('--raw-dir',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--seeds',type=int,nargs='+',required=True);p.add_argument('--models',nargs='+',default=['jt77500']);a=p.parse_args()
    output={}
    for model in a.models:
        metas=[json.loads((a.raw_dir/f'{model}_seed{s}.jsonl').read_text().splitlines()[0]) for s in a.seeds]
        stats=[m['latent_imitation'] for m in metas];counts=[s['post_alive_samples'] for s in stats];n=sum(counts)
        item={'seeds':a.seeds,'post_alive_samples':n,'interpretation':'same-state branch disagreement on this model trajectory, not an oracle performance bound or semantic identification','sampling':'all alive post-fault states across joints/severities','joint_target_delta_note':'action_scale times raw action disagreement; nominal target before action clipping, not actual joint angle or torque error'}
        for key in ['latent_coordinate_rmse','action_coordinate_rmse']:
            item[key]=math.sqrt(sum(s[key]**2*c for s,c in zip(stats,counts))/n)
        item['joint_target_coordinate_rmse_rad']=.25*item['action_coordinate_rmse']
        normalized=[(s,c) for s,c in zip(stats,counts) if 'teacher_latent_coordinate_rms' in s];nn=sum(c for s,c in normalized)
        if nn:
            item['normalization_seeds']=[seed for seed,s in zip(a.seeds,stats) if 'teacher_latent_coordinate_rms' in s]
            item['teacher_latent_coordinate_rms']=math.sqrt(sum(s['teacher_latent_coordinate_rms']**2*c for s,c in normalized)/nn)
            item['student_latent_coordinate_rms']=math.sqrt(sum(s['student_latent_coordinate_rms']**2*c for s,c in normalized)/nn)
            item['relative_latent_rmse']=math.sqrt(sum(s['latent_coordinate_rmse']**2*c for s,c in normalized)/sum(s['teacher_latent_coordinate_rms']**2*c for s,c in normalized))
            item['mean_latent_vector_cosine']=sum(s['mean_latent_vector_cosine']*c for s,c in normalized)/nn
        output[model]=item
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))
if __name__=='__main__':main()
