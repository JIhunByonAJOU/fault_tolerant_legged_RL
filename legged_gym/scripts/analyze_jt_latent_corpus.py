"""Audit all JT checkpoints on identical recorded inputs; no optimization."""
import argparse
import json
from pathlib import Path
import isaacgym  # noqa: F401
import torch
from legged_gym.learning.joint_teacher_student_actor_critic import JointTeacherStudentActorCritic
from legged_gym.learning.official_wim_teacher_actor_critic import OfficialWimTeacherActorCritic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    corpus = torch.load(args.corpus, map_location='cpu')
    run = Path('logs/jt_wim243_failure_fullrange_onset/Aug14_00-37-37_')
    teacher = OfficialWimTeacherActorCritic(235, 45, 12).eval()
    checkpoint = 'logs/official_wim_teacher243_failure_fullrange_fromscratch/Aug13_03-44-07_seed1-50000iter-fromscratch-fullrange/model_43000.pt'
    teacher.load_state_dict(torch.load(checkpoint, map_location='cpu')['model_state_dict'])
    obs = corpus['obs']; priv = corpus['privileged']
    masks = {'all': torch.ones(len(obs), dtype=torch.bool),
             'post_fault_1s_plus': (corpus['elapsed_since_onset'] >= 1) & (corpus['rate'] > 0),
             'd1_1s_plus': (corpus['elapsed_since_onset'] >= 1) & (corpus['rate'] == 1)}
    with torch.inference_mode():
        tf_latent = teacher.teacher_encoder(priv)
        tf_action = teacher.act_inference(obs[:, :235], priv)
        rows = []
        for f in sorted(run.glob('model_*.pt'), key=lambda f: int(f.stem.split('_')[-1])):
            it = int(f.stem.split('_')[-1])
            model = JointTeacherStudentActorCritic(2635, 45, 12).eval()
            model.load_state_dict(torch.load(str(f), map_location='cpu')['model_state_dict'], strict=True)
            zt = model.encode_privileged(priv)
            zs = model.encode_history(obs)
            at = model.act_inference_with_latent(obs, zt)
            ass = model.act_inference_with_latent(obs, zs)
            row = {'iteration': it, 'groups': {}}
            for name, m in masks.items():
                ztm, zsm = zt[m], zs[m]
                row['groups'][name] = dict(n=int(m.sum()),
                    latent_l2=float((zsm-ztm).norm(dim=-1).mean()),
                    latent_relative_rms=float((zsm-ztm).square().mean().sqrt()/ztm.square().mean().sqrt().clamp(min=1e-8)),
                    teacher_latent_rms=float(ztm.square().mean().sqrt()),
                    student_latent_rms=float(zsm.square().mean().sqrt()),
                    cosine=float(torch.nn.functional.cosine_similarity(zsm,ztm).mean()),
                    actor_action_mae=float((ass[m]-at[m]).abs().mean()),
                    actor_joint_target_mae_rad=float((ass[m]-at[m]).abs().mean()*.25),
                    student_vs_tf_action_mae=float((ass[m]-tf_action[m]).abs().mean()),
                    current_teacher_vs_tf_action_mae=float((at[m]-tf_action[m]).abs().mean()))
            rows.append(row)
            print(it, row['groups']['post_fault_1s_plus'], flush=True)
    Path(args.output).write_text(json.dumps({'corpus': args.corpus, 'rows': rows,
         'caveat': 'Identical TF-generated states; offline agreement is not closed-loop student performance.'}, indent=2))

if __name__ == '__main__':
    main()
