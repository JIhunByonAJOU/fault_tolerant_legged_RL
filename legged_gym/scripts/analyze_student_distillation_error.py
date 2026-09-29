"""Break held-out fixed-teacher imitation error down by post-fault age."""

import argparse
import json
from pathlib import Path

import isaacgym  # noqa: F401
import torch

from legged_gym.learning.joint_teacher_student_actor_critic import StudentHistoryEncoder
from legged_gym.learning.official_wim_teacher_actor_critic import OfficialWimTeacherActorCritic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', required=True)
    args = parser.parse_args()
    root = Path(args.experiment_dir)
    torch.set_num_threads(2)
    corpus_root = Path('logs/evaluations/jt-transition-audit-20260928')
    parts = []
    for file in ('common_teacher_inputs.pt', 'common_student_inputs.pt'):
        parts.append(torch.load(corpus_root / file, map_location='cpu'))
    corpus = {key: torch.cat([p[key] for p in parts])
              for key in ('obs', 'privileged', 'env_id', 'rate', 'elapsed_since_onset')}
    tf_file = Path('logs/official_wim_teacher243_failure_fullrange_fromscratch/'
                   'Aug13_03-44-07_seed1-50000iter-fromscratch-fullrange/model_43000.pt')
    tf = OfficialWimTeacherActorCritic(235, 45, 12).eval()
    tf.load_state_dict(torch.load(tf_file, map_location='cpu')['model_state_dict'])
    obs = corpus['obs']
    history = obs[:, 235:].reshape(-1, 50, 48)
    current = obs[:, :235]
    with torch.no_grad():
        target_z = tf.teacher_encoder(corpus['privileged'])
        target_action = tf.actor(torch.cat((current, target_z), dim=1))
    result = {}
    for method in ('latent_tf', 'action_tf'):
        encoder = StudentHistoryEncoder().eval()
        encoder.load_state_dict(torch.load(root / (method + '.pt'), map_location='cpu')['student_encoder_state_dict'])
        with torch.no_grad():
            z = torch.cat([encoder(history[i:i + 1024]) for i in range(0, len(history), 1024)])
            action = tf.actor(torch.cat((current, z), dim=1))
        held_out = (corpus['env_id'] // 72) % 4 == 3
        severity = corpus['rate']
        elapsed = corpus['elapsed_since_onset']
        masks = {
            'intact_pre_onset': held_out & (elapsed < 0),
            'dgt0_0_to_0_5s': held_out & (severity > 0) & (elapsed >= 0) & (elapsed < .5),
            'dgt0_0_5_to_1s': held_out & (severity > 0) & (elapsed >= .5) & (elapsed < 1),
            'dgt0_1_to_2s': held_out & (severity > 0) & (elapsed >= 1) & (elapsed < 2),
            'dgt0_2_to_5s': held_out & (severity > 0) & (elapsed >= 2) & (elapsed < 5),
            'dgt0_5_to_10s': held_out & (severity > 0) & (elapsed >= 5) & (elapsed < 10),
            'dgt0_10s_plus': held_out & (severity > 0) & (elapsed >= 10),
            'd1_0_to_1s': held_out & (severity == 1) & (elapsed >= 0) & (elapsed < 1),
            'd1_1s_plus': held_out & (severity == 1) & (elapsed >= 1),
        }
        result[method] = {}
        for name, mask in masks.items():
            if mask.sum() == 0:
                continue
            diff = z[mask] - target_z[mask]
            result[method][name] = {
                'n': int(mask.sum()),
                'latent_relative_rms': float(diff.square().mean().sqrt()
                    / target_z[mask].square().mean().sqrt().clamp(min=1e-8)),
                'action_mae': float((action[mask] - target_action[mask]).abs().mean()),
            }
    (root / 'error_by_elapsed.json').write_text(json.dumps(result, indent=2))
    for method, groups in result.items():
        print(method, groups)


if __name__ == '__main__':
    main()
