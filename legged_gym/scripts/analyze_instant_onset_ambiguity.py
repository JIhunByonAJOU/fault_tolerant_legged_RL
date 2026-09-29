"""Measure TF's instant response to GT failure with identical student input.

This is a counterfactual input intervention, not a rollout. At the exact onset
control interval, the evaluator updates privileged motor strength and failure
flag before the student observation/history can show a motor response.
"""

import argparse
import json
from pathlib import Path

import isaacgym  # noqa: F401
import torch

from legged_gym.learning.official_wim_teacher_actor_critic import OfficialWimTeacherActorCritic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    tf_file = Path('logs/official_wim_teacher243_failure_fullrange_fromscratch/'
                   'Aug13_03-44-07_seed1-50000iter-fromscratch-fullrange/model_43000.pt')
    tf = OfficialWimTeacherActorCritic(235, 45, 12).eval()
    tf.load_state_dict(torch.load(tf_file, map_location='cpu')['model_state_dict'])
    root = Path('logs/evaluations/jt-transition-audit-20260928')
    report = {}
    for file in ('common_teacher_inputs.pt', 'common_student_inputs.pt'):
        corpus = torch.load(root / file, map_location='cpu')
        select = (corpus['elapsed_since_onset'] < 0) & (corpus['rate'] > 0)
        obs = corpus['obs'][select, :235]
        privileged = corpus['privileged'][select]
        rates = corpus['rate'][select]
        joints = (corpus['env_id'][select] % 72) // 6
        if not torch.all(privileged[:, -1] == 0):
            raise RuntimeError('Pre-onset corpus unexpectedly has failure flag')
        if not torch.all((joints >= 0) & (joints < 12)):
            raise RuntimeError('Invalid joint assignment')
        counterfactual = privileged.clone()
        row = torch.arange(len(obs))
        counterfactual[row, 2 + joints] *= (1. - rates)
        counterfactual[:, -1] = 1.
        with torch.no_grad():
            z0 = tf.teacher_encoder(privileged)
            z1 = tf.teacher_encoder(counterfactual)
            a0 = tf.actor(torch.cat((obs, z0), dim=1))
            a1 = tf.actor(torch.cat((obs, z1), dim=1))
        report[file] = {}
        for rate in (.2, .4, .6, .8, 1.):
            mask = rates == rate
            report[file][str(rate)] = {
                'n': int(mask.sum()),
                'latent_l2': float((z1[mask] - z0[mask]).norm(dim=1).mean()),
                'action_mae': float((a1[mask] - a0[mask]).abs().mean()),
                'joint_target_mae_rad': float((a1[mask] - a0[mask]).abs().mean() * .25),
            }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
