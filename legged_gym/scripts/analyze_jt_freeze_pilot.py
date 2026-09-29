"""Common-input diagnostics for matched JT49000 teacher-freeze continuations."""

import argparse
import json
from pathlib import Path

import isaacgym  # noqa: F401
import torch

from legged_gym.learning.joint_teacher_student_actor_critic import JointTeacherStudentActorCritic


def load_model(path):
    model = JointTeacherStudentActorCritic(2635, 45, 12).eval()
    model.load_state_dict(torch.load(path, map_location='cpu')['model_state_dict'], strict=True)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    models = {key: load_model(value) for key, value in
              (('source', args.source), ('baseline', args.baseline), ('freeze', args.freeze))}
    corpus_root = Path('logs/evaluations/jt-transition-audit-20260928')
    report = {'checkpoints': vars(args), 'corpora': {}}
    for source_name in ('common_teacher_inputs.pt', 'common_student_inputs.pt'):
        data = torch.load(corpus_root / source_name, map_location='cpu')
        obs, privileged = data['obs'], data['privileged']
        groups = {
            'damaged_post_1s': (data['rate'] > 0) & (data['elapsed_since_onset'] >= 1),
            'd1_post_1s': (data['rate'] == 1) & (data['elapsed_since_onset'] >= 1),
        }
        with torch.inference_mode():
            reference = models['source'].encode_privileged(privileged)
            entries = {}
            for name, model in models.items():
                teacher = model.encode_privileged(privileged)
                student = model.encode_history(obs)
                ta = model.act_inference_with_latent(obs, teacher)
                sa = model.act_inference_with_latent(obs, student)
                entries[name] = {}
                for group, mask in groups.items():
                    own_delta = student[mask] - teacher[mask]
                    reference_delta = student[mask] - reference[mask]
                    entries[name][group] = {
                        'n': int(mask.sum()),
                        'teacher_rms': float(teacher[mask].square().mean().sqrt()),
                        'student_rms': float(student[mask].square().mean().sqrt()),
                        'own_teacher_relative_rms': float(own_delta.square().mean().sqrt()
                            / teacher[mask].square().mean().sqrt().clamp(min=1e-8)),
                        'source_teacher_relative_rms': float(reference_delta.square().mean().sqrt()
                            / reference[mask].square().mean().sqrt().clamp(min=1e-8)),
                        'teacher_drift_from_source_relative_rms': float(
                            (teacher[mask] - reference[mask]).square().mean().sqrt()
                            / reference[mask].square().mean().sqrt().clamp(min=1e-8)),
                        'own_actor_action_mae': float((sa[mask] - ta[mask]).abs().mean()),
                    }
        report['corpora'][source_name] = entries
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    for corpus, entries in report['corpora'].items():
        print(corpus)
        for name, groups in entries.items():
            print(name, groups['damaged_post_1s'])


if __name__ == '__main__':
    main()
