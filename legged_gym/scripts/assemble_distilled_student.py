"""Assemble a frozen TF43000 actor/teacher with an offline distilled student."""

import argparse
from pathlib import Path

import isaacgym  # noqa: F401
import torch

from legged_gym.learning.joint_teacher_student_actor_critic import JointTeacherStudentActorCritic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher-checkpoint', required=True)
    parser.add_argument('--student-checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    teacher = torch.load(args.teacher_checkpoint, map_location='cpu')
    student = torch.load(args.student_checkpoint, map_location='cpu')
    student_width = student['student_encoder_state_dict']['frame_encoder.0.weight'].shape[0]
    model = JointTeacherStudentActorCritic(2635, 45, 12, student_embedding_dim=student_width)
    copied = model.load_state_dict(teacher['model_state_dict'], strict=False)
    expected_missing = {'student_encoder.' + key for key in model.student_encoder.state_dict()}
    if set(copied.missing_keys) != expected_missing or copied.unexpected_keys:
        raise RuntimeError('TF->JT state mismatch: %s' % (copied,))
    model.student_encoder.load_state_dict(student['student_encoder_state_dict'], strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'iter': 43000,
                'infos': {'joint_schedule_origin': 43000,
                          'student_source': str(Path(args.student_checkpoint).resolve()),
                          'teacher_source': str(Path(args.teacher_checkpoint).resolve())}}, output)
    reloaded = torch.load(output, map_location='cpu')['model_state_dict']
    for key, value in teacher['model_state_dict'].items():
        if not torch.equal(reloaded[key], value):
            raise RuntimeError('Frozen teacher tensor changed: ' + key)
    print(output, 'frozen TF tensors verified', flush=True)


if __name__ == '__main__':
    main()
