# Student recovery diagnosis and frozen-Teacher baseline

Status: episode-level diagnosis, CPU trace smoke, and frozen-baseline CPU
training smoke complete; per-step GPU traces and production baseline training
not yet run.

## Observed failure mode

Source: frozen P5 paired JSONL, 3 seeds, 144 episodes per joint/severity cell
and policy. These are robot-episode aggregates, not step traces.

| Onset cell | Policy | Stable recovery | Median time if recovered | Survived without recovery |
| --- | --- | ---: | ---: | ---: |
| RL_hip d=1 | TF43000 | 60.42% | 5.04 s | 34.72% |
| RL_hip d=1 | JT71500 Student | 3.47% | 9.54 s | 90.28% |
| RL_thigh d=0.8 | TF43000 | 96.53% | 3.28 s | 3.47% |
| RL_thigh d=0.8 | JT71500 Student | 26.39% | 8.35 s | 72.22% |
| RR_calf d=0.8 | TF43000 | 95.14% | 1.00 s | 2.78% |
| RR_calf d=0.8 | JT71500 Student | 36.81% | 6.18 s | 59.72% |

The Student often continues moving without meeting the simultaneous 1-second
vx/yaw recovery criterion. This supports a post-failure control deficit, but
does **not** yet distinguish slow fault inference from persistent bias or
oscillation. Median recovery time is conditional on recovery and should not be
compared alone.

## Focused step traces

`evaluate_teacher243_failure_onset.py` now optionally writes per-step JSONL
for selected environment IDs without changing the canonical episode output.
Example diagnostic: rates `[0.8, 1.0]`, onset `[5.0]`, 4 replicates gives 96
environments. IDs `13, 14, 22` are respectively RL_hip d=1, RL_thigh d=0.8,
RR_calf d=0.8 in replicate 0; IDs `37, 38, 46` repeat them in replicate 1.
Run Teacher and Student in separate jobs, with the same seed and commands.
This is a **small diagnostic**, not the full P5 evaluation.

A one-replicate CPU trace smoke was run with seed 1 and saved under
`logs/evaluations/jt-recovery-diagnostic/`. It confirms the trace pipeline
works, but `RR_calf d=0.8` showed physically implausible tens-of-m/s velocity
**before** fault onset for both policies. Do not use that CPU sample or the
CPU traces as a performance comparison; rerun the selected cells on GPU once
the ongoing GPU evaluation is finished.

```bash
PATH=/home/jihun/Capstone2/miniconda3/envs/WIM/bin:$PATH \
PYTHONPATH=/home/jihun/legged_gym:/home/jihun/Capstone2/isaacgym/python \
PYTHONNOUSERSITE=1 \
python -m legged_gym.scripts.evaluate_teacher243_failure_onset \
  --task a1_official_wim_jt_failure_fullrange_onset \
  --checkpoint-path logs/jt_wim243_failure_fullrange_onset/Aug14_00-37-37_/model_71500.pt \
  --policy-mode student --seed 1 --headless \
  --rates 0.8 1.0 --onset-seconds 5.0 --replicates-per-condition 4 \
  --command-x 0.5 --post-seconds 20 \
  --trace-env-ids 13 14 22 37 38 46 \
  --trace-output logs/evaluations/jt-recovery-diagnostic/student_steps.jsonl \
  --output logs/evaluations/jt-recovery-diagnostic/student_episodes.jsonl
```

Use TF43000 with task `a1_official_wim_teacher243_failure_fullrange`,
`--policy-mode teacher`, and distinct `teacher_*` output paths for the paired
run. Inspect vx/yaw/action near the onset and whether the stable criterion
fails from delay, persistent bias, or oscillation. Do not use the final JT
checkpoint's Teacher path as a reference: it is incompatible with its actor.

## Frozen-TF student baseline

Separate task `a1_official_wim_frozen_tf_student_onset` uses the same
environment as canonical JT. Start from TF43000. The Teacher encoder, actor,
and action standard deviation stay frozen; the Student CNN and critic train.
For the first 2,000 iterations after warm start, Teacher controls actions
(`alpha=0`) while Student learns its latent. Over the next 8,000 iterations,
`alpha` reaches 1. Adaptation `beta=1` throughout. These schedule lengths are
project choices for a diagnostic baseline, not paper-prescribed values.

The actor stays fixed, so this test answers: can history inference reproduce
enough of TF43000 behavior without changing the policy? A poor result alone
would not prove that fault information is absent: a fixed actor may be brittle
to imperfect Student latents. Compare its Student-only policy using the same
P5 protocol. Only then decide between better fault inference/sampling and
changing the joint-optimization schedule.

After the trace diagnosis and GPU-resource gate, a first 10,000-iteration
diagnostic run can be started with:

```bash
PATH=/home/jihun/Capstone2/miniconda3/envs/WIM/bin:$PATH \
PYTHONPATH=/home/jihun/legged_gym:/home/jihun/Capstone2/isaacgym/python \
PYTHONNOUSERSITE=1 \
python -u -m legged_gym.scripts.train \
  --task a1_official_wim_frozen_tf_student_onset \
  --headless --sim_device cuda:0 --rl_device cuda:0 \
  --num_envs 4096 --max_iterations 10000 --seed 1 \
  --resume \
  --load_run /home/jihun/legged_gym/logs/official_wim_teacher243_failure_fullrange_fromscratch/Aug13_03-44-07_seed1-50000iter-fromscratch-fullrange \
  --checkpoint 43000
```

The terminal checkpoint is global iteration 53000; evaluate intermediate
Student-only checkpoints before extending toward a 30,000-iteration matched
budget. Use a copy of the P5 protocol with the baseline checkpoint SHA and
the canonical JT evaluation task name, because both policy tensor contracts
are identical. Do not select by training reward alone.

CPU-only 24-env, 2-iteration pipeline smoke finished under
`logs/frozen_tf_student_smoke/Sep21_15-20-55_cpu_20260921_v2/`. Comparing
`model_43000.pt` and `model_43002.pt` showed unchanged Teacher encoder (8/8
parameter tensors), actor (8/8), and action std (1/1); Student encoder (10/10)
and critic (8/8) changed. This verifies the freeze/update wiring, not policy
performance. An earlier smoke used the wrong installed package when invoked
by a script path; use `python -m` with the explicit `PYTHONPATH` above.

No production run should start while another GPU evaluation/training job is
active. The source TF checkpoint, JT71500, and frozen P5 protocol must remain
unchanged.
