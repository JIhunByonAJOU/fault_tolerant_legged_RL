# JT Student adaptation-loss floor ablation

Status: code and CPU checks only; **no training or performance result yet**.

## Question and single change

The canonical JT run starts from TF43000 and linearly changes `alpha` from 0 to
1 and `beta` from 1 to 0 during its first 10,000 iterations. Its remaining
20,000 iterations therefore have no teacher-latent adaptation-loss gradient.
This may contribute to the observed Student recovery gap, but has not been
established as its cause.

The separate `a1_official_wim_jt_beta_floor_onset` profile keeps the same
environment, network, TF43000 warm start, alpha schedule, PPO, and reward. Its
only scientific change is an **experimental** beta floor of 0.1:

`beta = 0.1 + 0.9 * (1 - alpha)`.

The original task and checkpoints are unchanged. The value 0.1 is a project
ablation choice, not a number reported by *Saving the Limping*. It might help,
hurt, or leave recovery unchanged; latent matching is not the final outcome.

## Before a production run

1. Trace `RL_hip d=1`, `RL_thigh d=0.8`, and `RR_calf d=0.8` to distinguish
   slow adaptation from persistent tracking bias.
2. Check that no other GPU training or evaluation is active.
3. Use TF43000, **not** JT71500, as the source for this matched ablation.
   JT71500 is suitable for Student-only fine-tuning, but its teacher path is
   no longer a reliable supervised target.

Example command, to run only after these gates:

```bash
PATH=/home/jihun/Capstone2/miniconda3/envs/WIM/bin:$PATH \
PYTHONPATH=/home/jihun/legged_gym:/home/jihun/Capstone2/isaacgym/python \
PYTHONNOUSERSITE=1 \
python -u -m legged_gym.scripts.train \
  --task a1_official_wim_jt_beta_floor_onset \
  --headless --sim_device cuda:0 --rl_device cuda:0 \
  --num_envs 4096 --max_iterations 30000 --seed 1 \
  --resume \
  --load_run /home/jihun/legged_gym/logs/official_wim_teacher243_failure_fullrange_fromscratch/Aug13_03-44-07_seed1-50000iter-fromscratch-fullrange \
  --checkpoint 43000
```

After training, select a Student checkpoint using the same fixed/onset
candidate protocol as the canonical JT run. Then add its checkpoint path and
SHA to a **copy** of `p5_paired_protocol.json` and evaluate `policy_mode=student`
with the canonical JT task name: both tasks have the same evaluation
environment and policy tensor contract. Do not edit the frozen P5 protocol.
Compare recovery, d=1 recovery, vx/yaw RMSE, and survival on matched seeds;
report joint-by-severity cells and held-out seeds before claiming an
improvement. A higher W&B reward or smaller latent L2 alone is not sufficient.
