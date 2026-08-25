# [JT] WIM243 FullRange random-onset training

## Selected parent

- Parent role: `[TF]` Teacher FailureEnv.
- Canonical selection: `[TF]-SCRATCH-FULL-43000`.
- Checkpoint: `logs/official_wim_teacher243_failure_fullrange_fromscratch/Aug13_03-44-07_seed1-50000iter-fromscratch-fullrange/model_43000.pt`.
- SHA-256: `944a697abb30dfc8023e15544d0909acfcdaa4d8c4c0f930656847398f150635`.
- The selection is fixed by
  `canonical_tf_selection_20260813/selection_summary.json`, not by final
  iteration number.

## Tensor contract

- Current policy observation: WIM 235D = current 48D + terrain heights 187D.
- Teacher input: privileged45 -> MLP -> teacher latent8.
- Student input: reset-safe history 50 x current 48D -> RMA-lineage 1-D CNN -> student latent8.
- Policy input: current WIM235 + fused latent8 = 243D.
- PPO storage observation: 235 + 50x48 = 2635D. The actor does not consume all
  2635 values directly; the CNN compresses the history before the policy MLP.

## Training contract

- Separate task: `a1_official_wim_jt_failure_fullrange_onset`.
- One actuator, `d={0,.2,.4,.6,.8,1.0}`.
- Every episode starts intact. Nonzero failure is applied once at a uniformly
  sampled 2--10 second onset.
- Warm-start teacher encoder, actor, critic, and action standard deviation from
  the selected `[TF]` checkpoint. Student CNN is new; optimizer is reset because
  the parameter topology changed.
- Joint schedule over 10,000 iterations:
  `alpha: 0 -> 1`, `beta: 1 -> 0`, both linear. Saving specifies only the
  direction of these schedules, not their exact numeric schedule.
- Training: 4,096 envs, 24 steps/env, 4 minibatches, 5 PPO epochs, seed1,
  save every 500. Total new transitions: 983,040,000.

## Required evaluation

Do not select the final checkpoint by iteration number. Rank saved checkpoints
using:

1. student-only (`alpha=1`) fixed FullRange matrix;
2. random-onset survival and stable-window recovery;
3. student-teacher latent L2 and action disagreement;
4. normal `d=0` gait preservation;
5. zero/shuffled history intervention.

The run is `[JT]`, not `[SB]` or `[SF]`, because teacher encoder, student CNN,
and policy are optimized together. `[SB]` and `[SF]` are reserved for
student-only evaluation or separately trained student baselines.

## Readiness

- Focused CPU tests: 22/22 PASS.
- Student-only inference is independent of privileged input and exactly matches
  the fused policy at `alpha=1`.
- GPU inference smoke: 64 environments x 1,000 steps PASS with finite 2635D
  stored observations, privileged45, actions, and rewards.
- Production run completed at
  `logs/jt_wim243_failure_fullrange_onset/Aug14_00-37-37_`:
  30,000 new iterations, 2,949,120,000 transitions, and 61 checkpoints.
- The selected Student checkpoint is `model_71500.pt`, SHA-256
  `1887f9b5abeea72ca87e0c92172687462714793f91c9011490a170cffe42e1f0`.
- Student-only fixed FullRange evaluation (3 seeds, 3,456 robot episodes):
  survival 95.60%, d=1 survival 89.58%, vx RMSE 0.102 m/s, yaw-rate RMSE
  0.127 rad/s.
- Student-only random-onset evaluation (3 seeds, 8,640 robot episodes):
  post-failure survival 94.80%, stable-window recovery 81.66%, and d=1
  survival/recovery 88.95%/45.14%.
- History interventions are causal: survival falls from 95.60% with real
  history to 79.98% with zero history and 3.24% with history shuffled across
  robots.
- Limitation: after the Student-only portion of joint optimization, the shared
  actor is no longer compatible with the privileged Teacher latent (fixed
  Teacher-reference survival 4.02%). The selected deployment Student is valid,
  but the final checkpoint does not preserve a usable Teacher inference path.
- Durable selection evidence:
  `logs/evaluations/jt-wim243-student-selection/Aug14_00-37-37_selection_summary.json`.
