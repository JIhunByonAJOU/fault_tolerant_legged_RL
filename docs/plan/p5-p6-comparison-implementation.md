# P5/P6 comparison implementation protocol

This implementation follows the approved comparison contract in
`docs/submission/ieie2026/private/benchmark/comparison_contract_v1.md`. It adds
two task profiles and does not alter the canonical JT task, its checkpoints, or
the frozen P5 evaluator.

## B1: current-repeat / temporal-history control

Task `a1_official_wim_jt_history_free_onset` uses the canonical random-onset
environment, current 235-D observation, privileged 45-D label, Actor, Critic,
fault distribution, rewards, PPO, and alpha/beta schedule. Its only observation
intervention is that the Student's 50x48 tensor contains 50 copies of the
current 48-D frame at every step, including the first observation after reset.
The policy still receives the current observation and previous action, so this
control is not described as memoryless. The Student CNN width is 64 and the
initial learning rate is `1e-5`.

## B2: RMA-style internal two-stage control

Task `a1_official_wim_separate_student_onset` compares the joint-training
pipeline with a separate adaptation pipeline. The original TF Actor, Critic,
Teacher encoder, and action standard deviation are frozen. The current Student
latent drives the frozen Actor during on-policy rollout. The Teacher latent is
computed only as a `no_grad` supervision label.

Only the Student encoder belongs to Adam. Each 4096-env x 24-step rollout is
reused for 5 epochs x 4 minibatches, giving 20 supervised updates at fixed
`1e-5` with gradient clipping at 1.0. The objective is the batch mean of the
unsquared per-sample latent L2 norm. Student-vs-Teacher action MSE is recorded as
a diagnostic with weight zero. PPO actor loss, value loss, entropy, PPO KL, and
adaptive-KL learning-rate changes do not enter the optimizer or metrics.

This is an `RMA-style internal two-stage control`, not an exact RMA
reproduction. It uses the project's 50 Hz WIM random-onset actuator-fault
environment and unsquared latent loss. RMA's on-policy Student rollout rationale
is documented in the local RMA paper record, lines 289-305 and Algorithm 1.
Because rollout, optimizer, and schedule differ from JT, B2 is a pipeline
comparison rather than a single-variable ablation.

The designated B2 pipeline fixes Adam at `1e-5`; canonical JT retains adaptive
KL, while original RMA reports learning rate `5e-4`, squared MSE, and 80 million
adaptation-training steps. Matching B1/B2 transition exposure therefore does
not establish an optimal RMA configuration or support an optimal-RMA claim.
B2 records latent loss, gradient norm, completed updates, fixed learning rate,
action-MSE diagnostic, and the Student parameter-step L2 norm at every rollout.

## Common initialization and budget ledger

Both tasks resolve the normal resume route to TF43000 and require SHA256
`944a697abb30dfc8023e15544d0909acfcdaa4d8c4c0f930656847398f150635`.
They reject another path or hash and reset the optimizer. A per-run comparison
manifest records the source, fresh Student hash, width, and initial
Teacher/Actor/Critic/action-std hashes. With the same seed, their fresh width-64
Student tensors are identical before training.

Comparison checkpoints carry a method profile ID, original TF SHA, seed,
Student width, schedule and budget, normalized full environment/training digest,
source-code digest, optimizer state, and initialization hashes. A loader accepts
the exact TF43000 as a fresh warmstart or a checkpoint from its own profile. It
rejects canonical JT and cross-method B1/B2 checkpoints even when tensor shapes
match. B2 also verifies that Actor, Critic, Teacher encoder, and action std match
their saved frozen hashes after load.

The production budget is 34,501 rollout batches from global origin 43,000.
The runner saves after the update under the loop label, so the fixed ledger is:

| checkpoint | completed rollout batches |
|---:|---:|
| `model_53000.pt` | 10,001 |
| `model_63000.pt` | 20,001 |
| `model_77500.pt` | 34,501 |

The runner's terminal `model_77501.pt` may duplicate the selected final state;
the predeclared comparison checkpoint is `model_77500.pt`. Seeds 2 and 3 are
launch-time overrides. Existing JT77500 seed-1 provenance stays fixed, and no
paper superiority claim changes before held-out results exist.

Each comparison checkpoint stores `next_iteration` and
`completed_new_batches`. Thus `model_53000.pt` resumes from 53,001 with 24,500
batches left, and analogous resumes use `77501 - next_iteration`. The terminal
file records the same next iteration and state as the selected final without an
extra rollout batch. Canonical checkpoint semantics remain unchanged.

Production lineage always normalizes its scientific target to 34,501 batches;
an interrupted invocation records its actual remaining budget separately. For
example, a resume from `model_53000.pt` may invoke `--max_iterations=24500`
without changing the production configuration digest.

Short comparison runs require explicit `--comparison_pilot`. A pilot must start
from the exact TF43000, uses 1–100 batches, and records `run_class=pilot` plus a
dynamic target `43000+N`. Pilot checkpoints are marked
`pilot_only_not_production`, cannot resume, and are rejected by production
loaders. Without the flag, a fresh run must retain the full 34,501-batch target.

Optional `--shared_gpu_step_sleep_ms` and
`--shared_gpu_minibatch_sleep_ms` pacing values default to zero. Positive values
sleep after each collected control step and optimizer minibatch, respectively;
they are recorded in resolved config and metrics without changing environment
count, rollout length, minibatch count, gradients, or checkpoint budget.
