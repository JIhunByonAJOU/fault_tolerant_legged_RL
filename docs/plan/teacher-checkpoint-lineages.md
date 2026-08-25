# Teacher checkpoint lineages

Use the canonical IDs in reports, evaluation folders, and future task names.
The raw checkpoint iteration is metadata, not the experiment identity.

| Canonical ID | Checkpoint | Initialization | Failure exposure | Role |
|---|---|---|---|---|
| `[TB]-WIM243-22500` | `model_22500.pt` | WIM235 policy expanded to Teacher243 and trained in BaseEnv | none | selected Teacher BaseEnv |
| `[TF]-WARM-MILD-25500` | `model_25500.pt` | warm-start from `[TB]-WIM243-22500` | episode-start, `d={0,.2,.4,.6,.8}`, +3k | early Teacher FailureEnv comparison |
| `[TF]-WARM-FULL-32500` | `model_32500.pt` | warm-start from `[TB]-WIM243-22500` | episode-start, `d={0,.2,.4,.6,.8,1}`, +10k | provisional selected Teacher FailureEnv |
| `[TF]-SCRATCH-FULL-50000` | `model_50000.pt` | from scratch | episode-start, `d={0,.2,.4,.6,.8,1}`, 50k | independent initialization control |
| `[TF]-WARM-ONSET-32500` | not trained | proposed warm-start from `[TB]-WIM243-22500` | random mid-episode onset, FullRange, +10k | optional ablation; not currently justified |

Future identifiers use `[SB]` for Student BaseEnv, `[SF]` for Student
FailureEnv, and `[JT]` for joint teacher-student training.

## P2.5 sudden-onset evaluation

- Evaluation ID: `EVAL-P25-ONSET-FT-WARM-FULL-32500-v1`
- This is inference of `[TF]-WARM-FULL-32500`, not a new checkpoint.
- Protocol: onset `Tf={2,5,10}s`, then a common 20-second post-failure horizon;
  12 actuators x `d={.2,.4,.6,.8,1}` x 3 seeds x 16 replicates.
- Artifact: `logs/evaluations/teacher243-failure-random-onset/model32500_onset2-5-10s_20260813/`.
- Overall post-onset survival: 94.50%; `d=1`: 88.48%.
- Strict stable-window recovery: 85.78% overall, conditional mean 3.45 s;
  `d=1`: 50.41%, conditional mean 6.22 s.
- Decision: retain `[TF]-WARM-FULL-32500` as the provisional FailureEnv Teacher.
  Do not train `[TF]-WARM-ONSET-32500` solely to improve the privileged teacher.
  Use random-onset episodes when implementing the history-based Student and
  joint T-S path, where observing the normal-to-failure transition is essential.
