# P5 final evaluation and generalization

## Current gate

P0--P4 are complete.  P5 is not a new-training phase: it first closes the
same-protocol evidence gap between the canonical privileged Failure Teacher and
the deployed history Student.

## P5-A: canonical paired comparison

Frozen protocol:

- Teacher: `TF43000`, privileged inference from the independent Failure Teacher
  checkpoint.
- Student: `JT71500`, Student-only history inference from the selected
  deployment checkpoint.
- Diagnostic only: `JT45000`; it must not replace `JT71500` based on viewer
  appearance or training reward.
- Fixed failure: 3 seeds, 12 joints, six degradation rates, 16 replicates, and
  1,000 policy steps.
- Random onset: 3 seeds, 12 joints, five nonzero degradation rates, onset at
  2/5/10 seconds, 16 replicates, and a 20-second post-failure horizon.
- Command: fixed forward velocity of 0.5 m/s.

The source of truth is
`legged_gym/evaluation/p5_paired_protocol.json`.  The runner verifies checkpoint
SHA-256 values before launching any simulation and writes one atomic JSONL
artifact per model, seed, and phase.

Run a minimal non-scientific smoke:

```bash
PYTHONPATH=/home/jihun/legged_gym:/home/jihun/Capstone2/isaacgym/python \
/home/jihun/Capstone2/miniconda3/bin/conda run -n WIM --no-capture-output \
python -m legged_gym.scripts.run_p5_paired_evaluation --smoke --resume
```

Run the required canonical Teacher/Student comparison:

```bash
PYTHONPATH=/home/jihun/legged_gym:/home/jihun/Capstone2/isaacgym/python \
/home/jihun/Capstone2/miniconda3/bin/conda run -n WIM --no-capture-output \
python -u -m legged_gym.scripts.run_p5_paired_evaluation \
  --models tf43000 jt71500 --resume
```

Summarize only after all required artifacts exist:

```bash
PYTHONPATH=/home/jihun/legged_gym \
/home/jihun/Capstone2/miniconda3/bin/conda run -n WIM --no-capture-output \
python -m legged_gym.scripts.summarize_p5_paired_evaluation
```

Primary metrics are survival, stable-window recovery, command velocity RMSE,
yaw-rate RMSE, vertical velocity RMS, contact-gated foot slip, forward
progress, and action saturation.  Candidate-minus-Teacher deltas are reported;
positive is favorable only for survival/recovery/progress and unfavorable for
error, slip, vertical motion, and saturation.

## P5-B: history and adaptation evidence

Retain the completed actual/zero/shuffled history intervention.  Add delayed or
masked history only if a narrower adaptation-time claim is required.  A
shuffled-history collapse supports robot-specific temporal-information use; it
does not by itself identify the semantics of individual latent dimensions.

## P5-C: command and terrain generalization

This is a separate protocol from P5-A.

- Do not call the current randomized terrain column index a terrain family.
  With curriculum disabled, terrain cells are sampled independently and the
  column label does not identify slope, stairs, or obstacles.
- A terrain claim requires an explicitly stratified evaluator with named
  terrain families and difficulty levels.
- In-distribution command-grid evaluation and out-of-range command stress tests
  must be labelled separately.
- The viewer's integrated command trace is not a world-frame ground-truth path.
  Base path RMSE requires a separately defined path and a high-level path
  follower and is optional, not a replacement for command-velocity RMSE.

## P5-D: presentation artifacts

- Student-only 30-second videos with the failed actuator colored.
- Normal-to-failure-to-recovery sequence under random onset.
- Same-condition Teacher/Student side-by-side videos.
- Joint-by-degradation heatmaps and fixed/onset metric tables generated from
  the canonical JSONL artifacts.

## P6 entry gate

Do not start P6 until the required P5-A comparison is complete and P5-C has an
explicit complete, limited, or omitted decision.  P6 freezes the configs,
checkpoint hashes, commands, result summaries, videos, and paper claim scope.
