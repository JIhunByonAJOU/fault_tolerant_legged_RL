# IEIE 2026 submission workflow

Final policy: JT77500 Student-only. Scientific scope: single-joint torque-multiplier loss in Isaac Gym A1 simulation. Work on `dev`; never move `paper_baseline`.

## Roles and responsibilities

- **Coordinator/writer** owns the manuscript, presentation, evidence-to-claim mapping and delivery manifest. Write only after the evaluator and literature reviewer supply evidence. Record every unresolved gate.
- **Evaluator** owns the frozen evaluation protocol, checkpoint/config hashes, numerical summaries, plots and reproducibility commands. Do not retrain. Do not stop existing GPU workloads. GPU PhysX is required for reported performance.
- **Literature/rules reviewer** owns official competition rules, template requirements, prior-art evidence, public-code compatibility and verified award records. Distinguish an award from an ordinary conference presentation and an abstract from a read full paper.
- **Professor reviewer** independently challenges novelty, information fairness, metric denominators, statistical claims, failure cases and conclusions. Cite concrete manuscript/slide passages and propose corrections.

## Evidence gates

1. Freeze the checkpoint and the evaluation definition before evaluating.
2. Validate complete condition grids and paired initial-state/protocol hashes.
3. Use normal locomotion as the principal internal control. Switching the final co-adapted actor to a Teacher latent is a representation intervention, not an original Teacher-policy baseline or a performance upper bound.
4. Published numbers from incompatible robots, damage models, terrain or metrics are contextual; never use them for a numerical superiority ranking.
5. Report safety, movement and tracking together. Body-frame velocity integrals are not world-frame displacement. Unrecorded stall/backward time is unavailable, not zero.
6. Distinguish evaluation seeds from independent training seeds. Selection-associated data are exploratory evidence, not an independent final test.
7. Render both official-template manuscripts and the presentation. Inspect page count, clipped text, formula placement, fonts and anonymization.
8. Submission requires all entries in the delivery manifest to pass, including author/member eligibility and publication eligibility. A visually finished draft is not automatically submission-ready.

## File boundaries

- `results/`: compact derived evidence and manifests; exclude raw traces and checkpoints.
- `sources/`: provenance, public-code audits and template structure; do not redistribute copyrighted papers or fonts.
- `reviewer/`: claim audits and concrete review questions.
- `private/`: manuscript/presentation content, identifying metadata and working sources; Git-ignored.
- `rendered/`: DOCX/PPTX/PDF deliverables and previews; Git-ignored.
- Root scripts consume structured private content and generate editable Office Math. No hidden dependence on ignored older presentation builders.

Do not publish the manuscript or new submission-specific results before the official unpublished-work requirement has been interpreted for this case. Prepare a concrete publish allowlist; source-control publication is a separate gate from local authoring.
