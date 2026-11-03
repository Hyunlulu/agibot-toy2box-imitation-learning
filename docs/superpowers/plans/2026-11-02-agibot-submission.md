# Agibot submission implementation plan

Goal: publish a self-contained course project with two final 5D checkpoints (training seeds101 and111) and a runnable evaluation command, and revise the presentation as requested.
Architecture: an isolated snapshot of IsaacLab v2.3.0 plus Agibot-only changes; automatic 5D-to-7D inference compatibility in the standard play entry point; portable generation/training helpers outside ignored logs.
Tech stack: Python3.11, IsaacLab/IsaacSim, robomimic, PyTorch, python-pptx.
Spec: user request dated 2026-11-02 and Week08 p.48.

Global constraints: preserve original environment/success/termination conditions and original datasets/results; final checkpoints epoch2000 training seeds101 and111; retain 23D observations; no Ant changes; only two submitted final checkpoints; do not publish to the teaching repository.
Review focus: automatic expansion only for the Agibot 5D policy; 7D compatibility; portable paths; refusal to overwrite evaluation evidence; arbitrary evaluator seed.

- [ ] Revise slides: remove old8, edit old2 assumption and old10 dimension diagram; maintain full evaluation videos; render and verify text/layout.
- [ ] Add a failing behavioral test for policy compatibility and output preservation; implement the standard play entry point compatibility and optional JSON output; run adapter tests and GPU smoke if available.
- [ ] Move tested generation/training helpers to scripts/imitation_learning/agibot_action5 and resolve paths relative to project; validate source and checkpoint hashes.
- [ ] Add the two final checkpoints, configs, results, README, evaluator commands and dependency notes. Package retraining data separately; preserve originals.
- [ ] Review and verify the exact upload tree; stage/commit when author/account available. Publish only to the user's connected account and verify the remote files.

Progress ledger:
- Isolated baseline copied from the original HEAD; only Agibot changes copied. Original working tree unchanged.
- GitHub connection missing; connection requested. Independent preparation continues.

- Ruling: isolated archive snapshot plus independent Git repository used instead of a linked worktree, so publication remotes and branches cannot alter the mixed original repository.
- Tests: existing adapter8 passed; new compatibility/result tests observed failing before implementation; combined14 now pass.
- Presentation revised to14 slides; PDF/native-text/embedded-video verification passed.

- Final review: independent reviewer identified canonical HDF5 output alias/duplicate-option preservation and unenforced16-env evidence. Fixed both with observed failing tests then green.
- Ruling: normalization finding elevated to Important because nonzero inserted rotations violate the explicit execution invariant; reject normalization flags for this raw-action5D model. Regression test now passes.
- Final suite:15 adapter/compatibility tests +10 conversion/rollout/generation tests =25 passed. GPU smoke3 matched original outcomes/steps exactly.

- 2026-11-03 LMS update: upload final seed101 and seed111 weights only; add both evaluation commands with separate result/video paths. This supersedes the earlier one-checkpoint instruction. Runtime code is unchanged.
