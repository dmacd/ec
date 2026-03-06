# RBII Loop V2 Plan and Session Decisions

Last updated: 2026-03-05

This document records the agreed design decisions for `RBIILoopV2` in
`dreamcoder/domains/rbii/rbii_loop_v2.py` so future Codex sessions do not
reintroduce reverted behavior.

## Scope

`RBIILoopV2` is a policy-factored scaffold for RBII with three interfaces:

1. `EnumerationController`: produce candidate proposals.
2. `CandidateWeightPolicy`: score/weight candidates.
3. `FreezePolicy`: decide when incumbents are frozen.

The active pool and frozen store are intentionally separate concerns.

## Hard Design Decisions (Do Not Revert Unless Explicitly Requested)

1. No hard minimum-weight cutoff in the active pool.
   - Low-weight predictors stay in the pool unless displaced by reranking or
     failing execution.

2. No forced normalization of pool weights after each step.
   - Weights remain on their natural multiplicative scale.
   - Any normalization must be an explicit future policy choice, not default loop
     behavior.

3. `CandidateProposal.witness_bits` is required and non-optional.
   - `None` is not treated as a normal case.
   - Bottom-solver proposals should raise if witness bits cannot be derived.

4. There is no persistent candidate buffer in V2.
   - Candidates are proposal batches for the current step only.
   - Do not reintroduce a buffer-capacity hyperparameter unless explicitly
     requested.

5. Pool update is a global rerank of:
   - existing pool members (with current weights), and
   - scored candidate admissions (with insertion weights).
   Then keep top `pool_target_size`.

6. Do not add a candidate to the pool if that program is already present in the
   current pool.
   - Program identity is currently keyed by `str(program)`.

7. Candidate admission must not write directly to `state.best_programs`.
   - Entering the active pool is not the same as being frozen.

8. `_apply_freeze_policy` is the only path that adds programs to
   `state.best_programs`.
   - Frozen-store writes go through `state.add_best_program(...)`.
   - The loop keeps an internal key->program_id index for already frozen entries.

9. No separate authoritative frozen program list in the loop.
   - The authoritative frozen store is `RBIIState.best_programs`.

10. Candidate filtering must enforce the manuscript criterion:
    - admit only candidates whose compression gain exceeds their bit-cost.
    - current policy uses `compression_gain = baseline_bits - loss_bits` and
      requires `compression_gain + compression_gain_slack_bits > witness_bits`.
    - `compression_gain_slack_bits` is a global permissiveness control (higher is
      more permissive).

11. Default loss mode is categorical log-loss.
    - pool updates are multiplicative Bayes-style updates:
      `weight <- weight * 2^(-loss_bits)`.
    - baseline bits use a uniform categorical baseline:
      `window_len * log2(|alphabet|)`.
    - `alphabet` must be explicitly provided in config as a validated
      `tuple[str, ...]`.
    - alphabet validation/canonicalization belongs in config construction, not
      in the loss-model hot path.

## Current Per-Step Flow

At each observed symbol:

1. Predict from weighted pool vote.
2. Update each pool member weight with categorical log-loss bits.
3. Append observation to `RBIIState`.
4. Build validation task/window (when past `min_time`).
5. Enumerate proposals (`EnumerationController`).
6. Score and weight current-step proposals (`CandidateWeightPolicy`), including
   compression-gain-vs-cost filtering.
7. Rerank pool + candidates together; keep best `pool_target_size`.
8. Update incumbent tracker.
9. Freeze according to `FreezePolicy` (only here can frozen store change).

## Notes on Keying and Duplicates

The current duplicate logic uses `str(program)` as a key in both pool dedupe and
frozen-store indexing. This is pragmatic and may conflate semantically different
program objects with identical printed forms. Keep this in mind when evaluating
future duplicate policies.

## Bottom Solver Integration Notes

Current bottom-solver adapter is intentionally narrow:

1. It proposes programs plus witness metadata.
2. It does not do final selection policy.
3. Weighting/reranking/freezing decisions happen in loop policies, not inside the
   enumerator.

## Open Work / Next Milestones

1. Replace simple candidate weight formula with the exact policy intended for the
   experiment (while preserving global rerank semantics).
2. Introduce explicit transformer-search enumerator implementation behind the
   same `EnumerationController` interface.
3. Add event logging for V2 decisions (candidate score, global rank position,
   freeze decision reason) to support debugging and timeline visualization.
4. Add focused tests for:
   - no pool insertion on duplicate candidate,
   - freeze-only writes to `best_programs`,
   - global rerank behavior across multiple timesteps.

## Guidance for Future Codex Sessions

Before changing V2 behavior, check this file and confirm whether the change is:

1. A bug fix that preserves the above invariants, or
2. An intentional policy change requested by the user.

If it is (2), update this file in the same PR/patch so the new behavior stays
documented.
