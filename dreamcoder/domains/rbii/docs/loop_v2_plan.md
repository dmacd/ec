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
    - the config precomputes `alphabet_set` and `baseline_bits_per_symbol`.
    - alphabet validation/canonicalization belongs in config construction, not
      in the loss-model hot path.

12. `RBIILoopV2` requires an explicit `RBIIConfigV2`.
    - There is no meaningful zero-argument default config because `alphabet` is
      required.

13. V2 event logging and visualization use V2-native semantics only.
    - Do not collapse active-pool admission and freezing into a single `enter`
      event.
    - No backward-compatibility shim to the old V1 viz schema is required
      unless explicitly requested.

## Current Per-Step Flow

At each observed symbol:

1. Predict from weighted pool mixture.
2. Compute mixture log-loss bits for the event log.
3. Update each pool member weight with categorical log-loss bits.
4. Append observation to `RBIIState`.
5. Emit `observe` for the current timestep.
6. Emit `pool_exit` for any predictor dropped during the online update.
7. Build validation task/window (when past `min_time`).
8. Enumerate proposals (`EnumerationController`).
9. Score and weight current-step proposals (`CandidateWeightPolicy`), including
   compression-gain-vs-cost filtering.
10. Rerank pool + candidates together; keep best `pool_target_size`.
11. Emit `pool_enter` for new active predictors at timestep `t + 1` and
    `pool_exit` for predictors displaced at timestep `t`.
12. Update incumbent tracker.
13. Freeze according to `FreezePolicy` (only here can frozen store change).
14. Emit `freeze` only when a program is newly added to
    `state.best_programs`.

## V2 Event Schema

Current V2 logs are JSONL and use these events:

1. `run_start`
2. `observe`
3. `pool_enter`
4. `pool_exit`
5. `freeze`
6. `run_end`

Key schema decisions:

1. Active-pool episodes are keyed by `active_id`.
   - `program_id` may be absent until freeze.
   - `active_id` is the identifier consumed by the visualizer for episode
     spans.

2. Frozen-store additions are keyed by `program_id` and come only from
   `freeze`.

3. `observe` carries:
   - `predicted`
   - `observed`
   - `logloss_bits`
   - incumbent `active_id` / `program_id` / `program`
   This is enough for the visualizer to reconstruct sequence rows and mark the
   incumbent episode used at each timestep.

4. `pool_enter` is logged at the first timestep when a predictor is active for
   prediction (`t + 1` after the observation that admitted it).

5. `pool_exit` is logged at the last timestep for which a predictor was active
   (`t` for a predictor dropped after observing timestep `t`).

## Viz Semantics

`rbii_viz_graph.py` is now V2-specific:

1. Episode spans come from `pool_enter` / `pool_exit`.
2. Frozen-store additions come from `freeze`.
3. Incumbent usage highlighting comes from `observe.active_id`.
4. Episode labels use `@active_id` and, when frozen, also show `[#program_id]`.
5. Frozen store column uses `#program_id`.
6. When multiple pool entries/exits happen on the same timestep, the visualizer
   staggers the bracket endpoints by a few pixels to keep same-row joins
   distinguishable.

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
3. If candidate-level debugging becomes necessary, add V2-native events for
   candidate scoring/rejection rather than reviving the old V1 event names.
4. Add focused tests for:
   - no pool insertion on duplicate candidate,
   - freeze-only writes to `best_programs`,
   - global rerank behavior across multiple timesteps,
   - event-log and viz smoke coverage.
   These now exist in `dreamcoder/domains/rbii/tests/test_loop_v2.py`; extend
   that file rather than adding scattered V2 test modules unless there is a
   clear reason to split by concern.

## Guidance for Future Codex Sessions

Before changing V2 behavior, check this file and confirm whether the change is:

1. A bug fix that preserves the above invariants, or
2. An intentional policy change requested by the user.

If it is (2), update this file in the same PR/patch so the new behavior stays
documented.
