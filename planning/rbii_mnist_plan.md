# Separate RBII-MNIST Feasibility Prototype (Isolated Under `rbii/`)

## Summary
Build a new MNIST-specific RBII implementation in a subpackage under `dreamcoder/domains/rbii/` so the current char-sequence RBII path remains untouched.
The first milestone will use DreamCoder enumeration now, fast CPU defaults, split-MNIST with returning contexts, log-loss-based selection, and report both log-loss and top-1 accuracy from day one.

## Scope Boundaries
1. Keep existing simple RBII files behaviorally unchanged:
   - `dreamcoder/domains/rbii/rbii_loop.py`
   - `dreamcoder/domains/rbii/rbii_primitives.py`
   - `dreamcoder/domains/rbii/rbii_state.py`
   - `dreamcoder/domains/rbii/rbii_test.py`
2. Add all new logic under a separate subpackage:
   - `dreamcoder/domains/rbii/mnist/`
3. Phase 1 excludes prototype->program recall index (`knnProg`/prestige); design interfaces so it drops in as Phase 2.

## New Files (Additive Only)
1. `dreamcoder/domains/rbii/mnist/__init__.py`
2. `dreamcoder/domains/rbii/mnist/types.py`
3. `dreamcoder/domains/rbii/mnist/state.py`
4. `dreamcoder/domains/rbii/mnist/model.py`
5. `dreamcoder/domains/rbii/mnist/stream.py`
6. `dreamcoder/domains/rbii/mnist/primitives.py`
7. `dreamcoder/domains/rbii/mnist/likelihood.py`
8. `dreamcoder/domains/rbii/mnist/loop.py`
9. `dreamcoder/domains/rbii/mnist/metrics.py`
10. `dreamcoder/domains/rbii/mnist/rbii_mnist_test.py`

## Public Interfaces / Types
1. `tmnist_state = baseType("mnist_state")`
2. `tmnist_pred = baseType("mnist_pred")`
3. `MNISTPrediction` dataclass:
   - `kind: "dist" | "label"`
   - `dist: Optional[torch.Tensor]`
   - `label: Optional[int]`
4. `MNISTEvalState` protocol:
   - `timestep`
   - `current_x()`
   - `label_at(i)`
   - `program_at(k)`
   - `context_id()`
5. `MNISTRBIIConfig` defaults:
   - `pool_target_size=4`
   - `validation_window=64`
   - `min_time=32`
   - `enum_timeout_s=1.0`
   - `eval_timeout_s=0.05`
   - `upper_bound=40.0`
   - `budget_increment=1.5`
   - `max_frontier=12`
   - `learning_rate=1e-3`
   - `label_smoothing_eps=1e-3`
   - `evict_max_bits=3.5`
   - `device="cpu"`

## Enumeration + Scoring Design
1. Request type for tasks: `arrow(tmnist_state, tmnist_pred)`.
2. Enumeration uses `enumerateForTasks(...)` with a custom likelihood model.
3. `MNISTLogLossLikelihoodModel.score(program, task)`:
   - evaluate program on each window example
   - convert prediction to proper 10-way distribution
   - compute mean `-log2 p(y_true)`
   - return `(True, -mean_bits)` when finite else `(False, -inf)`.
4. Output conversion rules:
   - `dist` output clamped/renormalized
   - `label` output converted to smoothed one-hot (`eps`).
5. Frontiers ranked by posterior with log-loss likelihood (no exact-match gate).

## RBII-MNIST Loop Behavior
1. Stream step uses known `(x_t, y_t, context_t)` but prediction sees `x_t` and past state only.
2. Prediction uses weighted mixture over active pool distributions.
3. Per step update:
   - compute per-program loss on current example
   - evict programs above `evict_max_bits` or invalid output
   - observe true label into state
   - refill via enumeration if pool below target.
4. Refill task uses recent causal views from `validation_window`.
5. New programs compiled, added to frozen store, admitted with initial weight.
6. Event logging to `experimentOutputs/rbii_mnist_events/`.

## Split-MNIST Returns Stream (Default)
1. Dataset source: torchvision MNIST at `data/mnist`, auto-download if missing.
2. Default contexts: `A={0,1,2,3,4}`, `B={5,6,7,8,9}`.
3. Default schedule: `A -> B -> A -> B`.
4. Fast CPU default: 150 steps per segment (600 total).

## Metrics and Acceptance Criteria
1. Always report:
   - online top-1 accuracy
   - online mean log-loss bits
   - per-context summary.
2. Reacquisition metrics for return episodes:
   - delay to recover near-prior performance
   - excess bits during recovery window.
3. Acceptance for Phase 1:
   - quick CPU run completes in minutes
   - metrics file emitted
   - at least one return episode reports finite reacquisition delay.

## Non-Regression Test Plan
1. Existing command unchanged:
   - `python -m dreamcoder.domains.rbii.rbii_test`
2. New command:
   - `python -m dreamcoder.domains.rbii.mnist.rbii_mnist_test --quick`
3. Existing RBII logs remain under `experimentOutputs/rbii_program_events/`.
4. MNIST logs under `experimentOutputs/rbii_mnist_events/`.

## Phase 2 Hook (Planned, Not Implemented in Milestone 1)
1. Add `RecallIndex` interface in MNIST package with no-op default now.
2. Later implement prototype->program retrieval and prestige updates without changing loop signatures.

## Assumptions and Defaults
1. Repo may be dirty in unrelated areas; do not modify existing RBII simple-path files.
2. `torch` / `torchvision` are available locally.
3. CPU-first defaults; GPU opt-in via config only.
4. Log-loss is the selection objective; accuracy tracked as secondary metric.
