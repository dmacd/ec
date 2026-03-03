# RBII Domain Quickstart

This directory contains the RBII character-sequence prototype:

- core loop: `dreamcoder/domains/rbii/rbii_loop.py`
- primitives/grammar: `dreamcoder/domains/rbii/rbii_primitives.py`
- runner script: `dreamcoder/domains/rbii/rbii_test.py`
- visualization tool: `dreamcoder/domains/rbii/rbii_viz_graph.py`

## Run the RBII test script

From repo root (`ec/`):

```bash
./ve/bin/python -m dreamcoder.domains.rbii.rbii_test
```

Outputs are written to:

- `experimentOutputs/rbii_program_events/run_XXXX/`

That run directory includes:

- `*.jsonl` event logs
- `*.svg` visualizations (auto-rendered at end of run)
- optional enumeration debug files (Python-enumerator mode only)

## Run RBII bottom-solver integration tests

```bash
./ve/bin/python -m pytest -q dreamcoder/domains/rbii/tests/test_bottom_solver_integration.py
```

## Bottom-solver multiprocessing notes

Parallel bottom solver (`enum_solver="bottom"`, `enum_cpus > 1`) sends worker
results through Python multiprocessing pickling. Two classes of serialization
issues mattered for RBII:

- non-picklable primitive values (fixed for `succ_char`)
- non-picklable runtime caches inside state views (`compiled_programs` lambdas)

These are now addressed in the RBII domain by:

- defining `succ_char` with a top-level picklable callable class
- serializing `RBIIState` without compiled lambda caches and recompiling on load

If a new primitive adds local/nested closures as primitive values, parallel
bottom solver can regress. See:

- `dreamcoder/domains/rbii/tests/test_bottom_solver_integration.py`
