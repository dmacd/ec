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

## Current bottom-solver multiprocessing failure (diagnosis)

When running `rbii_test` with:

- `enum_solver="bottom"`
- `enum_bottom_compile_me=False`
- `enum_cpus > 1`

the run can fail in multiprocessing result transport with:

```text
MaybeEncodingError: ... Can't pickle local object '_succ_char.<locals>.f'
```

### Why this happens

`solveForTask_bottom` parallelizes with `multiprocessing.Pool` and must pickle worker results to send them back to the parent process.  
Some returned programs include the `succ_char` primitive value implemented as a nested closure (`_succ_char.<locals>.f`), which is not picklable by default in this path.

### Practical workaround

Use serial bottom enumeration (`enum_cpus=1`) or switch to the Python enumerator mode.

If you want to keep bottom solver + multiprocessing, `succ_char` should be reworked to use a picklable top-level callable (or equivalent serialization-safe primitive implementation).
