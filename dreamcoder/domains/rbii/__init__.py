# dreamcoder/domains/rbii/__init__.py
"""
Minimal RBII prototype domain (Python-only) for the ec/DreamCoder repo.

Files:
- rbii_types.py      : explicit RBII task input type + state protocol
- rbii_state.py      : RBIIState + causal per-timestep state views
- rbii_primitives.py : DSL primitives + grammar builder (explicit state input)
- rbii_loop.py       : minimal RBII loop with pool eviction/refill
- rbii_loop_v2.py    : policy-factored RBII scaffold for richer scheduling/selection
"""
