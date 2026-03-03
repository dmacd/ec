# bin/rbii_test.py
# try:
#     import binutil  # required to import from dreamcoder modules
# except ModuleNotFoundError:

import bin.binutil  # alt import if called as module
import os
import subprocess
import sys

from dreamcoder.enumeration import EnumerationDebugHook
from dreamcoder.utilities import eprint

from dreamcoder.domains.rbii.rbii_primitives import RBIIPrimitiveConfig, make_rbii_grammar
from dreamcoder.domains.rbii.rbii_loop import RBIIConfig, RBIILoop
from dreamcoder.domains.rbii.rbii_state import RBIIState


class _FileEnumerationDebugHook(EnumerationDebugHook):
    def __init__(self, log_path: str):
        self.log_path = log_path

    def on_program(self, **payload):
        dt = float(payload.get("dt", 0.0))
        likelihood = payload.get("likelihood")
        program = payload.get("program")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"{dt:.6f}\t{likelihood}\t{program}\n")

    def on_end(self, **_payload):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("-------\n")


def _next_run_subdir(base_dir: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    max_idx = 0
    for entry in os.listdir(base_dir):
        p = os.path.join(base_dir, entry)
        if not os.path.isdir(p):
            continue
        if not entry.startswith("run_"):
            continue
        suffix = entry[len("run_") :]
        if not suffix.isdigit():
            continue
        max_idx = max(max_idx, int(suffix))
    run_dir = os.path.join(base_dir, f"run_{max_idx + 1:04d}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def _make_enum_debug_hook_factory(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def hook_factory(_current_index: int, _task):
        return _FileEnumerationDebugHook(log_path)

    return hook_factory


def run_sequence(name: str, seq: str, event_log_dir: str) -> None:
    eprint("\n" + "=" * 80)
    eprint(f"Sequence: {name}  len={len(seq)}")
    eprint(f"  {seq}")
    eprint("=" * 80)

    # Build grammar once per run.
    g = make_rbii_grammar(RBIIPrimitiveConfig(alphabet="abcde", max_int=6, log_variable=0.0))

    total_cpus = os.cpu_count() or 1
    enum_cpus = max(1, (total_cpus * 3) // 4)

    cfg = RBIIConfig(
        pool_target_size=3,
        validation_window=6,
        min_time=3,            # enough history for k=0,1,2 lookbacks
        enum_timeout_s=.6,
        # enum_timeout_s=0.6,
        eval_timeout_s=0.02,
        upper_bound=200,
        # upper_bound=30,
        budget_increment=1.5,
        max_frontier=10,
        verbose=True,
        event_log_dir=event_log_dir,
        event_log_name=name,
        log_candidate_events=True,
        enum_solver="bottom",
        enum_cpus=enum_cpus,
        enum_bottom_compile_me=False,
    )

    state = RBIIState()

    # Warmup: seed with the first min_time characters.
    warmup = min(cfg.min_time, len(seq))
    for i in range(warmup):
        state.observe(seq[i])

    eprint(f"Warmup seeded obs_history[:{warmup}] = {''.join(state.obs_history)!r}")

    if cfg.verbose:
        eprint(
            f"Enumeration solver={cfg.enum_solver} "
            f"compile_me={cfg.enum_bottom_compile_me} cpus={cfg.enum_cpus}/{total_cpus}"
        )

    enum_debug_factory = None
    if cfg.enum_solver == "python":
        enum_debug_log_path = os.path.join(event_log_dir, f"{name}_enumerate_debug.log")
        enum_debug_factory = _make_enum_debug_hook_factory(enum_debug_log_path)

    rbii = RBIILoop(
        grammar=g,
        state=state,
        cfg=cfg,
        enumeration_debug_hooks_factory=enum_debug_factory,
    )

    # Online loop: at each step observe the next symbol.
    for i in range(warmup, len(seq)):
        rbii.observe_and_update(seq[i])
    rbii.close()
    if rbii.event_log_path:
        eprint(f"Event log written: {rbii.event_log_path}")

    eprint("\nFinal:")
    eprint(f"  total_obs={len(state.obs_history)}")
    eprint(f"  stored_best_programs={len(state.best_programs)}")
    for j, p in enumerate(state.best_programs[:10]):
        eprint(f"    [{j}] {p}")
    if len(state.best_programs) > 10:
        eprint(f"    ... ({len(state.best_programs) - 10} more)")


def _render_viz_for_run(run_event_log_dir: str) -> None:
    """
    Auto-render SVG timelines for all JSONL logs in this run directory and
    write outputs into that same directory.
    """
    cmd = [
        sys.executable,
        "-m",
        "dreamcoder.domains.rbii.rbii_viz_graph",
        "--input-dir",
        run_event_log_dir,
        "--output-dir",
        run_event_log_dir,
    ]
    eprint(f"Rendering viz into run dir: {run_event_log_dir}")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        stdout = (e.stdout or "").strip()
        eprint("WARNING: Viz render failed.")
        if stdout:
            eprint(stdout)
        if stderr:
            eprint(stderr)
        return

    stdout = (result.stdout or "").strip()
    if stdout:
        for line in stdout.splitlines():
            eprint(line)


def main():
    base_event_log_dir = os.path.join("experimentOutputs", "rbii_program_events")
    run_event_log_dir = _next_run_subdir(base_event_log_dir)
    eprint(f"Run output dir: {run_event_log_dir}")

    # Simple predictable sequences
    run_sequence("all_a", "aaaaaaaaaaaaaaaaaaaa", run_event_log_dir)
    run_sequence("alternating_ab", "abababababababababab", run_event_log_dir)
    run_sequence("runs_of_3",
                 "aaabbbcccdddeeeaaabbbcccdddeeeaaabbbcccdddeeeaaabbbcccdddeee",
                 run_event_log_dir)
    _render_viz_for_run(run_event_log_dir)


if __name__ == "__main__":
    main()
