# bin/rbii_test.py
# try:
#     import binutil  # required to import from dreamcoder modules
# except ModuleNotFoundError:

import bin.binutil  # alt import if called as module
import os

from dreamcoder.utilities import eprint

from dreamcoder.domains.rbii.rbii_primitives import RBIIPrimitiveConfig, make_rbii_grammar
from dreamcoder.domains.rbii.rbii_loop import RBIIConfig, RBIILoop
from dreamcoder.domains.rbii.rbii_state import RBIIState


def run_sequence(name: str, seq: str) -> None:
    eprint("\n" + "=" * 80)
    eprint(f"Sequence: {name}  len={len(seq)}")
    eprint(f"  {seq}")
    eprint("=" * 80)

    # Build grammar once per run.
    g = make_rbii_grammar(RBIIPrimitiveConfig(alphabet="abcde", max_int=6, log_variable=0.0))

    event_log_dir = os.path.join("experimentOutputs", "rbii_program_events")
    cfg = RBIIConfig(
        pool_target_size=3,
        validation_window=6,
        min_time=3,            # enough history for k=0,1,2 lookbacks
        enum_timeout_s=5, #0.6,
        eval_timeout_s=0.02,
        upper_bound=200.0, # 30,
        budget_increment=1.5,
        max_frontier=10,
        verbose=True,
        event_log_dir=event_log_dir,
        event_log_name=name,
        log_candidate_events=True,
    )

    state = RBIIState()

    # Warmup: seed with the first min_time characters.
    warmup = min(cfg.min_time, len(seq))
    for i in range(warmup):
        state.observe(seq[i])

    eprint(f"Warmup seeded obs_history[:{warmup}] = {''.join(state.obs_history)!r}")

    rbii = RBIILoop(grammar=g, state=state, cfg=cfg)

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


def main():
    # Simple predictable sequences
    run_sequence("all_a", "aaaaaaaaaaaaaaaaaaaa")
    run_sequence("alternating_ab", "abababababababababab")
    run_sequence("runs_of_3",
                 "aaabbbcccdddeeeaaabbbcccdddeeeaaabbbcccdddeeeaaabbbcccdddeee")


if __name__ == "__main__":
    main()
