from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import bin.binutil  # noqa: F401

from dreamcoder.utilities import eprint

from .loop import MNISTRBIIConfig, MNISTRBIILoop
from .primitives import MNISTPrimitiveConfig, make_mnist_rbii_grammar
from .state import MNISTState
from .stream import build_split_mnist_return_stream, build_synthetic_return_stream


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RBII MNIST continual-learning feasibility harness")
    p.add_argument("--quick", action="store_true", help="Run a faster smoke configuration")
    p.add_argument("--data-dir", default="data/mnist")
    p.add_argument("--download", action="store_true", default=True)
    p.add_argument("--no-download", dest="download", action="store_false")
    p.add_argument("--per-context", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--synthetic", action="store_true", help="Use synthetic offline stream instead of MNIST")
    p.add_argument("--enum-timeout", type=float, default=1.0)
    p.add_argument("--eval-timeout", type=float, default=0.05)
    p.add_argument("--max-frontier", type=int, default=12)
    p.add_argument("--verbose", action="store_true", default=False)
    p.add_argument("--viz", action="store_true", default=True, help="Render SVG timeline after run")
    p.add_argument("--no-viz", dest="viz", action="store_false")
    p.add_argument("--viz-output-dir", default=os.path.join("experimentOutputs", "rbii_mnist_viz"))
    return p.parse_args()


def render_timeline_viz(log_path: str, output_dir: str) -> Path | None:
    if not log_path:
        return None

    cmd = [
        sys.executable,
        "-m",
        "dreamcoder.domains.rbii.rbii_viz_graph",
        log_path,
        "--output-dir",
        output_dir,
        "--show-timestep-labels",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        if proc.stdout.strip():
            eprint(proc.stdout.strip())
        if proc.stderr.strip():
            eprint(proc.stderr.strip())
        eprint("Timeline visualization failed.")
        return None

    if proc.stdout.strip():
        eprint(proc.stdout.strip())

    stem = Path(log_path).stem
    return Path(output_dir) / f"{stem}.svg"


def main() -> None:
    args = parse_args()

    per_context = args.per_context
    enum_timeout = args.enum_timeout
    eval_timeout = args.eval_timeout
    max_frontier = args.max_frontier

    if args.quick:
        per_context = min(per_context, 40)
        enum_timeout = min(enum_timeout, 0.4)
        eval_timeout = min(eval_timeout, 0.02)
        max_frontier = min(max_frontier, 8)

    schedule = ("A", "B", "A", "B")

    if args.synthetic:
        eprint("Building synthetic return stream...")
        stream = build_synthetic_return_stream(
            per_context=per_context,
            schedule=schedule,
            seed=args.seed,
        )
    else:
        eprint("Building split-MNIST return stream...")
        stream = build_split_mnist_return_stream(
            data_dir=args.data_dir,
            download=args.download,
            train=True,
            per_context=per_context,
            schedule=schedule,
            seed=args.seed,
        )

    eprint(f"Stream size: {len(stream)}")

    primitive_cfg = MNISTPrimitiveConfig(
        max_int=16,
        hidden_size=64,
        model_seed=args.seed,
        learning_rate=1e-3,
        device="cpu",
        log_variable=0.0,
    )
    grammar = make_mnist_rbii_grammar(primitive_cfg)

    run_name = f"rbii_mnist_{int(time.time())}"
    cfg = MNISTRBIIConfig(
        pool_target_size=4,
        validation_window=64,
        min_time=32,
        enum_timeout_s=enum_timeout,
        eval_timeout_s=eval_timeout,
        upper_bound=40.0,
        budget_increment=1.5,
        max_frontier=max_frontier,
        label_smoothing_eps=1e-3,
        evict_max_bits=3.5,
        weight_temperature=1.0,
        verbose=bool(args.verbose),
        event_log_dir=os.path.join("experimentOutputs", "rbii_mnist_events"),
        event_log_name=run_name,
        log_candidate_events=True,
    )

    state = MNISTState()
    loop = MNISTRBIILoop(grammar=grammar, state=state, cfg=cfg)

    try:
        for ex in stream:
            loop.observe_and_update(x=ex.x, y=ex.y, context=ex.context)
    finally:
        loop.close()

    summary = loop.metrics.summary()

    eprint("\nRun complete")
    eprint(f"  total_steps={summary['n']}")
    eprint(f"  accuracy={summary['accuracy']:.4f}")
    eprint(f"  mean_logloss_bits={summary['mean_logloss_bits']:.4f}")
    eprint(f"  event_log={loop.event_log_path}")
    eprint(f"  metrics_json={loop.metrics_path}")

    svg_path = None
    if args.viz and loop.event_log_path:
        svg_path = render_timeline_viz(loop.event_log_path, args.viz_output_dir)
        if svg_path is not None:
            eprint(f"  timeline_svg={svg_path}")

    returns = summary.get("reacquisition", {}).get("returns", [])
    if returns:
        eprint("  reacquisition episodes:")
        for r in returns:
            eprint(
                "   "
                f"ctx={r['context']} start={r['segment_start']} "
                f"delay={r['reacquisition_delay']} excess_bits={r['excess_bits']:.3f}"
            )


if __name__ == "__main__":
    main()
