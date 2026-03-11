# bin/rbii_test.py
# try:
#     import binutil  # required to import from dreamcoder modules
# except ModuleNotFoundError:

import bin.binutil  # alt import if called as module
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import random
import time
import types
import urllib.error
import urllib.request
import webbrowser
from urllib.parse import quote

from dreamcoder.enumeration import EnumerationDebugHook
from dreamcoder.utilities import eprint

from dreamcoder.domains.rbii.rbii_loop_v2 import (
    BottomSolverEnumerationController,
    RBIIConfigV2,
    RBIILoopV2,
)
from dreamcoder.domains.rbii.rbii_primitives import RBIIPrimitiveConfig, make_rbii_grammar
from dreamcoder.domains.rbii.rbii_loop import RBIIConfig, RBIILoop
from dreamcoder.domains.rbii.rbii_state import RBIIState


MIN_FREE_LOG_BYTES = 1 << 30
LOG_SPACE_CHECK_EVERY_WRITES = 1
ENABLE_ENUM_DEBUG_LOGS = os.environ.get("RBII_ENABLE_ENUM_DEBUG_LOGS") == "1"


def _ensure_six_meta_path_has_find_spec() -> None:
    for importer in sys.meta_path:
        if type(importer).__name__ != "_SixMetaPathImporter":
            continue
        if hasattr(importer, "find_spec"):
            continue

        def _find_spec(self, fullname, path=None, target=None):
            loader = self.find_module(fullname, path)
            if loader is None:
                return None
            return importlib.util.spec_from_loader(fullname, loader)

        importer.find_spec = types.MethodType(_find_spec, importer)


_ensure_six_meta_path_has_find_spec()


def _prune_old_logs_if_needed(
    base_dir: str,
    preserve_paths,
    min_free_bytes: int = MIN_FREE_LOG_BYTES,
) -> bool:
    base_dir = os.path.abspath(base_dir)
    preserve_paths = {os.path.abspath(path) for path in preserve_paths}
    os.makedirs(base_dir, exist_ok=True)

    def enough_space() -> bool:
        return shutil.disk_usage(base_dir).free >= min_free_bytes

    if enough_space():
        return True

    def is_preserved(path: str) -> bool:
        return any(
            path == preserved or path.startswith(preserved + os.sep)
            for preserved in preserve_paths
        )

    def is_enumeration_dump(path: str) -> bool:
        name = os.path.basename(path)
        return name.endswith(".log") and "_enumerate_debug" in name

    candidates = []
    for root, dirnames, filenames in os.walk(base_dir):
        root = os.path.abspath(root)
        if is_preserved(root):
            dirnames[:] = []
            continue

        dirnames[:] = [
            dirname
            for dirname in dirnames
            if not is_preserved(os.path.abspath(os.path.join(root, dirname)))
        ]

        for filename in filenames:
            path = os.path.abspath(os.path.join(root, filename))
            if not is_enumeration_dump(path):
                continue
            try:
                stat = os.stat(path, follow_symlinks=False)
            except FileNotFoundError:
                continue
            candidates.append((stat.st_mtime, -stat.st_size, path))

    candidates.sort()
    for _mtime, _neg_size, path in candidates:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as e:
            eprint("WARNING: failed to delete old log path:", path, e)
        if enough_space():
            return True

    return enough_space()


class _FileEnumerationDebugHook(EnumerationDebugHook):
    def __init__(
        self,
        log_path: str,
        cleanup_root: str | None = None,
        min_free_bytes: int = MIN_FREE_LOG_BYTES,
        check_every_writes: int = LOG_SPACE_CHECK_EVERY_WRITES,
    ):
        self.log_path = log_path
        self.cleanup_root = cleanup_root or os.path.dirname(log_path)
        self.min_free_bytes = int(min_free_bytes)
        self.check_every_writes = max(1, int(check_every_writes))
        self._writes_since_check = self.check_every_writes
        self._warned_insufficient_space = False

    def for_worker(self, worker_id):
        root, ext = os.path.splitext(self.log_path)
        return _FileEnumerationDebugHook(
            f"{root}.worker_{worker_id}{ext}",
            cleanup_root=self.cleanup_root,
            min_free_bytes=self.min_free_bytes,
            check_every_writes=self.check_every_writes,
        )

    def _write_line(self, line: str) -> None:
        self._writes_since_check += 1
        if self._writes_since_check >= self.check_every_writes:
            self._writes_since_check = 0
            if not _prune_old_logs_if_needed(
                self.cleanup_root,
                preserve_paths=[os.path.dirname(os.path.abspath(self.log_path))],
                min_free_bytes=self.min_free_bytes,
            ):
                if not self._warned_insufficient_space:
                    eprint(
                        "WARNING: enumeration log writing disabled; "
                        f"unable to keep {self.min_free_bytes} bytes free under "
                        f"{self.cleanup_root}"
                    )
                    self._warned_insufficient_space = True
                return

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)

    def on_program(self, **payload):
        dt = float(payload.get("dt", 0.0))
        likelihood = payload.get("likelihood")
        program = payload.get("program")
        self._write_line(f"{dt:.6f}\t{likelihood}\t{program}\n")

    def on_end(self, **_payload):
        self._write_line("-------\n")


LIVE_VIZ_HOST = "127.0.0.1"
LIVE_VIZ_PORT = 8765


def _live_viz_base_url() -> str:
    return f"http://{LIVE_VIZ_HOST}:{LIVE_VIZ_PORT}"


def _probe_live_viz_server():
    try:
        with urllib.request.urlopen(f"{_live_viz_base_url()}/healthz", timeout=0.5) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return None


def _ensure_live_viz_server(base_dir: str) -> str | None:
    expected_base_dir = os.path.abspath(base_dir)
    status = _probe_live_viz_server()
    if status is not None:
        actual_base_dir = os.path.abspath(str(status.get("base_dir", "")))
        if actual_base_dir != expected_base_dir:
            eprint(
                "WARNING: live viz server already running with different base dir:",
                actual_base_dir,
            )
            return None
        return _live_viz_base_url()

    cmd = [
        sys.executable,
        "-m",
        "dreamcoder.domains.rbii.rbii_viz_live",
        "--base-dir",
        expected_base_dir,
        "--host",
        LIVE_VIZ_HOST,
        "--port",
        str(LIVE_VIZ_PORT),
    ]
    subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    deadline = time.time() + 10.0
    while time.time() < deadline:
        status = _probe_live_viz_server()
        if status is None:
            time.sleep(0.25)
            continue
        actual_base_dir = os.path.abspath(str(status.get("base_dir", "")))
        if actual_base_dir == expected_base_dir:
            return _live_viz_base_url()
        eprint(
            "WARNING: live viz server started with unexpected base dir:",
            actual_base_dir,
        )
        return None

    eprint("WARNING: live viz server did not become ready.")
    return None


def _open_live_run_page(server_url: str | None, base_dir: str, run_dir: str) -> None:
    if not server_url:
        return
    rel_run_dir = os.path.relpath(run_dir, base_dir).replace(os.sep, "/")
    webbrowser.open(f"{server_url}/browse/{quote(rel_run_dir)}")


def _write_live_link_file(log_path: str, base_dir: str) -> None:
    rel_log_path = os.path.relpath(log_path, base_dir).replace(os.sep, "/")
    live_url = f"{_live_viz_base_url()}/view/{quote(rel_log_path)}"
    link_path = os.path.splitext(log_path)[0] + ".live.html"
    base_dir_abs = os.path.abspath(base_dir)
    with open(link_path, "w", encoding="utf-8") as handle:
        handle.write(
            f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>RBII Live Log Link</title>
    <style>
      body {{
        margin: 0;
        min-height: 100vh;
        display: grid;
        place-items: center;
        background: #f5f1ea;
        color: #2b241b;
        font-family: "Avenir Next", "Segoe UI", sans-serif;
      }}
      main {{
        max-width: 640px;
        padding: 32px;
        border: 1px solid #c9bba7;
        border-radius: 20px;
        background: #fffaf2;
      }}
      a {{
        color: #8f4b26;
        word-break: break-all;
      }}
      code {{
        font-family: Menlo, Consolas, Monaco, "Courier New", monospace;
      }}
    </style>
  </head>
  <body>
    <main>
      <h1>RBII Live Log</h1>
      <p>Open this log in the local live visualizer:</p>
      <p><a href="{live_url}">{live_url}</a></p>
      <p>If the page does not load, start:</p>
      <p><code>PYTHONPATH="$PWD" ./ve/bin/python -m dreamcoder.domains.rbii.rbii_viz_live --base-dir {base_dir_abs}</code></p>
    </main>
  </body>
</html>
"""
        )


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


def _make_enum_debug_hook_factory(log_path: str, cleanup_root: str):
    if not ENABLE_ENUM_DEBUG_LOGS:
        return None

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not _prune_old_logs_if_needed(
        cleanup_root,
        preserve_paths=[os.path.dirname(os.path.abspath(log_path))],
        min_free_bytes=MIN_FREE_LOG_BYTES,
    ):
        eprint(
            "WARNING: enumeration debug logging disabled; "
            f"unable to keep {MIN_FREE_LOG_BYTES} bytes free under {cleanup_root}"
        )
        return None

    def hook_factory(_current_index: int, _task):
        return _FileEnumerationDebugHook(log_path, cleanup_root=cleanup_root)

    return hook_factory


def run_sequence(
    name: str,
    seq: str,
    event_log_dir: str,
    base_event_log_dir: str,
    loop_version: str,
    enum_cpus: int,
) -> None:
    eprint("\n" + "=" * 80)
    eprint(f"Sequence: {name}  len={len(seq)}")
    eprint(f"  {seq}")
    eprint("=" * 80)

    # alphabet = "abcde"
    alphabet = "abc"

    # Build grammar once per run.
    g = make_rbii_grammar(
        RBIIPrimitiveConfig(alphabet=alphabet,
                            max_int=3,
                            log_variable=0.0)
    )

    total_cpus = os.cpu_count() or 1

    state = RBIIState()

    min_time = 3
    # Warmup: seed with the first min_time characters.
    warmup = min(min_time, len(seq))
    for i in range(warmup):
        state.observe(seq[i])

    eprint(f"Warmup seeded obs_history[:{warmup}] = {''.join(state.obs_history)!r}")

    enum_debug_log_path = os.path.join(event_log_dir, f"{name}_enumerate_debug.log")
    enum_debug_factory = _make_enum_debug_hook_factory(
        enum_debug_log_path,
        cleanup_root=base_event_log_dir,
    )
    event_log_path = None

    if loop_version == "v2":
        cfg = RBIIConfigV2(
            pool_target_size=5,
            validation_window=10,
            min_time=min_time,
            enum_timeout_s=5.0,
            eval_timeout_s=0.02,
            upper_bound=200.0,
            budget_increment=1.5,
            max_frontier=10,
            enum_cpus=enum_cpus,
            enum_bottom_compile_me=False,  # TOOD: what is this and why do I
          # need it?
            alphabet=tuple(alphabet),
            verbose=True,
            event_log_dir=event_log_dir,
            event_log_name=name,
            compression_gain_slack_bits=10,
        )
        enumerator = BottomSolverEnumerationController(
            enumeration_debug_hooks_factory=enum_debug_factory,
        )
        rbii = RBIILoopV2(
            grammar=g,
            state=state,
            cfg=cfg,
            enumerator=enumerator,
        )
        if cfg.verbose:
            eprint(
                f"Loop=v2 bottom compile_me={cfg.enum_bottom_compile_me} "
                f"cpus={cfg.enum_cpus}/{total_cpus}"
            )
        event_log_path = rbii.event_log_path
    else:
        cfg = RBIIConfig(
            pool_target_size=3,
            validation_window=6,
            min_time=min_time,
            enum_timeout_s=3,
            eval_timeout_s=0.02,
            upper_bound=200,
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
        if cfg.verbose:
            eprint(
                f"Loop=v1 solver={cfg.enum_solver} "
                f"compile_me={cfg.enum_bottom_compile_me} cpus={cfg.enum_cpus}/{total_cpus}"
            )
        rbii = RBIILoop(
            grammar=g,
            state=state,
            cfg=cfg,
            enumeration_debug_hooks_factory=enum_debug_factory,
        )
        event_log_path = rbii.event_log_path

    # Online loop: at each step observe the next symbol.
    for i in range(warmup, len(seq)):
        rbii.observe_and_update(seq[i])
    if hasattr(rbii, "close"):
        rbii.close()
    if event_log_path:
        eprint(f"Event log written: {event_log_path}")
        if loop_version == "v2":
            _write_live_link_file(event_log_path, base_event_log_dir)

    eprint("\nFinal:")
    eprint(f"  total_obs={len(state.obs_history)}")
    eprint(f"  stored_best_programs={len(state.best_programs)}")
    if hasattr(rbii, "pool"):
        eprint(f"  active_pool={len(rbii.pool)}")
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--loop",
        choices=["v1", "v2"],
        default="v1",
        help="Choose the legacy RBII loop or the policy-factored V2 loop.",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    total_cpus = os.cpu_count() or 1
    enum_cpus = max(1, (total_cpus * 3) // 4)
    if args.loop == "v2":
        base_event_log_dir = os.path.join("experimentOutputs", "rbii_program_events_v2")
    else:
        base_event_log_dir = os.path.join("experimentOutputs", "rbii_program_events")

    eprint("Loop version:", args.loop)
    eprint(f"Using {enum_cpus} enumeration CPUs (total available:"
           f" {total_cpus})")

    run_event_log_dir = _next_run_subdir(base_event_log_dir)
    eprint(f"Run output dir: {run_event_log_dir}")
    live_server_url = None
    if args.loop == "v2":
        live_server_url = _ensure_live_viz_server(base_event_log_dir)
        _open_live_run_page(live_server_url, base_event_log_dir, run_event_log_dir)


    def _run_sequence(name, seq):
        return run_sequence(
            name=name,
            seq=seq,
            event_log_dir=run_event_log_dir,
            base_event_log_dir=base_event_log_dir,
            loop_version=args.loop,
            enum_cpus=enum_cpus,
        )

    ## Simple predictable sequences
    # _run_sequence("all_a", "aaaaaaaaaaaaaaaaaaaa")
    # _run_sequence("alternating_ab", "abababababababababab")


    # Requires conditional behavior with the base RBII grammar:
    # if last == prev:
    #   if last == e: a
    #   else: succ_char(last)
    # else:
    #   last
    # _run_sequence(
    #   "cycle_requires_if",
    #   "abcde" * 6,
    # )
    # TODO: could still make this work if we make (last) a single primitive?
    #  i.e. emulate library compression

    # even simpler if
    _run_sequence(
      "random_requires_if",
      "aabcaaaabcabcaabcaaaaabcabcaaabcaabcaabcaaaaabcabc" * 6,
    )


    # This yields aabbccddee... cyclically. With max_int=6 there is no simple
    # fixed-lag shortcut for the full 10-symbol period, so it is a cleaner
    # conditional stress case than the ad hoc "force_if" sketch below.
    # _run_sequence(
    #     "pairs_cycle_requires_if",
    #     "aabbccddee" * 6,
    # )
    # _run_sequence("runs_of_3",
    #              "aaabbbcccdddeeeaaabbbcccdddeeeaaabbbcccdddeeeaaabbbcccdddeee",
    #               )
    #
    # ## runs of increasting length
    # _run_sequence("runs_of_increasing",
    #              "aaabbbcccdddeee"
    #                   "aaaabbbbccccddddeeee"
    #                   "aaaaabbbbbcccccdddddeeeee"
    #                   "aaaaaabbbbbbccccccddddddeeeeee",
    #              )

    # sequence that forces conditional to be the best

    ## random seq
    # random.seed(0)
    # def _random_sequence(length: int, alphabet: str) -> str:
    #     return "".join(random.choices(alphabet, k=length))
    #
    # _run_sequence("force_if",
    #              "".join(
    #                ["aaaab"+_random_sequence(random.randint(1,5), "cde")
    #                       for _ in range(10)]),
    #              )


    if args.loop == "v2":
        _render_viz_for_run(run_event_log_dir)
    else:
        eprint("Skipping viz render for loop=v1 (visualizer expects V2 logs).")


if __name__ == "__main__":
    main()
