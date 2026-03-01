from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class ProgramMeta:
    program_text: str
    duplicate_candidate: bool = False


@dataclass
class ProgramEpisode:
    program_id: int
    program_text: str
    start_t: int
    end_t: int
    duplicate_candidate: bool
    used_for_prediction: bool = False
    lane: int = 0


@dataclass
class EpisodeLayout:
    episode: ProgramEpisode
    label: str
    bracket_x: float
    router_x: float
    y1: float
    y2: float
    desired_center: float
    box_w: float
    box_h: float
    box_x: float
    box_y: float = 0.0
    box_center_y: float = 0.0
    anchor_y: float = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Render RBII event logs as bracket timelines (no Graphviz). "
            "By default, processes all JSONL logs in experimentOutputs/rbii_program_events."
        )
    )
    p.add_argument(
        "logs",
        nargs="*",
        help="Optional explicit JSONL log paths. If omitted, scans --input-dir.",
    )
    p.add_argument(
        "--input-dir",
        default="experimentOutputs/rbii_program_events",
        help="Directory scanned for *.jsonl when no explicit logs are passed.",
    )
    p.add_argument(
        "--output-dir",
        default="experimentOutputs/rbii_program_viz",
        help="Directory for generated SVG files.",
    )
    p.add_argument(
        "--format",
        default="svg",
        choices=["svg"],
        help="Render format (currently SVG only).",
    )
    p.add_argument(
        "--show-timestep-labels",
        action="store_true",
        help="Include timestep numbers to the left of sequence symbols.",
    )
    p.add_argument(
        "--label-mode",
        default="program",
        choices=["alias", "program", "both"],
        help="Program label mode for the inline rounded boxes.",
    )
    p.add_argument(
        "--max-program-label-len",
        type=int,
        default=84,
        help="Maximum length for program text shown in rounded boxes.",
    )
    p.add_argument(
        "--show-program-map",
        action="store_true",
        help="Append alias->program mapping block under the chart.",
    )
    p.add_argument(
        "--row-step",
        type=float,
        default=22.0,
        help="Vertical spacing per timestep row in pixels.",
    )
    p.add_argument(
        "--lane-step",
        type=float,
        default=34.0,
        help="Horizontal spacing between bracket lanes in pixels.",
    )
    p.add_argument(
        "--connector-len",
        type=float,
        default=68.0,
        help="Horizontal connector length from bracket toward label box in pixels.",
    )
    p.add_argument(
        "--min-box-margin",
        type=float,
        default=5.0,
        help="Minimum vertical gap in pixels between adjacent program boxes.",
    )
    p.add_argument(
        "--font-family",
        default="Helvetica,Arial,sans-serif",
        help="General UI font family used in the SVG.",
    )
    p.add_argument(
        "--code-font-family",
        default="Menlo,Consolas,Monaco,'Courier New',monospace",
        help="Monospace font family for sequence symbols and program code boxes.",
    )
    return p.parse_args()


def _truncate(s: str, n: int) -> str:
    if n <= 0:
        return ""
    if len(s) <= n:
        return s
    if n <= 3:
        return s[:n]
    return s[: n - 3] + "..."


def _resolve_logs(args: argparse.Namespace) -> List[Path]:
    if args.logs:
        return [Path(x) for x in args.logs]
    input_dir = Path(args.input_dir)
    return sorted([p for p in input_dir.glob("*.jsonl") if p.is_file()])


def _load_rows(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i} invalid JSON: {e}") from e
    return rows


def _collect_sequence(rows: Iterable[dict]) -> Tuple[int, int, Dict[int, str]]:
    obs_by_t: Dict[int, str] = {}
    start_t: Optional[int] = None
    end_t: Optional[int] = None
    all_t: List[int] = []

    for row in rows:
        t = row.get("timestep")
        if isinstance(t, int):
            all_t.append(t)

        ev = row.get("event")
        if ev == "run_start" and isinstance(t, int):
            start_t = t if start_t is None else min(start_t, t)
        elif ev == "run_end" and isinstance(t, int):
            end_t = (t - 1) if end_t is None else max(end_t, t - 1)
        elif ev == "observe" and isinstance(t, int):
            obs = row.get("observed")
            if isinstance(obs, str):
                obs_by_t[t] = obs

    if obs_by_t:
        seq_start = min(obs_by_t.keys())
        seq_end = max(obs_by_t.keys())
    elif all_t:
        seq_start = min(all_t)
        seq_end = max(all_t)
    else:
        seq_start = 0
        seq_end = 0

    if start_t is None:
        start_t = seq_start
    if end_t is None:
        end_t = seq_end

    return min(start_t, seq_start), max(end_t, seq_end), obs_by_t


def _collect_program_meta(rows: Iterable[dict]) -> Dict[int, ProgramMeta]:
    meta: Dict[int, ProgramMeta] = {}

    for row in rows:
        pid = row.get("program_id")
        if not isinstance(pid, int):
            continue

        prog = row.get("program")
        if not isinstance(prog, str):
            prog = f"<program {pid}>"
        dup = bool(row.get("duplicate_candidate", False))

        if pid not in meta:
            meta[pid] = ProgramMeta(program_text=prog, duplicate_candidate=dup)
        else:
            if not meta[pid].program_text.startswith("<program") and prog.startswith("<program"):
                pass
            else:
                meta[pid].program_text = prog
            meta[pid].duplicate_candidate = meta[pid].duplicate_candidate or dup

    return meta


def _build_episodes(
    rows: Iterable[dict],
    end_t: int,
    meta_by_pid: Dict[int, ProgramMeta],
) -> List[ProgramEpisode]:
    open_by_pid: Dict[int, ProgramEpisode] = {}
    episodes: List[ProgramEpisode] = []

    predict_times: Dict[int, List[int]] = defaultdict(list)
    for row in rows:
        if row.get("event") == "predict_used":
            pid = row.get("program_id")
            t = row.get("timestep")
            if isinstance(pid, int) and isinstance(t, int):
                predict_times[pid].append(t)

    for row in rows:
        ev = row.get("event")
        t = row.get("timestep")
        if not isinstance(t, int):
            continue

        pid = row.get("program_id")
        if ev in ("enter", "evicted") and not isinstance(pid, int):
            continue
        if not isinstance(pid, int):
            continue

        meta = meta_by_pid.get(pid, ProgramMeta(program_text=f"<program {pid}>"))

        if ev == "enter":
            if pid in open_by_pid:
                existing = open_by_pid.pop(pid)
                existing.end_t = max(existing.start_t, t - 1)
                episodes.append(existing)

            open_by_pid[pid] = ProgramEpisode(
                program_id=pid,
                program_text=meta.program_text,
                start_t=t,
                end_t=t,
                duplicate_candidate=meta.duplicate_candidate,
            )

        elif ev == "evicted":
            if pid in open_by_pid:
                ep = open_by_pid.pop(pid)
                ep.end_t = max(ep.start_t, t)
                episodes.append(ep)
            else:
                episodes.append(
                    ProgramEpisode(
                        program_id=pid,
                        program_text=meta.program_text,
                        start_t=t,
                        end_t=t,
                        duplicate_candidate=meta.duplicate_candidate,
                    )
                )

    for ep in open_by_pid.values():
        ep.end_t = max(ep.start_t, end_t)
        episodes.append(ep)

    # Ensure predict-used with missing enter/evicted still appears as a one-row episode.
    covered_predicts: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for ep in episodes:
        covered_predicts[ep.program_id].append((ep.start_t, ep.end_t))

    for pid, times in predict_times.items():
        spans = covered_predicts.get(pid, [])
        for t in times:
            if any(a <= t <= b for a, b in spans):
                continue
            meta = meta_by_pid.get(pid, ProgramMeta(program_text=f"<program {pid}>"))
            episodes.append(
                ProgramEpisode(
                    program_id=pid,
                    program_text=meta.program_text,
                    start_t=t,
                    end_t=t,
                    duplicate_candidate=meta.duplicate_candidate,
                    used_for_prediction=True,
                )
            )
            covered_predicts[pid].append((t, t))

    for ep in episodes:
        times = predict_times.get(ep.program_id, [])
        ep.used_for_prediction = ep.used_for_prediction or any(
            ep.start_t <= t <= ep.end_t for t in times
        )

    episodes.sort(key=lambda e: (e.start_t, e.end_t, e.program_id, e.program_text))
    return episodes


def _assign_lanes(episodes: List[ProgramEpisode]) -> None:
    lane_ends: List[int] = []

    for ep in episodes:
        assigned = None
        for lane, lane_end in enumerate(lane_ends):
            if ep.start_t > lane_end:
                assigned = lane
                break

        if assigned is None:
            assigned = len(lane_ends)
            lane_ends.append(ep.end_t)
        else:
            lane_ends[assigned] = ep.end_t

        ep.lane = assigned


def _program_aliases(episodes: List[ProgramEpisode]) -> Dict[str, int]:
    alias_by_text: Dict[str, int] = {}
    next_idx = 1
    for ep in episodes:
        if ep.program_text not in alias_by_text:
            alias_by_text[ep.program_text] = next_idx
            next_idx += 1
    return alias_by_text


def _episode_label(
    ep: ProgramEpisode,
    alias_by_text: Dict[str, int],
    label_mode: str,
    max_program_label_len: int,
) -> str:
    alias = f"(program {alias_by_text[ep.program_text]})"
    prog = _truncate(ep.program_text, max_program_label_len)

    if label_mode == "alias":
        return alias
    if label_mode == "both":
        return f"{alias} {prog}"
    return prog


def _estimate_text_width_px(text: str, font_size: float = 13.0) -> float:
    # Approximate width for monospace-ish rendering.
    return max(20.0, len(text) * (font_size * 0.62))


def _layout_episode_boxes(
    episodes: List[ProgramEpisode],
    labels: List[str],
    y_by_t: Dict[int, float],
    bracket_x0: float,
    lane_step: float,
    connector_len: float,
    label_gap: float,
    min_box_margin: float,
    font_size: float,
    pad_x: float,
    pad_y: float,
) -> List[EpisodeLayout]:
    layouts: List[EpisodeLayout] = []

    for ep, label in zip(episodes, labels):
        y1 = y_by_t.get(ep.start_t)
        y2 = y_by_t.get(ep.end_t)
        if y1 is None or y2 is None:
            continue
        if y2 < y1:
            y1, y2 = y2, y1

        bracket_x = bracket_x0 + (ep.lane * lane_step)
        desired_center = (y1 + y2) / 2.0

        text_w = _estimate_text_width_px(label, font_size)
        box_w = text_w + (2.0 * pad_x)
        box_h = font_size + (2.0 * pad_y)

        layouts.append(
            EpisodeLayout(
                episode=ep,
                label=label,
                bracket_x=bracket_x,
                router_x=0.0,
                y1=y1,
                y2=y2,
                desired_center=desired_center,
                box_w=box_w,
                box_h=box_h,
                box_x=0.0,
            )
        )

    if not layouts:
        return layouts

    # Use a shared connector routing corridor left of all boxes.
    max_bracket_x = max(x.bracket_x for x in layouts)
    router_x = max_bracket_x + max(14.0, lane_step * 0.6)
    box_x = router_x + max(30.0, connector_len) + label_gap

    for item in layouts:
        item.router_x = router_x
        item.box_x = box_x

    # Ensure boxes never overlap and keep at least min_box_margin vertical gap.
    ordered = sorted(layouts, key=lambda x: (x.desired_center, x.box_x))
    prev_bottom = -10**9
    for item in ordered:
        top = item.desired_center - (item.box_h / 2.0)
        min_top = prev_bottom + min_box_margin
        if top < min_top:
            top = min_top
        item.box_y = top
        item.box_center_y = top + (item.box_h / 2.0)
        prev_bottom = top + item.box_h

    # Re-anchor connector line to stay inside bracket span when labels are shifted.
    for item in layouts:
        span = item.y2 - item.y1
        if span <= 6.0:
            item.anchor_y = (item.y1 + item.y2) / 2.0
            continue
        lo = item.y1 + 3.0
        hi = item.y2 - 3.0
        item.anchor_y = min(max(item.box_center_y, lo), hi)

    return layouts


def _render_svg(
    title: str,
    timesteps: List[int],
    obs_by_t: Dict[int, str],
    episodes: List[ProgramEpisode],
    alias_by_text: Dict[str, int],
    args: argparse.Namespace,
) -> str:
    row_step = float(args.row_step)
    lane_step = float(args.lane_step)
    connector_len = float(args.connector_len)
    min_box_margin = float(args.min_box_margin)

    top = 32.0
    left = 30.0
    seq_x = left + (42.0 if args.show_timestep_labels else 22.0)
    bracket_x0 = seq_x + 55.0
    label_gap = 10.0
    bracket_arm = 10.0

    program_font_size = 13.0
    program_pad_x = 9.0
    program_pad_y = 6.0
    sequence_font_size = 24.0

    y_by_t = {t: top + i * row_step for i, t in enumerate(timesteps)}
    label_texts = [
        _episode_label(ep, alias_by_text, args.label_mode, args.max_program_label_len)
        for ep in episodes
    ]

    episode_layouts = _layout_episode_boxes(
        episodes=episodes,
        labels=label_texts,
        y_by_t=y_by_t,
        bracket_x0=bracket_x0,
        lane_step=lane_step,
        connector_len=connector_len,
        label_gap=label_gap,
        min_box_margin=min_box_margin,
        font_size=program_font_size,
        pad_x=program_pad_x,
        pad_y=program_pad_y,
    )

    map_lines: List[str] = []
    if args.show_program_map and alias_by_text:
        inv = sorted(alias_by_text.items(), key=lambda x: x[1])
        for prog, idx in inv:
            map_lines.append(f"program {idx}: {_truncate(prog, 120)}")

    seq_bottom = top + ((len(timesteps) - 1) * row_step) if timesteps else top
    box_bottom = max((x.box_y + x.box_h for x in episode_layouts), default=seq_bottom)
    chart_bottom = max(seq_bottom, box_bottom)

    map_height = 0.0
    if map_lines:
        map_height = 34.0 + 18.0 * len(map_lines)

    max_right = seq_x + 20.0
    for item in episode_layouts:
        right = item.box_x + item.box_w
        if right > max_right:
            max_right = right

    width = max_right + 34.0
    height = chart_bottom + 34.0 + map_height

    parts: List[str] = []
    parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}">'
    )
    parts.append(
        f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" fill="#f3f3f3"/>'
    )
    parts.append(
        f'<text x="{left:.1f}" y="20" font-family="{html.escape(args.font_family)}" '
        'font-size="14" fill="#444">'
        + html.escape(title)
        + "</text>"
    )

    # Sequence column text in terminal/monospace style.
    for t in timesteps:
        y = y_by_t[t]
        char = obs_by_t.get(t, "")

        if args.show_timestep_labels:
            parts.append(
                f'<text x="{left:.1f}" y="{y + 4:.1f}" font-family="{html.escape(args.font_family)}" '
                'font-size="11" fill="#777" text-anchor="start">'
                + html.escape(str(t))
                + "</text>"
            )

        parts.append(
            f'<text x="{seq_x:.1f}" y="{y + 5:.1f}" font-family="{html.escape(args.code_font_family)}" '
            f'font-size="{sequence_font_size:.0f}" fill="#222" text-anchor="middle">'
            + html.escape(char)
            + "</text>"
        )

    # Program brackets + connectors + rounded code boxes.
    for item in episode_layouts:
        ep = item.episode
        x = item.bracket_x
        x_router = item.router_x
        stroke = "#5f5f5f"
        line_w = 2.1 if ep.used_for_prediction else 1.5
        line_dash = "" if ep.used_for_prediction else ' stroke-dasharray="3 3"'

        parts.append(
            f'<path d="M {x:.1f} {item.y1:.1f} h {-bracket_arm:.1f} '
            f'M {x:.1f} {item.y1:.1f} V {item.y2:.1f} '
            f'M {x:.1f} {item.y2:.1f} h {-bracket_arm:.1f}" '
            f'fill="none" stroke="{stroke}" stroke-width="{line_w:.1f}"{line_dash}/>'
        )
        parts.append(
            f'<path d="M {x:.1f} {item.anchor_y:.1f} H {x_router:.1f} '
            f'V {item.box_center_y:.1f} H {item.box_x:.1f}" '
            f'fill="none" stroke="{stroke}" stroke-width="{line_w:.1f}"{line_dash}/>'
        )

        box_dash = ' stroke-dasharray="1.5 3.5" stroke-linecap="round"' if ep.duplicate_candidate else ""
        parts.append(
            f'<rect x="{item.box_x:.1f}" y="{item.box_y:.1f}" width="{item.box_w:.1f}" '
            f'height="{item.box_h:.1f}" rx="8" ry="8" fill="#f8f8f8" '
            f'stroke="#666" stroke-width="1.3"{box_dash}/>'
        )

        text_x = item.box_x + program_pad_x
        text_y = item.box_y + program_pad_y + (program_font_size * 0.82)
        parts.append(
            f'<text x="{text_x:.1f}" y="{text_y:.1f}" '
            f'font-family="{html.escape(args.code_font_family)}" '
            f'font-size="{program_font_size:.0f}" fill="#2e2e2e" text-anchor="start">'
            + html.escape(item.label)
            + "</text>"
        )

    if map_lines:
        y0 = chart_bottom + 36.0
        parts.append(
            f'<text x="{left:.1f}" y="{y0 - 12:.1f}" font-family="{html.escape(args.font_family)}" '
            'font-size="13" fill="#555">Program map</text>'
        )
        for i, line in enumerate(map_lines):
            yy = y0 + i * 18.0
            parts.append(
                f'<text x="{left:.1f}" y="{yy:.1f}" font-family="{html.escape(args.code_font_family)}" '
                'font-size="12" fill="#666">'
                + html.escape(line)
                + "</text>"
            )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _build_svg_for_log(log_path: Path, args: argparse.Namespace) -> str:
    rows = _load_rows(log_path)
    start_t, end_t, obs_by_t = _collect_sequence(rows)
    timesteps = list(range(start_t, end_t + 1))

    meta_by_pid = _collect_program_meta(rows)
    episodes = _build_episodes(rows, end_t, meta_by_pid)
    _assign_lanes(episodes)
    alias_by_text = _program_aliases(episodes)

    return _render_svg(
        title=f"RBII Program Timeline: {log_path.stem}",
        timesteps=timesteps,
        obs_by_t=obs_by_t,
        episodes=episodes,
        alias_by_text=alias_by_text,
        args=args,
    )


def main() -> int:
    args = parse_args()
    logs = _resolve_logs(args)
    if not logs:
        print(f"No logs found. Checked: {args.input_dir}")
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for log_path in logs:
        svg = _build_svg_for_log(log_path, args)
        out_path = out_dir / f"{log_path.stem}.svg"
        out_path.write_text(svg, encoding="utf-8")
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
