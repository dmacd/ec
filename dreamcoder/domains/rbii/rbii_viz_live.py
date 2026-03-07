from __future__ import annotations

import argparse
import json
import mimetypes
import os
import posixpath
import time
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, Optional

from .rbii_viz_index import LIVE_SCHEMA_VERSION, LogIndexCache


POLL_INTERVAL_S = 0.25
HEARTBEAT_INTERVAL_S = 10.0


class RBiiLiveApp:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir).resolve()
        self.asset_dir = Path(__file__).with_name("viz_live")
        self.index_cache = LogIndexCache()

    def resolve_relative_path(self, raw_path: str, *, expect_dir: Optional[bool] = None) -> Path:
        decoded = urllib.parse.unquote(raw_path or "")
        if decoded.startswith(("/", "\\")):
            raise ValueError("path must be relative to base_dir")
        if any(part == ".." for part in decoded.split("/")):
            raise ValueError("path escapes base_dir")
        normalized = posixpath.normpath("/" + decoded).lstrip("/")
        if normalized in (".", ""):
            target = self.base_dir
        else:
            target = (self.base_dir / normalized).resolve()
        try:
            target.relative_to(self.base_dir)
        except ValueError:
            raise ValueError("path escapes base_dir")
        if expect_dir is True and not target.is_dir():
            raise FileNotFoundError(str(target))
        if expect_dir is False and not target.is_file():
            raise FileNotFoundError(str(target))
        return target

    def relative_display_path(self, path: Path) -> str:
        rel = path.resolve().relative_to(self.base_dir)
        return rel.as_posix()

    def list_relative_dir(self, raw_path: str) -> dict:
        directory = self.resolve_relative_path(raw_path, expect_dir=True)
        entries = []
        for child in sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name)):
            if child.name.startswith("."):
                continue
            if child.is_dir():
                entries.append(
                    {
                        "name": child.name,
                        "path": self.relative_display_path(child),
                        "kind": "dir",
                        "mtime_ns": int(child.stat().st_mtime_ns),
                    }
                )
                continue
            if child.suffix != ".jsonl":
                continue
            meta = self.describe_log(child)
            entries.append(
                {
                    "name": child.name,
                    "path": self.relative_display_path(child),
                    "kind": "log",
                    "mtime_ns": int(child.stat().st_mtime_ns),
                    **meta,
                }
            )
        return {
            "path": "" if directory == self.base_dir else self.relative_display_path(directory),
            "entries": entries,
        }

    def describe_log(self, path: Path) -> dict:
        index = self.index_cache.get(path)
        meta = index.metadata()
        meta["path"] = self.relative_display_path(path)
        meta["name"] = path.name
        meta["size_bytes"] = int(path.stat().st_size)
        meta["mtime_ns"] = int(path.stat().st_mtime_ns)
        return meta

    def chunk_payload(self, raw_path: str, start_timestep: int, end_timestep: int) -> dict:
        log_path = self.resolve_relative_path(raw_path, expect_dir=False)
        index = self.index_cache.get(log_path)
        if start_timestep > end_timestep:
            start_timestep, end_timestep = end_timestep, start_timestep
        return {
            "path": self.relative_display_path(log_path),
            "start_t": start_timestep,
            "end_t": end_timestep,
            "step_groups": index.read_step_groups(start_timestep, end_timestep),
            "last_closed_timestep": index.last_closed_timestep,
            "run_complete": bool(index.run_complete),
        }

    def bootstrap_payload(self, mode: str, raw_path: str) -> dict:
        return {
            "mode": mode,
            "path": raw_path,
            "baseDir": str(self.base_dir),
            "schemaVersion": LIVE_SCHEMA_VERSION,
        }


def _json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def _parse_int_arg(query: Dict[str, list[str]], name: str, default: int = 0) -> int:
    raw = query.get(name, [str(default)])[0]
    return int(raw)


def make_handler(app: RBiiLiveApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "RBiiLive/1.0"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            try:
                if parsed.path == "/healthz":
                    self._send_json(
                        {
                            "ok": True,
                            "base_dir": str(app.base_dir),
                            "schema": LIVE_SCHEMA_VERSION,
                        }
                    )
                    return

                if parsed.path.startswith("/static/"):
                    rel = parsed.path[len("/static/") :]
                    self._send_static(rel)
                    return

                if parsed.path.startswith("/api/list"):
                    query = urllib.parse.parse_qs(parsed.query)
                    payload = app.list_relative_dir(query.get("path", [""])[0])
                    self._send_json(payload)
                    return

                if parsed.path.startswith("/api/meta"):
                    query = urllib.parse.parse_qs(parsed.query)
                    raw_path = query.get("path", [""])[0]
                    log_path = app.resolve_relative_path(raw_path, expect_dir=False)
                    self._send_json(app.describe_log(log_path))
                    return

                if parsed.path.startswith("/api/chunk"):
                    query = urllib.parse.parse_qs(parsed.query)
                    raw_path = query.get("path", [""])[0]
                    start_t = _parse_int_arg(query, "start_t", 0)
                    end_t = _parse_int_arg(query, "end_t", start_t)
                    self._send_json(app.chunk_payload(raw_path, start_t, end_t))
                    return

                if parsed.path.startswith("/api/stream"):
                    query = urllib.parse.parse_qs(parsed.query)
                    raw_path = query.get("path", [""])[0]
                    from_t = _parse_int_arg(query, "from_t", -1)
                    self._stream_steps(raw_path, from_t)
                    return

                if parsed.path == "/" or parsed.path.startswith("/browse/"):
                    raw_path = parsed.path[len("/browse/") :] if parsed.path.startswith("/browse/") else ""
                    self._send_page("browse", raw_path)
                    return

                if parsed.path.startswith("/view/"):
                    raw_path = parsed.path[len("/view/") :]
                    self._send_page("view", raw_path)
                    return

                self.send_error(HTTPStatus.NOT_FOUND)
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND)
            except ValueError as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, explain=str(exc))
            except BrokenPipeError:
                return

        def log_message(self, _format: str, *_args) -> None:
            return

        def _send_json(self, payload: dict, status: int = 200) -> None:
            raw = _json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _send_static(self, raw_rel: str) -> None:
            safe_rel = posixpath.normpath("/" + raw_rel).lstrip("/")
            asset_path = (app.asset_dir / safe_rel).resolve()
            try:
                asset_path.relative_to(app.asset_dir.resolve())
            except ValueError:
                raise ValueError("static path escapes asset dir")
            if not asset_path.is_file():
                raise FileNotFoundError(str(asset_path))
            content = asset_path.read_bytes()
            content_type, _ = mimetypes.guess_type(str(asset_path))
            self.send_response(200)
            self.send_header("Content-Type", content_type or "application/octet-stream")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _send_page(self, mode: str, raw_path: str) -> None:
            template_path = app.asset_dir / "index.html"
            template = template_path.read_text(encoding="utf-8")
            bootstrap = json.dumps(app.bootstrap_payload(mode, raw_path), sort_keys=True)
            content = template.replace("__RBII_BOOTSTRAP__", bootstrap).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _stream_steps(self, raw_path: str, from_timestep: int) -> None:
            log_path = app.resolve_relative_path(raw_path, expect_dir=False)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            last_sent = from_timestep
            last_heartbeat = 0.0
            while True:
                index = app.index_cache.get(log_path)
                if index.last_closed_timestep is not None and index.last_closed_timestep > last_sent:
                    groups = index.read_step_groups(last_sent + 1, index.last_closed_timestep)
                    if groups:
                        self._write_sse(
                            "steps",
                            {
                                "path": app.relative_display_path(log_path),
                                "step_groups": groups,
                                "run_complete": bool(index.run_complete),
                            },
                        )
                        last_sent = groups[-1]["timestep"]
                if index.run_complete:
                    self._write_sse(
                        "status",
                        {
                            "path": app.relative_display_path(log_path),
                            "run_complete": True,
                            "last_closed_timestep": index.last_closed_timestep,
                        },
                    )
                    break
                now = time.time()
                if now - last_heartbeat >= HEARTBEAT_INTERVAL_S:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                    last_heartbeat = now
                time.sleep(POLL_INTERVAL_S)

        def _write_sse(self, event_name: str, payload: dict) -> None:
            raw = json.dumps(payload, sort_keys=True)
            self.wfile.write(f"event: {event_name}\n".encode("utf-8"))
            self.wfile.write(f"data: {raw}\n\n".encode("utf-8"))
            self.wfile.flush()

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the RBII live timeline webapp.")
    parser.add_argument(
        "--base-dir",
        default=os.path.join("experimentOutputs", "rbii_program_events_v2"),
        help="Base directory for RBII V2 run outputs.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the browser after starting.")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    app = RBiiLiveApp(Path(args.base_dir))
    server = ThreadingHTTPServer((args.host, int(args.port)), make_handler(app))
    if args.open:
        webbrowser.open(f"http://{args.host}:{args.port}/browse/")
    server.serve_forever()


if __name__ == "__main__":
    main()
