from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LIVE_SCHEMA_VERSION = "rbii_v2_live_1"


@dataclass(frozen=True)
class StepGroupBounds:
    timestep: int
    start_byte: int
    end_byte: int


class LogIndex:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.schema_version: Optional[str] = None
        self.first_timestep: Optional[int] = None
        self.latest_observed_timestep: Optional[int] = None
        self.last_closed_timestep: Optional[int] = None
        self.run_complete: bool = False
        self._group_bounds_by_timestep: Dict[int, StepGroupBounds] = {}
        self._inode: Optional[Tuple[int, int]] = None
        self._size: int = 0
        self._open_group_timestep: Optional[int] = None
        self._open_group_start: Optional[int] = None

    @property
    def supports_live_pool_inspector(self) -> bool:
        return self.schema_version == LIVE_SCHEMA_VERSION

    def refresh(self) -> None:
        stat = self.path.stat()
        inode = (int(stat.st_dev), int(stat.st_ino))
        size = int(stat.st_size)

        if self._inode != inode or size < self._size:
            self._reset()
            scan_from = 0
        elif size == self._size:
            return
        else:
            scan_from = self._size

        self._scan(scan_from)
        self._inode = inode
        self._size = size

    def timesteps(self) -> List[int]:
        return sorted(self._group_bounds_by_timestep.keys())

    def read_step_groups(self, start_timestep: int, end_timestep: int) -> List[dict]:
        requested = [
            self._group_bounds_by_timestep[t]
            for t in self.timesteps()
            if start_timestep <= t <= end_timestep
        ]
        if not requested:
            return []

        groups: List[dict] = []
        with self.path.open("rb") as handle:
            for bounds in requested:
                handle.seek(bounds.start_byte)
                payload = handle.read(bounds.end_byte - bounds.start_byte)
                events = []
                for line in payload.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    events.append(json.loads(line.decode("utf-8")))
                if events:
                    groups.append({"timestep": bounds.timestep, "events": events})
        return groups

    def metadata(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "supports_live_pool_inspector": self.supports_live_pool_inspector,
            "run_complete": bool(self.run_complete),
            "first_timestep": self.first_timestep,
            "last_closed_timestep": self.last_closed_timestep,
            "latest_observed_timestep": self.latest_observed_timestep,
        }

    def _reset(self) -> None:
        self.schema_version = None
        self.first_timestep = None
        self.latest_observed_timestep = None
        self.last_closed_timestep = None
        self.run_complete = False
        self._group_bounds_by_timestep = {}
        self._inode = None
        self._size = 0
        self._open_group_timestep = None
        self._open_group_start = None

    def _scan(self, start_offset: int) -> None:
        with self.path.open("rb") as handle:
            handle.seek(start_offset)
            offset = start_offset
            while True:
                line_start = offset
                raw = handle.readline()
                if not raw:
                    break
                offset += len(raw)
                line = raw.strip()
                if not line:
                    continue
                row = json.loads(line.decode("utf-8"))
                self._consume_row(row, line_start, offset)

    def _consume_row(self, row: dict, line_start: int, line_end: int) -> None:
        event = row.get("event")
        timestep = row.get("timestep")

        if event == "run_start":
            schema_version = row.get("schema_version")
            if isinstance(schema_version, str):
                self.schema_version = schema_version
            return

        if event == "run_end":
            self.run_complete = True
            return

        if not isinstance(timestep, int):
            return

        if self.first_timestep is None:
            self.first_timestep = timestep
        else:
            self.first_timestep = min(self.first_timestep, timestep)

        if event == "observe":
            if self.latest_observed_timestep is None:
                self.latest_observed_timestep = timestep
            else:
                self.latest_observed_timestep = max(self.latest_observed_timestep, timestep)
            if self._open_group_timestep is None:
                self._open_group_timestep = timestep
                self._open_group_start = line_start
            elif self._open_group_timestep != timestep:
                self._open_group_timestep = timestep
                self._open_group_start = line_start
            return

        if event != "pool_snapshot":
            return

        if self._open_group_timestep != timestep or self._open_group_start is None:
            return

        self._group_bounds_by_timestep[timestep] = StepGroupBounds(
            timestep=timestep,
            start_byte=self._open_group_start,
            end_byte=line_end,
        )
        if self.last_closed_timestep is None:
            self.last_closed_timestep = timestep
        else:
            self.last_closed_timestep = max(self.last_closed_timestep, timestep)
        self._open_group_timestep = None
        self._open_group_start = None


class LogIndexCache:
    def __init__(self):
        self._by_path: Dict[Path, LogIndex] = {}

    def get(self, path: Path) -> LogIndex:
        path = Path(path)
        index = self._by_path.get(path)
        if index is None:
            index = LogIndex(path)
            self._by_path[path] = index
        index.refresh()
        return index
