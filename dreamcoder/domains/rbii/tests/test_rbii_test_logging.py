import os
from collections import namedtuple

from dreamcoder.domains.rbii import rbii_test
from dreamcoder.domains.rbii.rbii_test import (
    _FileEnumerationDebugHook,
    _make_enum_debug_hook_factory,
    _prune_old_logs_if_needed,
)


_Usage = namedtuple("_Usage", ["total", "used", "free"])


def test_prune_old_logs_if_needed_deletes_oldest_enum_dump_only(tmp_path, monkeypatch):
    base_dir = tmp_path / "logs"
    old_run = base_dir / "run_0001"
    newer_run = base_dir / "run_0002"
    current_run = base_dir / "run_0003"
    for path in [old_run, newer_run, current_run]:
        path.mkdir(parents=True)
        (path / "events.jsonl").write_text(path.name, encoding="utf-8")

    old_dump = old_run / "old_enumerate_debug.worker_0.log"
    newer_dump = newer_run / "newer_enumerate_debug.worker_0.log"
    current_dump = current_run / "current_enumerate_debug.worker_0.log"
    old_dump.write_text("old dump\n", encoding="utf-8")
    newer_dump.write_text("newer dump\n", encoding="utf-8")
    current_dump.write_text("current dump\n", encoding="utf-8")

    os.utime(old_run, (1, 1))
    os.utime(newer_run, (2, 2))
    os.utime(current_run, (3, 3))
    os.utime(old_dump, (1, 1))
    os.utime(newer_dump, (2, 2))
    os.utime(current_dump, (3, 3))

    def fake_disk_usage(_path):
        free = 700
        if not old_dump.exists():
            free += 400
        if not newer_dump.exists():
            free += 400
        return _Usage(total=10_000, used=0, free=free)

    monkeypatch.setattr(rbii_test.shutil, "disk_usage", fake_disk_usage)

    assert _prune_old_logs_if_needed(
        str(base_dir),
        preserve_paths=[str(current_run)],
        min_free_bytes=1_000,
    )

    assert old_run.exists()
    assert newer_run.exists()
    assert current_run.exists()
    assert not old_dump.exists()
    assert newer_dump.exists()
    assert current_dump.exists()
    assert (old_run / "events.jsonl").exists()


def test_file_enumeration_debug_hook_prunes_old_logs_before_writing(tmp_path, monkeypatch):
    base_dir = tmp_path / "logs"
    old_run = base_dir / "run_0001"
    current_run = base_dir / "run_0002"
    old_run.mkdir(parents=True)
    current_run.mkdir(parents=True)
    old_dump = old_run / "old_enumerate_debug.worker_0.log"
    old_dump.write_text("old\n", encoding="utf-8")
    (old_run / "events.jsonl").write_text("keep me\n", encoding="utf-8")

    def fake_disk_usage(_path):
        free = 700
        if not old_dump.exists():
            free += 500
        return _Usage(total=10_000, used=0, free=free)

    monkeypatch.setattr(rbii_test.shutil, "disk_usage", fake_disk_usage)

    hook = _FileEnumerationDebugHook(
        str(current_run / "enum.log"),
        cleanup_root=str(base_dir),
        min_free_bytes=1_000,
        check_every_writes=1,
    )
    hook.on_program(dt=0.25, likelihood=0.0, program="(lambda 0)")

    assert old_run.exists()
    assert not old_dump.exists()
    assert (old_run / "events.jsonl").exists()
    assert (current_run / "enum.log").read_text(encoding="utf-8") == "0.250000\t0.0\t(lambda 0)\n"


def test_make_enum_debug_hook_factory_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(rbii_test, "ENABLE_ENUM_DEBUG_LOGS", False)

    factory = _make_enum_debug_hook_factory(
        str(tmp_path / "run_0001" / "enum.log"),
        cleanup_root=str(tmp_path),
    )

    assert factory is None
