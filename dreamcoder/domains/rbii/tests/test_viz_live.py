import threading
import json
import urllib.request
from pathlib import Path

import pytest
from http.server import ThreadingHTTPServer

from dreamcoder.domains.rbii.rbii_loop_v2 import RBIILoopV2, RBIIConfigV2
from dreamcoder.domains.rbii.rbii_viz_index import LIVE_SCHEMA_VERSION, LogIndex
from dreamcoder.domains.rbii.rbii_viz_live import RBiiLiveApp, make_handler
from dreamcoder.domains.rbii.rbii_test import _write_live_link_file


class _FakeProgram:
    def __init__(self, name: str, value):
        self.name = name
        self._value = value

    def __str__(self):
        return self.name

    def evaluate(self, _environment):
        return self._value


class _StaticEnumerator:
    def __init__(self, proposals):
        self.proposals = list(proposals)

    def propose_batch(self, _ctx):
        return list(self.proposals)


class _AlwaysFreezePolicy:
    def should_freeze(self, ctx):
        from dreamcoder.domains.rbii.rbii_loop_v2 import FreezeDecision

        return FreezeDecision(
            should_freeze=(ctx.incumbent is not None),
            predictor=ctx.incumbent,
            reason="test_always",
        )


class _PassThroughCandidatePolicy:
    def __init__(self, weight: float = 1.0):
        self.weight = float(weight)

    def admit_and_weight(self, _ctx, candidates):
        from dreamcoder.domains.rbii.rbii_loop_v2 import WeightedAdmission

        return [
            WeightedAdmission(proposal=candidate, initial_weight=self.weight, reason="test")
            for candidate in candidates
        ]


def _seed_state(seq: str):
    from dreamcoder.domains.rbii.rbii_state import RBIIState

    state = RBIIState()
    for ch in seq:
        state.observe(ch)
    return state


def _write_live_log(tmp_path: Path) -> Path:
    from dreamcoder.domains.rbii.rbii_loop_v2 import CandidateProposal

    state = _seed_state("a")
    cfg = RBIIConfigV2(
        alphabet=("a", "b"),
        min_time=0,
        validation_window=1,
        pool_target_size=1,
        verbose=False,
        event_log_dir=str(tmp_path),
        event_log_name="live_log",
    )
    program = _FakeProgram("always_a", lambda _view: "a")
    proposal = CandidateProposal(program=program, witness_bits=0.0, fn=program.evaluate([]))
    loop = RBIILoopV2(
        grammar=None,
        state=state,
        cfg=cfg,
        enumerator=_StaticEnumerator([proposal]),
        candidate_policy=_PassThroughCandidatePolicy(weight=1.0),
        freeze_policy=_AlwaysFreezePolicy(),
    )
    loop.observe_and_update("a")
    loop.observe_and_update("a")
    loop.close()
    return tmp_path / "live_log.jsonl"


def test_log_index_builds_closed_timestep_groups(tmp_path: Path):
    log_path = _write_live_log(tmp_path)
    index = LogIndex(log_path)
    index.refresh()

    assert index.schema_version == LIVE_SCHEMA_VERSION
    assert index.supports_live_pool_inspector is True
    assert index.first_timestep == 0
    assert index.last_closed_timestep == 2
    assert index.latest_observed_timestep == 2
    assert index.run_complete is True

    groups = index.read_step_groups(0, 2)
    assert [group["timestep"] for group in groups] == [0, 1, 2]
    assert groups[0]["events"][0]["event"] == "observe"
    assert groups[0]["events"][-1]["event"] == "pool_snapshot"


@pytest.fixture()
def live_server(tmp_path: Path):
    _write_live_log(tmp_path)
    app = RBiiLiveApp(tmp_path)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    except PermissionError:
        pytest.skip("sandbox disallows binding a local HTTP test server")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    try:
        yield base_url
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def test_live_server_lists_logs_and_returns_meta(live_server):
    payload = _get_json(f"{live_server}/api/list?path=")
    log_entries = [entry for entry in payload["entries"] if entry["kind"] == "log"]
    assert [entry["name"] for entry in log_entries] == ["live_log.jsonl"]

    meta = _get_json(f"{live_server}/api/meta?path=live_log.jsonl")
    assert meta["schema_version"] == LIVE_SCHEMA_VERSION
    assert meta["supports_live_pool_inspector"] is True
    assert meta["last_closed_timestep"] == 2


def test_live_server_chunk_and_stream(live_server):
    chunk = _get_json(f"{live_server}/api/chunk?path=live_log.jsonl&start_t=1&end_t=2")
    assert [group["timestep"] for group in chunk["step_groups"]] == [1, 2]
    assert all(group["events"][-1]["event"] == "pool_snapshot" for group in chunk["step_groups"])

    request = urllib.request.Request(
        f"{live_server}/api/stream?path=live_log.jsonl&from_t=0",
        headers={"Accept": "text/event-stream"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        body = response.read().decode("utf-8")

    assert "event: steps" in body
    assert '"timestep": 1' in body
    assert '"timestep": 2' in body
    assert '"run_complete": true' in body


def test_live_server_rejects_path_escape(tmp_path: Path):
    app = RBiiLiveApp(tmp_path)
    with pytest.raises(ValueError):
        app.resolve_relative_path("../escape", expect_dir=False)


def test_live_app_lists_logs_and_chunks_without_http(tmp_path: Path):
    log_path = _write_live_log(tmp_path)
    app = RBiiLiveApp(tmp_path)

    listing = app.list_relative_dir("")
    log_entries = [entry for entry in listing["entries"] if entry["kind"] == "log"]
    assert [entry["name"] for entry in log_entries] == [log_path.name]
    assert log_entries[0]["schema_version"] == LIVE_SCHEMA_VERSION

    chunk = app.chunk_payload(log_path.name, 1, 2)
    assert [group["timestep"] for group in chunk["step_groups"]] == [1, 2]
    assert all(group["events"][-1]["event"] == "pool_snapshot" for group in chunk["step_groups"])


def test_write_live_link_file_creates_html_link(tmp_path: Path):
    log_path = tmp_path / "run_0001" / "seq.jsonl"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("", encoding="utf-8")

    _write_live_link_file(str(log_path), str(tmp_path))

    link_path = log_path.with_suffix(".live.html")
    html = link_path.read_text(encoding="utf-8")
    assert "RBII Live Log" in html
    assert "/view/run_0001/seq.jsonl" in html
