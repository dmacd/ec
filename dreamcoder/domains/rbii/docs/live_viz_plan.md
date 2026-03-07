# RBII Live Timeline Webapp

Last updated: 2026-03-06

This document records the implementation plan and contract for the RBII V2
live timeline webapp.

## Summary

The live visualizer is a single local web server rooted at a base directory.
It serves HTML/CSS/JS for a browser app that monitors long-running RBII V2 logs
incrementally.

Key decisions:

1. The server does not compute or send a full render model.
2. The browser evaluates loaded-range timeline state from raw timestep groups.
3. The web app uses HTML/CSS for layout, not SVG.
4. The browser only renders a virtualized visible window.
5. Timestep click semantics use the post-step pool at timestep `t`, which
   predicts `t + 1`.
6. The live pool inspector is supported only for V2 logs with the new live
   schema.

## Log Schema Requirements

V2 live logs use:

- `schema_version = "rbii_v2_live_1"` on `run_start`
- one `pool_snapshot` event per observed timestep
- warmup `pool_snapshot` rows with empty pools

### `pool_snapshot`

Each `pool_snapshot` row contains:

- `timestep`
- `prediction_timestep`
- `warmup`
- `pool`, sorted by descending current weight

Each pool member contains:

- `rank`
- `active_id`
- `program_id`
- `program`
- `weight`
- `source`
- `duplicate_candidate`
- `incumbent`

The snapshot is emitted after the full post-step update, including rerank and
freeze.

## Server Responsibilities

The live server lives in `dreamcoder/domains/rbii/rbii_viz_live.py`.

It is responsible for:

1. Base-dir-relative path safety.
2. Directory and log listing.
3. Lightweight per-log timestep indexing.
4. Serving timestep chunks.
5. SSE streaming of newly closed timestep groups.

It is not responsible for:

1. episode extraction,
2. row layout,
3. card layout,
4. pool inspector derivation,
5. whole-log render-model generation.

## Indexing Model

The indexer lives in `dreamcoder/domains/rbii/rbii_viz_index.py`.

It records byte ranges for closed timestep groups.

A timestep group:

- begins at `observe(t)`
- ends at `pool_snapshot(t)`

Only closed timestep groups are exposed through chunking and SSE.

## Browser Responsibilities

The browser app lives under `dreamcoder/domains/rbii/viz_live/`.

It is responsible for:

1. loading only the tail of long logs initially,
2. fetching older history incrementally,
3. evaluating rows, frozen additions, episodes, and pool snapshots for the
   loaded range,
4. rendering only the visible rows and visible cards,
5. connecting scopes/cards using the local connector library,
6. showing the post-step pool inspector for the selected timestep.

## Routing Contract

HTTP routes:

- `/healthz`
- `/browse/<relative-dir>`
- `/view/<relative-log-path>`
- `/api/list?path=<relative-dir>`
- `/api/meta?path=<relative-log-path>`
- `/api/chunk?path=<relative-log-path>&start_t=<int>&end_t=<int>`
- `/api/stream?path=<relative-log-path>&from_t=<int>`

All file paths are relative to the server base dir.

## Runner Integration

`rbii_test.py --loop v2` should:

1. reuse or start the single live server for the V2 base output dir,
2. open one browser tab to the run directory page,
3. write one `.live.html` link file next to each JSONL log.

## Defaults

- host: `127.0.0.1`
- port: `8765`
- base dir: `experimentOutputs/rbii_program_events_v2`
- initial tail load: `300` timesteps
- historical fetch window: `300` timesteps
- row height: `28px`
- render buffer: `100` rows above and below the viewport

## Testing Expectations

Focused test coverage should include:

1. schema-version and `pool_snapshot` emission,
2. indexer metadata and timestep grouping,
3. API list/meta/chunk/stream behavior,
4. path escape rejection,
5. runner link-file generation,
6. static V2 SVG compatibility with the new log rows.
