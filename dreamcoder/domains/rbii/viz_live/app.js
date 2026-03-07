(() => {
  const ROW_HEIGHT = 28;
  const INITIAL_WINDOW = 300;
  const HISTORY_WINDOW = 300;
  const RENDER_BUFFER = 100;
  const ROWS_WIDTH = 220;
  const LANE_STEP = 30;
  const MIN_CARD_WIDTH = 360;
  const MAX_CARD_WIDTH = 720;
  const CARD_HEIGHT = 42;
  const CARD_GAP = 5;
  const BRACKET_X0 = ROWS_WIDTH + 18;
  const BRACKET_ARM = 12;

  const state = {
    mode: window.RBII_BOOTSTRAP.mode,
    routePath: window.RBII_BOOTSTRAP.path || "",
    currentDirPath: "",
    currentLogPath: null,
    listEntries: [],
    meta: null,
    runComplete: false,
    groupsByTimestep: new Map(),
    loadedMin: null,
    loadedMax: null,
    firstTimestep: null,
    lastClosedTimestep: null,
    selectedTimestep: null,
    followLatest: true,
    paused: false,
    historyLoading: false,
    derived: null,
    eventSource: null,
    connectorObjects: [],
    renderQueued: false,
    directoryRefreshHandle: null,
    hoveredTimestep: null,
    hoveredActiveId: null,
    renderedStartIndex: null,
    renderedEndIndex: null,
    suppressScrollPause: false,
  };

  const visibleDom = {
    bracketNodesByActiveId: new Map(),
    cardNodesByActiveId: new Map(),
    connectorEntriesByActiveId: new Map(),
  };

  const refs = {
    browsePath: document.getElementById("browse-path"),
    entryList: document.getElementById("entry-list"),
    logTitle: document.getElementById("log-title"),
    logStatus: document.getElementById("log-status"),
    pauseButton: document.getElementById("pause-button"),
    followToggle: document.getElementById("follow-toggle"),
    unsupportedBanner: document.getElementById("unsupported-banner"),
    timelineScroll: document.getElementById("timeline-scroll"),
    timelineSurface: document.getElementById("timeline-surface"),
    rowsLayer: document.getElementById("rows-layer"),
    overlayLayer: document.getElementById("overlay-layer"),
    connectorLayer: document.getElementById("connector-layer"),
    inspectorHeader: document.getElementById("inspector-header"),
    inspectorSubtitle: document.getElementById("inspector-subtitle"),
    poolList: document.getElementById("pool-list"),
  };

  function encodePath(path) {
    return path
      .split("/")
      .filter((part) => part.length > 0)
      .map((part) => encodeURIComponent(part))
      .join("/");
  }

  async function fetchJson(url) {
    const response = await fetch(url, { credentials: "same-origin" });
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    return response.json();
  }

  function scheduleRender() {
    if (state.renderQueued) {
      return;
    }
    state.renderQueued = true;
    requestAnimationFrame(() => {
      state.renderQueued = false;
      render();
    });
  }

  function closeStream() {
    if (state.eventSource) {
      state.eventSource.close();
      state.eventSource = null;
    }
  }

  function clearConnectors() {
    const removed = new Set();
    for (const connector of state.connectorObjects) {
      if (removed.has(connector)) {
        continue;
      }
      connector.remove();
      removed.add(connector);
    }
    for (const entries of visibleDom.connectorEntriesByActiveId.values()) {
      for (const entry of entries) {
        if (removed.has(entry.connector)) {
          continue;
        }
        entry.connector.remove();
        removed.add(entry.connector);
      }
    }
    state.connectorObjects = [];
  }

  function poolHighlightedActiveIds() {
    if (Number.isInteger(state.hoveredTimestep) && state.derived) {
      const snapshot = state.derived.poolSnapshotsByTimestep.get(state.hoveredTimestep);
      if (snapshot && Array.isArray(snapshot.pool)) {
        return new Set(
          snapshot.pool
            .map((member) => member.active_id)
            .filter((activeId) => Number.isInteger(activeId)),
        );
      }
    }
    return new Set();
  }

  function pushVisibleNode(nodeMap, activeId, node) {
    if (!Number.isInteger(activeId)) {
      return;
    }
    const bucket = nodeMap.get(activeId) || [];
    bucket.push(node);
    nodeMap.set(activeId, bucket);
  }

  function pushConnectorEntry(activeId, entry) {
    if (!Number.isInteger(activeId)) {
      return;
    }
    const bucket = visibleDom.connectorEntriesByActiveId.get(activeId) || [];
    bucket.push(entry);
    visibleDom.connectorEntriesByActiveId.set(activeId, bucket);
  }

  function toggleVisibleClass(nodeMap, className, predicate) {
    for (const [activeId, nodes] of nodeMap.entries()) {
      const enabled = predicate(activeId);
      for (const node of nodes) {
        node.classList.toggle(className, enabled);
      }
    }
  }

  function applyHoverHighlights() {
    const poolIds = poolHighlightedActiveIds();
    const focusedActiveId = Number.isInteger(state.hoveredActiveId) ? state.hoveredActiveId : null;

    toggleVisibleClass(
      visibleDom.cardNodesByActiveId,
      "pool-highlighted",
      (activeId) => poolIds.has(activeId) || activeId === focusedActiveId,
    );
    toggleVisibleClass(
      visibleDom.bracketNodesByActiveId,
      "highlighted",
      (activeId) => poolIds.has(activeId) || activeId === focusedActiveId,
    );

    for (const [activeId, entries] of visibleDom.connectorEntriesByActiveId.entries()) {
      const highlighted = poolIds.has(activeId) || activeId === focusedActiveId;
      for (const entry of entries) {
        if (entry.highlighted === highlighted) {
          continue;
        }
        entry.connector.remove();
        entry.connector = createConnector(entry.anchor, entry.card, entry.middleX, {
          highlighted,
          solid: entry.solid,
        });
        entry.highlighted = highlighted;
      }
    }
  }

  function applyDashedConnectorStyle(connector, color) {
    for (const segment of connector._segments || []) {
      const width = parseFloat(segment.style.width || "0");
      const height = parseFloat(segment.style.height || "0");
      const direction = width >= height ? "to right" : "to bottom";
      segment.style.background = `repeating-linear-gradient(${direction}, ${color} 0 7px, transparent 7px 12px)`;
    }
  }

  function createConnector(anchor, card, middleX, options = {}) {
    const highlighted = Boolean(options.highlighted);
    const solid = Boolean(options.solid);
    const color = highlighted ? "#8f4b26" : "#6b6257";
    const connector = new window.LeaderLine(anchor, card, {
      container: refs.connectorLayer,
      color,
      size: highlighted ? 4 : 2,
      middleX,
    });
    if (!solid) {
      applyDashedConnectorStyle(connector, color);
    }
    return connector;
  }

  function formatBits(value) {
    if (typeof value !== "number" || Number.isNaN(value)) {
      return "-";
    }
    if (Math.abs(value) >= 0.01) {
      return `${value.toFixed(2)}b`;
    }
    return `${value.toExponential(2)}b`;
  }

  function formatWeight(value) {
    if (typeof value !== "number" || Number.isNaN(value)) {
      return "-";
    }
    if (Math.abs(value) >= 1e-3) {
      return value.toFixed(4);
    }
    return value.toExponential(2);
  }

  function recordMeta(metaByActiveId, raw) {
    const activeId = raw && raw.active_id;
    if (!Number.isInteger(activeId)) {
      return;
    }
    const existing = metaByActiveId.get(activeId) || {
      activeId,
      programText: `<active ${activeId}>`,
      programId: null,
      duplicateCandidate: false,
      source: null,
    };
    if (typeof raw.program === "string") {
      existing.programText = raw.program;
    }
    if (Number.isInteger(raw.program_id)) {
      existing.programId = raw.program_id;
    }
    if (raw.duplicate_candidate) {
      existing.duplicateCandidate = true;
    }
    if (typeof raw.source === "string") {
      existing.source = raw.source;
    }
    metaByActiveId.set(activeId, existing);
  }

  function deriveModel() {
    const timesteps = Array.from(state.groupsByTimestep.keys()).sort((a, b) => a - b);
    const metaByActiveId = new Map();
    const rowsByTimestep = new Map();
    const poolSnapshotsByTimestep = new Map();
    const storeAdditionsByTimestep = new Map();
    const incumbentTimes = new Map();
    const openByActiveId = new Map();
    const episodes = [];

    if (timesteps.length === 0) {
      return {
        timesteps,
        rowsByTimestep,
        poolSnapshotsByTimestep,
        storeAdditionsByTimestep,
        episodes,
      };
    }

    const firstGroup = state.groupsByTimestep.get(timesteps[0]);
    if (firstGroup) {
      const firstSnapshot = firstGroup.events.find((event) => event.event === "pool_snapshot");
      if (firstSnapshot && Array.isArray(firstSnapshot.pool)) {
        for (const member of firstSnapshot.pool) {
          if (!Number.isInteger(member.active_id)) {
            continue;
          }
          recordMeta(metaByActiveId, member);
          openByActiveId.set(member.active_id, {
            activeId: member.active_id,
            programId: Number.isInteger(member.program_id) ? member.program_id : null,
            programText: typeof member.program === "string" ? member.program : `<active ${member.active_id}>`,
            startT: firstSnapshot.timestep + 1,
            endT: firstSnapshot.timestep + 1,
            duplicateCandidate: Boolean(member.duplicate_candidate),
            source: typeof member.source === "string" ? member.source : null,
            usedForPrediction: false,
            lane: 0,
          });
        }
      }
    }

    for (const timestep of timesteps) {
      const group = state.groupsByTimestep.get(timestep);
      if (!group) {
        continue;
      }
      const events = group.events;
      const observe = events.find((event) => event.event === "observe") || null;
      const snapshot = events.find((event) => event.event === "pool_snapshot") || null;

      for (const event of events) {
        recordMeta(metaByActiveId, event);
        if (event.event === "pool_snapshot" && Array.isArray(event.pool)) {
          for (const member of event.pool) {
            recordMeta(metaByActiveId, member);
          }
        }
      }

      if (observe) {
        rowsByTimestep.set(timestep, {
          timestep,
          observed: observe.observed || "",
          loglossBits: typeof observe.logloss_bits === "number" ? observe.logloss_bits : null,
          activeId: Number.isInteger(observe.active_id) ? observe.active_id : null,
          warmup: Boolean(observe.warmup),
        });
        if (Number.isInteger(observe.active_id)) {
          const times = incumbentTimes.get(observe.active_id) || [];
          times.push(timestep);
          incumbentTimes.set(observe.active_id, times);
        }
      }

      if (snapshot) {
        poolSnapshotsByTimestep.set(timestep, snapshot);
      }

      const freezeProgramIds = [];
      for (const event of events) {
        if (event.event === "freeze" && Number.isInteger(event.program_id)) {
          freezeProgramIds.push(event.program_id);
        }
      }
      if (freezeProgramIds.length > 0) {
        freezeProgramIds.sort((a, b) => a - b);
        storeAdditionsByTimestep.set(timestep, freezeProgramIds);
      }

      for (const event of events) {
        if (!Number.isInteger(event.active_id)) {
          continue;
        }
        const activeId = event.active_id;
        const meta = metaByActiveId.get(activeId) || {
          activeId,
          programText: `<active ${activeId}>`,
          programId: null,
          duplicateCandidate: false,
          source: null,
        };
        const eventProgramId = Number.isInteger(event.program_id) ? event.program_id : meta.programId;

        if (event.event === "pool_enter") {
          if (openByActiveId.has(activeId)) {
            const existing = openByActiveId.get(activeId);
            existing.endT = Math.max(existing.startT, timestep - 1);
            episodes.push(existing);
          }
          openByActiveId.set(activeId, {
            activeId,
            programId: eventProgramId,
            programText: meta.programText,
            startT: timestep,
            endT: timestep,
            duplicateCandidate: Boolean(meta.duplicateCandidate),
            source: meta.source,
            usedForPrediction: false,
            lane: 0,
          });
        } else if (event.event === "pool_exit") {
          if (openByActiveId.has(activeId)) {
            const episode = openByActiveId.get(activeId);
            episode.programId = eventProgramId;
            episode.endT = Math.max(episode.startT, timestep);
            episodes.push(episode);
            openByActiveId.delete(activeId);
          } else {
            episodes.push({
              activeId,
              programId: eventProgramId,
              programText: meta.programText,
              startT: timestep,
              endT: timestep,
              duplicateCandidate: Boolean(meta.duplicateCandidate),
              source: meta.source,
              usedForPrediction: false,
              lane: 0,
            });
          }
        }
      }
    }

    const lastTimestep = timesteps[timesteps.length - 1];
    for (const episode of openByActiveId.values()) {
      if (episode.startT > lastTimestep) {
        continue;
      }
      episode.endT = Math.max(episode.startT, lastTimestep);
      episodes.push(episode);
    }

    const coveredByActiveId = new Map();
    for (const episode of episodes) {
      const spans = coveredByActiveId.get(episode.activeId) || [];
      spans.push([episode.startT, episode.endT]);
      coveredByActiveId.set(episode.activeId, spans);
    }

    for (const [activeId, times] of incumbentTimes.entries()) {
      const spans = coveredByActiveId.get(activeId) || [];
      for (const timestep of times) {
        const covered = spans.some(([startT, endT]) => startT <= timestep && timestep <= endT);
        if (covered) {
          continue;
        }
        const meta = metaByActiveId.get(activeId) || {
          activeId,
          programText: `<active ${activeId}>`,
          programId: null,
          duplicateCandidate: false,
          source: null,
        };
        episodes.push({
          activeId,
          programId: meta.programId,
          programText: meta.programText,
          startT: timestep,
          endT: timestep,
          duplicateCandidate: Boolean(meta.duplicateCandidate),
          source: meta.source,
          usedForPrediction: true,
          lane: 0,
        });
      }
    }

    episodes.sort((a, b) => {
      if (a.startT !== b.startT) {
        return a.startT - b.startT;
      }
      if (a.endT !== b.endT) {
        return a.endT - b.endT;
      }
      return a.activeId - b.activeId;
    });

    for (const episode of episodes) {
      const times = incumbentTimes.get(episode.activeId) || [];
      episode.usedForPrediction = episode.usedForPrediction || times.some((t) => episode.startT <= t && t <= episode.endT);
    }

    assignLanes(episodes);
    return {
      timesteps,
      rowsByTimestep,
      poolSnapshotsByTimestep,
      storeAdditionsByTimestep,
      episodes,
    };
  }

  function assignLanes(episodes) {
    const laneEnds = [];
    for (const episode of episodes) {
      let assignedLane = null;
      for (let lane = 0; lane < laneEnds.length; lane += 1) {
        if (episode.startT > laneEnds[lane]) {
          assignedLane = lane;
          break;
        }
      }
      if (assignedLane === null) {
        assignedLane = laneEnds.length;
        laneEnds.push(episode.endT);
      } else {
        laneEnds[assignedLane] = episode.endT;
      }
      episode.lane = assignedLane;
    }
  }

  function visibleWindow() {
    const timesteps = state.derived ? state.derived.timesteps : [];
    if (timesteps.length === 0) {
      return { startIndex: 0, endIndex: -1, rowTopByTimestep: new Map(), maxLane: 0 };
    }
    const scrollTop = refs.timelineScroll.scrollTop;
    const viewportHeight = refs.timelineScroll.clientHeight || 0;
    const rawStart = Math.floor(scrollTop / ROW_HEIGHT) - RENDER_BUFFER;
    const rawEnd = Math.ceil((scrollTop + viewportHeight) / ROW_HEIGHT) + RENDER_BUFFER;
    const startIndex = Math.max(0, rawStart);
    const endIndex = Math.min(timesteps.length - 1, rawEnd);
    const rowTopByTimestep = new Map();
    for (let index = startIndex; index <= endIndex; index += 1) {
      rowTopByTimestep.set(timesteps[index], index * ROW_HEIGHT);
    }
    let maxLane = 0;
    for (const episode of state.derived.episodes) {
      if (episode.endT < timesteps[startIndex] || episode.startT > timesteps[endIndex]) {
        continue;
      }
      maxLane = Math.max(maxLane, episode.lane);
    }
    return { startIndex, endIndex, rowTopByTimestep, maxLane };
  }

  function needsViewportRender() {
    if (!state.derived || state.derived.timesteps.length === 0) {
      return false;
    }
    if (state.renderedStartIndex === null || state.renderedEndIndex === null) {
      return true;
    }
    const scrollTop = refs.timelineScroll.scrollTop;
    const viewportHeight = refs.timelineScroll.clientHeight || 0;
    const visibleStart = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT));
    const visibleEnd = Math.min(
      state.derived.timesteps.length - 1,
      Math.ceil((scrollTop + viewportHeight) / ROW_HEIGHT),
    );
    const rerenderGuard = Math.max(16, Math.floor(RENDER_BUFFER / 2));
    return (
      visibleStart < (state.renderedStartIndex + rerenderGuard) ||
      visibleEnd > (state.renderedEndIndex - rerenderGuard)
    );
  }

  function selectedSnapshot() {
    if (!state.derived || state.selectedTimestep === null) {
      return null;
    }
    return state.derived.poolSnapshotsByTimestep.get(state.selectedTimestep) || null;
  }

  function renderEntries() {
    refs.entryList.innerHTML = "";
    refs.browsePath.textContent = state.currentDirPath || "/";

    if (state.listEntries.length === 0) {
      const empty = document.createElement("div");
      empty.className = "empty-state";
      empty.textContent = "No logs or run directories found here.";
      refs.entryList.appendChild(empty);
      return;
    }

    for (const entry of state.listEntries) {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "entry-card";
      if (entry.kind === "log" && entry.path === state.currentLogPath) {
        card.classList.add("active");
      }
      if (entry.kind === "dir" && entry.path === state.currentDirPath) {
        card.classList.add("active");
      }
      card.addEventListener("click", () => {
        if (entry.kind === "dir") {
          window.location.href = `/browse/${encodePath(entry.path)}`;
          return;
        }
        window.location.href = `/view/${encodePath(entry.path)}`;
      });

      const name = document.createElement("div");
      name.className = "entry-name";
      const title = document.createElement("span");
      title.textContent = entry.name;
      name.appendChild(title);
      const kind = document.createElement("span");
      kind.className = `status-pill ${entry.kind === "log" ? (entry.run_complete ? "complete" : entry.supports_live_pool_inspector ? "live" : "unsupported") : ""}`;
      kind.textContent = entry.kind === "dir" ? "dir" : entry.run_complete ? "complete" : entry.supports_live_pool_inspector ? "live" : "unsupported";
      name.appendChild(kind);
      card.appendChild(name);

      const path = document.createElement("div");
      path.className = "entry-path";
      path.textContent = entry.path;
      card.appendChild(path);

      if (entry.kind === "log") {
        const meta = document.createElement("div");
        meta.className = "entry-meta";
        meta.appendChild(chip(entry.last_closed_timestep === null ? "no rows" : `t=${entry.last_closed_timestep}`));
        meta.appendChild(chip(entry.size_bytes ? `${Math.round(entry.size_bytes / 1024)} KB` : "0 KB"));
        card.appendChild(meta);
      }

      refs.entryList.appendChild(card);
    }
  }

  function chip(text) {
    const node = document.createElement("span");
    node.className = "chip";
    node.textContent = text;
    return node;
  }

  function renderInspector() {
    refs.poolList.innerHTML = "";
    const snapshot = selectedSnapshot();
    if (state.selectedTimestep === null) {
      refs.inspectorHeader.textContent = "Select a timestep";
      refs.inspectorSubtitle.textContent = "";
      refs.poolList.innerHTML = '<div class="empty-state">Click a row in the timeline to inspect its post-step pool.</div>';
      return;
    }

    refs.inspectorHeader.textContent = `t = ${state.selectedTimestep}`;
    refs.inspectorSubtitle.textContent = snapshot && snapshot.warmup
      ? "warmup row"
      : `post-step pool, predicts t + 1 = ${state.selectedTimestep + 1}`;

    if (!snapshot || !Array.isArray(snapshot.pool) || snapshot.pool.length === 0) {
      refs.poolList.innerHTML = '<div class="empty-state">Pool empty at this timestep.</div>';
      return;
    }

    for (const member of snapshot.pool) {
      const card = document.createElement("div");
      card.className = "pool-card";
      const header = document.createElement("div");
      header.className = "pool-card-header";
      const title = document.createElement("span");
      title.textContent = `#${member.rank} @${member.active_id}${Number.isInteger(member.program_id) ? ` [#${member.program_id}]` : ""}`;
      header.appendChild(title);
      const weight = document.createElement("span");
      weight.textContent = formatWeight(member.weight);
      header.appendChild(weight);
      card.appendChild(header);

      const meta = document.createElement("div");
      meta.className = "pool-card-meta";
      if (member.incumbent) {
        meta.appendChild(chip("incumbent"));
      }
      if (member.duplicate_candidate) {
        meta.appendChild(chip("duplicate"));
      }
      if (member.source) {
        meta.appendChild(chip(member.source));
      }
      card.appendChild(meta);

      const program = document.createElement("div");
      program.className = "pool-program";
      program.textContent = member.program || "<unknown>";
      card.appendChild(program);
      refs.poolList.appendChild(card);
    }
  }

  function renderToolbar() {
    refs.logTitle.textContent = state.currentLogPath || "No log selected";
    refs.pauseButton.textContent = state.paused ? "Resume" : "Pause";
    refs.followToggle.checked = state.followLatest;

    let statusClass = "status-pill";
    let statusText = "idle";
    if (state.meta) {
      if (!state.meta.supports_live_pool_inspector) {
        statusClass += " unsupported";
        statusText = "unsupported";
      } else if (state.runComplete) {
        statusClass += " complete";
        statusText = "complete";
      } else {
        statusClass += " live";
        statusText = state.paused ? "paused" : "live";
      }
    }
    refs.logStatus.className = statusClass;
    refs.logStatus.textContent = statusText;
  }

  function renderTimeline() {
    clearConnectors();
    refs.rowsLayer.innerHTML = "";
    refs.overlayLayer.innerHTML = "";
    visibleDom.bracketNodesByActiveId = new Map();
    visibleDom.cardNodesByActiveId = new Map();
    visibleDom.connectorEntriesByActiveId = new Map();

    if (!state.derived || state.derived.timesteps.length === 0) {
      state.renderedStartIndex = null;
      state.renderedEndIndex = null;
      refs.timelineSurface.style.height = `${ROW_HEIGHT}px`;
      refs.timelineSurface.style.width = `${ROWS_WIDTH + 480}px`;
      return;
    }

    const view = visibleWindow();
    state.renderedStartIndex = view.startIndex;
    state.renderedEndIndex = view.endIndex;
    const timesteps = state.derived.timesteps;
    const startTimestep = timesteps[view.startIndex];
    const endTimestep = timesteps[view.endIndex];
    const totalHeight = timesteps.length * ROW_HEIGHT;
    const cardX = BRACKET_X0 + ((view.maxLane + 1) * LANE_STEP) + 28;
    const dynamicCardWidth = Math.max(
      MIN_CARD_WIDTH,
      Math.min(
        MAX_CARD_WIDTH,
        refs.timelineScroll.clientWidth - cardX - 32,
      ),
    );
    const totalWidth = Math.max(refs.timelineScroll.clientWidth - 2, cardX + dynamicCardWidth + 32);
    refs.timelineSurface.style.height = `${totalHeight}px`;
    refs.timelineSurface.style.width = `${totalWidth}px`;

    for (let index = view.startIndex; index <= view.endIndex; index += 1) {
      const timestep = timesteps[index];
      const row = state.derived.rowsByTimestep.get(timestep) || {
        timestep,
        observed: "",
        loglossBits: null,
        activeId: null,
        warmup: false,
      };
      const top = index * ROW_HEIGHT;
      const node = document.createElement("div");
      node.className = "timeline-row";
      if (timestep === state.selectedTimestep) {
        node.classList.add("selected");
      }
      node.style.top = `${top}px`;
      node.dataset.timestep = String(timestep);
      node.innerHTML = `
        <div class="row-timestep">t=${timestep}</div>
        <div class="row-symbol">${escapeHtml(row.observed || "")}</div>
        <div class="row-loss">${formatBits(row.loglossBits)}</div>
        <div class="row-freeze">${renderFreezeText(timestep)}</div>
      `;
      node.addEventListener("click", () => selectTimestep(timestep, { pause: true }));
      node.addEventListener("mouseenter", () => {
        state.hoveredTimestep = timestep;
        applyHoverHighlights();
      });
      node.addEventListener("mouseleave", () => {
        if (state.hoveredTimestep === timestep) {
          state.hoveredTimestep = null;
          applyHoverHighlights();
        }
      });
      refs.rowsLayer.appendChild(node);
    }

    const visibleEpisodes = state.derived.episodes.filter((episode) => episode.endT >= startTimestep && episode.startT <= endTimestep);
    const layouts = layoutEpisodes(visibleEpisodes, view, timesteps, {
      cardWidth: dynamicCardWidth,
      cardX,
    });

    for (const layout of layouts) {
      const bracket = document.createElement("div");
      bracket.className = "scope-bracket";
      bracket.style.left = `${layout.bracketX - BRACKET_ARM}px`;
      bracket.style.top = `${layout.y1}px`;
      bracket.style.height = `${Math.max(2, layout.y2 - layout.y1)}px`;
      bracket.addEventListener("click", () => selectTimestep(layout.episode.startT, { pause: true }));
      refs.overlayLayer.appendChild(bracket);
      pushVisibleNode(visibleDom.bracketNodesByActiveId, layout.episode.activeId, bracket);

      const anchor = document.createElement("div");
      anchor.className = "scope-anchor";
      anchor.style.left = `${layout.anchorX}px`;
      anchor.style.top = `${layout.anchorY}px`;
      refs.overlayLayer.appendChild(anchor);

      const card = document.createElement("div");
      card.className = "program-card";
      if (layout.episode.usedForPrediction) {
        card.classList.add("incumbent");
      } else {
        bracket.classList.add("secondary");
      }
      if (layout.episode.duplicateCandidate) {
        card.classList.add("duplicate");
      }
      card.style.width = `${layout.cardWidth}px`;
      card.style.left = `${layout.boxX}px`;
      card.style.top = `${layout.boxY}px`;
      card.dataset.timestep = String(layout.episode.startT);
      card.innerHTML = `
        <div class="program-card-row">
          <span class="program-id">@${layout.episode.activeId}${layout.episode.programId === null ? "" : ` [#${layout.episode.programId}]`}</span>
          <span class="program-flags"></span>
          <span class="program-text">${escapeHtml(layout.episode.programText)}</span>
        </div>
      `;
      const flags = card.querySelector(".program-flags");
      if (layout.episode.usedForPrediction) {
        flags.appendChild(chip("incumbent"));
      }
      if (layout.episode.duplicateCandidate) {
        flags.appendChild(chip("duplicate"));
      }
      card.addEventListener("mouseenter", () => {
        state.hoveredActiveId = layout.episode.activeId;
        applyHoverHighlights();
      });
      card.addEventListener("mouseleave", () => {
        if (state.hoveredActiveId === layout.episode.activeId) {
          state.hoveredActiveId = null;
          applyHoverHighlights();
        }
      });
      card.addEventListener("click", () => selectTimestep(layout.episode.startT, { pause: true }));
      refs.overlayLayer.appendChild(card);
      pushVisibleNode(visibleDom.cardNodesByActiveId, layout.episode.activeId, card);

      const connector = createConnector(anchor, card, layout.routerX, {
        highlighted: false,
        solid: layout.episode.usedForPrediction,
      });
      state.connectorObjects.push(connector);
      pushConnectorEntry(layout.episode.activeId, {
        anchor,
        card,
        connector,
        highlighted: false,
        middleX: layout.routerX,
        solid: layout.episode.usedForPrediction,
      });
    }
    applyHoverHighlights();
  }

  function layoutEpisodes(episodes, view, timesteps, options) {
    const { cardWidth, cardX } = options;
    const rowIndexByTimestep = new Map();
    timesteps.forEach((timestep, index) => rowIndexByTimestep.set(timestep, index));
    const layouts = [];
    const rowOffsetStep = 7;
    const startOffsets = eventOffsets(episodes, "startT", rowOffsetStep);
    const endOffsets = eventOffsets(episodes, "endT", rowOffsetStep);

    for (const episode of episodes) {
      const startIndex = rowIndexByTimestep.get(episode.startT);
      const endIndex = rowIndexByTimestep.get(episode.endT);
      if (startIndex === undefined || endIndex === undefined) {
        continue;
      }
      let y1 = startIndex * ROW_HEIGHT + (ROW_HEIGHT / 2) + (startOffsets.get(episode.activeId) || 0);
      let y2 = endIndex * ROW_HEIGHT + (ROW_HEIGHT / 2) + (endOffsets.get(episode.activeId) || 0);
      if (y2 < y1) {
        const tmp = y1;
        y1 = y2;
        y2 = tmp;
      }
      layouts.push({
        episode,
        y1,
        y2,
        desiredCenter: (y1 + y2) / 2,
        bracketX: BRACKET_X0 + (episode.lane * LANE_STEP),
        routerX: 0,
        boxX: 0,
        boxY: 0,
        anchorY: 0,
        anchorX: 0,
      });
    }

    const sortedForRouter = [...layouts].sort((a, b) => {
      if (a.episode.startT !== b.episode.startT) {
        return a.episode.startT - b.episode.startT;
      }
      if (a.episode.endT !== b.episode.endT) {
        return a.episode.endT - b.episode.endT;
      }
      return a.episode.activeId - b.episode.activeId;
    });
    const routerX0 = cardX - 18;
    const routerStep = 8;

    sortedForRouter.forEach((layout, index) => {
      layout.routerX = routerX0 + (index * routerStep);
      layout.boxX = cardX;
      layout.cardWidth = cardWidth;
      layout.anchorX = layout.bracketX;
    });

    const ordered = [...sortedForRouter].sort((a, b) => a.desiredCenter - b.desiredCenter);
    let previousBottom = -1e9;
    for (const layout of ordered) {
      let top = layout.desiredCenter - (CARD_HEIGHT / 2);
      const minTop = previousBottom + CARD_GAP;
      if (top < minTop) {
        top = minTop;
      }
      layout.boxY = top;
      previousBottom = top + CARD_HEIGHT;
      const boxCenter = top + (CARD_HEIGHT / 2);
      if (layout.y2 - layout.y1 <= 6) {
        layout.anchorY = (layout.y1 + layout.y2) / 2;
      } else {
        layout.anchorY = Math.max(layout.y1 + 3, Math.min(boxCenter, layout.y2 - 3));
      }
    }

    return layouts;
  }

  function eventOffsets(episodes, key, step) {
    const groups = new Map();
    for (const episode of episodes) {
      const timestep = episode[key];
      const bucket = groups.get(timestep) || [];
      bucket.push(episode);
      groups.set(timestep, bucket);
    }
    const offsets = new Map();
    for (const [timestep, group] of groups.entries()) {
      const ordered = [...group].sort((a, b) => {
        if (a.lane !== b.lane) {
          return a.lane - b.lane;
        }
        return a.activeId - b.activeId;
      });
      const mid = (ordered.length - 1) / 2;
      ordered.forEach((episode, index) => {
        offsets.set(episode.activeId, (index - mid) * step);
      });
    }
    return offsets;
  }

  function renderFreezeText(timestep) {
    const ids = state.derived.storeAdditionsByTimestep.get(timestep) || [];
    if (ids.length === 0) {
      return "";
    }
    return `freeze ${ids.map((id) => `#${id}`).join(", ")}`;
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function renderUnsupported() {
    if (!state.meta || state.meta.supports_live_pool_inspector) {
      refs.unsupportedBanner.classList.add("hidden");
      refs.unsupportedBanner.textContent = "";
      return;
    }
    refs.unsupportedBanner.classList.remove("hidden");
    refs.unsupportedBanner.textContent = "This log does not support the live pool inspector. Use a V2 log with schema_version rbii_v2_live_1.";
  }

  function render() {
    renderEntries();
    renderToolbar();
    renderUnsupported();
    renderTimeline();
    renderInspector();
  }

  function ingestStepGroups(groups) {
    for (const group of groups) {
      if (!Number.isInteger(group.timestep)) {
        continue;
      }
      state.groupsByTimestep.set(group.timestep, group);
      if (state.loadedMin === null || group.timestep < state.loadedMin) {
        state.loadedMin = group.timestep;
      }
      if (state.loadedMax === null || group.timestep > state.loadedMax) {
        state.loadedMax = group.timestep;
      }
    }
    state.derived = deriveModel();
  }

  async function loadChunk(startTimestep, endTimestep) {
    if (!state.currentLogPath) {
      return;
    }
    const payload = await fetchJson(`/api/chunk?path=${encodeURIComponent(state.currentLogPath)}&start_t=${startTimestep}&end_t=${endTimestep}`);
    state.lastClosedTimestep = payload.last_closed_timestep;
    state.runComplete = Boolean(payload.run_complete);
    ingestStepGroups(payload.step_groups || []);
    scheduleRender();
  }

  function latestTimestep() {
    if (!state.derived || state.derived.timesteps.length === 0) {
      return null;
    }
    return state.derived.timesteps[state.derived.timesteps.length - 1];
  }

  function selectTimestep(timestep, options = {}) {
    state.selectedTimestep = timestep;
    if (options.pause) {
      state.paused = true;
    }
    scheduleRender();
  }

  function scrollToLatest() {
    const last = latestTimestep();
    if (last === null || !state.derived) {
      return;
    }
    const index = state.derived.timesteps.indexOf(last);
    if (index < 0) {
      return;
    }
    state.suppressScrollPause = true;
    refs.timelineScroll.scrollTop = Math.max(0, (index * ROW_HEIGHT) - refs.timelineScroll.clientHeight + (ROW_HEIGHT * 4));
    requestAnimationFrame(() => {
      state.suppressScrollPause = false;
    });
  }

  async function ensureOlderHistory() {
    if (state.historyLoading || !state.currentLogPath || state.loadedMin === null || state.firstTimestep === null) {
      return;
    }
    if (state.loadedMin <= state.firstTimestep) {
      return;
    }
    const top = refs.timelineScroll.scrollTop;
    if (top > ROW_HEIGHT * 20) {
      return;
    }
    state.historyLoading = true;
    try {
      const oldHeight = refs.timelineSurface.offsetHeight;
      const previousTop = refs.timelineScroll.scrollTop;
      const nextStart = Math.max(state.firstTimestep, state.loadedMin - HISTORY_WINDOW - 1);
      const nextEnd = state.loadedMin - 1;
      await loadChunk(nextStart, nextEnd);
      await new Promise((resolve) => requestAnimationFrame(resolve));
      const newHeight = refs.timelineSurface.offsetHeight;
      refs.timelineScroll.scrollTop = previousTop + Math.max(0, newHeight - oldHeight);
    } finally {
      state.historyLoading = false;
    }
  }

  function openStream() {
    closeStream();
    if (!state.currentLogPath || !state.meta || !state.meta.supports_live_pool_inspector) {
      return;
    }
    const fromTimestep = state.loadedMax === null ? -1 : state.loadedMax;
    const source = new EventSource(`/api/stream?path=${encodeURIComponent(state.currentLogPath)}&from_t=${fromTimestep}`);
    source.addEventListener("steps", (event) => {
      const payload = JSON.parse(event.data);
      ingestStepGroups(payload.step_groups || []);
      state.runComplete = Boolean(payload.run_complete);
      if (!state.paused && state.followLatest) {
        const last = latestTimestep();
        if (last !== null) {
          state.selectedTimestep = last;
        }
        scheduleRender();
        requestAnimationFrame(scrollToLatest);
      } else {
        scheduleRender();
      }
    });
    source.addEventListener("status", (event) => {
      const payload = JSON.parse(event.data);
      state.runComplete = Boolean(payload.run_complete);
      scheduleRender();
      if (state.runComplete) {
        closeStream();
      }
    });
    source.onerror = () => {
      if (state.runComplete) {
        closeStream();
      }
    };
    state.eventSource = source;
  }

  async function openLog(path) {
    closeStream();
    state.currentLogPath = path;
    state.groupsByTimestep = new Map();
    state.loadedMin = null;
    state.loadedMax = null;
    state.derived = null;
    state.selectedTimestep = null;
    state.renderedStartIndex = null;
    state.renderedEndIndex = null;

    state.meta = await fetchJson(`/api/meta?path=${encodeURIComponent(path)}`);
    state.runComplete = Boolean(state.meta.run_complete);
    state.firstTimestep = Number.isInteger(state.meta.first_timestep) ? state.meta.first_timestep : 0;
    state.lastClosedTimestep = Number.isInteger(state.meta.last_closed_timestep) ? state.meta.last_closed_timestep : null;

    if (!state.meta.supports_live_pool_inspector || state.lastClosedTimestep === null) {
      scheduleRender();
      return;
    }

    const initialStart = Math.max(state.firstTimestep, state.lastClosedTimestep - INITIAL_WINDOW);
    await loadChunk(initialStart, state.lastClosedTimestep);
    const last = latestTimestep();
    if (last !== null) {
      state.selectedTimestep = last;
    }
    scheduleRender();
    requestAnimationFrame(scrollToLatest);
    openStream();
  }

  function chooseDefaultLog(entries) {
    const logs = entries.filter((entry) => entry.kind === "log");
    if (logs.length === 0) {
      return null;
    }
    logs.sort((a, b) => {
      if (a.run_complete !== b.run_complete) {
        return a.run_complete ? 1 : -1;
      }
      return (b.mtime_ns || 0) - (a.mtime_ns || 0);
    });
    return logs[0];
  }

  async function loadDirectory(path) {
    const payload = await fetchJson(`/api/list?path=${encodeURIComponent(path || "")}`);
    state.currentDirPath = payload.path || "";
    state.listEntries = payload.entries || [];
    scheduleRender();

    if (state.mode === "browse") {
      const defaultLog = chooseDefaultLog(state.listEntries);
      const currentEntry = state.listEntries.find(
        (entry) => entry.kind === "log" && entry.path === state.currentLogPath,
      ) || null;
      const shouldSwitch = (
        defaultLog &&
        (
          !state.currentLogPath ||
          currentEntry === null ||
          currentEntry.run_complete
        ) &&
        defaultLog.path !== state.currentLogPath
      );
      if (shouldSwitch) {
        await openLog(defaultLog.path);
      }
    }
  }

  function startDirectoryRefresh() {
    if (state.directoryRefreshHandle !== null) {
      clearInterval(state.directoryRefreshHandle);
    }
    state.directoryRefreshHandle = window.setInterval(() => {
      if (!state.currentDirPath && state.currentDirPath !== "") {
        return;
      }
      void loadDirectory(state.currentDirPath);
    }, 2000);
  }

  async function initialize() {
    refs.pauseButton.addEventListener("click", () => {
      state.paused = !state.paused;
      if (!state.paused) {
        const last = latestTimestep();
        if (last !== null) {
          state.selectedTimestep = last;
        }
        scheduleRender();
        requestAnimationFrame(scrollToLatest);
      } else {
        scheduleRender();
      }
    });

    refs.followToggle.addEventListener("change", (event) => {
      state.followLatest = Boolean(event.target.checked);
      if (state.followLatest && !state.paused) {
        requestAnimationFrame(scrollToLatest);
      }
    });

    refs.timelineScroll.addEventListener("scroll", () => {
      state.hoveredTimestep = null;
      state.hoveredActiveId = null;
      applyHoverHighlights();
      const nearBottom = (
        refs.timelineScroll.scrollTop + refs.timelineScroll.clientHeight
      ) >= (refs.timelineScroll.scrollHeight - (ROW_HEIGHT * 3));
      if (!state.suppressScrollPause && !nearBottom && !state.paused) {
        state.paused = true;
        scheduleRender();
      }
      if (needsViewportRender()) {
        scheduleRender();
      }
      void ensureOlderHistory();
    });

    if (state.mode === "view" && state.routePath) {
      const slash = state.routePath.lastIndexOf("/");
      const dirPath = slash >= 0 ? state.routePath.slice(0, slash) : "";
      startDirectoryRefresh();
      await loadDirectory(dirPath);
      await openLog(state.routePath);
      return;
    }

    startDirectoryRefresh();
    await loadDirectory(state.routePath || "");
  }

  window.addEventListener("beforeunload", () => {
    closeStream();
    if (state.directoryRefreshHandle !== null) {
      clearInterval(state.directoryRefreshHandle);
      state.directoryRefreshHandle = null;
    }
  });
  window.addEventListener("resize", () => scheduleRender());
  void initialize();
})();
