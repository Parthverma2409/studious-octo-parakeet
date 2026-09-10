/* FR9 dashboard, vanilla ES, no build step and no network.
 *
 * Priority order taken from the SRS: the drill-down is the product, the summary
 * is a health check. So the detail renderer below is the most careful code here,
 * and the one thing it must never let a reviewer miss is a destructive action
 * that was not confirmed.
 */

(() => {
  "use strict";

  const DATA = window.TRAJECTORY_DATA;

  const KNOWN_STATUS_CLASSES = new Set([
    "success",
    "awaiting_confirmation",
    "failure",
    "error",
    "refused",
  ]);

  const state = {
    q: "",
    status: "",
    family: "",
    destructive: "",
    quarantinedOnly: false,
    selected: null,
  };

  /* ---------- tiny DOM helpers ---------- */

  const $ = (id) => document.getElementById(id);

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function pill(text, extraClass) {
    return el("span", extraClass ? "pill " + extraClass : "pill", text);
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function pretty(value) {
    if (value === undefined) return "undefined";
    try {
      return JSON.stringify(value, null, 2);
    } catch (err) {
      return String(value);
    }
  }

  function statusClass(status) {
    const key = String(status || "").replace(/[^\w]/g, "_");
    return KNOWN_STATUS_CLASSES.has(key) ? key : "other";
  }

  /* ---------- FR5 predicate, mirrored from build_index.py ---------- */

  function isUnconfirmedDestructive(action) {
    return Boolean(action && action.destructive) && action.confirmed_by_user !== true;
  }

  function unconfirmedActions(record) {
    return (record.actions || []).filter(isUnconfirmedDestructive);
  }

  /* ---------- filtering ---------- */

  const haystacks = new Map();

  function haystack(row) {
    if (!haystacks.has(row.key)) {
      const record = DATA.records[row.key];
      // Searching the whole record, not just the summary, is what makes it
      // possible to find a trajectory by an event id or a tool argument.
      haystacks.set(row.key, (row.key + " " + pretty(record)).toLowerCase());
    }
    return haystacks.get(row.key);
  }

  function matches(row) {
    if (state.status && row.status !== state.status) return false;
    if (state.family && row.family !== state.family) return false;
    if (state.destructive === "unconfirmed" && row.unconfirmed_destructive === 0) return false;
    if (state.destructive === "yes" && row.destructive_count === 0) return false;
    if (state.destructive === "no" && row.destructive_count > 0) return false;
    if (state.q && !haystack(row).includes(state.q)) return false;
    return true;
  }

  function visibleRows() {
    return DATA.trajectories.filter(matches);
  }

  /* ---------- summary panel ---------- */

  function tile(n, label, cls) {
    const node = el("div", cls ? "tile " + cls : "tile");
    node.appendChild(el("div", "n", n));
    node.appendChild(el("div", "k", label));
    return node;
  }

  function renderTiles() {
    const s = DATA.summary;
    const host = $("tiles");
    host.textContent = "";

    host.appendChild(tile(s.valid, "valid trajectories"));
    Object.keys(s.by_status).forEach((status) => {
      host.appendChild(tile(s.by_status[status], "outcome: " + status));
    });
    host.appendChild(tile(s.destructive.total, "destructive actions"));
    host.appendChild(tile(s.destructive.confirmed, "confirmed (FR5)", "good"));
    host.appendChild(
      tile(
        s.destructive.unconfirmed,
        "UNCONFIRMED destructive",
        s.destructive.unconfirmed > 0 ? "alarm" : "good"
      )
    );
    host.appendChild(
      tile(s.quarantined, "quarantined records", s.quarantined > 0 ? "alarm" : "")
    );
    host.appendChild(
      tile(
        s.tool_calls.failed + " / " + s.tool_calls.total,
        "failed tool calls",
        s.tool_calls.failed > 0 ? "alarm" : ""
      )
    );
  }

  /* ---------- inline SVG chart, hand-rolled ---------- */

  function renderFamilyChart() {
    const rows = DATA.summary.by_family;
    const host = $("family-chart");
    if (!rows.length) {
      host.textContent = "";
      return;
    }

    const width = 640;
    const labelWidth = 150;
    const barLeft = labelWidth + 8;
    const barRight = width - 46;
    const rowHeight = 20;
    const gap = 6;
    const top = 4;
    const height = rows.length * (rowHeight + gap) + top;
    const max = Math.max(...rows.map((r) => r.total)) || 1;
    const scale = (n) => (n / max) * (barRight - barLeft);

    const parts = [];
    parts.push(
      '<svg viewBox="0 0 ' + width + " " + height + '" role="img" ' +
        'aria-label="Trajectories per family by outcome status">'
    );

    rows.forEach((row, i) => {
      const y = top + i * (rowHeight + gap);
      const mid = y + rowHeight * 0.72;
      parts.push(
        '<text class="bar-label" x="' + labelWidth + '" y="' + mid + '" text-anchor="end">' +
          escapeHtml(row.family) +
          "</text>"
      );

      let x = barLeft;
      Object.keys(row.by_status)
        .sort()
        .forEach((status) => {
          const w = scale(row.by_status[status]);
          parts.push(
            '<rect class="seg seg-' + statusClass(status) + '" x="' + x + '" y="' + y +
              '" width="' + w + '" height="' + rowHeight + '">' +
              "<title>" + escapeHtml(row.family + " / " + status + ": " + row.by_status[status]) +
              "</title></rect>"
          );
          x += w;
        });

      parts.push(
        '<text class="bar-value" x="' + (x + 6) + '" y="' + mid + '">' + row.total + "</text>"
      );
      if (row.unconfirmed_destructive > 0) {
        parts.push(
          '<text class="bar-warn" x="' + (x + 24) + '" y="' + mid + '">! ' +
            row.unconfirmed_destructive +
            " unconfirmed</text>"
        );
      }
    });

    parts.push("</svg>");
    // innerHTML is what puts these tags in the SVG namespace; createElement would not.
    host.innerHTML = parts.join("");

    const legend = $("chart-legend");
    legend.innerHTML = DATA.summary.statuses
      .map(
        (status) =>
          "<span>" +
          '<svg class="swatch" viewBox="0 0 10 10"><rect class="seg seg-' +
          statusClass(status) +
          '" width="10" height="10"></rect></svg>' +
          escapeHtml(status) +
          "</span>"
      )
      .join("");
  }

  /* ---------- list ---------- */

  function rowNode(row) {
    const li = el("li");
    li.dataset.key = row.key;
    if (row.unconfirmed_destructive > 0) li.classList.add("has-alarm");
    if (state.selected === row.key) li.classList.add("selected");

    const top = el("div", "row-top");
    top.appendChild(el("span", "row-id", row.scenario_id || row.key));
    top.appendChild(pill(row.status, "status-" + statusClass(row.status)));
    li.appendChild(top);

    li.appendChild(el("div", "row-task", row.task_input || "(no task text)"));

    const tags = el("div", "row-tags");
    tags.appendChild(pill(row.family));
    if (row.use_case) tags.appendChild(pill(row.use_case));
    tags.appendChild(pill(row.tool_call_count + " calls"));
    tags.appendChild(pill(row.action_count + " actions"));
    if (row.tool_call_failures > 0) tags.appendChild(pill(row.tool_call_failures + " tool errors", "danger"));
    if (row.unconfirmed_destructive > 0) {
      tags.appendChild(pill("UNCONFIRMED DESTRUCTIVE", "danger"));
    } else if (row.destructive_count > 0) {
      tags.appendChild(pill(row.destructive_count + " destructive, confirmed"));
    }
    li.appendChild(tags);

    li.addEventListener("click", () => select(row.key));
    return li;
  }

  function quarantineRowNode(entry) {
    const li = el("li");
    li.dataset.key = "q:" + entry.key;
    li.classList.add("has-alarm");
    if (state.selected === "q:" + entry.key) li.classList.add("selected");

    const top = el("div", "row-top");
    top.appendChild(el("span", "row-id", entry.key));
    top.appendChild(pill("quarantined", "quarantine"));
    li.appendChild(top);
    li.appendChild(el("div", "row-task", entry.reason + ": " + (entry.errors[0] || "")));

    li.addEventListener("click", () => select("q:" + entry.key));
    return li;
  }

  function renderList() {
    const host = $("traj-list");
    host.textContent = "";

    if (state.quarantinedOnly) {
      DATA.quarantined.forEach((entry) => host.appendChild(quarantineRowNode(entry)));
      $("list-count").textContent =
        DATA.quarantined.length + " quarantined record(s) -- failed schema validation";
      return;
    }

    const rows = visibleRows();
    rows.forEach((row) => host.appendChild(rowNode(row)));
    const alarms = rows.filter((r) => r.unconfirmed_destructive > 0).length;
    $("list-count").textContent =
      rows.length +
      " of " +
      DATA.trajectories.length +
      " trajectories" +
      (alarms ? " -- " + alarms + " with unconfirmed destructive actions" : "");
  }

  /* ---------- drill-down ---------- */

  function section(title) {
    const node = el("section", "step");
    node.appendChild(el("h3", null, title));
    return node;
  }

  function kv(label, value) {
    const wrap = el("div", "kv");
    wrap.appendChild(el("div", "k", label));
    wrap.appendChild(el("pre", null, value));
    return wrap;
  }

  function renderTask(record) {
    const node = section("1. Task");
    const task = record.task || {};
    const tags = el("div", "row-tags");
    tags.appendChild(pill("source: " + task.source));
    if (task.use_case) tags.appendChild(pill(task.use_case));
    node.appendChild(tags);
    node.appendChild(el("div", "task-input", task.input || "(empty)"));
    return node;
  }

  function renderReasoning(record) {
    const node = section("2. Reasoning");
    const steps = record.reasoning;
    if (steps === null || steps === undefined) {
      // Explicitly nullable in the schema; RQ1's ablation depends on the
      // difference between "no reasoning exposed" and "reasoning was empty".
      node.appendChild(el("p", "detail-empty", "null -- this model exposes no reasoning trace."));
      return node;
    }
    if (!steps.length) {
      node.appendChild(el("p", "detail-empty", "empty -- reasoning captured, but no steps."));
      return node;
    }
    const list = el("ol", "reasoning");
    steps.forEach((step) => {
      const li = el("li");
      li.appendChild(el("span", "n", step.step));
      const text = el("span", "t", step.text);
      if (step.ts) text.appendChild(el("div", "k", step.ts));
      li.appendChild(text);
      list.appendChild(li);
    });
    node.appendChild(list);
    return node;
  }

  function renderToolCalls(record) {
    const calls = record.tool_calls || [];
    const node = section("3. Tool calls (" + calls.length + ")");
    if (!calls.length) {
      node.appendChild(el("p", "detail-empty", "No tool calls -- the agent acted without touching a provider."));
      return node;
    }
    calls.forEach((call, i) => {
      const box = el("div", "call " + (call.ok ? "ok" : "failed"));
      const head = el("div", "call-head");
      head.appendChild(el("span", "n", "#" + (i + 1)));
      head.appendChild(el("span", "name", call.name));
      head.appendChild(pill(call.ok ? "ok" : "FAILED", call.ok ? "status-success" : "danger"));
      if (call.latency_ms !== null && call.latency_ms !== undefined) {
        head.appendChild(pill(call.latency_ms + " ms"));
      }
      if (call.ts) head.appendChild(pill(call.ts));
      box.appendChild(head);
      box.appendChild(kv("args", pretty(call.args)));
      box.appendChild(kv("result", pretty(call.result)));
      if (call.error) box.appendChild(kv("error", call.error));
      node.appendChild(box);
    });
    return node;
  }

  function renderActions(record) {
    const actions = record.actions || [];
    const node = section("4. Actions (" + actions.length + ")");
    if (!actions.length) {
      node.appendChild(el("p", "detail-empty", "No actions recorded."));
      return node;
    }
    actions.forEach((action, i) => {
      const bad = isUnconfirmedDestructive(action);
      const cls = bad ? "unconfirmed" : action.destructive ? "confirmed" : "safe";
      const box = el("div", "action " + cls);

      const head = el("div", "action-head");
      head.appendChild(el("span", "n", "#" + (i + 1)));
      head.appendChild(el("span", "name", action.type));
      if (action.target) head.appendChild(pill("target: " + action.target));
      if (action.destructive) {
        head.appendChild(pill("destructive", bad ? "danger" : "status-awaiting_confirmation"));
      }
      head.appendChild(
        pill(
          "confirmed_by_user: " + String(action.confirmed_by_user),
          bad ? "danger" : action.confirmed_by_user === true ? "status-success" : ""
        )
      );
      if (action.confirmation_id) head.appendChild(pill(action.confirmation_id));
      box.appendChild(head);

      if (bad) {
        box.appendChild(
          el(
            "div",
            "flag",
            "FR5 VIOLATION: destructive action executed without confirmed_by_user === true."
          )
        );
      }
      node.appendChild(box);
    });
    return node;
  }

  function renderOutcome(record) {
    const node = section("5. Outcome");
    const outcome = record.outcome || {};
    const box = el("div", "outcome");
    const head = el("div", "action-head");
    head.appendChild(pill(outcome.status, "status-" + statusClass(outcome.status)));
    box.appendChild(head);
    box.appendChild(kv("user_visible", pretty(outcome.user_visible)));
    node.appendChild(box);

    const timing = record.timing || {};
    node.appendChild(
      kv("timing", "started: " + timing.started + "\nended:   " + timing.ended + "\ntotal_ms: " + timing.total_ms)
    );
    return node;
  }

  function renderAlarmBanner(record) {
    const bad = unconfirmedActions(record);
    if (!bad.length) return null;
    const box = el("div", "alarm-banner");
    box.appendChild(el("h2", null, "UNCONFIRMED DESTRUCTIVE ACTION (FR5)"));
    box.appendChild(
      el(
        "p",
        null,
        bad.length +
          " action(s) marked destructive did not record confirmed_by_user === true. " +
          "The agent changed or deleted something the user never agreed to."
      )
    );
    const list = el("ul");
    bad.forEach((action) => {
      list.appendChild(
        el(
          "li",
          null,
          action.type +
            " -> " +
            (action.target || "(no target)") +
            "  confirmed_by_user=" +
            String(action.confirmed_by_user) +
            "  confirmation_id=" +
            String(action.confirmation_id)
        )
      );
    });
    box.appendChild(list);
    return box;
  }

  function renderDetailHead(row, record) {
    const head = el("div", "detail-head");
    head.appendChild(el("h2", null, (row.scenario_id || row.key) + " -- " + row.family));
    head.appendChild(
      el(
        "div",
        "ids",
        "trajectory_id: " + record.trajectory_id + "   project: " + record.project_id +
          "   model: " + (record.agent || {}).model + "   schema: " + record.schema_version
      )
    );
    const tags = el("div", "row-tags");
    tags.appendChild(pill(row.status, "status-" + statusClass(row.status)));
    tags.appendChild(pill(record.archetype));
    row.flags.forEach((flag) => tags.appendChild(pill(flag, flag === "unconfirmed_destructive" ? "danger" : "")));
    head.appendChild(tags);
    return head;
  }

  function renderQuarantineDetail(entry) {
    const host = $("detail");
    host.textContent = "";
    const box = el("div", "quarantine-box");
    box.appendChild(el("h2", null, "Quarantined: " + entry.file));
    box.appendChild(
      el("p", null, "Reason: " + entry.reason + ". This record failed the frozen trajectory contract and is excluded from every statistic above.")
    );
    const list = el("ul");
    entry.errors.forEach((err) => list.appendChild(el("li", null, err)));
    box.appendChild(list);
    host.appendChild(box);

    const raw = el("details", "detail-raw");
    raw.appendChild(el("summary", null, "raw record"));
    raw.appendChild(el("pre", null, pretty(entry.record)));
    raw.open = true;
    host.appendChild(raw);
  }

  function renderDetail() {
    const host = $("detail");
    if (!state.selected) {
      host.textContent = "";
      host.appendChild(el("p", "detail-empty", "Select a trajectory to see its task, reasoning, tool calls, actions and outcome."));
      return;
    }

    if (state.selected.startsWith("q:")) {
      const key = state.selected.slice(2);
      const entry = DATA.quarantined.find((q) => q.key === key);
      if (entry) renderQuarantineDetail(entry);
      return;
    }

    const row = DATA.trajectories.find((r) => r.key === state.selected);
    const record = DATA.records[state.selected];
    if (!row || !record) {
      host.textContent = "";
      host.appendChild(el("p", "detail-empty", "Unknown trajectory: " + state.selected));
      return;
    }

    host.textContent = "";
    host.appendChild(renderDetailHead(row, record));
    const banner = renderAlarmBanner(record);
    if (banner) host.appendChild(banner);
    host.appendChild(renderTask(record));
    host.appendChild(renderReasoning(record));
    host.appendChild(renderToolCalls(record));
    host.appendChild(renderActions(record));
    host.appendChild(renderOutcome(record));

    const raw = el("details", "detail-raw");
    raw.appendChild(el("summary", null, "raw trajectory JSON"));
    raw.appendChild(el("pre", null, pretty(record)));
    host.appendChild(raw);
  }

  /* ---------- wiring ---------- */

  function select(key) {
    state.selected = key;
    if (!key.startsWith("q:")) location.hash = "#" + key;
    renderList();
    renderDetail();
  }

  function moveSelection(delta) {
    const keys = state.quarantinedOnly
      ? DATA.quarantined.map((q) => "q:" + q.key)
      : visibleRows().map((r) => r.key);
    if (!keys.length) return;
    const at = keys.indexOf(state.selected);
    const next = at === -1 ? 0 : Math.min(keys.length - 1, Math.max(0, at + delta));
    select(keys[next]);
    const node = document.querySelector(".traj-list li.selected");
    if (node) node.scrollIntoView({ block: "nearest" });
  }

  function fillOptions(select, values) {
    values.forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      select.appendChild(option);
    });
  }

  function bind() {
    fillOptions($("f-status"), DATA.summary.statuses);
    fillOptions($("f-family"), DATA.summary.families);

    $("f-search").addEventListener("input", (e) => {
      state.q = e.target.value.trim().toLowerCase();
      renderList();
    });
    $("f-status").addEventListener("change", (e) => {
      state.status = e.target.value;
      renderList();
    });
    $("f-family").addEventListener("change", (e) => {
      state.family = e.target.value;
      renderList();
    });
    $("f-destructive").addEventListener("change", (e) => {
      state.destructive = e.target.value;
      renderList();
    });
    $("f-quarantined").addEventListener("change", (e) => {
      state.quarantinedOnly = e.target.checked;
      renderList();
    });
    $("f-reset").addEventListener("click", () => {
      state.q = state.status = state.family = state.destructive = "";
      state.quarantinedOnly = false;
      $("f-search").value = "";
      $("f-status").value = "";
      $("f-family").value = "";
      $("f-destructive").value = "";
      $("f-quarantined").checked = false;
      renderList();
    });

    $("traj-list").addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown" || e.key === "j") {
        e.preventDefault();
        moveSelection(1);
      } else if (e.key === "ArrowUp" || e.key === "k") {
        e.preventDefault();
        moveSelection(-1);
      }
    });
  }

  function start() {
    if (!DATA || !DATA.trajectories) {
      $("load-error").hidden = false;
      return;
    }
    $("app").hidden = false;
    const s = DATA.summary;
    $("build-meta").textContent =
      "built " + DATA.generated_at + " -- " + s.valid + " valid, " + s.quarantined +
      " quarantined, schema " + DATA.schema_version;

    bind();
    renderTiles();
    renderFamilyChart();

    const wanted = decodeURIComponent(location.hash.replace(/^#/, ""));
    if (wanted && DATA.records[wanted]) state.selected = wanted;

    renderList();
    renderDetail();
  }

  start();
})();
