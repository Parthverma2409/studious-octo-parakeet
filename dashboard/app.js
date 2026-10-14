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

  /* ---------- tooltip (one node, moved under the cursor) ---------- */

  function bindTooltip(host) {
    const tip = $("tip");
    host.addEventListener("mousemove", (e) => {
      const target = e.target.closest("[data-tip]");
      if (!target) {
        tip.hidden = true;
        return;
      }
      tip.innerHTML = target.dataset.tip;
      tip.hidden = false;
      const pad = 14;
      let x = e.clientX + pad;
      let y = e.clientY + pad;
      const w = tip.offsetWidth;
      const h = tip.offsetHeight;
      if (x + w > window.innerWidth - 8) x = e.clientX - w - pad;
      if (y + h > window.innerHeight - 8) y = e.clientY - h - pad;
      tip.style.left = x + "px";
      tip.style.top = y + "px";
    });
    host.addEventListener("mouseleave", () => {
      tip.hidden = true;
    });
  }

  function tipText(title, lines) {
    return "<b>" + escapeHtml(title) + "</b>" + lines.map((l) => "<br>" + escapeHtml(l)).join("");
  }

  function pct(n, d) {
    return d ? Math.round((n / d) * 100) + "%" : "0%";
  }

  /* ---------- KPI tiles ---------- */

  // Inline SVG paths, 24x24, stroked with currentColor by the stylesheet.
  const ICONS = {
    list: '<path d="M8 6h13M8 12h13M8 18h13"/><circle cx="3.5" cy="6" r="1"/><circle cx="3.5" cy="12" r="1"/><circle cx="3.5" cy="18" r="1"/>',
    check: '<path d="M20 6L9 17l-5-5"/>',
    shield: '<path d="M12 3l8 3v6c0 4.5-3.4 8-8 9-4.6-1-8-4.5-8-9V6z"/><path d="M9 12l2 2 4-4"/>',
    alert: '<path d="M12 3L2 20h20z"/><path d="M12 10v4M12 17v.5"/>',
    hand: '<path d="M7 11V6a1.5 1.5 0 0 1 3 0v5M10 10V4.5a1.5 1.5 0 0 1 3 0V10M13 10V6a1.5 1.5 0 0 1 3 0v6"/><path d="M16 12l1.6-2.2a1.5 1.5 0 0 1 2.4 1.8L16 18a5 5 0 0 1-9 0l-2.8-4.2A1.5 1.5 0 0 1 6.5 12L7 12.5"/>',
    tool: '<path d="M14.5 5.5a4 4 0 0 0 5 5L9 21a2 2 0 0 1-3-3z"/><path d="M14.5 5.5l2-2 4 4-2 2"/>',
    box: '<path d="M3 8l9-5 9 5v8l-9 5-9-5z"/><path d="M3 8l9 5 9-5M12 13v8"/>',
  };

  // Parsed through innerHTML so the HTML parser puts the element in the SVG
  // namespace for us; that keeps the namespace URI (a URL) out of this file.
  function icon(name, cls) {
    const wrap = el("span");
    wrap.innerHTML =
      '<svg viewBox="0 0 24 24"' + (cls ? ' class="' + cls + '"' : "") + ' aria-hidden="true">' +
      (ICONS[name] || "") + "</svg>";
    return wrap.firstChild;
  }

  function badge(text, cls, iconName) {
    const node = el("span", "badge " + cls);
    if (iconName) node.appendChild(icon(iconName));
    node.appendChild(document.createTextNode(text));
    return node;
  }

  function tile(opts) {
    const node = el("div", "tile" + (opts.cls ? " " + opts.cls : ""));
    const head = el("div", "tile-head");
    head.appendChild(el("span", "k", opts.label));
    head.appendChild(icon(opts.icon, "ico"));
    node.appendChild(head);

    const n = el("div", "n", opts.value);
    if (opts.unit) n.appendChild(el("small", null, opts.unit));
    node.appendChild(n);

    if (opts.sub) {
      const sub = el("div", "sub");
      sub.innerHTML = opts.sub;
      node.appendChild(sub);
    }
    if (opts.badge) node.appendChild(opts.badge);
    if (opts.meter) node.appendChild(opts.meter);
    return node;
  }

  // A segmented bar: [{cls, n, title}] scaled to the sum.
  function meter(segments) {
    const total = segments.reduce((a, s) => a + s.n, 0) || 1;
    const bar = el("div", "meter");
    segments.forEach((s) => {
      if (!s.n) return;
      const seg = el("span", s.cls);
      seg.style.width = (s.n / total) * 100 + "%";
      seg.dataset.tip = tipText(s.title, [s.n + " of " + total + " (" + pct(s.n, total) + ")"]);
      bar.appendChild(seg);
    });
    return bar;
  }

  function renderTiles() {
    const s = DATA.summary;
    const host = $("tiles");
    host.textContent = "";

    const success = s.by_status.success || 0;
    const refused = s.by_status.refused || 0;
    const failed = (s.by_status.failure || 0) + (s.by_status.error || 0);
    const waiting = s.by_status.awaiting_confirmation || 0;
    const ucs = Object.keys(s.by_use_case || {});

    host.appendChild(
      tile({
        label: "Trajectories",
        icon: "list",
        cls: "accent",
        value: s.valid,
        sub: "<b>" + s.families.length + "</b> families &middot; <b>" + ucs.length + "</b> use cases",
        meter: meter(
          s.statuses.map((st) => ({
            cls: "seg-" + statusClass(st),
            n: s.by_status[st] || 0,
            title: st,
          }))
        ),
      })
    );

    host.appendChild(
      tile({
        label: "Success rate",
        icon: "check",
        cls: "good",
        value: pct(success, s.valid),
        sub: "<b>" + success + "</b> of " + s.valid + " ended in success",
        badge:
          waiting > 0 ? badge(waiting + " awaiting confirmation", "warn") : badge("no pending", "good", "check"),
      })
    );

    const unconfirmed = s.destructive.unconfirmed;
    host.appendChild(
      tile({
        label: "FR5 gate",
        icon: unconfirmed > 0 ? "alert" : "shield",
        cls: unconfirmed > 0 ? "alarm" : "good",
        value: unconfirmed,
        unit: "unconfirmed",
        sub:
          "<b>" + s.destructive.total + "</b> destructive actions, <b>" + s.destructive.confirmed +
          "</b> confirmed by user",
        badge:
          unconfirmed > 0
            ? badge("VIOLATION", "bad", "alert")
            : badge("all confirmed", "good", "check"),
      })
    );

    host.appendChild(
      tile({
        label: "Refused",
        icon: "hand",
        cls: "neutral",
        value: refused,
        sub: pct(refused, s.valid) + " of trajectories declined to act",
      })
    );

    host.appendChild(
      tile({
        label: "Tool call failures",
        icon: "tool",
        cls: s.tool_calls.failed > 0 ? "warn" : "good",
        value: s.tool_calls.failed,
        unit: "/ " + s.tool_calls.total,
        sub: "<b>" + failed + "</b> trajectories ended in failure",
        badge:
          s.tool_calls.failed > 0
            ? badge(pct(s.tool_calls.failed, s.tool_calls.total) + " error rate", "warn")
            : badge("no errors", "good", "check"),
      })
    );

    host.appendChild(
      tile({
        label: "Quarantined",
        icon: "box",
        cls: s.quarantined > 0 ? "alarm" : "neutral",
        value: s.quarantined,
        sub:
          s.quarantined > 0
            ? "records failed the frozen schema"
            : "all <b>" + s.total + "</b> files passed schema validation",
      })
    );

    $("overview-sub").textContent =
      "model " + (DATA.trajectories[0] || {}).model + "  |  " + s.families.length +
      " scenario families  |  source: " + Object.keys(s.by_source || {}).join(", ");
  }

  /* ---------- outcome mix ---------- */

  function renderStatusChart() {
    const s = DATA.summary;
    const host = $("status-chart");
    host.textContent = "";

    const rows = s.statuses
      .map((st) => ({ status: st, n: s.by_status[st] || 0 }))
      .sort((a, b) => b.n - a.n);

    const band = el("div", "status-band");
    rows.forEach((r) => {
      if (!r.n) return;
      const seg = el("span", "seg-" + statusClass(r.status));
      seg.style.width = (r.n / s.valid) * 100 + "%";
      seg.dataset.tip = tipText(r.status, [r.n + " trajectories (" + pct(r.n, s.valid) + ")"]);
      band.appendChild(seg);
    });
    host.appendChild(band);

    const list = el("div", "status-rows");
    rows.forEach((r) => {
      const row = el("div", "status-row");
      const sw = el("span", "swatch seg-" + statusClass(r.status));
      sw.style.background = "";
      row.appendChild(sw);
      row.appendChild(el("span", "name", r.status.replace(/_/g, " ")));
      row.appendChild(el("span", "v", r.n));
      row.appendChild(el("span", "p", pct(r.n, s.valid)));
      list.appendChild(row);
    });
    host.appendChild(list);

    const foot = el("div", "status-foot");
    foot.appendChild(el("span", null, s.valid + " valid trajectories"));
    foot.appendChild(el("span", null, s.flagged_trajectories + " flagged"));
    host.appendChild(foot);
  }

  /* ---------- tool calls by name ---------- */

  function renderToolsChart() {
    const s = DATA.summary.tool_calls;
    const host = $("tools-chart");
    const names = Object.keys(s.by_name || {}).sort((a, b) => s.by_name[b] - s.by_name[a]);
    if (!names.length) {
      host.textContent = "";
      return;
    }

    const width = 420;
    const labelWidth = 170;
    const barLeft = labelWidth + 10;
    const barRight = width - 40;
    const rowHeight = 16;
    const gap = 8;
    const top = 2;
    const height = names.length * (rowHeight + gap) + top;
    const max = Math.max(...names.map((n) => s.by_name[n])) || 1;
    const scale = (n) => (n / max) * (barRight - barLeft);

    const parts = [];
    parts.push(
      '<svg viewBox="0 0 ' + width + " " + height + '" role="img" aria-label="Tool calls by name">'
    );
    names.forEach((name, i) => {
      const y = top + i * (rowHeight + gap);
      const mid = y + rowHeight * 0.72;
      const n = s.by_name[name];
      const w = Math.max(scale(n), 2);
      parts.push(
        '<text class="bar-label" x="' + labelWidth + '" y="' + mid + '" text-anchor="end">' +
          escapeHtml(name) + "</text>"
      );
      parts.push(
        '<rect class="track" x="' + barLeft + '" y="' + y + '" width="' + (barRight - barLeft) +
          '" height="' + rowHeight + '" rx="4"/>'
      );
      parts.push(
        '<rect class="seg seg-tool" x="' + barLeft + '" y="' + y + '" width="' + w +
          '" height="' + rowHeight + '" rx="4" data-tip="' +
          escapeHtml(tipText(name, [n + " calls (" + pct(n, s.total) + " of all calls)"])) + '"/>'
      );
      parts.push(
        '<text class="bar-value" x="' + (barLeft + w + 6) + '" y="' + mid + '">' + n + "</text>"
      );
    });
    parts.push("</svg>");
    host.innerHTML = parts.join("");

    $("tools-sub").textContent =
      s.total + " calls across " + names.length + " tools, " + s.failed + " failed";
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
    const labelWidth = 160;
    const barLeft = labelWidth + 10;
    const barRight = width - 120;
    const rowHeight = 18;
    const gap = 8;
    const top = 2;
    const segGap = 2; // surface gap between stacked segments
    const height = rows.length * (rowHeight + gap) + top;
    const max = Math.max(...rows.map((r) => r.total)) || 1;
    const scale = (n) => (n / max) * (barRight - barLeft);

    const parts = [];
    parts.push(
      '<svg viewBox="0 0 ' + width + " " + height + '" role="img" ' +
        'aria-label="Trajectories per family by outcome status">'
    );

    // Hairline gridlines at clean ticks, drawn first so the bars sit on top.
    const tick = max <= 10 ? 2 : max <= 25 ? 5 : 10;
    for (let v = tick; v <= max; v += tick) {
      const gx = barLeft + scale(v);
      parts.push('<line class="grid" x1="' + gx + '" x2="' + gx + '" y1="0" y2="' + height + '"/>');
    }

    rows.forEach((row, i) => {
      const y = top + i * (rowHeight + gap);
      const mid = y + rowHeight * 0.72;
      const totalW = scale(row.total);
      const clipId = "fam-clip-" + i;
      parts.push(
        '<text class="bar-label" x="' + labelWidth + '" y="' + mid + '" text-anchor="end">' +
          escapeHtml(row.family) +
          "</text>"
      );
      // One rounded data-end per row: square at the baseline, 4px radius at the tip.
      parts.push(
        '<clipPath id="' + clipId + '"><rect x="' + (barLeft - 8) + '" y="' + y + '" width="' +
          (totalW + 8) + '" height="' + rowHeight + '" rx="4"/></clipPath>'
      );
      parts.push('<g clip-path="url(#' + clipId + ')">');

      let x = barLeft;
      const statuses = Object.keys(row.by_status).sort();
      statuses.forEach((status, k) => {
        const n = row.by_status[status];
        const w = scale(n);
        const inner = k < statuses.length - 1 ? Math.max(w - segGap, 0) : w;
        parts.push(
          '<rect class="seg seg-' + statusClass(status) + '" x="' + x + '" y="' + y +
            '" width="' + inner + '" height="' + rowHeight + '" data-tip="' +
            escapeHtml(
              tipText(row.family, [
                status + ": " + n + " of " + row.total + " (" + pct(n, row.total) + ")",
              ])
            ) +
            '"/>'
        );
        x += w;
      });
      parts.push("</g>");

      parts.push(
        '<text class="bar-value" x="' + (x + 6) + '" y="' + mid + '">' + row.total + "</text>"
      );
      if (row.unconfirmed_destructive > 0) {
        parts.push(
          '<text class="bar-warn" x="' + (x + 28) + '" y="' + mid + '">! ' +
            row.unconfirmed_destructive +
            " unconfirmed</text>"
        );
      }
    });

    parts.push('<line class="axis" x1="' + barLeft + '" x2="' + barLeft + '" y1="0" y2="' + height + '"/>');
    parts.push("</svg>");
    // innerHTML is what puts these tags in the SVG namespace; createElement would not.
    host.innerHTML = parts.join("");

    const legend = $("chart-legend");
    legend.innerHTML = DATA.summary.statuses
      .map(
        (status) =>
          '<span><span class="swatch seg-' + statusClass(status) + '"></span>' +
          escapeHtml(status.replace(/_/g, " ")) +
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

  function section(title, count) {
    const node = el("section", "step");
    const m = /^(\d+)\.\s*(.*)$/.exec(title);
    const h = el("h3");
    h.appendChild(el("span", "num", m ? m[1] : ""));
    h.appendChild(document.createTextNode(m ? m[2] : title));
    if (count !== undefined) h.appendChild(el("span", "cnt", "(" + count + ")"));
    node.appendChild(h);
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
    const node = section("3. Tool calls", calls.length);
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
    const node = section("4. Actions", actions.length);
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

  function idPair(label, value) {
    const node = el("span");
    node.appendChild(el("b", null, label + " "));
    node.appendChild(document.createTextNode(String(value)));
    return node;
  }

  function renderDetailHead(row, record) {
    const head = el("div", "detail-head");
    const h2 = el("h2", null, row.scenario_id || row.key);
    h2.appendChild(el("span", "fam", "  /  " + row.family));
    head.appendChild(h2);
    const ids = el("div", "ids");
    ids.appendChild(idPair("trajectory", record.trajectory_id));
    ids.appendChild(idPair("project", record.project_id));
    ids.appendChild(idPair("model", (record.agent || {}).model));
    ids.appendChild(idPair("schema", record.schema_version));
    head.appendChild(ids);
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
      const ph = el("div", "detail-placeholder");
      ph.innerHTML =
        '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 5h16v14H4z"/><path d="M4 10h16M9 10v9"/></svg>' +
        "<p>Select a trajectory to see its task, reasoning, tool calls, actions and outcome.</p>" +
        '<p class="hint">Focus the list and use <kbd>j</kbd> / <kbd>k</kbd> or the arrow keys to move.</p>';
      host.appendChild(ph);
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


  /* ---------- views + router ---------- */

  const VIEWS = {
    overview: { title: "Overview", sub: "health of the captured trajectory corpus" },
    trajectories: { title: "Trajectories", sub: "browse, filter and drill into every captured run" },
    calendar: { title: "Calendar", sub: "every booking and calendar event the scenarios touch" },
  };
  let currentView = "overview";

  function showView(name) {
    if (!VIEWS[name]) name = "overview";
    currentView = name;
    Object.keys(VIEWS).forEach((v) => {
      $("view-" + v).hidden = v !== name;
      $("nav-" + v).classList.toggle("is-active", v === name);
    });
    $("view-title").textContent = VIEWS[name].title;
    $("view-sub").textContent = VIEWS[name].sub;
    if (name === "calendar") renderCalendar();
  }

  // Routes: #overview | #trajectories | #calendar | #calendar/YYYY-MM-DD | #<trajectory key>
  function route() {
    const hash = decodeURIComponent(location.hash.replace(/^#/, ""));
    if (!hash || hash === "overview") return showView("overview");
    if (hash === "trajectories") return showView("trajectories");
    if (hash.startsWith("calendar")) {
      const day = hash.split("/")[1];
      if (day && /^\d{4}-\d{2}-\d{2}$/.test(day)) {
        cal.day = day;
        cal.month = day.slice(0, 7);
      }
      return showView("calendar");
    }
    if (DATA.records[hash]) {
      state.selected = hash;
      renderList();
      renderDetail();
      return showView("trajectories");
    }
    showView("overview");
  }

  function fillChrome() {
    const s = DATA.summary;
    const model = (DATA.trajectories[0] || {}).model || "-";
    $("side-model").textContent = model;
    $("side-schema").textContent = DATA.schema_version;
    $("side-built").textContent = String(DATA.generated_at).slice(0, 10);
    $("nav-traj-count").textContent = s.valid;
    $("nav-cal-count").textContent = ((DATA.calendar || {}).events || []).length;

    const chip = $("gate-chip");
    chip.textContent = "";
    chip.appendChild(el("span", "dot"));
    if (s.destructive.unconfirmed > 0) {
      chip.classList.add("bad");
      chip.appendChild(document.createTextNode("FR5: " + s.destructive.unconfirmed + " unconfirmed destructive"));
    } else {
      chip.appendChild(document.createTextNode("FR5 gate clean"));
    }
  }

  /* ---------- overview: needs attention + upcoming ---------- */

  function renderAttention() {
    const table = $("attention");
    table.textContent = "";
    const rows = DATA.trajectories
      .map((r) => {
        const why = [];
        let weight = 0;
        if (r.unconfirmed_destructive > 0) {
          why.push(r.unconfirmed_destructive + " unconfirmed destructive (FR5)");
          weight += 100;
        }
        if (r.status === "failure" || r.status === "error") {
          why.push("ended in " + r.status);
          weight += 10;
        }
        if (r.tool_call_failures > 0) {
          why.push(r.tool_call_failures + " failed tool call" + (r.tool_call_failures > 1 ? "s" : ""));
          weight += 5;
        }
        if (r.status === "awaiting_confirmation") {
          why.push("still awaiting user confirmation");
          weight += 3;
        }
        return { r, why, weight };
      })
      .filter((x) => x.weight > 0)
      .sort((a, b) => b.weight - a.weight || a.r.key.localeCompare(b.r.key));

    $("attention-sub").textContent = rows.length
      ? rows.length + " of " + DATA.trajectories.length + " trajectories need a look"
      : "nothing flagged";

    if (!rows.length) {
      const wrap = el("tbody");
      const tr = el("tr");
      const td = el("td", "tbl-empty", "Every trajectory finished cleanly.");
      td.colSpan = 5;
      tr.appendChild(td);
      wrap.appendChild(tr);
      table.appendChild(wrap);
      return;
    }

    const thead = el("thead");
    const hr = el("tr");
    ["Scenario", "Family", "Status", "Why", "Calls"].forEach((h) => hr.appendChild(el("th", null, h)));
    thead.appendChild(hr);
    table.appendChild(thead);

    const tbody = el("tbody");
    rows.slice(0, 12).forEach(({ r, why }) => {
      const tr = el("tr", "row-link");
      tr.appendChild(el("td", "mono", r.scenario_id || r.key));
      tr.appendChild(el("td", null, r.family));
      const st = el("td");
      st.appendChild(pill(r.status, "status-" + statusClass(r.status)));
      tr.appendChild(st);
      tr.appendChild(el("td", "why", why.join("; ")));
      tr.appendChild(el("td", "num", r.tool_call_failures + " / " + r.tool_call_count));
      tr.addEventListener("click", () => select(r.key));
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
  }

  function renderUpcoming() {
    const host = $("upcoming");
    host.textContent = "";
    const events = ((DATA.calendar || {}).events || []).filter((e) => e.start);
    const now = (DATA.calendar || {}).now || "";
    const byDay = new Map();
    events.forEach((e) => {
      const day = e.start.slice(0, 10);
      if (day < now.slice(0, 10)) return;
      if (!byDay.has(day)) byDay.set(day, []);
      byDay.get(day).push(e);
    });
    const days = Array.from(byDay.keys()).sort().slice(0, 5);
    $("upcoming-sub").textContent = days.length
      ? "from the scenario clock, " + fmtDate(now)
      : "no dated events";

    days.forEach((day) => {
      const list = byDay.get(day);
      const row = el("div", "up-day");
      const d = el("div", "up-date");
      d.appendChild(el("span", "d", day.slice(8, 10)));
      d.appendChild(el("span", "m", monthShort(day)));
      row.appendChild(d);

      const items = el("div", "up-items");
      // Same title on the same day across many scenarios collapses to one line.
      const grouped = new Map();
      list.forEach((e) => {
        const k = e.type + "|" + e.title;
        if (!grouped.has(k)) grouped.set(k, { e, n: 0 });
        grouped.get(k).n += 1;
      });
      Array.from(grouped.values())
        .slice(0, 4)
        .forEach(({ e, n }) => {
          const it = el("div", "up-item");
          it.appendChild(el("span", "swatch type-" + typeClass(e.type)));
          it.appendChild(el("span", "t", e.title));
          it.appendChild(el("span", "n", n + " scenario" + (n > 1 ? "s" : "")));
          items.appendChild(it);
        });
      if (grouped.size > 4) {
        const more = el("span", "up-more", "+" + (grouped.size - 4) + " more");
        more.addEventListener("click", () => {
          location.hash = "#calendar/" + day;
        });
        items.appendChild(more);
      }
      row.appendChild(items);
      row.style.cursor = "pointer";
      row.addEventListener("click", (ev) => {
        if (ev.target.classList.contains("up-more")) return;
        location.hash = "#calendar/" + day;
      });
      host.appendChild(row);
    });
  }

  /* ---------- calendar ---------- */

  const MONTHS = ["January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December"];
  const DOWS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const TYPE_CLASSES = new Set(["flight", "hotel", "appointment", "train"]);

  const CAL_MARKS = {
    cancelled: '<path d="M6 6l12 12M18 6L6 18"/>',
    rescheduled: '<path d="M4 12h13M13 7l5 5-5 5"/>',
    confirmed: '<path d="M20 6L9 17l-5-5"/>',
    DELETE: '<path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/>',
    UPDATE: '<path d="M4 20h4l10-10-4-4L4 16z"/>',
    NOOP: '<circle cx="12" cy="12" r="4"/>',
    alarm: '<path d="M12 3L2 20h20z"/><path d="M12 10v4M12 17v.5"/>',
  };

  function mark(name) {
    const wrap = el("span");
    wrap.innerHTML = '<svg class="mk" viewBox="0 0 24 24" aria-hidden="true">' + (CAL_MARKS[name] || "") + "</svg>";
    return wrap.firstChild;
  }

  function typeClass(type) {
    return TYPE_CLASSES.has(type) ? type : "other";
  }

  function monthShort(iso) {
    return MONTHS[Number(iso.slice(5, 7)) - 1].slice(0, 3);
  }

  function fmtDate(iso) {
    if (!iso) return "";
    const d = new Date(iso.slice(0, 10) + "T00:00:00");
    return DOWS[(d.getDay() + 6) % 7] + " " + Number(iso.slice(8, 10)) + " " + monthShort(iso) + " " + iso.slice(0, 4);
  }

  function fmtTime(iso) {
    return iso && iso.length >= 16 ? iso.slice(11, 16) : "--:--";
  }

  // What the chip should say about an event, in one word, plus the flag.
  function eventState(e) {
    if (e.origin === "calendar" && e.action) {
      if (e.action.destructive && !e.action.confirmed) return { key: "alarm", label: "UNCONFIRMED " + e.action.type };
      return { key: e.action.type, label: e.action.type.toLowerCase() + (e.action.destructive ? ", confirmed" : "") };
    }
    if (e.origin === "calendar") return { key: "calendar", label: "in calendar" };
    return { key: e.status || "booking", label: e.status || "booking" };
  }

  const cal = {
    month: null, // "YYYY-MM"
    day: null, // "YYYY-MM-DD"
    mode: "month",
    origin: "",
    type: "",
    family: "",
    q: "",
  };

  function calEvents() {
    const all = ((DATA.calendar || {}).events || []).filter((e) => e.start);
    return all.filter((e) => {
      if (cal.origin && e.origin !== cal.origin) return false;
      if (cal.type && e.type !== cal.type) return false;
      if (cal.family && e.family !== cal.family) return false;
      if (cal.q) {
        const hay = (e.title + " " + (e.booking_ref || "") + " " + e.scenario_id + " " + (e.location || "") + " " + e.family).toLowerCase();
        if (!hay.includes(cal.q)) return false;
      }
      return true;
    });
  }

  function groupByDay(events) {
    const map = new Map();
    events.forEach((e) => {
      const day = e.start.slice(0, 10);
      if (!map.has(day)) map.set(day, []);
      map.get(day).push(e);
    });
    map.forEach((list) => list.sort((a, b) => (a.start > b.start ? 1 : a.start < b.start ? -1 : a.scenario_id.localeCompare(b.scenario_id))));
    return map;
  }

  function shiftMonth(delta) {
    const y = Number(cal.month.slice(0, 4));
    const m = Number(cal.month.slice(5, 7)) - 1 + delta;
    const d = new Date(y, m, 1);
    cal.month = d.getFullYear() + "-" + String(d.getMonth() + 1).padStart(2, "0");
    renderCalendar();
  }

  function renderCalStats(events) {
    const host = $("cal-stats");
    host.textContent = "";
    const inMonth = events.filter((e) => e.start.slice(0, 7) === cal.month);
    const bookings = inMonth.filter((e) => e.origin === "booking");
    const calRows = inMonth.filter((e) => e.origin === "calendar");
    const cancelled = bookings.filter((e) => e.status === "cancelled").length;
    const rescheduled = bookings.filter((e) => e.status === "rescheduled").length;
    const deleted = calRows.filter((e) => e.action && e.action.type === "DELETE").length;
    const alarms = calRows.filter((e) => e.action && e.action.destructive && !e.action.confirmed).length;

    const stat = (k, n, unit, cls) => {
      const box = el("div", "card cal-stat" + (cls ? " " + cls : ""));
      box.appendChild(el("div", "k", k));
      const v = el("div", "n", n);
      if (unit) v.appendChild(el("small", null, unit));
      box.appendChild(v);
      return box;
    };
    host.appendChild(stat("Events this month", inMonth.length, "", ""));
    host.appendChild(stat("Incoming bookings", bookings.length, cancelled ? cancelled + " cancelled" : "", ""));
    host.appendChild(stat("Reschedules", rescheduled, "", ""));
    host.appendChild(stat("Deleted by agent", deleted, "of " + calRows.length + " seeded", ""));
    host.appendChild(stat("Unconfirmed destructive", alarms, "", alarms ? "alarm" : ""));
  }

  function chipFor(e) {
    const st = eventState(e);
    const chip = el("div", "cal-chip type-" + typeClass(e.type));
    if (st.key === "alarm") chip.classList.add("is-alarm");
    if (st.key === "cancelled") chip.classList.add("is-cancelled");
    if (st.key === "DELETE") chip.classList.add("is-deleted");
    if (CAL_MARKS[st.key]) chip.appendChild(mark(st.key));
    chip.appendChild(el("span", "t", e.title));
    chip.dataset.tip = tipText(e.title, [
      fmtDate(e.start) + " " + fmtTime(e.start) + (e.end ? " - " + fmtTime(e.end) : ""),
      e.type + (e.booking_ref ? " " + e.booking_ref : ""),
      e.scenario_id + " (" + e.family + "): " + st.label,
    ]);
    return chip;
  }

  function renderCalGrid(byDay) {
    const host = $("cal-grid");
    host.textContent = "";
    DOWS.forEach((d) => host.appendChild(el("div", "cal-dow", d)));

    const y = Number(cal.month.slice(0, 4));
    const m = Number(cal.month.slice(5, 7)) - 1;
    const first = new Date(y, m, 1);
    const lead = (first.getDay() + 6) % 7; // Monday first
    const daysInMonth = new Date(y, m + 1, 0).getDate();
    const total = Math.ceil((lead + daysInMonth) / 7) * 7;
    const today = ((DATA.calendar || {}).now || "").slice(0, 10);

    for (let i = 0; i < total; i++) {
      const d = new Date(y, m, i - lead + 1);
      const iso = d.getFullYear() + "-" + String(d.getMonth() + 1).padStart(2, "0") + "-" + String(d.getDate()).padStart(2, "0");
      const list = byDay.get(iso) || [];
      const cell = el("div", "cal-cell");
      cell.setAttribute("role", "gridcell");
      cell.dataset.day = iso;
      if (d.getMonth() !== m) cell.classList.add("is-outside");
      if (iso === today) cell.classList.add("is-today");
      if (iso === cal.day) cell.classList.add("is-selected");
      if (list.some((e) => eventState(e).key === "alarm")) cell.classList.add("has-alarm");

      const row = el("div", "cal-date-row");
      row.appendChild(el("span", "cal-date", d.getDate()));
      if (list.length) row.appendChild(el("span", "cal-total", list.length));
      cell.appendChild(row);

      // Same title at the same time collapses into one chip: 40 scenarios reuse
      // one fixture flight, and forty identical chips tell the reviewer nothing.
      const seen = new Map();
      list.forEach((e) => {
        const k = e.type + "|" + e.title + "|" + fmtTime(e.start);
        const st = eventState(e).key;
        if (!seen.has(k) || st === "alarm") seen.set(k, e);
      });
      const uniq = Array.from(seen.values());
      const limit = 3;
      uniq.slice(0, limit).forEach((e) => cell.appendChild(chipFor(e)));
      if (uniq.length > limit) cell.appendChild(el("div", "cal-more", "+" + (uniq.length - limit) + " more"));

      cell.addEventListener("click", () => {
        cal.day = iso;
        location.hash = "#calendar/" + iso;
        renderCalendar();
      });
      host.appendChild(cell);
    }
  }

  function eventCard(e, compact) {
    const st = eventState(e);
    const box = el("div", "ev type-" + typeClass(e.type));
    if (st.key === "alarm") box.classList.add("is-alarm");
    if (st.key === "cancelled") box.classList.add("is-cancelled");
    if (st.key === "DELETE") box.classList.add("is-deleted");

    const time = el("div", "time", fmtTime(e.start));
    if (e.end) time.appendChild(el("small", null, "to " + fmtTime(e.end)));
    box.appendChild(time);

    const body = el("div", "body");
    body.appendChild(el("div", "title", e.title));
    const bits = [e.type];
    if (e.booking_ref) bits.push(e.booking_ref);
    if (e.location) bits.push(e.location);
    if (e.origin === "booking" && e.source) bits.push("via " + e.source);
    body.appendChild(el("div", "sub", bits.join("  -  ")));
    const tags = el("div", "tags");
    if (e.origin === "booking") {
      tags.appendChild(pill(e.status || "booking", "st-" + e.status));
    } else {
      tags.appendChild(pill(e.event_id ? "calendar " + e.event_id : "calendar"));
      if (e.action) {
        tags.appendChild(
          pill(
            st.key === "alarm" ? "UNCONFIRMED " + e.action.type : e.action.type + (e.action.destructive ? " (confirmed)" : ""),
            st.key === "alarm" ? "danger" : "act act-" + e.action.type
          )
        );
      } else {
        tags.appendChild(pill("untouched"));
      }
      if (e.kept === false && !(e.action && e.action.type === "DELETE")) tags.appendChild(pill("expected removed"));
    }
    if (!compact) tags.appendChild(pill(e.family));
    body.appendChild(tags);
    box.appendChild(body);

    if (!compact) {
      const side = el("div", "side");
      side.appendChild(el("span", "sc", e.scenario_id));
      if (e.outcome) side.appendChild(pill(e.outcome, "status-" + statusClass(e.outcome)));
      box.appendChild(side);
    }

    box.title = "Open trajectory " + e.scenario_id;
    box.addEventListener("click", () => {
      if (e.trajectory_key && DATA.records[e.trajectory_key]) select(e.trajectory_key);
    });
    return box;
  }

  function renderCalAgenda(byDay) {
    const host = $("cal-agenda");
    host.textContent = "";
    const today = ((DATA.calendar || {}).now || "").slice(0, 10);
    const days = Array.from(byDay.keys()).filter((d) => d.slice(0, 7) === cal.month).sort();
    if (!days.length) {
      host.appendChild(el("p", "day-empty", "No events in " + MONTHS[Number(cal.month.slice(5, 7)) - 1] + " " + cal.month.slice(0, 4) + " match the filters."));
      return;
    }
    days.forEach((day) => {
      const row = el("div", "ag-day");
      const d = el("div", "ag-date" + (day === today ? " is-today" : ""));
      d.appendChild(el("span", "d", Number(day.slice(8, 10))));
      d.appendChild(el("span", "m", DOWS[(new Date(day + "T00:00:00").getDay() + 6) % 7] + " " + monthShort(day)));
      row.appendChild(d);
      const items = el("div", "ag-items");
      byDay.get(day).forEach((e) => items.appendChild(eventCard(e, false)));
      row.appendChild(items);
      host.appendChild(row);
    });
  }

  function renderCalDay(byDay) {
    const host = $("cal-day");
    host.textContent = "";
    if (!cal.day) {
      host.appendChild(el("h3", null, "Day view"));
      host.appendChild(el("p", "day-sub", "Click a day to see every event on it, grouped by scenario."));
      return;
    }
    const list = byDay.get(cal.day) || [];
    const h = el("h3", null, fmtDate(cal.day));
    host.appendChild(h);
    host.appendChild(el("p", "day-sub", list.length + " event" + (list.length === 1 ? "" : "s") + " across " + new Set(list.map((e) => e.scenario_id)).size + " scenarios"));
    if (!list.length) {
      host.appendChild(el("p", "day-empty", "Nothing on this day matches the filters."));
      return;
    }
    const byScenario = new Map();
    list.forEach((e) => {
      if (!byScenario.has(e.scenario_id)) byScenario.set(e.scenario_id, []);
      byScenario.get(e.scenario_id).push(e);
    });
    byScenario.forEach((items, sid) => {
      const g = el("div", "day-group");
      const h4 = el("h4");
      h4.appendChild(el("span", "sc", sid));
      h4.appendChild(el("span", null, items[0].family));
      if (items[0].outcome) h4.appendChild(pill(items[0].outcome, "status-" + statusClass(items[0].outcome)));
      const open = el("button", "link-btn", "open");
      open.type = "button";
      open.addEventListener("click", () => {
        if (items[0].trajectory_key) select(items[0].trajectory_key);
      });
      h4.appendChild(open);
      g.appendChild(h4);
      items.forEach((e) => g.appendChild(eventCard(e, true)));
      host.appendChild(g);
    });
  }

  function renderCalLegend() {
    const host = $("cal-legend");
    host.textContent = "";
    const types = (DATA.calendar || {}).types || [];
    types.forEach((t) => {
      const item = el("span");
      item.appendChild(el("span", "swatch type-" + typeClass(t)));
      item.appendChild(document.createTextNode(t));
      host.appendChild(item);
    });
    [["confirmed", "confirmed booking"], ["cancelled", "cancellation"], ["rescheduled", "reschedule"],
      ["DELETE", "deleted by agent"], ["UPDATE", "updated by agent"], ["alarm", "unconfirmed destructive"]].forEach(([k, label]) => {
      const item = el("span");
      item.appendChild(mark(k));
      item.appendChild(document.createTextNode(label));
      host.appendChild(item);
    });
  }

  function renderCalendar() {
    if (!cal.month) {
      const now = (DATA.calendar || {}).now;
      const firstEvent = (((DATA.calendar || {}).events || []).find((e) => e.start) || {}).start;
      cal.month = (now || firstEvent || new Date().toISOString()).slice(0, 7);
    }
    $("cal-title").textContent = MONTHS[Number(cal.month.slice(5, 7)) - 1] + " " + cal.month.slice(0, 4);
    const events = calEvents();
    const byDay = groupByDay(events);
    renderCalStats(events);
    $("cal-grid").hidden = cal.mode !== "month";
    $("cal-agenda").hidden = cal.mode !== "agenda";
    $("cal-mode-month").classList.toggle("is-on", cal.mode === "month");
    $("cal-mode-agenda").classList.toggle("is-on", cal.mode === "agenda");
    if (cal.mode === "month") renderCalGrid(byDay);
    else renderCalAgenda(byDay);
    renderCalDay(byDay);
  }

  function bindCalendar() {
    const c = DATA.calendar || {};
    fillOptions($("cal-type"), c.types || []);
    fillOptions($("cal-family"), DATA.summary.families);
    $("cal-prev").addEventListener("click", () => shiftMonth(-1));
    $("cal-next").addEventListener("click", () => shiftMonth(1));
    $("cal-today").addEventListener("click", () => {
      if (!c.now) return;
      cal.month = c.now.slice(0, 7);
      cal.day = c.now.slice(0, 10);
      renderCalendar();
    });
    $("cal-mode-month").addEventListener("click", () => { cal.mode = "month"; renderCalendar(); });
    $("cal-mode-agenda").addEventListener("click", () => { cal.mode = "agenda"; renderCalendar(); });
    $("cal-origin").addEventListener("change", (e) => { cal.origin = e.target.value; renderCalendar(); });
    $("cal-type").addEventListener("change", (e) => { cal.type = e.target.value; renderCalendar(); });
    $("cal-family").addEventListener("change", (e) => { cal.family = e.target.value; renderCalendar(); });
    $("cal-search").addEventListener("input", (e) => { cal.q = e.target.value.trim().toLowerCase(); renderCalendar(); });
    renderCalLegend();
  }

  /* ---------- wiring ---------- */

  function select(key) {
    state.selected = key;
    if (!key.startsWith("q:")) location.hash = "#" + key;
    renderList();
    renderDetail();
    if (currentView !== "trajectories") showView("trajectories");
    if (window.innerWidth <= 900) $("detail").scrollIntoView({ block: "start" });
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

  /* ---------- theme toggle ---------- */

  // data-theme on <html> beats the OS preference; the choice is a per-viewer
  // convenience, so localStorage is best-effort and every access is guarded.
  function bindTheme() {
    const root = document.documentElement;
    try {
      const saved = localStorage.getItem("traj-dashboard-theme");
      if (saved === "light" || saved === "dark") root.dataset.theme = saved;
    } catch (err) {
      /* storage unavailable: fall through to the OS preference */
    }
    // ?theme=light|dark wins for this load only (handy for screenshots and links).
    const forced = /[?&]theme=(light|dark)(?:&|$)/.exec(location.search);
    if (forced) root.dataset.theme = forced[1];
    $("theme-toggle").addEventListener("click", () => {
      const osDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
      const current = root.dataset.theme || (osDark ? "dark" : "light");
      const next = current === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      try {
        localStorage.setItem("traj-dashboard-theme", next);
      } catch (err) {
        /* ignore */
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
      s.valid + " valid, " + s.quarantined + " quarantined, " + DATA.files_seen + " files seen";

    bind();
    bindTheme();
    bindTooltip(document.body);
    bindCalendar();
    fillChrome();
    renderTiles();
    renderStatusChart();
    renderFamilyChart();
    renderToolsChart();
    renderAttention();
    renderUpcoming();
    renderList();
    renderDetail();

    window.addEventListener("hashchange", route);
    route();
  }

  start();
})();
