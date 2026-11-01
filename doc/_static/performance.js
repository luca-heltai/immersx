/* Static, dependency-free viewer. Render archive strings as text, never HTML. */
(() => {
  "use strict";
  const root = document.getElementById("performance-history");
  if (!root) return;
  const byId = (id) => document.getElementById(`performance-${id}`);
  const element = (tag, text, parent) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  };
  const link = (text, href, parent) => {
    const node = element("a", text, parent);
    node.href = href;
    return node;
  };
  const commitURL = (sha) => `https://github.com/luca-heltai/immersx/commit/${encodeURIComponent(sha)}`;
  const fmt = (v) => v === null || v === undefined ? "N/A" : Number(v).toPrecision(5);
  const percent = (v) => v === null || v === undefined ? "N/A" : `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
  const svgNS = "http://www.w3.org/2000/svg";
  function chart(rows, phase) {
    const target = byId("chart");
    target.replaceChildren();
    const ratio = phase === "native_ratio";
    if (!rows.length) {
      element("p", ratio
        ? "No validated equivalent native comparison for these filters. Choose absolute timings to inspect other cases."
        : "No validated timings for these filters.", target);
      return;
    }
    const value = (r) => ratio ? r.native_ratio : r.phases[phase].median_ms;
    const unit = ratio ? "× native deal.II" : "ms";
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", "0 0 700 220");
    svg.setAttribute("role", "img");
    const title = document.createElementNS(svgNS, "title");
    title.textContent = ratio
      ? "ImmersX / native deal.II, dimensionless ratio in commit order. 1× means equal time."
      : `${phase}, milliseconds in commit order. Exact values and MAD appear in the table.`;
    svg.appendChild(title);
    const max = Math.max(ratio ? 1.2 : 0, ...rows.map(value));
    const y = (v) => 190 - 160 * v / max;
    if (ratio) {
      const reference = document.createElementNS(svgNS, "line");
      reference.setAttribute("x1", "30"); reference.setAttribute("x2", "670");
      reference.setAttribute("y1", y(1)); reference.setAttribute("y2", y(1));
      reference.setAttribute("stroke", "currentColor");
      reference.setAttribute("stroke-dasharray", "5 5");
      svg.appendChild(reference);
      const label = document.createElementNS(svgNS, "text");
      label.setAttribute("x", "30"); label.setAttribute("y", y(1) - 6);
      label.setAttribute("fill", "currentColor"); label.setAttribute("font-size", "12");
      label.textContent = "1× native deal.II";
      svg.appendChild(label);
    }
    const coords = rows.map((r, i) => [30 + 640 * i / Math.max(1, rows.length - 1), y(value(r))]);
    const line = document.createElementNS(svgNS, "polyline");
    line.setAttribute("points", coords.map((p) => p.join(",")).join(" "));
    line.setAttribute("fill", "none");
    line.setAttribute("stroke", "currentColor");
    svg.appendChild(line);
    coords.forEach(([x, y], i) => {
      const dot = document.createElementNS(svgNS, "circle");
      dot.setAttribute("cx", x); dot.setAttribute("cy", y); dot.setAttribute("r", "4");
      dot.setAttribute("fill", "currentColor");
      const label = document.createElementNS(svgNS, "title");
      label.textContent = `${rows[i].sha.slice(0, 10)}: ${fmt(value(rows[i]))} ${unit}`;
      dot.appendChild(label); svg.appendChild(dot);
    });
    target.appendChild(svg);
    element("p", `0–${fmt(max)} ${unit}; ${rows.length} valid measurements, oldest to newest. Gaps are listed above.`, target);
    if (ratio) element("p", "Ratio of paired warm medians: 1× = equal time; 1.2× = 20% overhead; below 1× = faster than native deal.II. Ratios reduce machine-speed effects, but are not hardware-independent.", target);
    if (rows.length === 1) element("p", "Only one compatible measurement: no historical trend yet.", target);
  }
  async function main() {
    const response = await fetch(new URL("data/history.json", window.location.href));
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (data.schema_version !== 1) throw new Error("Unsupported history schema");
    byId("status").textContent = data.coverage.length
      ? `${data.coverage.length} monitored commits. Current: ${data.current.slice(0, 12)}. Monitoring baseline: ${data.baseline.slice(0, 12)}.`
      : "No measurements yet. This local build has no official history attached.";
    const coverage = element("ul", undefined, byId("coverage"));
    data.coverage.slice().reverse().forEach((c) => {
      const li = element("li", undefined, coverage);
      link(c.sha.slice(0, 12), `#commit-${c.sha}`, li);
      element("span", ` — ${c.status} (${c.attempts.length} attempts) `, li);
      link("source", commitURL(c.sha), li);
      li.id = `commit-${c.sha}`;
    });
    const selectors = {};
    const dimensions = {
      case: (r) => r.case,
      dimension: (r) => String(r.dimension),
      "build type": (r) => r.configuration.build_type,
      ranks: (r) => String(r.ranks),
      configuration: (r) => r.series,
    };
    const params = new URLSearchParams(window.location.hash.replace(/^#/, ""));
    Object.entries(dimensions).forEach(([name, getter]) => {
      const label = element("label", `${name} `, byId("filters"));
      const select = element("select", undefined, label);
      select.setAttribute("aria-label", name);
      const all = element("option", "All", select); all.value = "";
      [...new Set(data.rows.map(getter))].sort().forEach((value) => {
        const option = element("option", value, select); option.value = value;
      });
      select.value = params.get(name) || "";
      select.addEventListener("change", render);
      selectors[name] = select;
      element("span", " ", byId("filters"));
    });
    const metric = byId("metric");
    metric.value = params.get("metric") === "absolute" ? "absolute" : "ratio";
    metric.addEventListener("change", render);
    function render() {
      const filter = new URLSearchParams();
      if (metric.value === "absolute") filter.set("metric", "absolute");
      Object.entries(selectors).forEach(([key, select]) => { if (select.value) filter.set(key, select.value); });
      if (filter.size) history.replaceState(null, "", `#${filter}`);
      else if (window.location.hash.includes("=")) history.replaceState(null, "", window.location.pathname + window.location.search);
      const rows = data.rows.filter((r) => Object.entries(selectors).every(([key, s]) => !s.value || dimensions[key](r) === s.value));
      const target = byId("results"); target.replaceChildren();
      const table = element("table", undefined, target);
      element("caption", "Absolute timings, native overhead and commit-to-commit changes", table);
      const head = element("tr", undefined, element("thead", undefined, table));
      ["Commit / case", "Phase", "Median ± MAD (ms)", "ImmersX / native", "Previous", "Baseline", "Status"].forEach((s) => {
        const th = element("th", s, head); th.scope = "col";
      });
      const body = element("tbody", undefined, table);
      rows.slice().reverse().forEach((r) => {
        Object.entries(r.phases).forEach(([phase, timing]) => {
          const tr = element("tr", undefined, body);
          const name = element("td", undefined, tr);
          link(r.sha.slice(0, 10), `#attempt-${r.id}`, name);
          element("span", ` ${r.case} (${r.dimension}D, ${r.ranks} ranks)`, name);
          element("td", phase, tr).title = JSON.stringify(r.timing_contract || {scope: "legacy-unspecified"});
          element("td", `${fmt(timing.median_ms)} ± ${fmt(timing.mad_ms)}`, tr);
          element("td", phase === "warm_immersx" ? fmt(r.native_ratio) : "N/A", tr).title = r.native_na_reason || "Ratio is defined only for the paired warm phase";
          const change = r.changes[phase] || {};
          element("td", percent(change.previous_percent), tr).title = change.previous_sha || "No compatible predecessor";
          element("td", percent(change.baseline_percent), tr).title = change.baseline_sha || "No compatible baseline";
          element("td", r.status === "failed" ? "Numerical failure" : r.attempt_status === "failed" ? "Failed attempt" : change.regression ? "Diagnostic regression" : "Validated", tr);
        });
      });
      const chooser = byId("series"), old = chooser.value;
      chooser.style.maxWidth = "100%";
      chooser.replaceChildren();
      const groups = new Map();
      rows.filter((r) => r.status === "passed" && r.attempt_status === "passed").forEach((r) => {
        const phases = metric.value === "ratio"
          ? (Number.isFinite(r.native_ratio) && r.native_ratio > 0 ? ["native_ratio"] : [])
          : Object.keys(r.phases).sort((a, b) => (b === "warm_immersx") - (a === "warm_immersx"));
        phases.forEach((phase) => {
          const key = `${r.series}/${phase}`;
          if (!groups.has(key)) groups.set(key, []);
          groups.get(key).push(r);
        });
      });
      const commitOrder = new Map(data.coverage.map((c, i) => [c.sha, i]));
      const ordered = [...groups].sort((a, b) => b[1].length - a[1].length ||
        commitOrder.get(b[1][b[1].length - 1].sha) - commitOrder.get(a[1][a[1].length - 1].sha));
      ordered.forEach(([key, rs]) => {
        const first = rs[0], last = rs[rs.length - 1];
        const machine = `${first.environment.cpu || "Unknown CPU"}; ${first.environment.os || "unknown kernel"}`;
        const range = `${first.sha.slice(0, 10)} → ${last.sha.slice(0, 10)}`;
        const label = `${first.case} ${first.dimension}D ${first.ranks} ranks — ${rs.length} points — ${machine} — ${range} — ${key}`;
        const option = element("option", label, chooser); option.value = key;
        option.title = label;
      });
      if (groups.has(old)) chooser.value = old;
      const draw = () => chart(groups.get(chooser.value) || [],
        metric.value === "ratio" ? "native_ratio" : chooser.value.split("/")[1]);
      chooser.onchange = draw; draw();
    }
    data.attempts.slice().reverse().forEach((a) => {
      const details = element("details", undefined, byId("attempts"));
      details.id = `attempt-${a.id}`;
      element("summary", `${a.sha.slice(0, 12)} — ${a.status} — ${a.started_at} — ${a.environment.diagnostic !== false ? "diagnostic" : "dedicated"}`, details);
      link("Permalink", `#attempt-${a.id}`, details);
      element("span", " · ", details);
      if (/^https:\/\/github\.com\//.test(a.run_url)) link("Actions run", a.run_url, details);
      element("span", " · ", details);
      link("Attempt JSON", `data/attempts/${encodeURIComponent(a.id)}.json`, details);
      element("pre", JSON.stringify({harness_sha: a.harness_sha, harness_version: a.harness_version,
        benchmark_sources: a.benchmark_sources, environment: a.environment, configuration: a.configuration,
        errors: a.errors, missing_cases: a.missing_cases,
        checks: a.records.map((r) => ({case: r.case, dimension: r.dimension, timing_contract: r.timing_contract,
          checks: r.checks, native_na_reason: r.native_na_reason, raw: r.raw}))}, null, 2), details);
      // Known filenames only: no arbitrary URLs supplied by benchmark output.
      ["configure.log", "CMakeCache.txt", "compile_commands.json", ...["weak_term_assembly", "lifted_integration_support", "particle_penalty"].flatMap((n) => {
        const target = `${n}_benchmark${a.configuration.build_type === "Debug" ? "_debug" : ""}`;
        const mpi = n === "particle_penalty" ? (a.configuration.particle_penalty_ranks || [])
          .filter((r) => Number.isInteger(r) && r > 1)
          .flatMap((r) => [`${target}-mpi${r}.jsonl`, `${target}-mpi${r}.stderr`]) : [];
        return [`${target}-build.log`, `${target}.jsonl`, `${target}.stderr`, ...mpi];
      })].filter((file) => (a.raw_files || []).includes(file)).forEach((file) => { link(file, `data/raw/${encodeURIComponent(a.id)}/${file}`, details); element("br", undefined, details); });
    });
    render();
    function reveal() {
      const target = document.getElementById(window.location.hash.slice(1));
      if (target) { if (target.tagName === "DETAILS") target.open = true; target.scrollIntoView(); }
    }
    window.addEventListener("hashchange", reveal); reveal();
  }
  main().catch((error) => { byId("status").textContent = `History unavailable: ${error.message}. No data have been substituted.`; });
})();
