# Performance

These measurements are diagnostics, not speed guarantees. GitHub-hosted CPUs
and host load vary. Compare only matching configurations and environments.
A native ratio measures overhead against deal.II at the **same commit**;
a regression compares ImmersX across **different commits**.

The default chart shows **ImmersX / native deal.II**, the ratio of paired warm
median times. Select **Absolute timings** to inspect individual timing phases.
A ratio of 1 means equal time; 1.2 means 20% overhead; below 1 means ImmersX is
faster. The dashed line marks 1×. Ratios reduce sensitivity to machine speed,
but cache, memory and compiler differences can still affect overhead. The
viewer therefore keeps incompatible environments and configurations in separate
series, even in ratio mode. Cases without equivalent native timings and failed
attempts are excluded from ratio charts, not plotted as zero.

The overview shows every recorded case/dimension/rank/build combination without
selecting a series. Each card shows its most complete compatible history; ties
prefer the newest commit. Expand additional series to inspect other compatible
configurations, each in its own chart. Cases with
failed measurements or unavailable native comparisons remain visible with an
explanation. The filters and chart metric apply to both the overview and detail.

The detailed chart initially selects the compatible series with the most measurements;
ties prefer the series reaching the newest commit. The series selector shows
the CPU, kernel, and point count. Hosted runners can use different CPUs or
kernels on successive runs, so the archive can contain many commits split
across shorter compatible series. A single sample is a baseline point, not
a historical trend.

The table shows medians and median absolute deviations (MAD), in milliseconds.
Positive changes mean slower execution. A diagnostic regression requires an
increase greater than all of: 10%, 0.1 ms, and three times the sum of both MADs.
The comparison baseline is the first valid result in each compatible series.
Failed attempts never become timing baselines.

```{raw} html
<div id="performance-history">
  <p id="performance-status" role="status">Loading performance history…</p>
  <noscript>JavaScript is needed for filters and charts. Download the JSON history below.</noscript>
  <p><a href="data/history.json">Download complete history (JSON)</a></p>
  <div id="performance-filters"></div>
  <h2>Commit coverage</h2>
  <div id="performance-coverage"></div>
  <h2>Timing history</h2>
  <label for="performance-metric">Chart metric</label>
  <select id="performance-metric">
    <option value="ratio">ImmersX / native deal.II</option>
    <option value="absolute">Absolute timings</option>
  </select>
  <h2>All benchmark cases</h2>
  <div id="performance-overview"></div>
  <h2>Selected series and measurements</h2>
  <label for="performance-series">Chart series</label>
  <select id="performance-series"></select>
  <div id="performance-chart"></div>
  <div id="performance-results" style="overflow-x:auto"></div>
  <h2>Attempts and validation</h2>
  <div id="performance-attempts"></div>
</div>
<link rel="stylesheet" href="../_static/performance.css">
<script src="../_static/performance.js" defer></script>
```

The [step-80 particle penalty cases](particle-penalty) compare scalar and vector
bulk–bulk integration with a pinned native deal.II implementation, including
codimension two and independent volume meshes.

Legacy lifted and bulk–bulk trace cases have no native timing oracle recorded. Boundary
flux times different operations in ImmersX and deal.II, so its ratio is also
N/A. Cold and setup timings, numerical checks and tolerances are available in
the attempt details and downloadable raw output. Legacy CP8 trace assembly
phases stop before matrix materialization. PR #250 includes materialization and
adds explicit scope and operation counts for the nonlinear end-to-end comparison:
three ImmersX calls versus one fused MeshWorker traversal, not the same kernel.
The viewer keeps benchmark source versions and explicit scopes in separate
series. Missing scope means unspecified legacy timing, not full assembly.

See [the pipeline guide](pipeline) for environment details, commit coverage,
local execution, backfill and recovery.

The [one-vessel elastodynamics cost analysis](metric-flow-x-elastodynamics)
profiles coupled Jacobian setup and nested iterative solves, and compares the
parameter-selected serial UMFPACK path.

```{toctree}
:maxdepth: 1
:titlesonly:

particle-penalty
elastodynamics
metric-flow-x-elastodynamics
pipeline
```
