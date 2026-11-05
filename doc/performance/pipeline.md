# Performance pipeline

The [history page](index) is part of the Sphinx site, under
`/immersx/performance/`. It uses a static JSON archive and a dependency-free
JavaScript viewer. No database, external chart service, or dedicated runner is
required. The pipeline only becomes active after this stack reaches `master`.

## Execution and permissions

`Documentation` calls the reusable `Performance measurements` workflow.
Explicit job dependencies enforce this order:

1. Plan monitored commits, build the existing benchmark targets with
   `IMMERSX_BUILD_BENCHMARKS=ON`, run them, and normalize their JSONL.
2. Upload the batch, including failed attempts. On trusted `master` runs only,
   append it to `performance-data`. Generate the report.
3. Check out the same source SHA, verify the report's `source_sha`, and build
   Sphinx/Doxygen. Copy report assets **after** `scripts/build_doc.sh` recreates
   the site directory, before uploading Pages.
4. Publish only if the run still describes the remote `master` head.

Numerical, configuration, build, execution, timeout and missing-case failures
make the measurement job fail **after** its artifact is uploaded. Downstream
archive and documentation jobs still run when a batch exists, so the failure
is visible on the site. Infrastructure failure before planning cannot invent
an attempt: the next plan exposes and recovers the gap. A broken archive upload
blocks documentation publication rather than publishing misleading old data.

PRs run on GitHub-hosted runners with read-only permissions. They get an
experimental report artifact, but cannot publish history or Pages. The writer
job is separate, runs only on `master` push, schedule or manual events, and does
not execute benchmark binaries. The reusable workflow also forces other events
to the hosted runner and preview mode. In Actions, the writer checks the batch's
source/harness SHA, selected commits, run ID and attempt number before opening
an archive checkout. The benchmark checkout does not retain Git credentials,
and the container receives no write token. No repository settings, secrets or runners
are provisioned by these workflows.

## Which commits are monitored?

“Every commit on master” means every commit in **first-parent order**, including
merge commits and every first-parent commit in a multi-commit push. Individual
commits reachable only through the second parent of a merge are not measured
separately. This measures each successive state of the main development line.

The initial monitoring baseline is explicitly the first first-parent commit
that adds `scripts/performance/config.json` to `master` (the activation commit).
That commit is included. The planner persists its full SHA in
`performance-data:baseline.json`; it never starts from the beginning of the
repository, and it never imports the checked-in historical CP8 reports as new
measurements. The timing baseline is different: it is the first valid result
in each compatible environment/case series.

Every plan enumerates the full monitored interval through the triggering SHA.
It selects the oldest missing commits, at most eight per run. All remaining
commits are still listed as missing. A daily scheduled run drains a backlog;
pushes and manual runs do the same. Already failed commits are retained and
require `retry=true` or an explicit SHA to retry, so a persistent failure cannot
starve later commits. There is no deletion of pending work when a push arrives.
A non-fast-forward change to the monitored first-parent history is rejected and
requires human review; the pipeline never silently resets its baseline.

The concurrency group serializes official documentation runs and does not
cancel the running one. It is **not** a durable queue. GitHub's current
[concurrency documentation](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency)
(described here as checked on 2026-10-13) says the default `queue: single`
retains at most one pending run and replaces it with a newer one; ordering is
not guaranteed. Optional `queue: max` allows at most 100 pending runs, not an
unbounded queue. This implementation does not depend on that newer option.
Git history plus the archive ledger make replaced, skipped, canceled or timed
out runs recoverable. Scheduled Actions can also be delayed or dropped; manual
backfill remains available.

Late batches append immutable attempts but cannot shorten `plan.json` or move
its current SHA backward. Official report exports use that retained ledger,
not an older incoming plan; source verification blocks documentation from a
superseded batch. The report's current commit is the newest monitored
commit, **even if missing or failed**, not the most recent successful sample.
For a given SHA, the latest attempt by start timestamp wins, including failures.
Completion order does not replace newer attempts. Old attempts remain available
by permalink. The deployment freshness check and shared official concurrency
group prevent an older workflow from overwriting a newer deployment. If master
advances after that check, the running deployment may briefly precede the new
one; it cannot overtake a newer workflow in the same group.

## Environment and measurement contract

The suite also runs `particle_penalty_benchmark`: ten scalar/vector nonmatching
cases use the step-80 native penalty oracle. See the
[particle penalty comparison](particle-penalty) for geometry, correctness and
timing details. These cases appear in the same history filters and chart, with
native timing ratios rather than N/A. Each penalty case runs on one and two
MPI ranks, giving 39 required case/dimension/rank combinations. Harness version
4 separates the expanded MPI suite from older runs.

The initial workflow uses GitHub-hosted `ubuntu-24.04`, Docker, and the same
`dealii/dealii:v9.7.1-noble` image as the C++ CI. It resolves and records the
image digest before executing it. These are **diagnostic** measurements:
CPU models and host load can change. Availability of a dedicated runner is not
assumed. The container may expose compatibility failures in CP8; those are
archived as failures, not replaced with local measurements.

The default configuration is Release, three refinements, nine repetitions,
one thread per rank. The harness sets `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `DEAL_II_NUM_THREADS` to one.
Weak-term is serial; lifted initializes MPI internally with one rank. The
particle penalty executable runs directly and under `mpirun -np 2`, using
distributed background meshes, partitioned integration-cell ownership and
distributed matrices/vectors. It synchronizes before each timing sample and
reports the maximum rank wall time. The rank filter keeps MPI and one-rank
series separate. A two-rank comparison is not a scaling study.
Both Debug and Release target names are supported by the local harness.

Each attempt records the compiler version, actual benchmark compile commands,
CMake options, deal.II version, image digest or local installation identifier,
OS/kernel, CPU model, architecture, threads, ranks and configuration. Raw
`CMakeCache.txt`, compile commands, configure/build logs, JSONL and stderr are
retained. The record carries DoF and cell counts, quadrature metadata and
repetitions. CP8 omits some fields: the adapter supplies the known quadrature
recipe and, for lifted only, derives uniform Q1 background counts from its
refinement parameter. Matching Q1 uses `QGauss(2)`, heterogeneous Q1/Q2 uses
`QGauss(3)`, nonmatching uses `QGauss<1>(2)` and fixed three refinements;
trace and lifted details follow the benchmark source recorded by the code SHA.
The rotated trace uses a fixed one-refinement background.

Compatibility hashes include the full environment, configuration, harness
version, case, mesh, rank count, explicit timing scope and operation counts.
New attempts also record the Git blob IDs of both benchmark sources; each series
includes the source ID of its executable. Incompatible CPUs, containers,
compilers, flags, builds, meshes or repetition counts never share a series.
A kernel update or any edit to that benchmark source starts a new series, even
if the edit is only a comment. Ordinary library changes keep the series.

The viewer orders compatible series by decreasing measurement count, with ties
preferring the series whose last measurement is on the newest commit. It keeps
a user's selected series while filters change if that series is still
available. Labels show the CPU, kernel, measurement count, commit range, and
compatibility hash. Hosted runners change CPU models and kernels, so a long
archive can contain several short series. A one-point series has no trend;
use another compatible series or collect more measurements on the same
environment.

The shipped harness is version 4. Earlier archives remain readable and are
never rewritten or relabelled. Version 2 separated explicit timing contracts
from the old CP8 recipe; version 4 adds the distributed penalty suite. The additive timing fields from PR #250 are preserved and included
in compatibility: `timing_scope`, `immersx_assembly_calls` and
`native_cell_traversals`. Missing scope is `legacy-unspecified`, not an inferred
claim about materialization or synchronization. Source isolation also separates
PR #250's trace materialization changes, which retain the old phase names and
do not add a scope field. Historical JSONL used in parser tests gets no new code
SHA. Increase `harness_version` whenever normalization or timing semantics
change outside the source/scope partition already recorded here.

For stable measurements, an administrator can provision an isolated runner
with Docker and set the reusable workflow's `runner`, `environment` (a unique
machine identifier) and `stable: true` inputs from a **trusted master-only
caller**. Do not expose that selection to PRs. The shipped caller
always uses the hosted default and labels results diagnostic. For a controlled
local installation, `measure --stable --environment <machine-id>` explicitly
labels a dedicated measurement. Changing only the label does not stabilize a
machine: CPU frequency policy, affinity and competing workloads remain the
operator's responsibility. Stable-runner provisioning is an external
prerequisite, not something this PR configures.

## Archive and schema version 1

`performance-data` is a separate Git branch, created by the first official
writer if repository policy permits `GITHUB_TOKEN` to create/update it. It has:

```text
baseline.json                 fixed monitoring baseline SHA and policy
plan.json                     latest non-regressing monitored commit ledger
attempts/<sha>-<run>-<attempt>.json
raw/<sha>-<run>-<attempt>/      original stdout, stderr and build provenance
```

Attempts are immutable. Re-importing identical content is a no-op; conflicting
content under the same identity is an error. Actions retries use a new attempt
number. Local attempts get a timestamp-derived identifier. Attempt files use
exclusive creation; a conflicting identity cannot overwrite an existing file.
Writers use ordinary fast-forward pushes and re-read/merge on contention, never
force push. Both raw
and normalized records persist independently of Actions artifact expiry.
Backup and retention of this branch are normal repository maintenance tasks;
there is deliberately no automatic history pruning.

The versioned JSON envelope has `schema_version`, code `sha`, `harness_sha`,
`harness_version`, `benchmark_sources` (new attempts), `run`, `attempt`,
`started_at`, `finished_at`, `configuration`, `environment`, `status`, `errors`,
`missing_cases`, `records` and `raw_path`.
Each normalized case contains:

- semantic case name, dimension, ranks and mesh/DoF/quadrature metadata;
- `phases`: named `{median_ms, mad_ms}` pairs, preserving native phases too;
- `checks`: measured errors and strict tolerances (CP8: `1e-12`, lifted measure
  absolute tolerance `1e-10` against `0.2`);
- `timing_contract` (new records), with explicit scope and operation counts;
- `status`, `native_ratio`, `native_na_reason`, and the complete original object
  in `raw`, including setup/cold timings and additional fields.

`history.json` is a derived export with `source_sha`, monitoring `baseline`,
`current`, per-commit `coverage`, timing `rows` and all `attempts`. Missing
commits have no fabricated timing rows. Failed attempts have explicit errors
and missing-case lists. Ratios are recomputed only for known equivalent
operations, including the MeshWorker end-to-end nonlinear comparison. A native
`null` or `"N/A"` in a non-equivalent case stays unavailable; it is never zero.
The viewer defaults to the dimensionless ImmersX/native ratio, with a dashed
1× reference, and offers absolute timings as a separate chart mode. Both modes
use the same compatibility partitions and exclude failed attempts and records.
Ratios can reduce machine-speed effects but do not justify merging measurements
from different hardware. Ratio charts do not alter the absolute-time regression
flags described below.
Malformed JSON, missing required fields, nonfinite or negative timings/errors,
zero median times, inconsistent ratios and wrong repetition/rank counts are
rejected. Unknown additional fields are preserved, not treated as evidence of
correctness. CP8's known PETSc unused-option messages are retained in raw stdout
but excluded from JSON parsing; arbitrary malformed lines still fail validation.

Regression flags are diagnostic only: the ImmersX median must increase by more
than 10%, 0.1 ms, **and** three times the sum of previous and current MADs. This
is a heuristic, not a confidence interval. Native overhead and historical
regressions are separate columns. Failed attempts do not contribute historical
comparison baselines. No automatic performance-based merge gate is enabled.

GitHub [artifact documentation](https://docs.github.com/en/actions/using-workflows/storing-workflow-data-as-artifacts)
states that v4 artifacts are immutable, transfers can use explicit `needs`, and
retention cannot exceed repository/organization policy. The
[upload-artifact limits](https://github.com/actions/upload-artifact#limitations)
include 500 artifacts per job and storage quota limits. This pipeline uses one
batch and one report per run, named with run ID and attempt, retained for 30
days subject to policy. They are transport/recovery copies, not the archive.

## Local commands

Use an out-of-source build, Python 3.10+, CMake, the project dependencies and a
deal.II installation. The measurement command builds both existing executables;
it does not implement assembly itself. A clean committed source revision is
measured in a temporary detached worktree. Uncommitted C++ changes are not part
of that revision. The harness itself is identified by the checkout SHA, so
commit harness changes before collecting provenance-bearing measurements.

```bash
python3 -m unittest discover -s scripts/performance/tests -v
python3 scripts/performance/pipeline.py plan --preview \
  --archive /tmp/immersx-perf-empty --output /tmp/immersx-perf-plan.json
python3 scripts/performance/pipeline.py measure \
  --plan /tmp/immersx-perf-plan.json --output /tmp/immersx-perf-batch \
  --environment local-diagnostic --image 'local-dealii-installation-id'
python3 scripts/performance/pipeline.py export \
  --archive /tmp/immersx-perf-batch --plan /tmp/immersx-perf-plan.json \
  --output /tmp/immersx-perf-report
IMMERSX_PERFORMANCE_REPORT=/tmp/immersx-perf-report ./scripts/build_doc.sh
```

After a documentation build, optional Chromium checks exercise the viewer under
`/immersx/`, including empty/partial history, filters, failed attempts, permalinks,
unsafe strings and missing downloads:

```bash
uv run --with playwright python -m playwright install chromium
uv run --with playwright python scripts/performance/tests/viewer.py
```

`measure` deliberately exits successfully after recording individual failed
attempts, allowing archive/export to continue; inspect attempt `status` locally.
Actions adds a separate validation step that returns failure for these records.
Without `IMMERSX_PERFORMANCE_REPORT`, the documentation command generates an
explicit empty history. It requires neither benchmark binaries nor credentials.
Serve `build/docs/site` over HTTP to use `fetch`; opening `file://` pages can be
blocked by browser policy. Paths are relative and work under `/immersx/`.

## Backfill and recovery

After the workflow reaches the default branch:

```bash
# Drain the oldest missing commits (eight per invocation).
gh workflow run doxygen.yml --ref master
# Also retry failures; previous attempts remain in history.
gh workflow run doxygen.yml --ref master -f retry=true
# Force a new attempt for one full SHA inside the monitored interval.
gh workflow run doxygen.yml --ref master -f commit=<full-master-sha>
```

Re-running a failed Actions workflow also increments its attempt identity.
For a lost archive upload, download the original batch before its retention
expires and, from a trusted checkout with write credentials, run:

```bash
gh run download <run-id> --name performance-batch-<run-id>-<attempt> \
  --dir /tmp/recovered-batch
python3 scripts/performance/archive.py publish \
  --incoming /tmp/recovered-batch --plan /tmp/recovered-batch/plan.json
```

Only restore an original trusted-master batch to the official branch. Do not
publish PR artifacts, synthetic fixtures or local experiments there. Follow
recovery with a manual documentation run on current `master`; do not redeploy an
old Pages artifact. If branch policy denies writes, an administrator must grant
the intended branch permission or arrange a trusted writer; this implementation
does not bypass protection or change repository settings.

The performance overview draws every recorded case/dimension/rank/build combination
without selecting a series. Each environment and timing contract remains a
separate chart. Native-unavailable and failed measurements remain visible as
explanations rather than zero-valued samples. The detailed table and attempt
records retain the full validation evidence.
