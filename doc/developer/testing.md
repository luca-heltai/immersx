# Testing

The repository tests run continuously on GitHub Actions:

[![Tests](https://github.com/luca-heltai/immersx/actions/workflows/tests.yml/badge.svg)](https://github.com/luca-heltai/immersx/actions/workflows/tests.yml)
[![Documentation](https://github.com/luca-heltai/immersx/actions/workflows/doxygen.yml/badge.svg)](https://github.com/luca-heltai/immersx/actions/workflows/doxygen.yml)
[![Indentation](https://github.com/luca-heltai/immersx/actions/workflows/indentation.yml/badge.svg)](https://github.com/luca-heltai/immersx/actions/workflows/indentation.yml)

See the [GitHub Actions workflows](https://github.com/luca-heltai/immersx/actions)
for current runs, logs, and artifacts.

CTest is the authoritative test runner. From the build directory:

```bash
ctest --output-on-failure                 # full suite
ctest -L quick --output-on-failure        # normal development gate
ctest -L unit --output-on-failure
ctest -L integration --output-on-failure
ctest -L validation --output-on-failure
ctest -L application --output-on-failure
```

Test layout:

- `unit` contains focused checks of individual data structures, parsing,
  algebra, geometry, and other reusable components.
- `integration` contains interactions among ImmersX components, distributed
  execution, and solver/adapter composition.
- `validation` contains manufactured-solution, analytic/native oracle, and
  physics-validation checks on small meshes.
- `application` contains executable and end-to-end application behavior.

Both `gtests/` and `tests/` use these primary categories. `quick` is an
orthogonal label applied to all unit and integration tests, while `mpi` marks
tests that require more than one MPI rank. GoogleTest categories are CTest
groups over one aggregate executable, so sources are compiled once and CTest
remains the single orchestration layer.

To inspect the inventory and labels:

```bash
ctest -N
ctest -N -V
```

For focused development, use CTest when possible:

```bash
ctest -R 'ImmersX\.integration\.Serial' --output-on-failure
ctest -R 'ImmersX\.integration\.MPI' --output-on-failure
```

GoogleTest targets are enabled only when `GTest` is available at configure time.

## Correctness-test budget

Each GoogleTest case must finish within 120 seconds, including fixture setup
and teardown. The common driver stops a case after 115 seconds on every MPI
rank, leaving five seconds within the budget for launcher shutdown. A watchdog
timeout fails the test process; the timing report also rejects completed cases
above the 120-second limit. Application launchers register a private process group
so a timeout also stops their subprocesses and closes their output pipes.
A CTest group can contain several
cases, so its aggregate duration is distinct from an individual case's budget.

Use the smallest fixture that exercises the contract. Keep nonzero loads,
moving boundary data, hanging nodes, distributed ownership, and independent
oracles where they are needed. When consolidating tests, identify the surviving
case for each assertion. Do not retain a large refinement or parameter sweep
merely to exercise a path already covered on a small mesh. Manual accuracy and
scalability studies belong outside the GoogleTest regression suite.

## CI selection

Pull requests normally run the `quick` label in Debug and Release, including
serial and MPI coverage. Documentation-only PRs run documentation, mathematics,
formatting, and lightweight tooling checks without compiling numerical or Coral
targets. The allowlist covers documentation pages and assets, the README,
changelog, Sphinx configuration, Doxyfile, and documentation requirements.
Executable examples, parameter files, and unknown inputs retain numerical checks.

`scripts/ci/select_changes.py` compares the actual PR head with its merge base,
including both paths of renames and deleted files. Missing history, malformed
diffs, empty changes, and unknown paths retain every normal PR job. Each workflow
uploads its `ci-plan` artifact with the changed paths and selection reasons.

C++ component ownership is recorded in observation mode. It does not exclude
tests: changes to implementation, headers, parameters, meshes, or baselines still
run the complete quick matrix. A component hint is not a transitive dependency
proof. Compiler and link dependencies, runtime fixture ownership, and comparisons
with complete runs are required before enabling finer exclusions.

Pushes to `master` run every CTest category and both Coral configurations.
The independent package job executes the complete `MetricFlowXPackage` matrix
once, including its Debug, Release, and DebugRelease scenarios. Ordinary Debug
and Release jobs disable only that duplicate registration through
`IMMERSX_ENABLE_METRIC_FLOW_X_PACKAGE_TESTS=OFF`; the local default remains `ON`.
CI does not rerun the aggregate GoogleTest binary outside CTest.

The ordinary jobs explicitly disable the optional legacy 1D solver rather than
downloading archives that fail their toolchain link check. Its client interfaces
are still compiled without linking that library. A compatible installation can
enable the drivers in a local configuration; those application targets then
provide the same interface compilation checks.

Use the GitHub CI workflow's manual dispatch with `full_suite=true` to run all
categories on a chosen branch. The final `ci` jobs require every selected job to
succeed and accept only explicitly planned skips. Workflows remain triggered on
every PR, so documentation-only changes still produce a final check result.

Documentation previews do not wait for performance measurements. They contain
an explicitly unmeasured performance report, and still run the viewer and MathJax
checks. Numerical PRs run small benchmark correctness cases through CTest;
master and scheduled documentation runs retain complete performance measurements
and append-only publication. Benchmark compilation uses a shared ccache with
source/build prefix maps; compilation logs and cache statistics remain in the
raw measurement artifacts.

The documentation builder uses two Sphinx workers. Set `SPHINX_JOBS` explicitly
when a local allocation requires a different worker budget.

## Timing and focused reruns

The Debug and Release jobs preserve the CTest manifest, JUnit results, and
temporary test logs as `ctest-debug` and `ctest-release` artifacts for 14 days.
These artifacts also contain compile commands, Ninja header dependencies,
`.ninja_log` command durations, and the target graph for reviewing component
selection. Runtime fixture dependencies
still require explicit ownership. GoogleTest XML and `test-timings.json` record
the case observations. The slowest
cases are listed in the job summary. MPI XML describes rank zero; the MPI/CTest
exit status remains authoritative for failures on any rank.
The report lists the longest build commands separately: elapsed command times
can overlap and must not be summed as build wall time. The Debug job reserves
4 GB for ccache; compare hits, misses, and cleanup counts on equivalent runs
when checking whether that working set is sufficient.
The numerical jobs use two compiler workers to contain peak memory use. Their
whole-job deadlines, 120 minutes for Debug and 90 minutes for Release, include
cold compilation and all selected tests. The per-case watchdog is
independent of those workflow deadlines.
`Testing/Temporary/LastTest.log` includes GoogleTest case timings even for
successful groups. Partial logs are uploaded on failure when the runner can
still execute cleanup steps; a hard job timeout can prevent the upload.

Integration GoogleTests are partitioned by suite within each category and
serial/MPI mode. Their union is checked against the compiled registry, and the
existing category coverage guard remains enabled. The executable is still built
once. The common output lock remains in place because generated parameter files
and fixtures still share output paths; removing it requires isolating those
paths first.

For example, a focused CTest expression can select one integration suite while
retaining its serial or MPI mode. Use `ctest -N` to obtain the exact names in the
current configuration before rerunning a failed group.

For application smoke tests and package checks, see
[Test a documented workflow](../how-to/testing).

## Retained cost audit

The Debug CI job on revision `c7c1256` took 56 min 47 s:

| Phase | Wall time |
| --- | ---: |
| Configuration and build | 29 min 05 s |
| CTest | 25 min 15 s |
| Setup, cache handling, and other steps | 2 min 27 s |

Harmonic fiber reinforcement, lifted elasticity, and lifted Poisson accounted
for 73% of CTest time, counting serial and two-rank execution. The Debug compiler
cache reported 38.33% hits and 112 cleanups at its former 1 GB limit. This
supports increasing the cache capacity; it does not attribute every miss to
eviction. Compiler command durations overlap, so their sum is not build wall
time.

The {download}`source audit <../../benchmarks/reports/gtest-cost-audit-2026-11-02.json>`
records decisions for all 115 original numerical GoogleTest source files,
including retained independent oracles and the surviving checks for consolidated
cases. It separately reviews the common driver and timing infrastructure.
The {download}`build audit <../../benchmarks/reports/gtest-build-costs-2026-11-02.json>`
retains compiler/cache evidence and the cross-checkout cache verification.
The {download}`case observations <../../benchmarks/reports/gtest-runtime-2026-11-02.jsonl>`
retain baseline and integrated candidate timings by configuration and rank count.

Local runs use GNU 11.4 and deal.II 9.8 on gpu02, with OpenMP and BLAS limited to
one thread. They are distinct from hosted CI timings. Baseline cases interrupted
at 120 seconds are lower bounds, and the two original Release serial AL parity
failures remain recorded. The removed fine-grid accuracy study and the omitted
tails of convergence and parameter sweeps are explicitly documented; small
correctness regressions do not establish the same accuracy or scaling coverage.

The integrated local matrix on `7438d953` passes all 164 CTest groups and 1,829
GoogleTest executions in Debug/Release, serial and two-rank MPI, with no skips or
timeouts. The largest recorded case takes 28.529 seconds. Summed case times
exclude process startup and CTest overhead:

| Configuration | Baseline case seconds | Candidate case seconds |
| --- | ---: | ---: |
| Debug, serial | ≥ 1,725.509 | 623.372 |
| Debug, MPI with two ranks | ≥ 1,255.765 | 403.465 |
| Release, serial | 359.231 | 168.825 |
| Release, MPI with two ranks | 253.927 | 112.609 |

These are separate runs with changed fixtures and coverage, including the
baseline failures and censored timings described above. They are an inventory
of observed costs, not a matched benchmark or a hosted-CI speedup measurement.
MPI case durations describe rank zero; successful launcher exits and the
per-rank watchdog also cover the other ranks. Optional MetricFlowX, Coral,
external FVM runtime, standalone benchmark, and deal.II-style application-test
configurations were not run in this local matrix.

```{toctree}
:maxdepth: 1
:titlesonly:

../how-to/testing
```
