# AGENTS.md

This file contains repository-specific instructions for coding agents working on
ImmersX.

The detailed contributor guide is
[`doc/developer/contributing.md`](doc/developer/contributing.md).
Read both files before making changes. When the two differ, this file is
normative for agent behavior.

## 1. Before starting work

Before editing the repository:

1. Inspect the current repository state:
   ```bash
   git status
   git branch --show-current
   git rev-parse HEAD
   git worktree list
   ```
2. Fetch and inspect the current remote `master`. Do not assume a previously
   observed SHA is still current.
3. Read this file and `doc/developer/contributing.md`.
4. Inspect the relevant implementation, tests, and documentation before
   proposing a new abstraction.
5. Preserve unrelated local changes. Never reset, clean, stash, overwrite, or
   discard user work unless explicitly requested.

For substantial refactors, first identify the smallest application or test that
can serve as a vertical correctness gate.

## 2. General design principles

ImmersX is a deal.II-based application/library. Prefer deal.II concepts and
facilities over introducing equivalent ImmersX abstractions.

In particular:

- Reuse deal.II data structures, algorithms, solver interfaces, and algebraic
  abstractions whenever they already express the required concept.
- Do not create wrappers whose only purpose is to rename or type-erase an
  existing deal.II abstraction.
- Prefer `dealii::LinearOperator`, `dealii::BlockLinearOperator`,
  `dealii::PackagedOperation`, and related deal.II facilities over local
  equivalents when applicable.
- Introduce ImmersX-specific abstractions only for genuinely ImmersX-specific
  semantics.
- Prefer small composable value types, free functions, builders, and callbacks
  over inheritance.
- Do not introduce base classes or virtual interfaces without a concrete need.
- Avoid speculative infrastructure. Add abstractions only when a real
  application or test exercises them.
- Prefer vertical, application-driven changes over broad horizontal
  generalization.

When a deal.II facility might already solve the problem, inspect the installed
deal.II headers/API before implementing a local substitute.

## 3. Architecture

Use the following ownership boundaries when designing or reviewing code.

### Problem

A Problem owns physical and discretization-specific information, including as
appropriate:

- mesh and DoFHandler;
- finite element and mapping;
- material and physical parameters;
- native matrices/operators;
- forcing and boundary data;
- accepted physical state.

A Problem should not know global coupled-system block numbers.

### Field

A Field is a semantic identity for state and residual rows.

A Field is not:

- a global matrix block number;
- a native Problem block number;
- an execution-adapter block number.

Keep semantic Field identity separate from execution storage.

### FE expression / Observable

A primitive FE expression is a small typed `Observable<Field, Operation>`
derived from a Field. It delegates FE value, gradient, and first-order shape
function operations to deal.II's `FEValuesViews` rather than rebuilding FE
algebra locally.

An Observable is a typed physical quantity derived from one or more Fields. It
may compose pointwise operations or a backend lift, but it does not introduce
another hierarchy or own the physical coupling relation between Problems.

`Representation` is no longer a current architectural concept. New extension
points must use the Field, Observable, WeakTerm, and Constraint/Interaction
boundaries described here.

Lifting, point search, retained stencils, and redistribution are execution or
backend details. They should be hidden behind the Observable or Interaction
that needs them rather than exposed as a second architectural concept.

### WeakTerm

A WeakTerm owns the residual and linearization contribution of a variational
expression. Linear weak terms may use cached deal.II operators; nonlinear terms
must evaluate their current state and provide the corresponding derivative.

### Constraint / Interaction

A Constraint or Interaction owns terms that exist because two or more systems
are related.

An Interaction may:

- contribute to residual rows owned by Problems;
- introduce auxiliary Fields, such as Lagrange multipliers;
- own coupling/search/transfer operators and associated geometry state.

### Execution adapter

An execution adapter owns solver and time-integration policy.

Examples include IDA and, potentially, ARKode, KINSOL, or a linear execution
path.

Execution adapters may internally own:

- semantic composition state;
- execution block layouts;
- global vectors;
- differential/algebraic masks;
- solver callbacks.

These implementation details should normally not appear in application-level
code.

Prefer application syntax where Problems and Interactions are added directly to
a single execution adapter.

Do not introduce a separate user-visible `CoupledSystem` merely to hold this
information unless a concrete future use case requires it.

## 4. Semidiscrete equations

Use the canonical residual form

```text
F(t, y, ydot) = 0.
```

When exposing linearization information, keep

```text
dF/dy
```

and

```text
dF/dydot
```

separate.

Solver-specific combinations belong to the execution adapter. For example, IDA
constructs

```text
J = dF/dy + alpha * dF/dydot.
```

A Problem or Interaction contributor should not know IDA's `alpha`.

Residual/Jacobian evaluation at a candidate state does not mean that state has
been accepted. Do not update accepted history or previous-solution state merely
because a nonlinear/time integrator evaluates a residual.

## 5. API stability

Interfaces introduced after PR #69 should currently be considered fluid.

Do not retain, deprecate, wrap, or duplicate a post-PR-69 interface merely for
backward compatibility. If a cleaner design requires changing or removing such
an interface, do so and migrate its in-repository users.

Prefer the smallest clean public API over compatibility with intermediate
post-PR-69 designs.

This rule should be revisited once the new composition/execution API is declared
stable.

## 6. Parallel and distributed code

Do not assume reduced-dimensional or zero-dimensional objects are inherently
serial.

For distributed geometric ownership/search, use the relevant deal.II
distributed facilities. In particular, point-cloud style data associated with a
background distributed mesh should use `Particles::ParticleHandler` ownership
and search facilities where appropriate rather than ad-hoc serial ownership.

Keep native Problem block numbering local to the Problem. Distributed execution
block numbering belongs to the execution adapter.

## 7. Git workflow and commits

During implementation, create small, meaningful intermediate commits at logical
milestones unless the task explicitly requests otherwise.

A good intermediate commit should:

- represent one coherent change;
- build or pass the relevant focused tests where practical;
- avoid mixing unrelated cleanup;
- have a descriptive commit message.

Before **every commit**, without exception:

1. Run from the repository root:
   ```bash
   ./scripts/indent
   ```
2. Review the complete resulting diff:
   ```bash
   git diff
   git diff --check
   ```
3. Confirm that no unrelated user changes or generated files are included.
4. Run the focused validation appropriate for that commit.

`./scripts/indent` formats the repository globally. Review formatter-induced
changes carefully before committing.

Agents may create local branches and local commits as part of implementation.

Do **not**:

- push branches;
- push commits;
- open or update pull requests;
- merge into `master`;
- force-update remote refs;

unless explicitly requested by the user.

For parallel work, prefer separate git worktrees and separate branches. Do not
have multiple workers modify the same branch or share a build directory.

Integrate parallel work through meaningful commits and cherry-picks rather than
copying untracked source trees manually.

## 8. Local development resources

The following local resources may be used when available:

```text
/Users/heltai/c++/immersx-dev-1
/Users/heltai/c++/immersx-dev-2
/Users/heltai/c++/immersx-dev-3
```

These are git worktrees of the ImmersX repository and may be used for parallel
branches.

Agents also have write permission on:

```text
~/.ccache
```

Use a shared ccache if useful, but never share CMake build directories between
worktrees.

Before using a worktree, inspect its current branch, HEAD, and local changes.
Do not overwrite unrelated work.

## 9. Build configuration

Always use an out-of-source CMake build.

Typical Debug configuration:

```bash
cmake -S . -B build-debug \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug -j
```

Set `DEAL_II_DIR` when required by the local installation.

Use ccache when available:

```bash
export CCACHE_DIR="$HOME/.ccache"
cmake -S . -B build-debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
```

Do not place generated build products in the source tree.

Each parallel worktree must have its own build directory. Sharing the ccache is
fine; sharing a CMake build directory is not.

## 10. Test inputs and generated files

The top-level CMake configuration preprocesses every `.in` file below
`gtests/`, `tests/`, and `data/`.

The generated file is placed in the corresponding location below the build tree
with only the final `.in` suffix removed.

This applies to, among others:

- `.prm.in`;
- `.txt.in`;
- `.vtk.in`;
- `.vtu.in`;
- `.pvtu.in`.

Templates may use:

```text
@IMMERSX_SOURCE_DIR@
@IMMERSX_BINARY_DIR@
@TEST_DATA_DIR@
@TEST_OUTPUT_DIR@
```

CMake expands them with `configure_file(... @ONLY)`.

Runtime code and tests must consume the generated build-tree files.

Do not introduce:

- `../data/...` runtime paths;
- cwd-dependent paths;
- source-tree output paths;
- build-to-source symlink workarounds;
- generated `.prm`, VTK, logs, or result files in the source tree.

GoogleTests should use `gtests/test_paths.h` for test data, generated parameter
files, and output directories.

## 11. GoogleTest conventions

CTest is the authoritative orchestration layer. Tests have one primary label
from `unit`, `integration`, `validation`, or `application`. The `quick` label
is applied to all unit and integration tests, and `mpi` is orthogonal. Pull
requests run `ctest -L quick`; pushes to `master` run the full CTest manifest.
CI must not rerun the aggregate GoogleTest executable outside CTest.

A GoogleTest intended for MPI execution must contain `MPI_` in the test name.

The common test driver uses this naming convention to select serial versus MPI
tests.

Examples:

```cpp
TEST(MyFeature, SerialBehavior)
```

and

```cpp
TEST(MyFeature, MPI_DistributedBehavior)
```

Tests must not depend on their working directory.

When relevant, validate the GoogleTest executable from:

1. the build directory;
2. the repository root;
3. an unrelated directory.

At minimum, substantial distributed changes should exercise both:

```bash
/path/to/build-debug/gtests/gtests_debug
```

and

```bash
mpirun -np 2 /path/to/build-debug/gtests/gtests_debug
```

Use focused `--gtest_filter` runs while iterating, then run the relevant broader
suite before finishing.

## 12. deal.II-style regression tests

The `tests/` tree uses the deal.II testsuite infrastructure.

Follow `doc/contributing.md` for source-driven and parameter-driven regression
tests.

Use generated build-tree input files and deterministic output.

Run relevant regression tests with `ctest`, for example:

```bash
ctest --test-dir build-debug -V -R '<regex>'
```

## 13. Validation expectations

Use the smallest useful test while iterating, but finish a substantial change
with validation appropriate to its scope.

For core, distributed, solver, packaging, or cross-cutting changes, normally
include:

1. focused unit/integration tests;
2. Debug build;
3. serial GoogleTests;
4. two-rank MPI GoogleTests;
5. relevant `ctest` tests;
6. Release build/tests when the change affects templates, configuration,
   packaging, or portable C++ behavior;
7. working-directory robustness checks when files or runtime paths are involved.

Do not weaken or delete an independent correctness oracle merely because a new
execution path duplicates the same computation.

When possible, separate:

- physics/discretization correctness;
- execution-adapter correctness;
- solver convergence/performance.

## 14. CMake and installation

Keep CMake targets relocatable and package-friendly.

Do not encode source-tree or build-tree assumptions into installed targets.

Public headers belong under:

```text
include/immersx/
```

and public API belongs in namespace:

```cpp
namespace ImmersX
{
  // ...
}
```

Installation rules should place:

- headers;
- libraries;
- executables;
- data/resources intended for installation;

in standard CMake/GNUInstallDirs-style locations.

Tests and development-only fixtures should not accidentally become runtime
dependencies of installed applications.

## 15. Documentation

Documentation sources live under `doc/`.

When adding a new documentation page, add it to the `doc/index.md` toctree.

Document the normal user-facing API first. Avoid teaching internal plumbing as
the primary workflow when a higher-level API exists.

For reusable extension points, document both:

1. ordinary application usage;
2. how to implement a new Problem, Field-based FE expression/Observable,
   WeakTerm, or Constraint/Interaction.

Do not document sealed legacy internals as recommended extension points. New
extension APIs must not reintroduce the obsolete parallel hierarchy.

Keep documented commands consistent with the actual CMake and test workflow.

## 16. Scope discipline

Do not opportunistically generalize unrelated parts of the repository.

If a task reveals a broader architectural issue:

- fix it when necessary for the requested vertical slice;
- otherwise record it separately rather than expanding the current change.

Avoid speculative support for future solvers, geometries, or discretizations
unless the current application requires it.

Prefer deleting obsolete post-PR-69 infrastructure over retaining multiple
parallel ways to do the same thing.

## 17. Final review

Before reporting a substantial task complete:

1. Inspect:
   ```bash
   git status
   git log --oneline --decorate -n 10
   ```
2. Run the relevant final tests.
3. Run:
   ```bash
   git diff --check
   ```
4. Ensure every intermediate commit is meaningful and reviewable.
5. Confirm that generated outputs are outside the source tree.
6. Summarize:
   - branches/worktrees used;
   - commits created;
   - files changed;
   - tests run and their results;
   - known limitations or follow-up work.
7. Confirm that nothing was pushed and no pull request was opened unless the
   user explicitly requested it.
