# ElasticStatic production slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Make \`ElasticStaticProblem\` a parameter-driven strong-BC static elasticity Problem with legacy-useful mesh, material, RHS, and output capabilities.

**Architecture:** Add a composed \`ElasticStaticParameters\` object and keep all mesh, FE, constraints, native operators, loads, accepted state, and output ownership inside \`ElasticStaticProblem\`. Reuse deal.II triangulation/grid/output facilities and the existing \`MaterialProperties\`/\`ModulatedParsedFunction\` concepts; keep adapter composition outside the Problem and do not introduce an elastic base class.

**Tech Stack:** C++17, deal.II 9.2+, MPI distributed and fullydistributed triangulations, ParameterAcceptor, GoogleTest, CTest, CMake-generated \`.prm\` fixtures, \`LinearAdapter\`.

**Spec:** \`docs/superpowers/specs/2026-08-25-elastic-static-production-design.md\`

## Global Constraints

- Do not implement weak Dirichlet conditions or nonlinear constitutive laws.
- Do not make \`ElasticStaticProblem\` own a solver, adapter, interaction, multiplier field, or global block number.
- Use generated build-tree inputs and outputs; never add cwd-dependent runtime paths or source-tree generated files.
- Use \`./scripts/indent\`, review \`git diff\`/\`git diff --check\`, and run focused validation before every commit.
- Do not push, merge, rebase onto, or modify the base branch.

---

### Task 1: Commit the capability baseline and design record

**Files:**
- Create: \`docs/superpowers/specs/2026-08-25-elastic-static-production-design.md\`
- Modify: \`doc/core-architecture.md\`

**Interfaces:**
- Produces the developer-facing feature matrix and the approved ownership/design constraints used by all later tasks.

- [x] **Step 1: Record the pre-change matrix and design**

The design file contains the five-class matrix, parameter/mesh/material/assembly design, testing gates, and explicit non-goals. \`doc/core-architecture.md\` links to it from the merged implementation-status section.

- [x] **Step 2: Verify documentation changes**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only the two intended documentation paths are changed or untracked.

- [ ] **Step 3: Commit**

```bash
./scripts/indent
git diff
git diff --check
git add doc/core-architecture.md docs/superpowers/specs/2026-08-25-elastic-static-production-design.md
git commit -m "docs: record post-115 elasticity capability baseline"
```

### Task 2: Define the parameterized ElasticStatic public API and red tests

**Files:**
- Modify: \`include/immersx/physics/elastic_static.h\`
- Create: \`include/immersx/physics/elastic_static_parameters.h\` only if the parameter declarations cannot remain focused in the main public header
- Modify: \`gtests/elastic_static_problem_01.cc\`
- Modify: \`gtests/CMakeLists.txt\` only if a separate translation unit is required

**Interfaces:**
- Produces \`ElasticStaticParameters<dim, spacedim>\` with a configurable subsection, \`fe_degree\`, \`initial_refinement\`, \`domain_type\`, \`name_of_grid\`, \`arguments_for_grid\`, \`grid_scale\`, \`triangulation_type\`, \`dirichlet_ids\`, \`neumann_ids\`, \`output_directory\`, \`output_name\`, parsed vector \`rhs\`, \`bc\`, and \`neumann_bc\`, default/material-id material configuration, and accessors for the selected material and boundary functions.
- Produces \`ElasticStaticProblem(const ElasticStaticParameters<dim, spacedim> &, MPI_Comm)\` and \`setup()\`; the old hard-coded constructor is migrated from in-tree callers rather than preserved as a compatibility layer.

- [ ] **Step 1: Write the failing parameter-registration tests**

Add tests that clear \`ParameterAcceptor\`, construct two parameter objects under \`/Left/Elastic static/\` and \`/Right/Elastic static/\`, parse distinct FE degree, grid, boundary-id, and parsed vector entries, and assert the resulting public members/functions are distinct and parsed. Add one test that parses a material-id tag mapping and asserts \`get_material_properties(1)\` returns the configured Lamé values.

- [ ] **Step 2: Run the focused test to verify the expected failure**

Configure a Debug build if needed, then run:

```bash
build-debug/gtests/gtests_debug --gtest_filter='ElasticStaticProblem.Parameter*'
```

Expected: compilation failure because \`ElasticStaticParameters\` and its accessors do not exist yet. If the build tree is absent, configure it using the repository’s Debug command before rerunning.

- [ ] **Step 3: Implement only the parameter declarations and parser**

Add normalized subsection handling, \`ParameterAcceptor\` entries matching the spec, \`ModulatedParsedFunction<spacedim>\` members for RHS/Dirichlet/Neumann data, and the two-pass material override creation using configurable material roots. Update \`MaterialProperties\` only as needed to accept a caller-provided root subsection while retaining its existing legacy default root.

- [ ] **Step 4: Run the focused tests to verify green**

```bash
build-debug/gtests/gtests_debug --gtest_filter='ElasticStaticProblem.Parameter*'
```

Expected: all new parameter tests pass with no parser warnings or subsection collisions.

- [ ] **Step 5: Commit**

```bash
./scripts/indent
git diff
git diff --check
git add include/immersx/physics/elastic_static.h include/immersx/physics/elastic_static_parameters.h gtests/elastic_static_problem_01.cc gtests/CMakeLists.txt
git commit -m "feat: add parameterized static elasticity configuration"
```

### Task 3: Add parsed strong BC, RHS, Neumann, and material-id assembly

**Files:**
- Modify: \`include/immersx/physics/elastic_static.h\`
- Create: \`source/elastic_static.cc\`
- Modify: \`gtests/elastic_static_problem_01.cc\`

**Interfaces:**
- \`ElasticStaticProblem::stiffness_operator()\`, \`forcing()\`, \`constraints()\`, \`dof_handler()\`, \`triangulation()\`, \`locally_owned_dofs()\`, and \`locally_relevant_dofs()\` remain read-only public native seams.
- \`ElasticStaticProblem::set_forcing(const dealii::Function<spacedim> &)\` remains a programmatic convenience that reassembles the load without erasing affine Dirichlet corrections.
- Assembly evaluates the configured material using \`cell->material_id()\`, body force using the parsed RHS, traction on configured boundary ids using the parsed Neumann function, and strong Dirichlet values using the parsed vector BC.

- [ ] **Step 1: Write the failing assembly/oracle tests**

Add independent tests for:

```cpp
EXPECT_NEAR(problem.constraints().get_inhomogeneity(boundary_dof), 1.0, tol);
EXPECT_GT(problem.forcing().l2_norm(), 0.0);
EXPECT_LT((problem.stiffness_operator() * solution - problem.forcing()).l2_norm(), tol);
```

Use a constant parsed body force and nonzero parsed boundary function for the first test, a constant parsed traction on a boundary id for the second, and a two-cell material-id mesh with differing Lamé values for the material test. Keep the residual calculation independent of \`LinearAdapter\`.

- [ ] **Step 2: Run the tests to verify the expected failure**

```bash
build-debug/gtests/gtests_debug --gtest_filter='ElasticStaticProblem.Parsed*:*ElasticStaticProblem.Material*'
```

Expected: compile or assertion failure because the current implementation has no parameter-driven assembly or material lookup.

- [ ] **Step 3: Implement Problem-owned mesh/FE/constraint/assembly state**

Move the templated implementation into \`source/elastic_static.cc\` where practical, add explicit 2D and 3D instantiations, and assemble stiffness plus all load terms in one constrained local-to-global pass. Preserve nonzero Dirichlet affine corrections by never resetting the accumulated load after constrained stiffness assembly.

- [ ] **Step 4: Run serial assembly tests**

```bash
build-debug/gtests/gtests_debug --gtest_filter='ElasticStaticProblem.Parsed*:*ElasticStaticProblem.Material*'
```

Expected: parsed data and the independent \`K u - f\` oracle pass for the serial distributed backend.

- [ ] **Step 5: Commit**

```bash
./scripts/indent
git diff
git diff --check
git add include/immersx/physics/elastic_static.h source/elastic_static.cc gtests/elastic_static_problem_01.cc
git commit -m "feat: assemble parameterized static elasticity loads"
```

### Task 4: Implement both triangulation backends and output

**Files:**
- Modify: \`include/immersx/physics/elastic_static.h\`
- Modify: \`source/elastic_static.cc\`
- Modify: \`gtests/elastic_static_problem_01.cc\`
- Create: \`gtests/parameters/elastic_static_problem.prm.in\`
- Modify: \`apps/app_coupled_poisson_elasticity.cc\`
- Modify: \`apps/coupled_poisson_elasticity.h\` only where constructor/FE access changes require it

**Interfaces:**
- \`ElasticStaticProblem\` owns a deal.II triangulation variant and exposes \`triangulation()\` as \`parallel::TriangulationBase<dim, spacedim> const &\`.
- \`ElasticStaticProblem::output_results() const\` writes configured displacement/material-id output without changing solver state.
- The existing coupled Poisson–elasticity application constructs both Problems from parameter objects and has no post-parse configuration overrides.

- [ ] **Step 1: Write backend and output red tests**

Add a parameterized test suite instantiated for \`distributed\` and \`fullydistributed\`; each test parses a generated two-cell mesh, calls \`setup()\`, asserts \`n_active_cells() > 0\`, assembles a finite stiffness operator, solves through \`LinearAdapter\`, and checks the independent residual oracle. Add an output test that checks only build-tree output existence through \`TestPaths::output_directory\`.

- [ ] **Step 2: Run serial and MPI tests to verify the expected failure**

```bash
build-debug/gtests/gtests_debug --gtest_filter='ElasticStaticProblem.Triangulation*'
mpirun -np 2 build-debug/gtests/gtests_debug --gtest_filter='ElasticStaticProblem.MPI_*'
```

Expected: the fullydistributed parameter path and output API are absent or fail before implementation.

- [ ] **Step 3: Implement variant mesh creation**

Follow the existing Poisson/Elastodynamics variant pattern. Generated grids use deal.II’s name-and-arguments helper; file grids use \`GridIn\`; scale precedes refinement; fullydistributed construction uses a serial triangulation and \`copy_triangulation\`. Avoid a new ImmersX mesh abstraction.

- [ ] **Step 4: Implement output and migrate the coupled application**

Add \`DataOut\` output for the displacement field and material ids. Give the coupled application \`PoissonParameters\` and \`ElasticStaticParameters\` configurable subsections, call \`initialize_parameters()\` before setup, remove assignments to parsed fields after initialization, and retain the adapter-based coupling flow.

- [ ] **Step 5: Run backend and application tests**

```bash
build-debug/gtests/gtests_debug --gtest_filter='ElasticStaticProblem.*'
mpirun -np 2 build-debug/gtests/gtests_debug --gtest_filter='ElasticStaticProblem.MPI_*'
ctest --test-dir build-debug -V -R 'coupled_poisson_elasticity|ElasticStatic'
```

Expected: serial and two-rank backend tests pass and the application consumes the generated \`.prm\` from the build tree.

- [ ] **Step 6: Commit**

```bash
./scripts/indent
git diff
git diff --check
git add include/immersx/physics/elastic_static.h source/elastic_static.cc gtests/elastic_static_problem_01.cc gtests/parameters/elastic_static_problem.prm.in apps/app_coupled_poisson_elasticity.cc apps/coupled_poisson_elasticity.h
git commit -m "feat: support distributed static elasticity applications"
```

### Task 5: Add the standalone parameter-driven application and final validation

**Files:**
- Create: \`apps/app_elastic_static.cc\`
- Create: \`gtests/parameters/elastic_static.prm.in\`
- Modify: \`gtests/app_executables_test.cc\` if application smoke coverage is registered there
- Modify: \`doc/tutorials/elasticity.md\` or add a focused developer/tutorial page and include it from the relevant toctree

**Interfaces:**
- The application reads dimensions and a generated parameter file, constructs \`ElasticStaticParameters\`, initializes all parameters once, adds the Problem to \`LinearAdapter\`, solves, accepts the state, writes output, and reports the independent residual norm.

- [ ] **Step 1: Write the failing executable/fixture test**

Add a generated \`.prm.in\` with nonzero parsed RHS, parsed Dirichlet data, explicit boundary ids, material defaults, distributed backend, output directory under \`TEST_OUTPUT_DIR\`, and a small refinement. Add an application test that launches the binary from \`/tmp\` with the generated parameter path and checks the diagnostics/output file.

- [ ] **Step 2: Run the test to verify the expected failure**

```bash
ctest --test-dir build-debug -V -R 'ElasticStatic|app_elastic_static'
```

Expected: the executable target or parameterized run is missing before implementation.

- [ ] **Step 3: Implement the application**

Use the existing \`app_*.cc\` and \`LinearAdapter\` conventions. Do not overwrite parsed values after \`initialize_parameters()\`. Keep all generated output below the configured build-tree output directory.

- [ ] **Step 4: Run final validation**

```bash
export CCACHE_DIR="$HOME/.ccache"
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug -DENABLE_GOOGLE_TESTING=ON -DENABLE_DEAL_II_APP_TESTING=ON -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build build-debug -j
build-debug/gtests/gtests_debug
mpirun -np 2 build-debug/gtests/gtests_debug
ctest --test-dir build-debug --output-on-failure
(cd /tmp && /Users/heltai/c++/immersx-task-1/build-debug/gtests/gtests_debug --gtest_filter='ElasticStaticProblem.*')
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DENABLE_GOOGLE_TESTING=ON -DENABLE_DEAL_II_APP_TESTING=ON -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build build-release -j
```

Record any unavailable optional VTK/CTest target explicitly rather than claiming it passed.

- [ ] **Step 5: Commit**

```bash
./scripts/indent
git diff
git diff --check
git add apps/app_elastic_static.cc gtests/parameters/elastic_static.prm.in gtests/app_executables_test.cc doc/tutorials/elasticity.md doc/tutorials/index.md
git commit -m "feat: add parameter-driven static elasticity application"
```
