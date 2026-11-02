# Elastodynamics performance

The baseline for the native and IDA execution paths is ImmersX revision
`6bd61abda38cd1c7908eea60623023f6d566ac56`. The
{download}`retained report <../../benchmarks/reports/elastodynamics-baseline-2026-11-02.json>`
records the source revision, binary and input checksums, timers, and iteration
counts. Its measurements describe that revision, not the commit that adds the
report or later optimizations.

## Baseline problem

The domain is the square $[0,0.1]^2$, with Q1 vector elements, density 1000,
Lamé coefficients $\mu=1000$ and $\lambda=0$, and viscosity $\eta=25$.
The displacement and velocity boundary data are zero on every side. Initial
displacement is $(\sin(10\pi x)\sin(10\pi y),0)$, initial velocity and applied
loads are zero. There are no inclusions or multipliers.

The simulation ends at $t=0.02$, with output every 0.0025. Elasticity uses
Newmark with $\beta=1/4$, $\gamma=1/2$; native Elastodynamics uses the
equivalent trapezoidal rule, both with eight steps of 0.0025. IDA uses adaptive
BDF with absolute and relative tolerances $10^{-4}$ and initial step $10^{-5}$.
The two IDA variants differ only in maximum BDF order: the default of one,
or five. These are different integrators; their timings do not demonstrate
equal temporal accuracy.

Release binaries were built on gpu04 and run sequentially on gpu02, an AMD EPYC
7763 node, with one MPI rank and OMP/OpenBLAS/MKL thread limits of one. No
concurrent build ran on the measurement node. This is a serial comparison,
not a scaling study or a measurement of GPU acceleration.

## Baseline measurements

With 16,384 quadrilateral cells and 33,282 spatial DoFs:

| Execution | Internal steps | Linear solves | Time in linear solves | Application timer |
|---|---:|---:|---:|---:|
| Elasticity, Newmark | 8 | 8 | 0.944 s | 2.40 s |
| Native Elastodynamics, trapezoidal | 8 | 8 | 6.08 s | 8.26 s |
| IDA, maximum BDF order 1 | 266 | 282 | 21.5 s | 31.1 s |
| IDA, maximum BDF order 5 | 52 | 68 | 7.64 s | 12.8 s |

There is one measurement per variant on this mesh. All variants write nine
outputs; Elasticity also computes stresses. Application timers include output
and exclude part of process/MPI startup. IDA's adapter timers are nested inside
the Problem timer: do not add `Solve DAE` to its component timers, or add
`Assemble body force` a second time to residual evaluation.

On the preceding mesh, with 4,096 cells and 8,450 spatial DoFs, three runs per
variant give these ranges:

| Phase | Baseline time |
|---|---:|
| Eight Newmark solves | 0.141–0.146 s |
| Eight native trapezoidal solves | 0.575–0.583 s |
| Native system assembly, eight calls | 0.226–0.229 s |
| Newmark matrix assembly, one call | 0.0260–0.0271 s |
| IDA order 1 linear solves, 276 calls | 4.36–4.53 s |
| IDA maximum order 5 linear solves, 64 calls | 1.26–1.30 s |

The exported native and Newmark displacement fields agree to the precision of
Float32 output on all four measured meshes. This does not bound their internal
double-precision difference. On the largest mesh, IDA order 1 and maximum order
5 differ from Newmark by about 0.295% and 0.205% in relative nodal displacement
norm. Newmark is not an exact temporal reference.

## Reproducing a comparison

Build `elasticity`, `elastodynamics`, and `elastodynamics_adapter` in Release
with the same toolchain and dependency installation for each revision. Retain
the baseline build separately, including its shared ImmersX library: replacing
that library would change the old executables' behavior.

Run on an allocated compute node:

```bash
python3 scripts/benchmark_elastodynamics.py \
  --build /absolute/path/to/release-build \
  --source-revision COMPILED_REVISION \
  --output /absolute/path/to/new-results \
  --refinements 6 7 --repetitions 3
```

The output directory must not exist. The script keeps every input and log,
records binary and library checksums, and writes `measurements.json` after each
case. Supply the revision actually compiled; the script deliberately does not
infer binary provenance from the checkout's current HEAD. Runtime dependency
paths, if needed, must be supplied in the environment. Compare phase timers and
iteration counts as well as total time; report numerical differences whenever
the time integrator or tolerances change.

## Reuse and block factorization

Spatial setup invalidates the Problem's caches. Between spatial setups,
`assemble_operators()` retains the mass, stiffness, and damping matrices.
The raw and homogeneously condensed operators use the same cell matrices
from a single quadrature pass.
Volumetric loads and tractions have separate caches: a static contribution is
retained, while a time-dependent contribution is evaluated at each new requested
time. Time dependence is detected using the configured time variable name,
including names other than `t`. Hanging-node topology is retained; only boundary
data that depend on time require new interpolation. The initial velocity and
the velocity derived by the native integrator remain distinct from the
adapter's explicitly prescribed velocity boundary function.

The native time matrix is an algebraic combination of the retained spatial
matrices. It is reused while the step and integration rule are unchanged.
The right-hand side uses matrix-vector products and affine lifting, including
nonzero Dirichlet data and hanging-node constraints. Changing the step or IDA's
derivative weight requires a new algebraic combination, not new spatial
quadrature. `assembly_statistics()` exposes monotone counters for spatial
assembly, load updates, constraint updates, and native time-matrix construction.

On free variables, the IDA Jacobian is

$$
J=\begin{pmatrix}\alpha M&-M\\K&D+\alpha M\end{pmatrix},
\qquad S=D+\alpha M+\alpha^{-1}K.
$$

Here the exact Schur complement remains sparse because the upper-right block
is proportional to mass. The more general native time matrix is

$$
A=\begin{pmatrix}aM&-bM\\cK&aM+dD\end{pmatrix},
\qquad S=aM+dD+\frac{bc}{a}K,
$$

with $a=1/h$ and $b=c=d=1$ for backward Euler, or $b=c=d=1/2$ for
trapezoidal integration. The preconditioner applies a complete block
factorization:

$$
w_u\approx(aM)^{-1}r_u,\qquad
w_v\approx S^{-1}(r_v-cKw_u),\qquad
z_u=w_u+\frac{b}{a}w_v,\quad z_v=w_v.
$$

Both inner solves use CG with AMG and a relative residual reduction of $10^{-2}$.
That inner tolerance is not the requested accuracy of the physical solution:
the outer FGMRES retains its configured stopping criterion. FGMRES uses right
preconditioning and permits variable inner solves. Symmetric constraint
projection keeps the inner matrices SPD; the exact outer system retains its
constraint treatment. Schur AMG receives rigid-body modes on full-dimensional
vector spaces and component-wise constant modes for embedded spaces. Mass AMG
uses its default scalar seed: the Cartesian mass graph already separates the
components, so rotational seeds would enlarge its coarse spaces unnecessarily.
The mass hierarchy is retained; the Schur hierarchy changes with its
coefficients. The [ML user guide](https://trilinos.github.io/pdfs/mlguide3.pdf)
discusses nullspace information for smoothed aggregation.

This choice follows the block-factorization and nested-solve approach discussed
by [Liu and Marsden (2019)](https://arxiv.org/abs/1809.06350) and the general
Schur-complement treatment in
[Wathen (2015)](https://people.maths.ox.ac.uk/wathen/preconditioning.pdf).
[Saad (1993)](https://www-users.cse.umn.edu/~saad/PDF/umsi-91-279.pdf)
provides the flexible outer iteration. The sparse physical Schur above follows
directly from this linear system; it does not require an approximation to
$M^{-1}M$. This is a structure-based choice, not a claim that one AMG setup or
inner tolerance is optimal for every mesh, material contrast, or coupled model.

## Measured changes

The measured implementation is revision
`930f0c4a954b772c1edb97cfce249e8d25db5129`. The
{download}`optimization report <../../benchmarks/reports/elastodynamics-optimization-2026-11-02.json>`
retains medians, ranges, binary checksums, original-binary replay, field
comparisons, and validation provenance. Source references for both measured
revisions are retained separately from later documentation changes.

For the same 33,282-DoF problem on gpu02, optimized times are medians of
three runs. Integrator settings, tolerances, output intervals, accepted step
counts, and numbers of linear solves are unchanged within each row:

| Execution | Original application time | Optimized median (range) | Speedup |
|---|---:|---:|---:|
| Elasticity control | 2.40 s | 2.43 s (2.40–2.46) | approximately 1 |
| Native trapezoidal | 8.26 s | 2.57 s (2.57–2.68) | 3.21 |
| IDA, maximum BDF order 1 | 31.1 s | 20.1 s (19.9–20.2) | 1.55 |
| IDA, maximum BDF order 5 | 12.8 s | 6.57 s (6.55–6.70) | 1.95 |

The original binaries and shared library still match their retained checksums.
A fresh original-binary replay gives 2.42 s for Elasticity, 7.79 s for native
Elastodynamics, and 30.7 s for IDA order 1, consistent with the original scale of
the comparison. The report retains the fourth replay as well. The baseline
largest-mesh timings have fewer samples than the optimized medians; these
measurements do not estimate long-term timing variance.

The eight native linear solves fall from 6.08 s to a median 1.25 s, including
preconditioner setup. Outer iterations fall from 269–296 to 5–9 per solve;
each new outer iteration includes inexact CG/AMG solves. Spatial assembly is
performed once, the native time matrix is built once, and the static load is
prepared once. IDA order 1 still takes 266 steps and 282 linear solves; its
remaining runtime includes that additional integration work. Its force-update
count falls from 284 to one.

The first Schur implementation, revision `0fd9473`, took 3.06 s, 23.2 s, and
7.47 s in single runs of the three elastodynamics variants. Building raw and
condensed matrices in one cell pass and reserving rigid modes for Schur AMG
improved those observations to the final ranges above. This comparison guided
the implementation choice; it is not an exhaustive AMG parameter search.

Across both measured meshes and all three optimized repetitions, the maximum
absolute difference from the original exported displacement or velocity is
$2.99\times10^{-8}$. Relative nodal differences are at most
$3.24\times10^{-9}$ for displacement and $6.35\times10^{-11}$ for velocity.
These comparisons use Float32 output and do not bound internal double-precision
differences. Independent manufactured-solution tests check native backward
Euler and trapezoidal lifting with nonzero moving boundaries and hanging nodes.

The focused CTest executables include the affected physics and execution-core
tests. All 12 unit/integration/validation groups pass in Debug and Release,
serially and with two MPI ranks, including simplex, embedded dimensions, and
fiber-coupled cases. The new public header also compiles independently in both
variants. Shared ccache contents were verified across gpu02/gpu04; representative
Release and Debug compilations in two distinct source/build trees each give a
direct cache hit with stable source, build, macro, and debug prefix mappings.
