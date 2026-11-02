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
