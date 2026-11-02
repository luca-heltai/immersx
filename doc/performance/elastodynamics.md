# Elastodynamics performance

The baseline for the native and IDA execution paths is ImmersX revision
`6bd61abda38cd1c7908eea60623023f6d566ac56`. The
[retained report](../../benchmarks/reports/elastodynamics-baseline-2026-11-02.json)
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
