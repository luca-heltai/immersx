# Issue 203: constrained elastodynamics investigation

Date: 2026-09-17
Committer: lheltai
Pull request: [PR 205](https://github.com/luca-heltai/immersx/pull/205)
Issue: [#203](https://github.com/luca-heltai/immersx/issues/203)

## Reproduction

The case uses the v0.3 release assets `BG001_back_actuator.inp` and
`BG001_vessels.vtk`, a distributed three-dimensional mesh, and:

```bash
mpirun -np 16 --map-by :OVERSUBSCRIBE ./elasticity_debug \
  /tmp/prm_brain_no_ext_wave.prm
```

The run takes about 21--23 minutes on `lnx5` with 16 MPI ranks.

## Experiments

| Change | Result |
| --- | --- |
| FGMRES instead of CG | Growth remained. The solver choice alone was not the cause. |
| Zero Dirichlet data | Growth remained. Boundary-value accumulation was not the cause. |
| Absolute displacement tolerance `1.e-12`, zero reduction | Growth remained. Stopping accuracy was not the root cause. |
| Distribute velocity after the Newmark corrector | No measurable change. |
| Reduced right-hand side `0; 0` | The solution stayed identically zero. The reduced coupling force excites the unstable state. |
| Compute `B^T lambda` but omit its injection | The solution stayed identically zero. The force-injection path reproduces the failure. |
| Reverse the sign of `B^T lambda` | The same growth occurred. The sign alone is not the cause. |
| Remove rigid-body modes from constrained Newmark AMG | The residual improved, but physical growth remained. |
| Modified Gram--Schmidt in FGMRES | The true residual reached about `1.e-13`, but physical growth remained. |

## Decisive observation

Before the row fix, the final step had approximately:

```text
||a||_2             = 1.38866e+08
||a_constrained||_2 = 4.32064e+02
||residual||_2      = 4.57797e-13
u max               = 7.15326
```

The residual was small for a vector whose constrained acceleration entries
were not zero. The code overwrote only constrained diagonal entries with one;
it did not clear the rest of those rows. The later
`acceleration_constraints.distribute(a)` therefore changed the converged
vector and corrupted the next Newmark state.

The fix uses:

```cpp
matrix.clear_row(dof, 1.);
```

for the dynamic mass, damping, Newmark, and stiffness matrices. This clears a
constrained row, sets its diagonal to one, and retains constrained columns.
Retaining those columns is intentional because time-dependent boundary data
must still contribute through the right-hand side.

## Solver follow-up

After the row fix is confirmed, compare CG and FGMRES on the same corrected
matrix and parameter file. Record convergence status, actual residual,
constrained acceleration norm before and after distribution, iterations,
final displacement, and whether the 16-rank run remains bounded.

CG is appropriate only for the reduced free-free operator when it is symmetric
positive definite. The full row-retained operator is generally nonsymmetric,
so FGMRES is the safe default until a reduced symmetric operator is supplied.

## Evidence

Long MPI logs remain in `/tmp` during the investigation. This curated entry is
the durable record; future entries should preserve concise excerpts and
commands while leaving generated logs and result files outside the source
tree.
