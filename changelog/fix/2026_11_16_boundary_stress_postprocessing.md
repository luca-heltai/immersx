PR: pending

- The boundary tractions written to `<output name>_forces.txt` by
  `ElasticityProblem` and to `<output name>_forces_<real|imag>_cycle_<n>.txt`
  by `ElasticityFreqProblem` now use the strain `sym(grad u)` instead of
  `grad u`. Before, the skew-symmetric part of the displacement gradient
  entered `sigma n`.
- The `uDotN_<id>` columns of the same files are now summed over all MPI
  ranks. Before, rank 0 wrote only its local contribution.
