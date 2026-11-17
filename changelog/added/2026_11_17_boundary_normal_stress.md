PR: pending

- The forces files written by `ElasticityProblem` and `ElasticityFreqProblem`
  have a new column `sigmaNN_<id>` per boundary id, after `uDotN_<id>`: the
  mean normal stress `n . sigma n` over that boundary. On a closed curved
  boundary, such as the inner face of a cylinder shell, the existing mean
  traction vector averages to about zero; the mean normal stress does not.
