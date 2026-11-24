PR: pending

- The wall particles of the tensor-product coupling in `elasticity` and `coupled_elasticity` carry the multiplier lifted to the wall (`lambda`), the displacement at the wall and, in `coupled_elasticity`, the normal stress at the wall point (`wall_pressure`). `Immersed Problem/Wall particle output/Properties` selects them; an empty list writes no particle file. Point coupling is unchanged.
- `coupled_elasticity` computes the wall pressure before writing the output of a step, so the files of a step hold that step's pressure.
- The new `wall_pressure` value of `LiftedParticleOutputParameters` is written only by problems that supply it.
