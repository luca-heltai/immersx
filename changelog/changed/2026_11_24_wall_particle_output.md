PR: #510

- The wall particles of the tensor-product coupling in `elasticity` and `coupled_elasticity` carry the multiplier lifted to the wall (`lambda`), the displacement at the wall and, in `coupled_elasticity`, the vessel pressure of each wall point (`pressure_at_immersed_surface`). `Immersed Problem/Wall particle output/Properties` selects them; an empty list writes no particle file. Point coupling is unchanged.
- `coupled_elasticity` computes the vessel pressure before writing the output of a step, so the files of a step hold that step's pressure.
- The new `pressure_at_immersed_surface` value of `LiftedParticleOutputParameters` is written only by problems that supply it.
