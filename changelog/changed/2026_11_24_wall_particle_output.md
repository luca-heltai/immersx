PR: #510

- The wall particles of the tensor-product coupling in `elasticity` and `coupled_elasticity` carry the multiplier lifted to the wall (`lambda`), the displacement at the wall and, in `coupled_elasticity`, the vessel pressure of each wall point (`vessel_pressure`). `Immersed Problem/Wall particle output/Properties` selects them; an empty list writes no particle file. Point coupling is unchanged.
- `coupled_elasticity` computes the vessel pressure before writing the output of a step, so the files of a step hold that step's pressure.
- The new `vessel_pressure` value of `LiftedParticleOutputParameters` is written only by problems that supply it.
- The pressure that `coupled_elasticity` passes to the 1D flow model is named `solid_pressure` in the application (previously `vessel_wall_pressure`). Its value is unchanged: $-10\,p_v$ in dyn/cm².
- `CoupledElasticityProblem::pressure_at_immersed_surface_by_inclusion` is renamed `pressure_at_immersed_surface_by_support_point`, since with tensor-product coupling it holds one value per degree of freedom of the reduced grid, not per inclusion. Its values are unchanged.
