# Constraint and vessel API names

Use names that identify the operation rather than the generic word “coupling”.
A multiplier constraint, a prescribed wall pressure, vessel postprocessing, and
exchange with an external flow solver are separate operations.

| Operation | Normal API | Compatibility name |
| --- | --- | --- |
| Map multiplier coefficients to bulk loads | `ElasticityProblem::multiplier_load_matrix()` | Public `coupling_matrix` storage |
| Assemble that matrix from a reduced space | `ReducedCoupling::assemble_multiplier_load_matrix()` | `assemble_coupling_matrix()` overloads |
| Inspect the reduced Poisson load matrix | `ReducedPoisson::assembled_multiplier_load_matrix()` | `assembled_coupling()` |
| Inspect its global Frobenius norm | `multiplier_load_matrix_frobenius_norm()` | `coupling_matrix_frobenius_norm()` |
| Solve elasticity and recover vessel wall pressure/radius | `VesselElasticityProblem` in `<immersx/physics/vessel_elasticity.h>` | `CoupledElasticityProblem` and its header |
| Select prescribed vessel pressure rather than a displacement constraint | `uses_prescribed_vessel_pressure()` / `set_prescribed_vessel_pressure(bool)` | Public `pressure_coupling` storage |
| Select that mode in a parameter file | `Prescribed vessel pressure` | `Pressure coupling` |

## Multiplier constraints

The multiplier load matrix has bulk rows and multiplier columns. It is the
transpose of the discrete constraint operator: if the constraint is `B u = g`,
the bulk reaction is `B^T lambda`. Transverse reduction is already part of this
operator. It is not a time exchange interface with a flow solver.

The old Elasticity matrix member remains the storage returned by the new
accessor. Both names observe identical entries. The old assembly overloads
forward to the new overloads, including the affine background correction to the
reduced right hand side. No transpose, normalization or numerical operation
changes when using the new names.

## Vessel elasticity and wall pressure

`VesselElasticityProblem<dim, spacedim>` is the same C++ type as
`CoupledElasticityProblem<dim, spacedim>`. Existing specializations, derived
classes, constructors and binary instantiations therefore retain their type
identity. Use the vessel name for the tissue problem even when no external flow
solver is present. The `coupled_elasticity` executable remains an external
staggered coupling application; its exchanged pressure is named
`vessel_wall_pressure` in the implementation.

The tissue problem recovers normal wall stress from its multiplier field.
The material also fills the vessels, so the multiplier is the jump between the
normal stress of the tissue and that of the interior material. With
`Remove interior stress from pressure = true`, the default, the analytic
interior stress is added back and the result is the normal stress of the
tissue alone. `pressure_at_immersed_surface` and
`pressure_at_immersed_surface_by_inclusion` name the vessel and inclusion data;
they do not imply that an external solver owns the pressure.
[Pressure on the vessel walls](../concepts/vessel-pressure.md) defines both
values, their sign, and the interior correction.

## Prescribed pressure mode

Use `parameters.set_prescribed_vessel_pressure(true)` or:

```text
subsection Immersed Problem
  set Prescribed vessel pressure = true
end
```

This mode projects the supplied vessel pressure and applies its transpose load
to the displacement equation. With the mode disabled, the static path solves
the multiplier saddle-point system instead. The mode does not select an
external solver or turn data exchange on or off. Existing restrictions on
quasistatic and dynamic point formulations remain unchanged. The load acts on
the extended solid, so the supplied value is a jump of normal stress across the
wall; it receives no interior-stress correction.

`Pressure coupling` is a ParameterHandler alias for the same entry. It accepts
existing files without a deprecation warning. If a file deliberately sets both
spellings, the last assignment wins, as for repeated assignments to one entry.
The C++ compatibility member and both accessors also share the same boolean;
there is no second value to synchronize.

Modern `Constraint`, `Interaction`, `ReducedCoupling`, and application names
that describe an actual pairing of physical systems retain their meanings.
This migration does not replace the mathematical concept of coupling or
introduce another application hierarchy.
