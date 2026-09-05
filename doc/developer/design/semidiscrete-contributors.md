# Semidiscrete contributors

ImmersX composes native Problems and relation-specific Interactions through an
execution adapter. The current public path is:

```text
deal.II FE space -> Field -> Observable / FE expression
                 -> weak_term -> Constraint / Interaction
                 -> LinearAdapter or IDAAdapter
```

## Application authors

A Problem owns its mesh, finite element, native operators, physical data, and
accepted state. An adapter owns solver policy and execution storage. Fields are
semantic state/residual identities; they are not global block numbers.

```cpp
IDAAdapter<FieldVector, GlobalVector> ida(time_parameters, MPI_COMM_WORLD);
auto fluid = ida.add(flow_problem, "fluid");
auto wall  = ida.add(elastic_problem, "wall");

auto pressure = value(fluid.fields().pressure);
auto traction = pressure * ImmersX::normal(surface);
ida.add(ImmersX::weak_term(traction,
                           wall.fields().displacement),
        "traction");
```

The adapter maps semantic Fields to execution storage and keeps differential
and algebraic metadata private. Candidate residual evaluations do not update a
Problem's accepted history.

## Observables and weak terms

`FESpaceView` is a non-owning view of a deal.II DoFHandler, mapping, and
constraints. A `Field` adds a semantic name and extractor. `Observable<Field,
Operation>` is a typed FE expression whose value type and supported operations
come from deal.II's `FEValuesViews`.

```cpp
auto displacement = solid_space.field(
  solid_fields.displacement, "displacement",
  dealii::FEValuesExtractors::Vector(0));
auto strain = ImmersX::symmetric_gradient(displacement);
auto pressure = ImmersX::value(flow_fields.pressure);

auto term = ImmersX::weak_term(pressure * ImmersX::normal(surface),
                               displacement);
```

The same operations apply to frozen coefficients. A frozen FE expression has
no active dependencies; it does not require a parallel class hierarchy.
Imported finite-element data is therefore ordinary FE-space metadata plus
externally owned coefficients.

The backend may use direct same-DoFHandler assembly, paired cell traversal,
particle search, or tensor-product lifting. Those are execution details and do
not change the mathematical contributor API.

## Constraints and Interactions

A `Constraint` or domain-specific Interaction contributes residual rows and
the corresponding linearization. Auxiliary multipliers are ordinary
algebraic Fields on their own FE spaces. Reactions are assembled from the
transpose of the same weak-term operator that defines the constraint.

MetricFlowX remains an external/native Problem. Its vessel-wall path exposes
the Area state as a Field, maps it through the nonlinear radial-displacement
Observable, and contributes a `MetricFlowXVesselWallConstraint` coupling to the
solid Field and pressure multiplier. The native MetricFlowX Problem is not
wrapped in a generic ImmersX Problem base class.

```cpp
auto flow = adapter.add(ImmersX::metric_flow_x(flow_problem), "blood-flow");
auto solid = adapter.add(solid_problem, "elastodynamics");
MetricFlowXVesselWallConstraint coupling(solid_field,
                                         wall_observable,
                                         search_parameters);
coupling.assemble();
auto fields = adapter.add(coupling,
                          "vessel-wall",
                          solid.fields().displacement,
                          solid.fields().velocity,
                          flow.fields().state);
```

The canonical residual is `F(t, y, ydot) = 0`; Problems and Interactions
provide `dF/dy` and `dF/dydot` separately. Solver-specific combinations belong
to the execution adapter.
