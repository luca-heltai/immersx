# Robin boundary conditions across applications

Scalar diffusion uses `kappa * grad(u) . n + beta*u = g`. Solid displacement
uses `stress(u)*n + beta*u = g`. Incompressible flow uses
`(2*nu*epsilon(u) - p*I)*n + beta*u = g`. The coefficient is scalar; the
datum has one component for a scalar field and one per displacement or velocity
component for a vector field. Robin data act on physical exterior faces, not on
immersed constraints or exchange interfaces.

The application entry points below share their physical Problem parameters.
Selecting an execution adapter does not remove the Problem boundary terms.

| Application | Physical solver or field | Robin path |
| --- | --- | --- |
| `coupled_elasticity` | `CoupledElasticityProblem / lib1dsolver CoupledModel1d` | Legacy solid Robin; external FVM native RCR |
| `coupled_poisson` | `PoissonSolver bulk and embedded + continuity constraint` | Scalar Robin on both fields |
| `coupled_poisson_elasticity` | `PoissonSolver<1,3> + ElasticStaticProblem<3>` | Scalar and solid Robin; interaction traction stays separate |
| `elastic_static` | `ElasticStaticProblem` | Vector Robin in native static solve |
| `elastic_static_adapter` | `ElasticStaticProblem / LinearAdapter` | Vector Robin residual and state Jacobian |
| `elasticity` | `ElasticityProblem` | Legacy native vector Robin |
| `elasticity_explicit` | `ElasticityExplicit` (independent equations) | Vector spring and load in explicit dynamics |
| `elasticity_freq` | `ElasticityFreqProblem` | Complex harmonic vector Robin |
| `elastodynamics` | `ElastodynamicsSolver` | Native transient Robin |
| `elastodynamics_adapter` | `ElastodynamicsSolver / IDAAdapter` | Time-dependent Robin residual and Jacobian |
| `fiber_reinforced_elasticity_freq` | `FiberReinforcedElasticityFreq` | Independent: both fields; shared: matrix faces and physical fiber end sections |
| `fiber_reinforced_elastodynamics` | `FiberReinforcedElastodynamics` | Matrix/fiber physical boundary contributions |
| `laplacian` | `PoissonProblem (legacy)` | Legacy scalar Robin |
| `lifted_elasticity` | `LiftedElasticity` / `static_elasticity_term` | Bulk vector Robin; lifted volume interactions stay distinct |
| `lifted_poisson` | `run_lifted_poisson` | Bulk scalar Robin |
| `liver_vascular` | `Vascularization::generate / solve_reference` | Graph pressure/conductance Robin; native resistive territories |
| `metric_flow_x` | `MetricFlowX::BloodFlowSystem / IDAAdapter` | Native terminal RCR; C=0 is pressure/flow Robin |
| `metric_flow_x_elastodynamics` | `MetricFlowX::BloodFlowSystem + ElastodynamicsSolver / IDAAdapter` | Solid vector Robin and native terminal RCR, independently |
| `navier_stokes` | `NavierStokesSolver / IDAAdapter` | Velocity traction Robin, pressure gauge, time and exact Jacobian |
| `poisson` | `PoissonSolver` | Native scalar Robin |
| `poisson_adapter` | `PoissonSolver / LinearAdapter` | Scalar Robin assembled operator and load |
| `pseudocoupling1D` | `CoupledElasticityProblem / lib1dsolver CoupledModel1d` | Legacy solid Robin; external native terminal closures |
| `reduced_poisson` | `ReducedPoisson` | Scalar Robin on the full bulk boundary |
| `vascular_network` | `Vascularization::run_application / solve_reference` | Graph pressure/conductance Robin at physical ports or unpaired terminals |

## Face conditions

Use `Robin boundary ids`, `Robin coefficient`, and `Robin boundary conditions`
in the selected Problem parameter subsection. Existing function groups retain
their own subsection paths; vector data must have the ambient component count.
A Robin boundary id must be disjoint from essential and Neumann boundary ids.
Coefficients must be finite and nonnegative, and every component of the datum
must be finite. Zero coefficient is the prescribed flux or traction limit.

For example, within a two-dimensional Navier–Stokes configuration:

```text
subsection Navier-Stokes
  set Dirichlet boundary ids = 0,1,2
  set Robin boundary ids = 3
  subsection Domain
    set Grid generator = hyper_cube
    set Grid generator arguments = 0:1:true
  end
  subsection Robin coefficient
    set Variable names = x,y,t
    set Function expression = 2+t
  end
  subsection Robin boundary conditions
    set Variable names = x,y,t
    set Function expression = 0;1+t
  end
end
```

The Navier–Stokes boundary datum is a traction. It is not multiplied by the
fluid density used for body acceleration. A physical Robin traction face fixes
the absolute pressure reference, including the zero-coefficient limit; the
solver then imposes no additional pressure pin. Both the native backward Euler
path and the IDA contributor evaluate the boundary data at their current time.
The IDA state Jacobian includes the boundary velocity operator.

In shared harmonic reinforcement, matrix faces use the matrix boundary data.
Fiber Robin data act on the matrix displacement trace at the physical fiber
end sections; they create no independent fiber unknown. In independent
formulations, each physical field has its own boundary conditions. Harmonic
problems apply the spring operator and traction to the real and imaginary
field components with their existing phase convention.

The legacy scalar applications register their functions under
`Immersed Problem/Functions` (`laplacian`) and `Reduced Poisson/Functions`
(`reduced_poisson`). For example, on a colorized unit square, this configuration
prescribes the constant solution `u=1` with zero volume forcing:

```text
subsection Immersed Problem
  set Dirichlet boundary ids = 0
  set Robin boundary ids = 1,2,3
  subsection Dirichlet boundary conditions
    set Function expression = 1
  end
  subsection Functions
    subsection Robin coefficient
      set Function expression = 2
    end
    subsection Robin boundary conditions
      set Function expression = 2
    end
    subsection Robin coefficient 1
      set Function expression = 3
    end
    subsection Robin boundary conditions 1
      set Function expression = 3
    end
  end
end
```

Replace `Immersed Problem` with `Reduced Poisson` for the reduced solver.
Per-id subsections override individual function entries; omitted entries retain
the global configuration, including after a reparse. A changed coefficient does
not automatically change the prescribed datum. The experimental legacy
`MATRIX_FREE_PATH` preprocessor branch is not exposed by the CMake applications;
these boundary conditions are implemented in their assembled solvers.

## Pressure and flow at network terminals

The vascular network and liver applications solve a Poiseuille pressure system
as part of their physical reference calculation. They are not only geometric
generators. Their terminal Robin relation is

```text
outward_flow = beta * pressure - g
```

`Vascularization/Physiology/Reference Robin conditions` maps a physical node id
to its conductance and source. Conductance has units m^3/(Pa s), and the source
has units m^3/s. A resistor to a reservoir satisfies `beta = 1/R` and
`g = P_reservoir/R`. A positive conductance anchors the pressure of its connected
component. Zero conductance prescribes a flux and still requires another
pressure anchor. These terminal conditions do not alter physical node ids,
vessel resistances, or the resistive links between tissue territories.

## Native flow terminal models

The standalone MetricFlowX application reads native MetricFlowX parameters.
The coupled MetricFlowX–elastodynamics application keeps those terminal data
separate from the solid Robin faces. Its flow contributor delegates residual
and both Jacobian blocks to `BloodFlowSystem`; it does not replace the native
terminal equations.

For a purely resistive terminal, select `Outlet boundary condition type = RCR`
and `Terminal parameter source = Parameters` under `MetricFlowSystem<1, 3>`.
Set `Default terminal C = 0`, nonnegative `Default terminal R1` and
`Default terminal R2` with positive sum, and `Default terminal P_out`. The native
condition then reads

```text
p_terminal - P_out = (R1 + R2) * Q_out
```

With positive compliance, the RCR capacitor pressure is a differential unknown.
`Terminal parameters CSV file` supplies per-boundary overrides. The generated
coupled-network path supplies its `Generated terminal resistance` as native
`R2`, with `R1 = C = 0`, and uses `Generated outlet pressure` as `P_out`.

The optional `coupled_elasticity` and `pseudocoupling1D` applications pass the
one-dimensional case file to `CoupledModel1d::init`. That backend retains its
native terminal resistance and RCR models. The ImmersX mechanical Robin data
belong to the tissue problem and do not overwrite those flow terminal data.
These applications require the optional `1dsolver` library; the solid Problem
can also be used and tested independently of that library.

Both legacy coupled drivers pass the physical flow time to the tissue solve.
Changing the time refreshes the mechanical boundary coefficient, applied load
and essential lifting together. This preserves time-dependent Robin data even
when the tissue is solved as a sequence of static problems.

See [Elasticity](../tutorials/elasticity) and
[Elastodynamics](../tutorials/elastodynamics) for the solid formulations.
