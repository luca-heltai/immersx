# Harmonic reinforcement with or without a lift

The matrix always has dimension `spacedim`. The root `dimension` parameter is
now the dimension of the reinforcement mesh; `space dimension` is the ambient
dimension. The application accepts all six pairs with
`1 <= dimension <= space dimension <= 3`.

- If `dim < spacedim`, the reinforcement is lifted through a cross section of
  dimension `spacedim-dim` to obtain a volume.
- If `dim == spacedim`, its independently supplied volume mesh is used directly.
  There is no `Lift` subsection in this case.

`Fiber Reinforced Elasticity Frequency/Formulation` selects the displacement
model. Both choices use the reinforcement's excess material coefficients,
which are added to the matrix response already present in that volume.

## Run the examples

Configure with MPI's C++ compiler, then build the application:

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=mpicxx -DDEAL_II_DIR=/path/to/deal.II \
  -DENABLE_COUPLED_PROBLEMS=OFF -DIMMERSX_BUILD_CORAL_PLUGINS=OFF
cmake --build build-debug --target fiber_reinforced_elasticity_freq_debug -j 4
mpirun -np 2 build-debug/fiber_reinforced_elasticity_freq_debug \
  build-debug/tutorials/fiber_reinforced_elasticity_freq/parameters.prm
```

The Release target is `fiber_reinforced_elasticity_freq`. CMake configures the
parameter templates into the build tree, with absolute output paths. The
executable and input can also be addressed by absolute paths from another
working directory.

Six complete inputs are provided under
`tutorials/fiber_reinforced_elasticity_freq/`:

| Generated input | Reinforcement | Formulation |
| --- | --- | --- |
| `parameters.prm` | 1D in 2D, with a 1D lift | `independent` |
| `shared.prm` | 1D in 2D, with a 1D lift | `shared` |
| `full_dimension.prm` | 2D in 2D, without a lift | `independent` |
| `fiber_3d.prm` | 1D in 3D, with a 2D lift | `independent` |
| `fiber_3d_shared.prm` | 1D in 3D, with a 2D lift | `shared` |
| `sheet.prm` | 2D in 3D, with a 1D lift | `independent` |

The [fiber in a 3D matrix tutorial](fiber-in-3d-frequency.md) explains the
codimension-two geometry and both displacement formulations.

The canonical lifted input is:

```{literalinclude} ../../tutorials/fiber_reinforced_elasticity_freq/parameters.prm.in
:language: ini
```

Each example clamps the matrix's left side, applies a traction with a phase of
15 degrees on the right, and sweeps 0, 0.1, and 0.2 Hz with nonzero damping.
Change `Formulation` to switch models. Use a separate output directory for each
run so old datasets from another model are not mistaken for current results.

## Lift geometry and material units

The `Lift` subsection is registered only for reduced reinforcement meshes.
Its cross section has dimension `spacedim-dim`. The supported `hyper_cube`
section is `[-1,1]^(spacedim-dim)`, scaled by `Thickness = h`; its physical
measure is therefore `(2h)^(spacedim-dim)`. This is an interval in codimension
one and a square in codimension two. A 2D sheet in 3D extrudes along its local
normal, using both tangent directions.

`Lift/Representative quadrature` and `Lift/Cross section` control the
quadrature points and repetitions. Increase them to resolve the background FE
traces and material interfaces. Points are located in the distributed matrix
mesh through deal.II's particle facilities; every lifted point must lie inside
the matrix's searchable domain. Quadrature is not split at matrix-cell
interfaces, so increasing only the Gauss order need not give monotone
convergence.

Material coefficients are per unit physical volume. The lift supplies the
section measure; do not multiply the material coefficients by it again. The
current harmonic reconstruction uses the constant section mode:
`Maximum basis degree = 0` and `Selected modes = 0`. The thickness is a positive
numeric constant. The reduced harmonic model assumes flat cell manifolds;
curved geometries can be approximated with piecewise-flat meshes. Geometry
remains fixed during a sweep. The quadrature uses
representative measure times section measure; curvature corrections for offset
Jacobians and derivatives of transported frames are not included.

## Frequency and phase convention

For $u(t)=\operatorname{Re}(\widehat u\exp(i\omega t))$, each material contributes

$$
H(\omega)=K-\omega^2M+i\omega D,\qquad \omega=2\pi f,
$$

with

$$
\begin{aligned}
K(u,v)&=\int 2\mu\,\varepsilon(u):\varepsilon(v)
                   +\lambda\,\operatorname{div}u\operatorname{div}v,\\
M(u,v)&=\int\rho\,u\cdot v,\\
D(u,v)&=\beta K(u,v)+\alpha M(u,v)
                  +\int\eta\,\varepsilon(u):\varepsilon(v).
\end{aligned}
$$

The real matrix representation is

$$
\begin{pmatrix}K-\omega^2M&-\omega D\\
\omega D&K-\omega^2M\end{pmatrix}.
$$

The root `Frequencies` list is shared by matrix and reinforcement. Their
`Material properties` and `Functions` subsections remain independent. Spatial
load amplitudes are evaluated at `t=0`, then combined with phase angles in
degrees. Reinforcement material ids are taken from its own mesh, including in
the shared formulation.

## Independent displacements

`Formulation = independent` retains separate matrix and reinforcement Fields,
with real and imaginary multiplier constraints. For reduced reinforcement,
the lift reconstructs each ambient displacement component as constant across
the section. Its native tangential strain, mass, damping, body forces, and
endpoint/boundary tractions are weighted by the physical section measure.
A full-dimensional reinforcement uses its ordinary volumetric strain and
volume measure.

Writing $E$ for constant normal extension, the continuity equations are

$$
\int_{\Omega_f}E\lambda\cdot(\widehat u_m-E\widehat u_f)=0.
$$

This is a weak section-average constraint. It does not require every matrix
point in a section to have the same displacement. For full-dimensional
reinforcement, $E$ is the identity. Both real and imaginary equations are
imposed. The complex system is

$$
\begin{pmatrix}
H_m&0&C_m^T\\
0&H_f&-C_f^T\\
C_m&-C_f&0
\end{pmatrix}
\begin{pmatrix}\widehat u_m\\\widehat u_f\\\widehat\lambda\end{pmatrix}
=
\begin{pmatrix}\widehat f_m\\\widehat f_f\\0\end{pmatrix}.
$$

The multiplier defaults to vector-valued continuous `FE_Q` on the reinforcement
mesh. `Multiplier FE degree` overrides its degree; `Multiplier space/Finite
element` selects an explicit element with `spacedim` components. The mesh and
space must yield independent constraints. The default direct solver treats
the complete saddle-point system without assuming positive definite or
individually invertible displacement blocks. An undamped resonance may still
make the complete system singular.

This is a vector small-strain model with constant section reconstruction. It
has no independent rotations or higher section modes.

## Shared displacement

`Formulation = shared` registers only the matrix's real and imaginary
spatial displacement Fields. The reinforcement mesh supplies integration
geometry and material ids; it carries no independent displacement unknowns
and introduces no multiplier. The equation is

$$
(H_m+H_f^{\mathrm{bulk}})\widehat u
 =\widehat f_m+\widehat f_f^{\mathrm{bulk}}.
$$

Reinforcement stiffness, mass, and damping use the matrix displacement's full
ambient gradients at physical reinforcement quadrature points. Its body force
is evaluated at those same points, with its own phase and material overrides.
The `Fiber` boundary-id sets must be empty: essential and Neumann boundary
conditions belong to the matrix in this formulation. The default reinforcement
boundary-id sets are empty.

This formulation allows the matrix displacement to vary across the lifted
section. It is therefore a different reduced model from constant-extension
independent displacements, even with the same coefficients and geometry.

## Output and current limits

Both formulations write `frequencies.csv` and matrix VTU/PVTU datasets with real
and imaginary displacement, component amplitudes, and phases in degrees.
Independent displacements also write reinforcement datasets on its native
mesh and multiplier datasets under `interaction/frequency_<index>`.

`residual_norm()` reports the complete algebraic residual.
`continuity_norm()` reports the combined multiplier-row residual in the
independent formulation and is zero when no constraint is present.
`reinforcement_measure()` reports the integrated physical reinforcement volume.
These diagnostics do not estimate the discretization error.

Each Problem uses one prepared mesh during the frequency sweep. Set
`Number of refinement cycles = 1` and choose initial refinements before running.
The adapter path supports strong Dirichlet conditions. Weak Dirichlet
conditions remain available in standalone `elasticity_freq`; normal-flux
constraints require a full-dimensional 2D or 3D Problem.

## Compose a harmonic Problem

An uncoupled harmonic Problem can also be registered directly with
`LinearAdapter`:

```cpp
using namespace ImmersX;
ElasticityFreqProblemParameters<2> parameters;
initialize_parameters(parameter_file);
ElasticityFreqProblem<2> problem(parameters);
problem.make_grid();
problem.setup_fe();
problem.setup_dofs();
problem.prepare_frequency(1.0);

LinearAdapterParameters solver_parameters;
solver_parameters.solver = LinearSolver::direct;
LinearAdapter<ImmersXLA::MPI::Vector, ImmersXLA::MPI::BlockVector>
  adapter(solver_parameters, problem.communicator());
const auto fields = adapter.add(problem, "solid").fields();
auto state = adapter.make_state();
adapter.solve(state);
problem.accept_frequency_solution(adapter.field(state, fields.real),
                                  adapter.field(state, fields.imaginary));
problem.output_results(0);
```

Include `immersx/physics/elasticity_freq_contributor.h` and
`immersx/core/linear_adapter.h`. The two algebraic Fields can be used in ordinary
FE expressions and continuity constraints. A new Interaction can contribute to
either residual row or introduce its own multiplier Field. Mesh ownership,
material lookup, and essential constraints remain with the Problem; execution
block numbers remain inside the adapter.

Each frequency has constant coefficients during one linear solve. Prepare the
new frequency and create a fresh adapter composition for the next solve so its
cached weak terms use the new coefficients. For a new constitutive term, add
its real and imaginary weak contributions with this same phase convention;
frequency-dependent solver policy belongs in the execution adapter.
