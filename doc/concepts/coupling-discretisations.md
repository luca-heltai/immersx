# Coupling discretisations

An embedded object, such as a vessel, a fiber or a set of inclusions, has a
smaller dimension than the bulk mesh around it, or is not resolved by that
mesh. ImmersX represents such an object, and couples it to the bulk, in one of
four ways:

- point coupling: discrete inclusions with a center, a direction and a
  radius, and a Fourier expansion on each inclusion;
- tensor-product coupling: a finite element space on a reduced grid,
  multiplied by a basis on a reference cross section;
- lifts: the same tensor-product geometry, used as an Observable or WeakTerm
  instead of a separate coupling object;
- direct overlap continuity: a multiplier on the embedded mesh itself, with
  no cross section.

All four evaluate bulk shape functions at points that the bulk mesh does not
own. Particle search and, optionally, local co-refinement handle that part.
This page explains each discretisation, the parameter sections it reads, and
which application uses it. The weak forms are in
[Mathematical background](mathematical-background.md).

## Applications and their coupling

| Application | Coupling | Selected by | Example input |
|---|---|---|---|
| `elasticity` | point or tensor-product | `Immersed Problem/Coupling type = Point` or `TensorProduct` | `gtests/parameters/elasticity_04_higher_order_modes.prm.in` (point), `data/tests/elasticity_tensor_product_coupling_3d.prm.in` (tensor-product) |
| `coupled_elasticity` | point or tensor-product | `Immersed Problem/Coupling type` | `gtests/parameters/coupled_elasticity_vs_elasticity.prm.in` |
| `laplacian` | point | always | `gtests/parameters/laplacian_simple.prm.in` |
| `reduced_poisson` | tensor-product; point cloud for `reduced dimension = 0` | `reduced dimension` | `tutorials/reduced_poisson/single_cylinder_variable_radius_3d.prm`, `tutorials/reduced_poisson/random_particles_2d.prm` |
| `lifted_poisson` | modal lift with a multiplier | always | `gtests/parameters/lifted_poisson.prm.in` |
| `lifted_elasticity` | lift | `Lifted elasticity/Coupling = none`, `multiplier` or `pressure` | `data/lifted_elasticity/multiplier_3d.prm.in`, `data/lifted_elasticity/pressure_radius_3d.prm.in` |
| `coupled_poisson` | direct overlap continuity | always | `tutorials/coupled_poisson/coupled_poisson.prm.in` |
| `coupled_poisson_elasticity` | lifted pressure and traction weak term | always | `tutorials/coupled_poisson_elasticity/coupled_poisson_elasticity.prm.in` |
| `metric_flow_x_elastodynamics` | lift of the vessel wall with a multiplier | always | `tutorials/metric_flow_x_elastodynamics/two_way.prm.in` |
| `fiber_reinforced_elastodynamics` | direct overlap continuity on the fiber | always | `tutorials/fiber_reinforced_elastodynamics/parameters.prm.in` |
| `fiber_reinforced_elasticity_freq` | lift when the reinforcement has lower dimension; direct overlap otherwise | dimensions and `Formulation` | `tutorials/fiber_reinforced_elasticity_freq/fiber_3d.prm.in`, `tutorials/fiber_reinforced_elasticity_freq/full_dimension.prm.in` |

The other applications (`poisson`, `elastic_static`, `elastodynamics`,
`elasticity_freq`, `elasticity_explicit`, `navier_stokes`, `metric_flow_x`,
`liver_vascular`, `vascular_network`) solve a single domain and have no
coupling discretisation. [Applications](../reference/applications.md) lists the
supported dimensions of every application.

## Point coupling

Point coupling is implemented by `Inclusions<spacedim>` and reads
`/Immersed Problem/Immersed inclusions`.

### Geometry

Each inclusion is one row. In 2D a row is `cx cy r`: a circle with center
$(c_x, c_y)$ and radius $r$. In 3D a row is `cx cy cz dx dy dz r vessel_id`: a
cylinder segment with center $c$, direction $d$ and radius $r$. The length of
the segment is $|d|$. Rows come from the inline list `Inclusions`; the rows of
`Inclusions file` are appended to it, and both are used.

### Multiplier space

The multiplier on inclusion $i$ is a finite Fourier expansion in the angle
$\theta$ around the inclusion,

$$
\lambda_i(\theta) = \sum_{c \in S} \lambda_{i,c}\, \phi_c(\theta),
$$

with one degree of freedom per selected coefficient $c$. Index $c$ refers to
Fourier mode $c / n$ and vector component $c \bmod n$, where $n$ is the number
of vector components. Mode 0 is the constant, mode $2k-1$ is $\cos k\theta$
and mode $2k$ is $\sin k\theta$. The selection $S$ is `Selection of Fourier
coefficients`. When it is empty, `Number of fourier coefficients` $= N$
selects all components of the first $N$ modes. With `Cluster inclusions with
segments = true`, the multiplier has one set of coefficients per segment
instead of per inclusion.

The prescribed data on each inclusion are the coefficients of the same
expansion: one line per inclusion in `Data file`, or `Reference inclusion
data` for every inclusion when `Data file` is empty.

### Quadrature

`Inclusions refinement` sets the number $N_q$ of equispaced angles on each
inclusion. Point $q$ of inclusion $i$ has the weight

$$
w_{i,q} = \frac{2 \pi r_i}{N_q}\, |d_i|,
$$

where $|d_i| = 1$ in 2D. The points are inserted as particles in the bulk
mesh, so each rank evaluates the bulk shape functions at the points that fall
in its cells. In `elasticity` the coupling matrix divides the weight by the
measure of the inclusion surface, so the multiplier pairs with the mean of
the displacement over the inclusion.

### Assembly

`ElasticityProblem::assemble_coupling` calls `assemble_point_coupling` when
`Coupling type = Point`. `PoissonProblem::assemble_coupling` does the same
for `laplacian`.

## Tensor-product coupling

Tensor-product coupling is implemented by
`ReducedCoupling<reduced_dim, dim, spacedim, n_components>`. Here
`reduced_dim` is the dimension of the reduced grid $\gamma$, `dim` is the
dimension of the represented object, and the cross section has dimension
`dim - reduced_dim`. A vessel modelled as a curve in 3D with a circular wall
uses `ReducedCoupling<1, 2, 3>`. The class reads `/Tensor product coupling/`
with three subsections:

- `Representative domain`: the reduced grid, its finite element and
  quadrature, the thickness and the input fields;
- `Cross section`: the reference cross section, its basis and its quadrature;
- `Particle coupling`: the search tree used to locate points in the bulk.

### Multiplier space

The multiplier space is the tensor product

$$
\Lambda_h = V_h(\gamma) \otimes \operatorname{span}\{\psi_c : c \in S\}.
$$

$V_h(\gamma)$ is `FE_Q` of degree `Finite element degree` on the reduced grid.
The modes $\psi_c$ are built on the reference cross section $\hat D$ from the
polynomials of degree at most `Maximum basis degree`, times the vector
components. They are orthonormalised in $L^2(\hat D)$ and scaled to a maximum
absolute value of one, the same normalisation as the Fourier modes of point
coupling. Modes that vanish on $\hat D$ are dropped. `Selected modes` is the
list $S$; an empty list selects every mode.

`Geometry type` chooses $\hat D$: `hyper_ball`, `hyper_sphere` or
`hyper_cube`. When the cross section has one dimension less than the space,
`hyper_ball` builds the boundary sphere, so a curve in 3D gets a circle: the
vessel wall. `Refinement level` refines $\hat D$.

### Quadrature

Two quadratures are combined:

- `Representative domain/Quadrature type`, `Number of quadrature points` and
  `Number of quadrature repetitions` integrate along the reduced grid;
- `Cross section/Quadrature type`, `Number of points` and `Number of
  repetitions` integrate over $\hat D$.

Both are `QIterated` formulas of a `QuadratureSelector` rule. Zero points
means $2p + 1$ Gauss points, where $p$ is `Finite element degree` on the
reduced grid and `Maximum basis degree` on the cross section.

Each reduced-grid point $x_\gamma$ with tangent $t$ and weight $w_\gamma$
generates one physical point per cross-section point $\hat x$ with weight
$\hat w$:

$$
x = x_\gamma + R(t)\, h\, \hat x,
\qquad
w = w_\gamma\, \hat w\, h^{\,d_D},
$$

where $R(t)$ rotates the reference section normal to $t$, $h$ is the
thickness and $d_D$ is the cross-section dimension.

### Geometry and thickness

The reduced grid is a VTK file, `Reduced grid name`. Its point and cell
fields are exposed through `Input file fields` and can appear in `Thickness`.
`Thickness` is either a number or an expression of those fields, for example
`Thickness = radius`. `Reduced right hand side` gives one expression per
selected mode.

A legacy `Inclusions file` can replace the VTK grid for `<1, 2, 3>`. Each row
becomes one cell of the reduced grid with fields `radius` and `vessel_id`, and
`Thickness = radius` reads the radius column. The legacy `Number of fourier
coefficients` and `Selection of Fourier coefficients` are converted to the
reference basis, and are used only when `Cross section/Selected modes` is
empty.

### Point clouds

With `reduced_dim = 0` the reduced grid is a cloud of points without a finite
element: each point carries one degree of freedom per selected mode, and the
representative weight is one. `reduced_poisson` uses this case for particles
in 2D and 3D, read from a `.vtk`, `.vtu` or `.pvtu` file. A legacy 2D
`Inclusions file` is accepted for `<0, 1, 2>`.

### Assembly

`ReducedCoupling` assembles the multiplier load matrix, the coupling mass
matrix and the reduced right-hand side. `elasticity` calls it from
`assemble_tensor_product_coupling` when `Coupling type = TensorProduct`, and
`reduced_poisson` from `assemble_coupling_system`. Tensor-product coupling
needs deal.II with VTK; without VTK, `Coupling type = TensorProduct` is an
error. [Reduced coupling](../how-to/reduced-coupling.md) covers simplex bulk
meshes and the point-cloud background hook.

## Lifts

A lift uses the tensor-product geometry, but not a `ReducedCoupling` object.
It maps a quantity defined on the reduced grid to the lifted points, or a
bulk quantity back to the reduced grid, and is used inside Observables,
WeakTerms and Interactions.

`TensorProductLift<reduced_dim, surface_dim, spacedim, n_components>` reads its
own section, for example `/MetricFlowX vessel wall lift/`:

- `Thickness`: a number, or an expression of the reduced-grid fields;
- `Representative quadrature/Type`, `Number of points` and `Number of
  repetitions`: the quadrature along the reduced grid;
- `Cross section/`: the same entries as in tensor-product coupling.

The applications use lifts as follows:

- `lifted_elasticity` lifts the displacement and either constrains selected
  modes with a multiplier or applies a modal pressure load, depending on
  `Coupling`. See [Lifted static elasticity](../how-to/lifted-static-elasticity.md).
- `lifted_poisson` uses `ModalLiftedObservable` with
  `LiftedMultiplierInteraction`, and takes the radius from a field or from
  `Thickness`. See [Lifted Poisson](../how-to/lifted-poisson.md).
- `coupled_poisson_elasticity` lifts the Poisson pressure from the curve to the
  vessel wall (`/Pressure lift/`) and applies it as a traction on the solid.
- `metric_flow_x_elastodynamics` adds a scalar multiplier on the MetricFlowX
  centerline. The lift (`/MetricFlowX vessel wall lift/`, modes 3 and 7)
  transfers the cross-sectional area and the multiplier to the vessel wall,
  where the multiplier weakly matches the radial displacement of the solid to
  the area. See [MetricFlowX and elastodynamics
  coupling](../tutorials/metric-flow-x-elastodynamics-coupling.md).
- `fiber_reinforced_elasticity_freq` lifts a lower-dimensional reinforcement
  to a thin volume through `Domain::lift()`. See
  [Lifted integration](../how-to/lifted-integration.md).

## Direct overlap continuity

When the embedded object is meshed in its own right, the multiplier can live
on that mesh with no cross section. `make_continuity_constraint` pairs the
multiplier with the difference of the two fields on the overlap,
integrated with a Gauss rule on the multiplier mesh. `coupled_poisson` uses
it with the multiplier space `/Coupled Poisson/Multiplier space/`, `FE_DGQ(0)`
by default. `fiber_reinforced_elastodynamics` uses it for the fiber velocity,
and `fiber_reinforced_elasticity_freq` uses it when the reinforcement has the
same dimension as the matrix. [Support-aware weak
terms](../how-to/support-aware-weak-terms.md) explains overlap and outside
supports.

## Particle search and co-refinement

Every coupling above evaluates bulk shape functions at points owned by
another mesh. `ParticleCoupling` builds a tree of the bounding boxes of the
locally owned bulk cells, extracts the level set by `RTree extraction level`,
and inserts the points with `Particles::ParticleHandler`, so each point is
evaluated on the rank that owns the containing cell. Point coupling has its
own handler, whose tree level is `Immersed inclusions/Bounding boxes
extraction level`.

Accuracy depends on the ratio between the bulk mesh size and the size of
the embedded object. `LocalCoRefinement` refines the bulk, the embedded mesh,
or both, near the embedded object, and `Embedded partitioning policy =
immersed` assigns each embedded cell to the rank that owns the bulk cell at
its center. Tensor-product coupling reads these settings from
`/Tensor product coupling/Local refinement parameters`; the lifted
applications read them from their `Co-refinement` section. See [Local
co-refinement](../how-to/local-co-refinement.md).

## Point and tensor-product coupling compared

Both couplings describe a vessel by its centerline, radius and modes on its
cross section, and `elasticity` and `coupled_elasticity` accept both through
`Coupling type`. They differ in where the multiplier lives:

- point coupling has independent coefficients on each inclusion, with no
  continuity between neighbouring inclusions;
- tensor-product coupling has a continuous finite element along the reduced
  grid, so the coefficients vary continuously along the vessel.

Point coupling reads circles or cylinder segments directly and integrates
with equispaced angles. Tensor-product coupling reads a VTK reduced grid with
fields and integrates with a Gauss rule on both the reduced grid and the
cross section, which makes the thickness a field rather than one radius per
inclusion.
