# MetricFlowX and elastodynamics coupling

This page describes the equations assembled by the
`metric_flow_x_elastodynamics` application. The model couples a 1D-in-3D
MetricFlowX blood-flow Problem to a 3D elastodynamics Problem through the
vessel wall. Both Problems and the wall interaction are added to one IDA
execution adapter.

## State variables

Write the solid displacement and velocity as $u(x,t)$ and $v(x,t)$.
The MetricFlowX state contains at least the cross-sectional area $A(s,t)$
and axial flow velocity $U(s,t)$, where $s$ is the vessel centerline
coordinate. The wall interaction adds a scalar multiplier field
$\lambda(s,t)$ on that same one-dimensional centerline.

The elastodynamics Problem uses the first-order semidiscrete form

$$
  M(\dot u-v)=0,\qquad
  M\dot v + K u + C v - f=0,
$$

where $M$, $K$, and $C$ are its mass, stiffness, and damping
operators, and $f$ is the body-force vector. MetricFlowX supplies its native
flow residual $F_{\mathrm{flow}}(A,U,\ldots)$.

## Area-to-wall kinematics

The resting area $A_0$ comes from the MetricFlowX vessel metadata. Define

$$
  R_0=\sqrt{A_0/\pi},\qquad
  \delta R(A)=\sqrt{A/\pi}-R_0,\qquad
  P(A)=\delta R(A)n,
$$

where $n$ is the unit radial direction on the reconstructed wall. Thus
$P(A_0)=0$. The tensor-product lift uses cross-section modes 3 and 7 to
represent this radial direction. It transfers the centerline area and
multiplier fields to the wall using the configured cross-section quadrature
and source thickness.

The kinematic condition is imposed weakly. For every scalar centerline test
function $\mu_h$, its lift $\mu_h^\ell n$ tests the radial displacement
mismatch:

$$
  \int_{\Gamma} u_h\cdot(\mu_h^\ell n)\,d\Gamma
  -
  \int_{\Gamma} P(A_h)\cdot(\mu_h^\ell n)\,d\Gamma=0.
$$

Equivalently, the second integrand is $\delta R(A_h)\mu_h^\ell$. These
integrals describe the continuous pairing; the discrete assembly uses the
wall lift's quadrature, mode reconstruction, and geometric weights. The
constraint is enforced through these lifted weak pairings.

In operator notation, the assembled constraint is

$$
  B^T u_h-P_h(A_h)=0\quad\text{in }\Lambda_h^*,
$$

where $\Lambda_h$ is the scalar finite-element space for $\lambda_h$,
and $\Lambda_h^*$ is its dual. The assembled residual is a vector with one
scalar entry per multiplier degree of freedom. Both terms in that residual
are dual-space vectors, even though the underlying displacement and wall
map are vector-valued fields on the wall.

## Multiplier and pressure signs

The multiplier is a scalar radial traction amplitude with pressure units for
the chosen reference-wall pairing. It can vary along the centerline. Its
identification with physical fluid pressure depends on the fictitious-interior
convention. The uncorrected extended solid includes interior material reaction;
the optional correction below removes its projected elastic and viscous part
while retaining inertia. The historical output name `pressure_jump`
denotes this amplitude. The multiplier couples the solid and flow residuals
with opposite signs:

$$
  F_{\mathrm{solid}}+B\lambda=0,
  \qquad
  F_{\mathrm{flow}}(A,U,\ldots)+F_{\Delta p}(\lambda)=0.
$$

In the current assembly, positive $\lambda$ enters the solid row through
$+B\lambda$, in its momentum equation. The native external-pressure interface receives $p_{\mathrm{add}}=-\lambda$
through pressure assembly, including its volume, numerical flux,
boundary, and junction terms. Native assembly supplies the density scaling.

## Coupled residual

With the optional modal correction disabled, collecting the solid state, flow
state, and multiplier into $y$ gives the schematic residual

$$
  F(t,y,\dot y)=
  \begin{bmatrix}
    M(\dot u-v)\\
    M\dot v+Ku+Cv-f+B\lambda\\
    F_A^{\mathrm{native}}\\
    F_U^{\mathrm{native}}+F_U^{\mathrm{pressure}}(-\lambda)\\
    B^T u-P_h(A)
  \end{bmatrix}=0.
$$

The area affects the solid through the nonlinear radius map in the kinematic
constraint. The multiplier balances solid momentum and supplies native external
pressure. Algebraic pressure trace equations also receive this pressure.
Physical continuity contains no multiplier reaction. Differentiating
compatibility gives $B^Tv-DP_h(A)\dot A=0$, which connects wall velocity
to the area rate without replacing the displacement constraint.

The application registers the multiplier and shows compatibility, traction,
and native pressure terms together. The generic symmetric constraint would
add transpose reactions to displacement and area residuals, which are not the momentum
rows of this first-order system.

The executable requires MetricFlowX, deal.II with SUNDIALS, and MPI. The
application and its build requirements are listed in the
[application reference](../reference/applications).

## Remove the two fictitious interior modes

For a single straight, connected, fully covered, uniform-radius reference
cylinder, enable:

~~~text
subsection MetricFlowX coupling
  set Remove fictitious modal response = true
end
~~~

This subtracts the elastic and viscous response of two spatially constant
affine strains: radial/circumferential strain and axial strain. It keeps the
native three-dimensional mass and mass-proportional Rayleigh damping. The
additional pressure received by MetricFlowX remains minus the multiplier; no
extra pressure correction is applied. The default leaves the original
extended-solid model unchanged.

The mode coefficients and their rates come from solid displacement and
velocity strains, using the solid's own positive volume Gauss weights inside
the fixed cylinder. The same weights and the exact transpose define the
correction. For homogeneous isotropic interior material this leaves a
nonnegative elastic and viscous energy remainder under native quadrature.
It removes only two modes, rather than all interior volume equations.
The quadrature samples an approximate lumen volume, which is reported beside
the analytical reference volume in energies.csv. Refine the solid mesh and
check this geometry error as well as solution errors.

The implementation rejects branching, curved, disconnected, nonuniform-radius,
partly uncovered, or heterogeneously stiff/damped interiors. Hanging-node
constraints and prescribed
boundary values are supported; other multipoint constraints are rejected.
Mesh constraints use the same condensation in the native solid operators and
the modal moments. Coverage is checked
at the existing wall quadrature points; it is not a proof of geometric
containment between those points. Use a contained cylinder in a convex solid
for the documented case. The correction has no new axial multiplier or vessel
mesh motion and does not add a continuity source.

The retained lumen density is the native solid material density at the sampled
points. Choose blood density there if that is the intended effective model.
The MMS uses solid density $\rho_s=1$ and fluid density $\rho_f=1060$; its
retained lumen density therefore differs from the configured fluid density.
It is not a resolved three-dimensional incompressible blood model. All native axial extension
inertia remains too: it is not tied to the one-dimensional axial velocity, and
overlap with one-dimensional axial inertia is not ruled out. This is an
effective closure without three-dimensional incompressibility, axial no-slip
or ALE compatibility. The traction convention remains
small strain on fixed reference geometry. See the
[manufactured-solution derivation](metric-flow-x-elastodynamics-mms.md) for
normalization, the reciprocal axial work, independent targets, and limitations.

Iterative actions retain two distributed strain-moment vectors. The solid
preconditioner uses the native sparse mass, stiffness and damping matrices;
the exact Jacobian still includes the factored correction. A direct solve
requests explicit materialization of the two outer products, whose storage can
grow quadratically with the number of lumen-support DoFs. Use the measured
memory requirements when choosing the finest direct-solver input.

An interaction can select this second-order preconditioning path with
the builder's second_order_preconditioner(displacement_id, mass, stiffness,
damping) method. The supplied operators approximate the inverse only; they do
not change residual or Jacobian contributions. The default continues to build
the preconditioner from the full model blocks.

On the iterative path, register the nonlocal correction through its
LinearOperator views. Matrix-backed term registrations are retained as matrices
when the adapter snapshots a Jacobian. The coupled application retains matrix
provenance for this term only when an explicit direct solve is selected.
