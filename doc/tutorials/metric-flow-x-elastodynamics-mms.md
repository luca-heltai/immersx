# One vessel in a cube: a manufactured solution

This example removes the fictitious interior solid's two affine elastic and
viscous modes while retaining full inertia. It uses distinct radial and axial
strains, nonzero viscosity, and nonzero forcing. Enable the correction with

~~~text
subsection MetricFlowX coupling
  set Remove fictitious modal response = true
end
~~~

Without this setting the existing extended-solid model is unchanged. The current
implementation supports one straight, circular, uniform-radius reference cylinder
contained in the solid. It projects onto two spatially constant modes over the
whole cylinder, rather than independent modes at every section. It does not
support a branching-network correction. The solid must be conforming: prescribed
Dirichlet values are supported, but hanging-node and other multipoint constraint
lines are rejected. Interior elastic and viscous coefficients must be homogeneous
and match the configured default material.

## Model and pressure normalization

The frozen reference radius is $a=R_0=\sqrt{A_0/\pi}$. Geometric $A_0$ differs
from the tube law's constitutive $A_{\rm ref}$: the two areas need not coincide,
and resting pressure need not be zero. The interior affine displacement is

$$
u_r=gr,\qquad u_x=hx,\qquad \varepsilon(u)=\operatorname{diag}(h,g,g).
$$

The native stress is

$$
\sigma_e=2\mu_s\varepsilon(u)+\lambda_s\operatorname{div}(u)I,\qquad
\sigma_v=(\eta_s+2\beta_R\mu_s)\varepsilon(v)
+\beta_R\lambda_s\operatorname{div}(v)I.
$$

Here $\eta_s$ multiplies $\varepsilon(v)$, not $2\varepsilon(v)$.
The Lamé coefficient $\lambda_s$ differs from the multiplier $\lambda$.
The modal normal stresses are

$$
\begin{aligned}
q_r={}&2(\lambda_s+\mu_s)g+\lambda_sh
+[\eta_s+2\beta_R(\lambda_s+\mu_s)]\dot g
+\beta_R\lambda_s\dot h,\\
q_z={}&2\lambda_sg+(\lambda_s+2\mu_s)h
+2\beta_R\lambda_s\dot g
+[\eta_s+\beta_R(\lambda_s+2\mu_s)]\dot h.
\end{aligned}
$$

The effective momentum equation is $R_\Omega(w)+b(\lambda,w)-C_f(w)=0$.
For exact reference geometry and sectionwise constant modes,

$$
C_f(w)=\int_\gamma A_0(2q_rg_w+q_zh_w)\,ds.
$$

The radial contribution equals $\int_{\Gamma_0}q_r(w\cdot n)\,d\Gamma$.
The axial contribution uses the resultant $N_z=A_0q_z$ and is not a pressure
on the lateral wall. Integration by parts gives $-\partial_sN_z$ and endpoint
forces. Since the correction is subtracted, constant $N_z$ contributes
$-N_zw_x$ at the right endpoint and $+N_zw_x$ at the left. Keeping
$\lambda_sh$ only in radial pressure would omit the reciprocal axial term.

The coupling integrates physical reference-wall weights with a normalized radial
vector lift. Thus $\lambda$ has pressure units; native pressure needs no extra
bare $2\pi R_0$ factor. The corrected convention sends $p_{\rm add}=-\lambda$
to native flow and subtracts the whole correction only in solid momentum.
Subtracting $q_r$ again from native pressure would count it twice.
Fixed-centerline continuity remains source-free.

These are small-strain equations with reference-geometry traction. Scalar
physical pressure expressed on reference coordinates is not automatically nominal
traction. Exact current-area power transfer would require the corresponding
$R_0/R$ conversion. This example retains the fixed-reference approximation.

The semidiscrete rows keep native mass and separate state from derivative
linearization. With assembled modal operators $C_e,C_v$, they are

$$
\begin{aligned}
F_u&=M(\dot u-v),\\
F_v&=M\dot v+(K-C_e)u+(D-C_v)v-f+B\lambda,\\
F_A&=F_A^{\rm native},\\
F_U&=F_U^{\rm native}+F_U^{\rm pressure}(-\lambda),\\
F_\lambda&=B^Tu-P_h(A).
\end{aligned}
$$

Native algebraic flow traces also retain their pressure contributions. The
correction changes only the displacement and velocity state derivatives in
solid momentum. Its derivative-state block is zero: $dF_v/d\dot v=M$ remains
native. The execution adapter forms its own
$J=dF/dy+\alpha\,dF/d\dot y$. No multiplier force enters continuity,
and no additional axial multiplier or vessel kinematic constraint is introduced.

## Discrete projection and retained energy

The implementation selects native solid Gauss points inside the reference
cylinder. With physical weights $w_q$ and $V_h=\sum_qw_q$, and unit tangent $t$,

$$
g_h=\frac{1}{2V_h}\sum_qw_q[
\operatorname{tr}\varepsilon(u_q)-t\cdot\varepsilon(u_q)t],\qquad
h_h=\frac{1}{V_h}\sum_qw_q\,t\cdot\varepsilon(u_q)t.
$$

Rates use the same projection of $\varepsilon(v)$, without area differencing or
an area acceleration. Although the area map is
$g(A)=\sqrt{A/A_0}-1$, discrete BDF differentiation does not obey an exact
chain rule: differentiating $g(A)$ need not equal $g'(A)\dot A$.
The residual uses the transpose of these same functionals
and the same points and weights:

$$
C_{f,h}(w)=V_h(2q_rg_{w,h}+q_zh_{w,h}).
$$

This global two-mode projection uses sampled $V_h$, not analytical $A_0L$.
Point membership approximates the curved boundary and changes discontinuously
under mesh refinement. Measure sampled-volume error as a geometry error.
The polygonal compatibility lift uses separate quadrature, so its wall
displacement average need not equal the strain projection at finite resolution.

The elastic density and modal matrices are

$$
\psi(g,h)=2\mu_sg^2+\mu_sh^2+\frac{\lambda_s}{2}(2g+h)^2,
$$

$$
H_e=\begin{pmatrix}
4(\lambda_s+\mu_s)&2\lambda_s\\
2\lambda_s&\lambda_s+2\mu_s
\end{pmatrix},\qquad
H_v=\begin{pmatrix}
2\eta_s+4\beta_R(\lambda_s+\mu_s)&2\beta_R\lambda_s\\
2\beta_R\lambda_s&\eta_s+\beta_R(\lambda_s+2\mu_s)
\end{pmatrix}.
$$

Generalized forces are $(2q_r,q_z)$. For usual positive elastic moduli,
$\det H_e=4\mu_s(3\lambda_s+2\mu_s)>0$. For nonnegative native viscous
coefficients $H_v$ is positive semidefinite. With homogeneous coefficients,
native positive Gauss weights and the matching projection/adjoint, subtraction
leaves nonnegative discrete strain-fluctuation energy. Positivity of the small
matrix alone would not prove positivity of the remaining three-dimensional
operator.

Nonmodal interior strains remain as numerical extension and stabilization.
All native mass remains, including axial and transverse acceleration. Interior
density is therefore part of the effective model. The vessels contain blood,
and native one-dimensional flow does not directly transfer transverse momentum
to the surrounding solid; retained three-dimensional inertia is the selected
effective representation of this contribution. Use blood density there when
different from tissue. This MMS uses the artificial verification density
$\rho_s=1$. Retained mass is not a three-dimensional incompressible blood solver.

Originally the extended-solid multiplier balanced fictitious interior elastic,
viscous and inertial reactions. Its negative supplied effective pressure feedback
for the extended-solid problem, rather than independently establishing pressure
for a hollow solid. Full removal of the interior volume equations
was considered but would also remove mass and could create internal null modes.
Keeping viscosity was another possible model. The selected model keeps mass and
removes only projected elasticity and viscosity. Radial-only removal was rejected:
its affine elastic remainder
$4\lambda_sgh+(\lambda_s+2\mu_s)h^2$ can be negative. Both reciprocal
cross-terms are retained here.

## Independent manufactured target

The cube is $\Omega=[-\ell,\ell]^3$, $\ell=0.120685$, $L=2\ell$.
The cylinder axis is $(x,0,0)$, $A_0=3.0605\,10^{-4}$ and $a^2=A_0/\pi$.
Set $\lambda_s=\mu_s=\rho_s=1$, $\eta_s=0.1$, $\beta_R=0.2$ and
Rayleigh mass damping $\alpha_R=0$. Define

$$
g=t^3,\quad h=2t^4,\quad
W=\frac{a^2}{\max(y^2+z^2,a^2)}(0,y,z).
$$

An independent continuum solution and body force are

$$
u^\star=(hx,gW_y,gW_z),\quad
v^\star=(\dot h x,\dot gW_y,\dot gW_z),\quad
f^\star=\rho_s(\ddot h x,\ddot gW_y,\ddot gW_z).
$$

Displacement is continuous but its radial derivative jumps. The exact interior
strain lies in the two modes, so its effective elastic and viscous stress is zero.
Outside the cylinder, let

$$
K=2\mu_sg+(\eta_s+2\beta_R\mu_s)\dot g,\quad
B=\lambda_sh+\beta_R\lambda_s\dot h,\quad
C=(\lambda_s+2\mu_s)h+
[\eta_s+\beta_R(\lambda_s+2\mu_s)]\dot h.
$$

The nonzero exterior stresses are

$$
\sigma_{xx}=C,\quad
\sigma_{yy}=B+Ka^2(z^2-y^2)/r^4,\quad
\sigma_{zz}=B-Ka^2(z^2-y^2)/r^4,\quad
\sigma_{yz}=\sigma_{zy}=-2Ka^2yz/r^4.
$$

Stress divergence vanishes in both subdomains away from the interface, leaving
the prescribed body force to supply acceleration. Radial jump balance gives

$$
\lambda^\star=B-K=2t^4-0.4t^3-1.5t^2.
$$

For $h=0$, this becomes $-2t^3-1.5t^2$; without viscosity it is $-2t^3$.
The old $-6t^3$ target included fictitious elasticity and is stale for this model.

Dirichlet cases prescribe exact displacement and velocity on every face.
Neumann cases prescribe the exterior tensor times the outer normal. On the end
planes the effective axial traction is $Cn_x$ outside the disk and zero inside.
The axial correction cancels the native interior endcap load. Initial fields are
zero. At $T=0.1$, $q_r=0.0328$, $q_z=0.0202$,
$\lambda^\star=-0.0152$ and $C=0.0062$ Pa.

## Source-free flow and energy oracles

The compatibility and flow targets remain

$$
A^\star=A_0(1+g)^2,\quad Q^\star=(\ell-x)\dot A^\star,\quad
U^\star=(\ell-x)\frac{6t^2}{1+t^3}.
$$

They satisfy $\partial_tA^\star+\partial_x(A^\star U^\star)=0$.
The inlet supplies $L\dot A^\star$, the reflected outlet flux is zero, and added
volume is $LA_0[(1+t^3)^2-1]$. Axial solid strain introduces neither a moving
one-dimensional mesh nor a continuity source. Pressure is spatially uniform;
with the retained native tube-law/material data the total pressure is
$P^\star=p_{\rm tube}(A^\star)-\lambda^\star$. Native momentum is

$$
\partial_tU+U\partial_sU+\rho_f^{-1}\partial_sP=S_U.
$$

The uniform pressure still requires cancellation of native volume, face and
boundary pressure contributions. With zero flow friction the velocity source is

$$
S_U=(\ell-x)\frac{12t-42t^4}{(1+t^3)^2},\qquad S_A=0.
$$

This uniform pressure does not test every pressure-gradient contribution.
Independent pressure, residual-row and Jacobian checks remain necessary.

An analytical energy oracle uses the exterior square cross-section:

$$
A_{\rm ext}=4\ell^2-\pi a^2,\qquad
I=\pi a^2-\frac{a^4(\pi+2)}{2\ell^2}.
$$

$$
\begin{aligned}
E_{\rm eff}^\star&=L[2\mu_sg^2I+(\mu_s+\lambda_s/2)h^2A_{\rm ext}],\\
E_f^\star&=A_0L\psi(g,h),\\
D_{\rm eff}^\star&=L[2(\eta_s+2\beta_R\mu_s)\dot g^2I+
(\eta_s+\beta_R(\lambda_s+2\mu_s))\dot h^2A_{\rm ext}].
\end{aligned}
$$

Removed viscous dissipation is $A_0L(\dot g,\dot h)^TH_v(\dot g,\dot h)$.
Discrete removed energy instead uses $V_h$ and projected strains. Native elastic
energy includes the removed modes and is not the effective corrected energy.

Retained kinetic energy is also explicit. With Catalan constant
$G=0.915965594177219015$ and
$J=a^4[\pi/2+2\pi\log(2\ell/a)-4G]$, the full-cube value is

$$
T^\star=\frac{\rho_s}{2}\left[
\dot h^2\frac{8\ell^5}{3}+\dot g^2LJ\right].
$$

The interior disk contributes to $J$ and remains in this kinetic energy.

## Configure, run and assess convergence

Templates in tutorials/metric_flow_x_elastodynamics/ have separate outputs:

| Template | Solid cells | Centerline cells | Purpose |
| --- | ---: | ---: | --- |
| mms_cube.prm.in | $4\times16\times16$ | 2 | Two-mode base case |
| mms_cube_refined.prm.in | $8\times32\times32$ | 4 | Second spatial level |
| mms_cube_finer.prm.in | $16\times64\times64$ | 8 | Third spatial level |
| mms_cube_tight.prm.in | $4\times16\times16$ | 2 | Tolerance sensitivity |
| mms_cube_radial.prm.in | $4\times16\times16$ | 2 | $h=0$ reduction |
| mms_cube_neumann.prm.in | $4\times16\times16$ | 2 | Endpoint traction |
| mms_cube_neumann_refined.prm.in | $8\times32\times32$ | 4 | Refined Neumann |
| mms_cube_mpi.prm.in | $4\times16\times16$ | 2 | Optional MPI, iterative |

Serial defaults use UMFPACK, without MUMPS. The separate mms_cube_mumps.prm.in
requires MUMPS. All cases use degree one. The DG multiplier mesh is coarser
than the continuous solid axial trace mesh. Composite representative quadrature
uses four Gauss points repeated twice per centerline cell; wall refinement
increases from two to three to four across spatial levels.

Configure deal.II with SUNDIALS and VTK and a compatible installed MetricFlowX:

~~~bash
cmake -S . -B build-mms -DCMAKE_BUILD_TYPE=Release \
  -DMetricFlowX_DIR=/path/to/metric-flow-x/lib/cmake/MetricFlowX
cmake --build build-mms --target metric_flow_x_elastodynamics -j2
python3 scripts/mfx_modal_convergence.py --build build-mms --run \
  --cases base refined finer
~~~

Require the MetricFlowX support enabled configuration message. Supply
DEAL_II_DIR if needed. For Debug pass
--executable build-mms/metric_flow_x_elastodynamics_debug to the runner.
It resolves all paths and runs in dedicated build-tree directories, producing
a JSON report of final times, solid displacement/velocity $L^2$ errors,
area/native-velocity/multiplier errors, added volume, volume defect and exact
continuum energy targets. It also reports projected strain/rate errors and
sampled-volume error from the native energy diagnostics. Logarithmic error
reductions are descriptive measurements, not evidence of an asymptotic rate
when sampled geometry is nonmonotone. Only the Python standard library is needed.

~~~bash
python3 scripts/mfx_modal_convergence.py --build build-mms --run \
  --cases tight radial
~~~

The report is build-mms/test_output/mfx-modal-convergence/report.json.
Omit --run to reprocess saved runner output. Generated parameter files also run
directly without path editing:

~~~bash
cd build-mms
./metric_flow_x_elastodynamics tutorials/metric_flow_x_elastodynamics/mms_cube.prm
mpirun -np 2 ./metric_flow_x_elastodynamics \
  tutorials/metric_flow_x_elastodynamics/mms_cube_mpi.prm
~~~

The postprocessor integrates exported Cartesian Q1 solid fields with three-point
Gauss quadrature per direction and linear centerline fields with three-point
Gauss quadrature. It rejects unsupported cells. Export precision and quadrature
of the unfitted strain jump limit these reproducible error diagnostics.

The tight case reduces adaptive IDA tolerances by 100 on the base spatial mesh.
This is tolerance sensitivity, not fixed-timestep convergence. Stopping at saved
output times preserves saved algebraic states but does not fix internal steps.
Separate time error, sampled-cylinder volume, wall geometry and spatial error
before claiming rates. Solver completion, smoother plots or serial/MPI agreement
alone do not establish convergence, inf-sup stability or physiological validity.
Historical results for the uncorrected, zero-viscosity $h=0$ model do not validate
this corrected model and are not reused as its error table.
