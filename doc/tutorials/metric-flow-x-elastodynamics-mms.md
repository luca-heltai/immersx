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
support a branching-network correction. The solid supports prescribed
Dirichlet values and hanging-node constraints; other multipoint constraint
lines are rejected. Interior elastic and viscous coefficients must be homogeneous
and match the configured default material.
[Pressure on the vessel walls](../concepts/vessel-pressure.md) compares this
correction with the analytic interior stress that `VesselElasticityProblem`
adds to its reported pressure.

## Model and pressure normalization

The frozen reference radius is $a=R_0=\sqrt{A_0/\pi}$. Geometric $A_0$ differs
from the tube law's constitutive $A_{\rm ref}$: the two areas need not coincide,
and resting pressure need not be zero. The interior affine displacement is

$$
u_r=gr,\qquad u_x=hx,\qquad \varepsilon(u)=\mathrm{diag}(h,g,g).
$$

The native stress is

$$
\sigma_e=2\mu_s\varepsilon(u)+\lambda_s\mathrm{div}(u)I,\qquad
\sigma_v=(\eta_s+2\beta_R\mu_s)\varepsilon(v)
+\beta_R\lambda_s\mathrm{div}(v)I.
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
\mathrm{tr}\varepsilon(u_q)-t\cdot\varepsilon(u_q)t],\qquad
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
$\rho_s=1$, while MetricFlowX retains $\rho_f=1060$. No density is changed by
the correction. The manufactured total pressure is uniform in space, so its
pressure-gradient term vanishes for this fluid density too. Retained mass is
not a three-dimensional incompressible blood solver.

Keeping all native mass also keeps axial extension inertia. The lumen's axial
solid velocity is not constrained to the one-dimensional flow velocity $U$.
There is no three-dimensional blood incompressibility constraint, axial
no-slip condition or ALE compatibility. Retained mass is an effective
extension/closure, not a consistent reconstruction of blood momentum by
construction; overlap with the one-dimensional axial inertia is not ruled out.
The lack of one-dimensional transverse momentum exchange motivates transverse
inertia, but does not alone justify this additional axial inertia.

The affine interior extension and the distinction between pressure and
stress-jump forcing are derived by [Heltai and Caiazzo](https://doi.org/10.1002/cnm.3264);
see the [author preprint](https://arxiv.org/pdf/1908.11243), sections 2.2–2.3,
equations (5)–(8). Their finite outer-cylinder zero-Dirichlet solution contains
both inverse-radius and linear-radius terms. This cube MMS instead prescribes
nonzero boundary values and uses an independently manufactured exterior
$a^2/r$ profile; it is not that finite outer-cylinder solution.

Originally the extended-solid multiplier balanced fictitious interior elastic,
viscous and inertial reactions. Its negative supplied effective pressure feedback
for the extended-solid problem, rather than independently establishing pressure
for a hollow solid. Full removal of the interior volume equations
was considered but would also remove mass, could create internal null modes,
and can introduce cut/independent-quadrature conditioning problems.
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

The input functions use numerical time measured in seconds. In dimensional
notation, $g=(t/t_0)^3$ and $h=2(t/t_0)^4$ with $t_0=1\,{\rm s}$, so
$\dot g=3(t/t_0)^2/t_0$ and $\dot h=8(t/t_0)^3/t_0$.
The polynomial expressions below use $t_0=1$ numerically, as do the parameter
files; their numerical values are unchanged.

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
| mms_cube.prm.in | $4\times16\times16$ | 1 | Two-mode base case |
| mms_cube_refined.prm.in | $8\times32\times32$ | 2 | Second spatial level |
| mms_cube_finer.prm.in | $16\times64\times64$ | 4 | Third spatial level, iterative |
| mms_cube_tight.prm.in | $4\times16\times16$ | 1 | Tolerance sensitivity |
| mms_cube_radial.prm.in | $4\times16\times16$ | 1 | $h=0$ reduction |
| mms_cube_neumann.prm.in | $4\times16\times16$ | 1 | Endpoint traction |
| mms_cube_neumann_refined.prm.in | $8\times32\times32$ | 2 | Refined Neumann |
| mms_cube_mpi.prm.in | $4\times16\times16$ | 1 | Optional MPI, iterative |
| mms_cube_finer_original_pairing.prm.in | $16\times64\times64$ | 8 | Original pairing diagnostic |

Base, refined and small verification cases use UMFPACK. Finer uses factored
iteration and supports serial execution; the measured finest recipe uses four
MPI ranks without MUMPS. The iterative case compares solvers on the base mesh.
The separate mms_cube_mumps.prm.in requires MUMPS.

All cases use degree one. Each ordinary flow cell spans four solid axial cells.
Axial wall quadrature uses four Gauss points on each of four equal subintervals,
one per solid cell. Wall refinements increase from two to three to four across
the spatial levels. The original pairing is retained only as a diagnostic.

Configure deal.II with SUNDIALS and VTK and a compatible installed MetricFlowX.
The correction requires the native `add_external_pressure_state_jacobian()`
API as well as the external-pressure residual and pressure-coefficient Jacobian
APIs. Pressure-dependent HLL weights contribute to the state derivative even
when the imposed pressure is held fixed. Omitting that derivative gives an
incomplete coupled Jacobian. An incompatible package is rejected before mesh
setup when this correction is enabled; the legacy option remains available.
Rebuild ImmersX with matching MetricFlowX headers and libraries after updating
the package. The reproducibility revision is MetricFlowX main
[`e98b2333ed41d0fcef027849e0efb460d9ada64d`](https://github.com/luca-heltai/metric-flow-x/commit/e98b2333ed41d0fcef027849e0efb460d9ada64d)
(version 1.0.0), installed separately from earlier packages.


~~~bash
cmake -S . -B build-mms -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_COUPLED_PROBLEMS=OFF \
  -DMetricFlowX_DIR=/path/to/metric-flow-x/lib/cmake/MetricFlowX
cmake --build build-mms --target metric_flow_x_elastodynamics -j2
python3 scripts/mfx_modal_convergence.py --build build-mms --run \
  --cases base refined
~~~

Require the MetricFlowX support enabled configuration message.
ENABLE_COUPLED_PROBLEMS=OFF excludes the separate optional 1dsolver applications;
it does not disable MetricFlowX. Supply
DEAL_II_DIR if needed. For Debug pass
--executable build-mms/metric_flow_x_elastodynamics_debug to the runner.
It resolves all paths and runs in dedicated build-tree directories, producing
a JSON report of final times, solid displacement/velocity $L^2$ errors,
area/native-velocity/multiplier errors, separate radial and axial solid errors,
added volume, volume defect and exact
continuum energy targets. It also reports projected strain/rate errors and
sampled-volume error from the native energy diagnostics. Logarithmic error
reductions are descriptive measurements, not evidence of an asymptotic rate
when sampled geometry is nonmonotone. Only the Python standard library is needed.

~~~bash
python3 scripts/mfx_modal_convergence.py --build build-mms --run \
  --cases tight radial
~~~

For MPI, use the same output layout and postprocessor without a custom wrapper:

~~~bash
python3 scripts/mfx_modal_convergence.py --build build-mms --run \
  --cases finer --mpi-ranks 4
python3 scripts/mfx_modal_convergence.py --build build-mms --cases finer
~~~

The default is one rank with no launcher. More than one rank is supported for
the iterative finer, finer-original, iterative and mpi cases; the runner rejects direct-solver
cases before starting. It finds mpirun or mpiexec on PATH, which must match the
MPI installation used by deal.II and MetricFlowX. No MUMPS is needed.
Each run records its argument list and rank count in run_metadata.json;
reprocessing retains that execution metadata. Supply separate --output paths
for independent serial/MPI comparisons.

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

The tight case reduces adaptive IDA tolerances by 10 on the base spatial mesh.
This is tolerance sensitivity, not fixed-timestep convergence. Stopping at saved
output times preserves saved algebraic states but does not fix internal steps.
Separate time error, sampled-cylinder volume, wall geometry and spatial error
before claiming rates. Solver completion, smoother plots or serial/MPI agreement
alone do not establish convergence, inf-sup stability or physiological validity.
Historical results for the uncorrected, zero-viscosity $h=0$ model do not validate
this corrected model and are not reused as its error table.


## Measured spatial and tolerance errors

These Release measurements use deal.II 9.8.0 and the pinned MetricFlowX
revision above. All three ordinary cases reach $T=0.1$ with common IDA
absolute/relative tolerances $10^{-9}/10^{-7}$, no field overrides, and enforced
output stops at 0, 0.05 and 0.1 s. The independent target remains
$\lambda^\star=-0.0152$ Pa. Field errors are $L^2$ errors expressed as percentages of the corresponding
analytical target norm. Transverse denotes the $(y,z)$ vector components.

| Solid cells | Flow cells | Full displacement error | Transverse displacement error | Transverse velocity error | Multiplier $L^2$ error | Mean multiplier [Pa] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| $4\times16\times16$ | 1 | 12.6239 | 102.6355 | 103.2000 | 538.7894 | -0.0970959961 |
| $8\times32\times32$ | 2 | 2.6550 | 21.6981 | 21.8717 | 91.3982 | -0.0287964060 |
| $16\times64\times64$ | 4 | 1.5406 | 12.5917 | 12.6652 | 55.0186 | -0.0233295346 |

Axial displacement errors are 1.5840%, 0.2664% and 0.1236%. The much smaller
transverse target norm explains why full-vector errors alone understate radial
error. Native area errors are $2.8100\times10^{-5}$%,
$2.4758\times10^{-5}$% and $2.2906\times10^{-5}$%; native velocity errors are
$2.7723\times10^{-5}$%, 0.0007419% and 0.0008300%. Thus not every flow error
decreases under refinement.

| Solid axial cells | Projected $g_h$ | Projected $\dot g_h$ | $V_h/(A_0L)-1$ [%] | Added-volume target defect [%] |
| --- | ---: | ---: | ---: | ---: |
| 4 | 0.00107257361 | 0.0321731279 | -25.6409 | 0.0137296 |
| 8 | 0.000945325679 | 0.0283631156 | +11.5387 | 0.0113534 |
| 16 | 0.000989720631 | 0.0296910081 | +2.2438 | 0.0101653 |

The independent targets are $g=0.001$ and $\dot g=0.03$. The axial projected
modes reproduce $h=0.0002$ and $\dot h=0.008$ near roundoff in these Dirichlet
cases; the computed axial fields are not exact. Added-volume defect compares
$\Delta V_h$ with the exact MMS inlet integral. It is not an independent audit
of the numerical boundary flux.

| Solid axial cells | Effective elastic-energy error [%] | Effective dissipation error [%] | Retained kinetic-energy error [%] |
| --- | ---: | ---: | ---: |
| 4 | 71.1393 | 42.9777 | 2.0200 |
| 8 | 11.1940 | 6.6950 | 0.3164 |
| 16 | 6.9076 | 4.1417 | 0.1773 |

These are instantaneous comparisons, not a discrete energy-balance proof.
The finest multiplier error remains 55%; the sampled geometry bias changes
sign. The observed reductions do not establish an asymptotic order, general
coupled stability or physiological validity.

The finest measurement uses four MPI ranks. Observed per-rank high-water marks
are 2.994 to 3.016 GiB; aggregate RSS sampled once per second peaks at 12.010 GiB.
Summing rank high-water marks gives a 12.010 GiB bound; shared mappings can be
counted repeatedly. Elapsed application time was 43 min 30 s with concurrent
work on the node. These are resource measurements, not an MPI scaling study.

### Tolerance sensitivity

The supplied tight case uses the same base mesh, quadrature and output stops,
with common tolerances reduced to $10^{-10}/10^{-8}$. No per-field override
replaces those controls.

| Control | Multiplier $L^2$ error [%] | Mean multiplier [Pa] | Added-volume target defect [%] |
| --- | ---: | ---: | ---: |
| Ordinary | 538.789448 | -0.0970959961 | 0.0137296 |
| Tight | 538.781900 | -0.0970948488 | 0.0042248 |

Direct field-to-field differences are 0.012318% of the displacement target norm,
0.00118586% for velocity, and 0.00754861% for the multiplier. The mean multiplier
changes by $1.1474\times10^{-6}$ Pa. Projected $g_h$ changes from
0.00107257361 to 0.00107247293 and $\dot g_h$ from 0.0321731279 to 0.0321731316;
the axial means remain at their targets. Effective elastic-energy error changes
from 71.1393% to 71.1227%, kinetic-energy error from 2.0200% to 2.0217%, and
dissipation error remains about 42.9777%. This is adaptive tolerance sensitivity,
not a fixed-timestep order study or improvement in every diagnostic.

Separate tests of explicit absolute tolerances $10^{-8}$ m for displacement,
$10^{-7}$ m/s for velocity and $10^{-6}$ Pa for the multiplier reduced work,
but increased the original base added-volume target defect from 0.01373% to
0.30838%. At the same finest saved time $T=0.05$, original strict and relaxed
multiplier errors were 268.505% and 267.418%, while volume defects were 0.05273%
and 0.53293%. Their stop histories differed. These overrides have not become
ordinary defaults.

## Why the original pairing oscillates

The original sequence used 2/4/8 flow cells with two solid axial cells per
flow cell. With prescribed end values, its actual base wall-traction operator
has rank 3 for 4 multipliers. The cellwise-linear vector $q=(1,-1,1,-1)$ has
relative traction residual $1.84\times10^{-16}$. However, the native
flow-pressure block acts on $q$ with normalized residual 0.7071. This is not a
null column of all coupled rows and does not prove a singular DAE.

The saved pressure component along this exact $q$ accounts for less than
0.000011% of squared pressure error in the finest $T=0.05$ controls. The large
pressure fluctuations cannot be attributed to that component alone.

A more specific compatibility constraint is visible in the assembled operator.
Let $r_j$ be the polygon-weighted radial trace at axial plane $j$, normalized
by the uniform target $gR_0$. The original two-solid-cell macrocell gives

$$
5r_0+6r_1+r_2=12,\qquad r_0+6r_1+5r_2=12.
$$

Subtracting these rows forces $r_0=r_2$. Successive macrocells propagate the
interpolated end trace $b$ through every even plane, with odd planes $2-b$.
Actual base traces are approximately
$[0.364696,1.634373,0.367061,1.634374,0.364696]$.
Including the small measured nonuniform-area moments reproduces them within
$2.47\times10^{-5}$; the refined counterpart agrees within $1.59\times10^{-5}$.
The radial derivative jump and prescribed end interpolation therefore generate
mesh-scale trace variation even when the projected mean strain is accurate.

The ordinary sequence now uses four solid axial cells per flow cell. Its base
moment rows are proportional to $[11,18,12,6,1]$ and $[1,6,12,18,11]$:
their difference includes interior values and no longer propagates the end
trace to every macrocell endpoint. Four axial quadrature repetitions preserve
the original physical points and weights; the measured traction embedding
agrees within $5.65\times10^{-16}$. Uniform area, linear velocity and constant
multiplier targets remain representable. No material, density, source or
analytical oracle is changed.

| Solid axial cells | Original multiplier error [%] | Ordinary multiplier error [%] |
| --- | ---: | ---: |
| 4 | 712.7506 | 538.7894 |
| 8 | 195.4006 | 91.3982 |
| 16 | 289.6060 | 55.0186 |

Both columns use common tolerances $10^{-9}/10^{-7}$ and 0.05 s output stops.
The pairing does not improve every field: base axial displacement error rises
from 0.3660% to 1.5840%. It removes the demonstrated trace identity and reduces
the measured multiplier errors without establishing a general inf-sup result.

The optional finer-original case retains the original finest pairing:

~~~bash
python3 scripts/mfx_modal_convergence.py --build build-mms --run \
  --cases finer-original --mpi-ranks 8 --output build-mms/test_output/original-pairing
~~~

Independent reconstruction of saved-field virtual work closes within
$1.3\times10^{-5}$ and $4.8\times10^{-5}$ on the original two Dirichlet meshes,
with compatible DG1 moments. This gives no evidence for an extra $2\pi R_0$
factor. It uses Float32 fields, polygon quadrature and acceleration estimated
from saved velocities; it is not an exact solver-residual or stability proof.

Native area or total-pressure accuracy alone does not certify the smaller
traction amplitude. Near $T=0.05$, the native tube-law derivative is about
$1.0592\times10^8$ Pa/m²; relative area error $10^{-6}$ changes tube pressure
by about 0.0324 Pa, against $|\lambda^\star|=0.0037875$ Pa. Reconstructing that
difference from Float32 area is precision-limited; the reported multiplier
errors use its directly exported field.

The optional mms_cube_finer_direct.prm.in case is outside the ordinary sequence.
An earlier SDK 0.1.0 diagnostic stopped its initial UMFPACK factorization after
15 min 21 s at 59.7 GiB peak RSS, before an accepted step. It is not a completed
measurement of the current package or ordinary pairing.

## Co-refinement on the physical lift

The `local`, `local-refined` and `local-finer` cases start from one solid
cube and one fluid cell, with zero initial refinement. They use the continuum
solution, material data and time controls above. The fluid mesh is selected
by the compatibility loop rather than prescribed as a spatial level.
Area, native velocity and multiplier spaces remain DG1.

~~~text
subsection MetricFlowX coupling
  subsection Co-refinement
    set Enabled = true
    set Refinement strategy = both
    set Lower factor = 0.25
    set Upper factor = 4
    set Max refinement level = 8
    set Max section refinement level = 6
    set Space pre-refinement cycles = 0
    set Embedded pre-refinement cycles = 0
    set Space post-refinement cycles = 0
    set Embedded post-refinement cycles = 0
  end
end
~~~

`Enabled` defaults to false. The current path uses native VTK flow input,
a hexahedral solid and a positive imported `a0` field. Generated and archived
network preparation is separate. The `space` strategy permits solid refinement;
`embedded` permits fluid and section refinement; `both` permits all three.

### Compatibility criterion

For each active fluid cell, build the same representative and section
quadrature used by the final physical lift, evaluating the retained reference
area at the current source-cell quadrature points. Locate the solid cells
that contain each lifted point using a cell-box search followed by inversion
of the solid's linear mapping. Shared-face ties can reach more than one cell.
There is no radius expansion of the search boxes.

For every reached solid cell, compare three physical lengths:

- $h_s$: diagonal of the solid cell's axis-aligned bounding box;
- $h_a$: length of the source fluid cell;
- $h_\theta$: positive physical integration length carried by the section point,
  computed as its lifted surface JxW divided by its representative axial JxW.

Thus $h_a$ follows the fluid mesh, while $h_\theta$ changes with the section mesh,
quadrature degree and quadrature repetitions. The radius enters the physical
section transform; it is not a replacement for section resolution.
Both ratios must satisfy the open interval
$L<h_a/h_s<U$ and $L<h_\theta/h_s<U$, with the configured lower and upper factors.
Equality requests refinement.

A ratio at/below $L$ marks the reached solid cell. An axial ratio at/above $U$
marks the source fluid cell. A section ratio at/above $U$ requests one global
section refinement, independently of the fluid mesh. The solid pass is applied
first; the refined solid is queried before deciding on fluid/section refinement.
After fluid refinement, source quadrature points, reference areas and lifted
supports are rebuilt. The loop stops when no permitted pass can change a mesh.

The mesh-level cap limits adaptive solid and fluid passes. The section cap
limits adaptive circumferential passes. Explicit global pre/post refinement
and local solid post passes follow the ordinary refinement-parameter convention
and can exceed an adaptive cap. A cap or a one-sided strategy can leave
violations; the final counts retain them rather than declaring compatibility.
Coverage concerns quadrature points, not every point on the swept surface.
These scale tests do not establish an inf-sup condition.

### Mesh and field lifecycle

Both domains stay in their preparation lifecycle throughout the loop.
Point fields follow FE_Q(1) interpolation and cell fields inherit their
FE_DGQ(0) values through refinement. The radius is evaluated from this retained
data; the VTK file is not reread to replace refined fields. The solid and fluid
are finalized only after the last comparison, before the native application
spaces, multiplier space and coupling operators are built.

The solid uses vertex level-difference balancing without island-removal
smoothing during co-refinement. Hanging nodes use the same homogeneous mesh
condensation in native matrices, loads and modal moment vectors. Prescribed
boundary columns remain. Eliminated algebraic slots carry only constraint
inhomogeneities; accepted physical state and FE output reconstruct the full
mesh constraints. The projection retains physical mass.

### Convergence inputs and diagnostics

The three cases request initial section refinement levels 0/1/2 with a
two-point Gauss rule per section cell. All use the same ratio interval and
start from a 1×1×1 solid. The axial quadrature remains four Gauss points with
four repetitions per fluid cell. The adaptive loop must refine both physical
meshes in every case. The two coarser cases use serial UMFPACK; the finest uses
factored iteration.

~~~bash
python3 scripts/mfx_modal_convergence.py --build build-mms --run \
  --cases local local-refined local-tight
python3 scripts/mfx_modal_convergence.py --build build-mms --run \
  --cases local-finer --mpi-ranks 4
python3 scripts/mfx_modal_convergence.py --build build-mms \
  --cases local local-refined local-finer local-tight
~~~

Each case writes separate build-tree output. The `flow/co_refinement.csv` and
JSON report contain the initial/final cell counts, active solid levels,
hanging-node DoFs, solid/fluid/section passes, final section level, physical
point/cell pair count, coverage failures, both ratio ranges and all four
violation counts. The local runner checks that both meshes changed and that
both ratio intervals are satisfied. The public CTest smoke cases use
`--final-time 0.02`; the convergence commands retain $T=0.1$.

The `local-tight` case changes all base time tolerances by one order of
magnitude at fixed spatial inputs. Comparing spatial levels changes the
solid, fluid and section together, so an error reduction does not isolate
one discretization error.


### Measured co-refinement sequence

The following runs reached $T=0.1$ on gpu04-codex with a Release build,
deal.II 9.8 and MetricFlowX 1.0.0 at
`e98b2333ed41d0fcef027849e0efb460d9ada64d`. The first two levels used serial
UMFPACK; the last used four MPI ranks and the factored iterative solver.
These are the bidirectional runs, with both physical meshes initially having
one cell. No radius padding or extra local post passes were used.

| Case | Solid cells | Active solid levels | Solid DoFs | Hanging DoFs | Fluid cells | Section level |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `local` | 400 | 2–4 | 1947 | 912 | 4 | 0 |
| `local-refined` | 848 | 2–5 | 3963 | 2088 | 8 | 1 |
| `local-finer` | 5104 | 2–6 | 20523 | 8568 | 16 | 2 |

Every case has zero uncovered lift points and zero violations in all four
directional checks. The axial ratio is 2.30940; the section ratios are
0.267105, 0.289113 and 0.294777, respectively, inside $(0.25,4)$.
The multiplier remains DG1, with 8, 16 and 32 DoFs respectively.

The relative $L^2$ errors are dimensionless fractions:

| Case | Displacement | Solid velocity | Area | Fluid velocity | Multiplier |
| --- | ---: | ---: | ---: | ---: | ---: |
| `local` | 1.67008e-1 | 1.26267e-1 | 2.14001e-6 | 1.93200e-4 | 1.04015e1 |
| `local-refined` | 2.17051e-2 | 1.65046e-2 | 3.47648e-7 | 2.16763e-5 | 9.59823e-1 |
| `local-finer` | 1.56724e-2 | 1.18697e-2 | 4.49357e-7 | 2.66362e-5 | 7.25085e-1 |

The solid and multiplier errors decrease at both transitions, but their second
reduction is much weaker. The finest multiplier error is still 72.5%; area
and fluid-velocity errors increase slightly in the last transition. This
sequence does not establish an asymptotic rate or converged pressure accuracy.
The absolute sampled reference-volume errors decrease from 25.64% to 11.54%
and 2.24%; they measure a separate geometric quadrature error.

At fixed base spatial inputs, `local-tight` changes the displacement $L^2$
error by 0.0118% and the multiplier error by 0.00117% relative to `local`.
The largest relative change among the five reported $L^2$ errors is 0.675%
for area. This is a base-mesh tolerance check, not a temporal error bound for
the finest MPI run. The [machine-readable measurements](../_static/metric-flow-x-corefinement/measurements.json)
retain the unrounded errors, mesh diagnostics, solver/rank metadata and code
versions. The commands above reproduce the complete four-case `report.json`.
