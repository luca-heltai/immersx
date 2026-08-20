# Standalone elastodynamics convergence cases

These three inputs are first-order-in-time convergence cases based on the MMS
fields documented under `tutorials/elasticity/`:

- `strong_dirichlet.prm`
- `neumann.prm`
- `kelvin_voigt.prm`

The standalone `ElastodynamicsSolver` deliberately implements essential
Dirichlet data only. Consequently, the Neumann input retains its manufactured
field but uses strong Dirichlet constraints on every boundary; it is not a
test of a Neumann operator. The Kelvin--Voigt input exercises the assembled
viscous operator through a nonzero velocity field.

The strong and Neumann cases use a time-independent manufactured displacement
and one backward-Euler step. The Kelvin--Voigt case uses `d=t*phi` and
`v=phi`, with nonzero shear damping. All cases use five global mesh levels.
The convergence tables therefore measure spatial displacement error while
exercising the first-order block solve and the separate mass/stiffness/damping
operators.

Run, for example:

```text
./build/elastodynamics_debug tutorials/elastodynamics/strong_dirichlet.prm
```

The tables are printed on rank zero. Output fields are disabled in these
inputs; the ignored `output/elastodynamics_convergence/` directories only
contain the used parameter files.
