# Configure reduced coupling

Reduced coupling represents a lower-dimensional geometry in a bulk finite-
element problem. The reduced-coupling section controls the representative
grid, thickness, transverse cross-section basis, quadrature, and particle
search. The [Reduced Poisson tutorial](../tutorials/reduced-poisson) introduces
these choices through a minimal cylinder; use this page when changing the
configuration.

The important distinction is between:

- the reduced or representative dimension, where coefficients are stored;
- the represented physical support, where the lifted field is evaluated; and
- the ambient `space dimension`, where the support is embedded.

Imported point and cell fields can be named in expressions. A variable radius,
for example, is a field in the input mesh and is selected by the `Input file
fields` option before it is used by `Thickness`. Distributed point ownership
and search are handled by the particle-coupling infrastructure.
