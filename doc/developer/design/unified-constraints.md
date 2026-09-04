# Unified weak-term constraints

An algebraic constraint is assembled from ordinary `Field` and `Observable`
objects. A Lagrange multiplier is an algebraic `Field` on its own finite-
element space; it is not a `Problem` and does not borrow a participant's
DoFHandler.

For example, two fields can be constrained through one independently chosen
multiplier space:

```{code-block} cpp
auto lambda = multiplier_space.field("lambda");
auto first  = weak_term(value(u1), lambda);
auto second = weak_term(value(u2), lambda);
auto constraint = make_constraint(first - second, rhs);

auto fields = adapter.add(constraint, "continuity");
```

The constraint contributes the signed weak-term sum minus `rhs` to the
multiplier row. Each participant receives the corresponding transpose
reaction. Scalar and vector constraints use the same API; the finite element
and extractor determine the value type.

The target multiplier `Field` can use a different DoFHandler and finite
element from every participant. When the geometries are shared, weak-term
assembly uses the direct cell or paired-cell path. Only nonmatching geometry
uses the cached particle backend. These choices are implementation details and
are not part of application-level constraint construction.
