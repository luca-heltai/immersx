# Overview

ImmersX is a deal.II-based C++ framework for finite-element problems on bulk,
embedded, and mixed-dimensional geometries. The repository includes scalar
Poisson problems, static and dynamic elasticity, incompressible flow, reduced
Poisson coupling, and coupled 3D/1D applications.

Applications are built from three kinds of code:

- a Problem or native provider owns a discretization and its physical data;
- an Observable or Interaction describes a quantity or relation between
  fields;
- an execution adapter assembles the semantic contributions and applies a
  solver or time-integration policy.

The [architecture page](architecture) defines these concepts and their
ownership. The [mathematical background](mathematical-background) explains the
weak formulations used by the reduced and immersed coupling code. Runnable
examples are listed in the [tutorials](../tutorials/index), and executable
capabilities are listed in the [application reference](../reference/applications).

The main physical and coupling headers are under `include/immersx/physics/`
and `include/immersx/coupling/`. The [generated API reference](../api/library_root)
contains the complete public signatures.
