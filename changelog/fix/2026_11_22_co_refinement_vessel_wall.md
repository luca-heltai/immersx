PR: #494

- Tensor-product co-refinement (`ReducedCoupling` and `ReducedPoisson`) pairs space cells with the vessel surface at the `Thickness` distance from each 1D cell instead of with the cell's bounding box. `Space post-refinement cycles` and the ratio-driven passes now refine around the vessel wall, so they can be used to converge wall quantities (#465).
- `TensorProductSpace::preprocess_serial_triangulation` takes a second argument that returns the thickness range on a serial reduced cell.
