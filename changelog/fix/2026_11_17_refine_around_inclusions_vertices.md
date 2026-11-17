PR: #462

- `CoupledElasticityProblem::refine_and_transfer_around_inclusions()` now
  refines the cells around the vertices of each inclusion cell. It passed
  local vertex numbers to `GridTools::find_cells_adjacent_to_vertex`, which
  expects global vertex indices, so the refined region depended on the vertex
  numbering and the partition.
- The two layers of neighbour cells are now flagged on the cells each rank
  owns. Refine flags set on ghost cells were ignored, so neighbours owned by
  another rank were not refined.
