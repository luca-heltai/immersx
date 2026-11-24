PR: pending

- The per-cell `update_vessel_radius_data` with point coupling checks the number of samples against the cells of each vessel in Release builds too. Before, a mismatch read outside the data.
- `coupled_elasticity` passes an exception of the 1D model or of the data exchange from rank zero to all ranks, so that an MPI run stops with the message instead of waiting in the next collective call.
