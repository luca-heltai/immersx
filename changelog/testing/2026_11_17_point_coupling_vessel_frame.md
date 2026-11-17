PR: #457

- Added `gtests/elasticity_point_coupling_frame_01.cc`: for a rigid
  translation, the point-coupling constraint with boundary data and no data
  line holds on a vessel along z and on a tilted vessel.
- Added `ElasticityTensorProductCouplingValidation.PointBoundaryFunctionMatchesDataLineAndTensorProduct`:
  the inflated vessel along x, imposed through the point-coupling boundary
  function, gives the same solution as the data line and matches tensor
  coupling.
