PR: #459

- Added `gtests/vessel_multiplier_convergence_01.cc` (nightly, validation
  suite `VesselMultiplierValidation`): an immersed vessel in a cube with a
  manufactured field, refined near the wall. It checks that the multiplier
  converges to the exact traction jump and that it does not depend on a
  uniform axial strain along the vessel.
