PR: #459

- Added the tutorial "A vessel in a cylinder: immersed vs resolved"
  (`doc/tutorials/cylinder-vessel-vs-hole.md`), with inputs in
  `tutorials/cylinder_vessel/` and the reduced grid
  `data/tests/cylinder_vessel_x.vtk.in`. It compares the wall stress of
  a resolved cylindrical hole with the pressure of an immersed vessel
  (tensor-product coupling) against the exact Lamé solution, and shows that
  the multiplier is the traction jump while the pressure is the tissue-side
  stress.
- Added "Convergence of the vessel multiplier"
  (`doc/tutorials/vessel-multiplier-convergence.md`): the static multiplier
  converges at first order in h/a near the wall, does not depend on an axial
  strain, and Newmark steps add no visible time error; this accounts for the
  size of the MetricFlowX multiplier error of #379. The page also covers the
  displacement at the wall, the refinement of the centerline (interior values
  do not depend on it; the nodes on the cube faces do not converge), meshes
  fitted to the wall (second order instead of first), five
  figures, and the input files in `tutorials/vessel_multiplier/`.
