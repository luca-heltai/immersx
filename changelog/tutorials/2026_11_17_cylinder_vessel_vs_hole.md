PR: #459

- Added the tutorial "A vessel in a cylinder: immersed vs resolved"
  (`doc/tutorials/cylinder-vessel-vs-hole.md`), with inputs in
  `tutorials/cylinder_vessel/` and the reduced grid
  `data/tests/cylinder_vessel_x.vtk.in`. It compares the wall stress of
  a resolved cylindrical hole with the pressure of an immersed vessel
  (tensor-product coupling) against the exact Lamé solution, and shows that
  the multiplier is the traction jump while the pressure is the tissue-side
  stress.
