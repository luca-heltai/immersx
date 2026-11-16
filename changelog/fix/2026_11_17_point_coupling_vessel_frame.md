PR: pending

- Point coupling in 3D: inclusions without a data line now impose the
  `Immersed inclusions/Boundary data` function in the vessel frame. Before,
  the multiplier of vessel-frame component `c` was paired with the global
  component `c` of the function, so tilted vessels, and vessels along -z,
  received the wrong boundary displacement. Inclusions with a data line and
  2D problems are unchanged.
