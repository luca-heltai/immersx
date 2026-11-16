PR: #441

- Steady row scaling reads native sparse-row views for noncontiguous MPI
  ownership between semantic fields, without constructing sparse iterators
  for unowned rows or allocating a second full coupled matrix.
