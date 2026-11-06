# Liver result figures

These figures use saved numerical results. They do not rerun the coupled model
or read a three-dimensional VTK series.

From the repository root, install the plotting dependencies in a virtual
environment and render the figures:

```sh
python3 -m venv /tmp/immersx-liver-plots
/tmp/immersx-liver-plots/bin/pip install -r tutorials/liver_icar/plots/requirements.txt
/tmp/immersx-liver-plots/bin/python tutorials/liver_icar/plots/plot_results.py
```

The script resolves paths relative to its own file. It can run from another
working directory. `--inputs DIRECTORY` selects a relocated input snapshot;
`--output-dir DIRECTORY` selects the output directory. The default outputs are
under `doc/_static/liver-icar/`. Each figure is written as PNG and SVG. The
render manifest records dependency versions and SHA256 hashes of the script,
input provenance, and generated figures. The SVG has no generated timestamp;
repeated renders with the recorded dependency versions produce identical bytes.
The script verifies all input hashes before plotting.

## Figures and scope

| Output prefix | Saved result and transformations |
| --- | --- |
| `scale-profiles` | Measured total coupled DoFs and observed minimum–maximum of numerical interval length divided by local background-cell scale. The dashed line is the 1:1 size reference. The three background meshes are independent, with 14 physical vessel edges and 6 tissue leaves in the common CCO architecture. These extrema do not prove inf-sup stability or mesh convergence. |
| `small-haemodynamics` | The 81 saved times of the small P2/DG1 cycle, 7.2–8.0 s, source `a92bee56`, MPI16. HA and PV flows point into the liver; HV points out. The CSV retains native outward-normal signs. Computed root pressures are taken from the terminal output. The prescribed HA flow is linearly interpolated at saved times, snapping the input endpoints only for time roundoff below `1e-12` s. No waveform smoothing is applied. |
| `small-conservation` | Original microvascular KCL and cumulative root-volume defects divided by their original per-sample allowed defects. The dashed ratio 1 is the original acceptance bound. The original positivity bound is zero. Volume uses the original trapezoidal saved-time quadrature. Exact-zero KCL defects remain in the CSV and are omitted on logarithmic axes. Positive areas refer to saved DG1 endpoint samples; there is no between-time guarantee. |
| `medium-solid-samples` | Eleven saved times over 20 ms, 7.2–7.22 s, source `b4d85cb6`, MPI8. The plotted displacement and velocity are maxima of vector norms over exported nodal samples. They are not exact extrema of the P2 finite-element field or strain estimates. |

The pressure conversion is `133.322387415 Pa/mmHg`, consistent with the
configured portal reservoir `1066.57909932 Pa = 8 mmHg`. Flow is converted from
m³/s to mL/min by multiplying by `60000000`; area from m² to mm² by `1000000`;
displacement and velocity from SI to mm and mm/s by `1000`.

## Input provenance

[inputs/provenance.json](inputs/provenance.json) records the full runtime
revision, original artifact-relative path, source-file SHA256, selected fields,
units, sign convention, and SHA256 of each compact plotting snapshot. It links
the baseline and scaling releases. The original audit policy is copied to
[inputs/small-audit-policy.json](inputs/small-audit-policy.json). Numeric source
precision is retained in the snapshots; only labels of the geometric range
endpoints are rounded to six decimal places.

The raw source files remain in the release artifacts. These compact inputs are
sufficient to reproduce the figures without downloading meshes or VTK series.
They represent two different saved runs: the full small cycle predates the
later clean-API scaling runs. Their source revisions are not relabelled as the
current publication revision.
