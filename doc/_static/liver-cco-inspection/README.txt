Partial anatomical CCO checkpoint A3073. No coupled solution, inf-sup certificate or completed scaling cases.

Reproduce interval figure from the two small CSVs with Python3, NumPy, Matplotlib:
  python3 plot_intervals.py
The script finds CSVs beside itself and can run from any directory.

Reproduce coverage figure from coverage-report.json:
  python3 plot_coverage.py /absolute/path/to/this-directory
This overwrites only its generated figure/CSV outputs and writes coverage-figure-provenance.json.

Reproduce growth projections with Python3, NumPy, Matplotlib and VTK. Extract the A3073 inspection archive linked in issue343, then use its geometry files:
  python3 plot_growth.py /absolute/assets/geometry/vascular-physical-A3073.vtu /absolute/assets/geometry/liver-background.vtu /absolute/path/to/new-output-directory
The exact two source hashes are checked. Output directory must be new. This does not rerun CCO or resume its state.

PNG/SVG widths are for display, not physical radii. Native ratio histograms exclude the seven exterior intervals. The upper check uses Hmin separately. DoF counts are solid u/v only before hanging-node elimination.
