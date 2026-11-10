Regenerate the two PNG/SVG figures from the bundled CSV:
python3 plot_perfusion.py
Requires NumPy and Matplotlib. No solver or HPC input is needed for rendering.

The 512- and 2048-territory cases are valid rigid reference solves with
demand-scaled compartment resistances. The 2048 case additionally doubles
the portal arm coefficient. This is a comparison of model variants, not
an isolated refinement or convergence study.

The CSV retains territory and terminal IDs, demand fractions, actual flows
and all three arm resistances. Failed attempts have status rows only.
The source records and import changes are retained in provenance.json.
