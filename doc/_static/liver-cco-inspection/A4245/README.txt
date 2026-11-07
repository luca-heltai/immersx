A4245: completed arterial stage of the anatomical V21 calculation.
The parent run timed out during portal growth; hepatic venous growth had not begun.
No completed three-network archive, coupled time steps or restart state is supplied.

Figure inputs are saved scalar/segment tables. H is the longest edge of an affine
generation tetrahedron. Occupancy is a measured histogram, not a per-cell quota.
The sufficient whole-cell distance bound certifies only the generated arterial
terminals on this 4,245-cell coarse volume. No native quadrature/routing replay,
triple-system coverage, finer-volume coverage, perfusion or inf-sup claim is made.

Reproduce both figures with Python, NumPy, Matplotlib and VTK:
  python3 plot.py /absolute/path/to/liver-cco-A4245-v21-20261107 /absolute/path/to/new-output

The first argument is the extracted inspection archive containing
liver-generation.vtu. The output directory must be new. Tables are read beside
this script, so the command can be run from any working directory. Input hashes
are checked before plotting. The recorded environment used Python3, NumPy2.2.6,
Matplotlib3.10.9 and VTK9.2.6. Rendering does not run CCO or resume its state.

The two PNGs are byte-identical to the visually checked inspection export.
SVG and CSV copies use LF endings with trailing layout whitespace removed;
the numerical values and coordinates are unchanged. provenance.json records
the original and documentation hashes. Line widths are constant for visibility
and do not represent the physical Radius fields stored in the archive.

Anatomical source: IRCAD, 3D-IRCADb-1 case12; Soler and coauthors (2010),
3D image reconstruction for comparison of algorithm database: A patient specific
anatomical and medical image database, IRCAD technical report.
https://www.ircad.fr/research-and-development/data-sets/liver-segmentation-3d-ircadb-01/
https://creativecommons.org/licenses/by-nc-nd/4.0/
