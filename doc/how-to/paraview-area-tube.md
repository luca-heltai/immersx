# Visualize vessel areas as tubes in ParaView

The Python plugin `scripts/paraview_area_tube.py` combines Calculator,
Extract Surface, and Tube in one **Area Tube** filter. It computes
`radius = sqrt(area / pi)` and uses **By Absolute Scalar** so the tube radius
matches the computed value without normalization.

## Load and apply the plugin

1. In ParaView, open **Tools → Manage Plugins → Load New** and select
   `scripts/paraview_area_tube.py` from your ImmersX checkout.
2. Select a dataset containing line cells and apply **Area Tube** from the
   filter search.
3. Set **AreaArray** to the cross-sectional area array name (default `area`).
4. Set **NumberOfSides** (default 24) and **Capping** (enabled by default),
   then click **Apply**.

ParaView must include Python support. No ImmersX build or C++ compilation is
needed. With a remote server, make the plugin available and load it on the
server as well.

## Input and units

The area must be a scalar **Point Data** array with finite, nonnegative values.
Zero area gives zero radius. The output includes the computed `radius` array;
an existing point array with that name is replaced in the output.

For cell-based areas, apply **Cell Data to Point Data** first. This interpolates
area before taking the square root and can smooth discontinuities at branches.
The filter accepts a single dataset; apply **Merge Blocks** first for composite
inputs. It generates tubes around line cells, not around surface edges.

Use compatible units: if coordinates are in metres, area must be in square
metres. Tube radii are expressed in coordinate units. Tube caps close individual
lines; this filter does not construct a watertight junction between branches.

## Python usage

Run with ParaView's Python console or `pvpython`:

```python
from paraview.simple import LoadPlugin

# Replace this with the absolute path to your checkout.
LoadPlugin("/path/to/immersx/scripts/paraview_area_tube.py", remote=True)
# After loading the plugin, import its generated proxy constructor.
from paraview.simple import AreaTube

tubes = AreaTube(Input=reader)  # reader is your existing line-data source
tubes.AreaArray = "area"
tubes.NumberOfSides = 24
tubes.Capping = 1
tubes.UpdatePipeline()
```
