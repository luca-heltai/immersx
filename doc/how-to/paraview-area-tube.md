# Visualize vessel radii and areas as tubes in ParaView

Load `scripts/paraview_variable_radius_tube.py` through **Tools → Manage
Plugins → Load New**, then apply **Variable Radius Tube** to a dataset with
line cells. ParaView must include Python support. Load the plugin on the
server as well when using a remote server.

## Choose a radius or area

With **InputArray** left at **Automatic**, the filter first looks for a scalar
array named `radius`, ignoring case. It searches Point Data before Cell Data
and uses array order to break ties within an association. If no radius array
exists, it uses **AreaArray** (default `area`), again preferring Point Data.

Select **InputArray** to choose a particular Point or Cell Data array.
**Interpretation** is **Automatic**, **Radius**, or **Area**. Automatic treats
`radius`, `Radius`, `RADIUS` and other case variants as radius; other selected
names mean area. Explicit Radius or Area overrides this interpretation. An
explicit selection is never replaced by an unrelated radius array.

All selected scalars must be finite and nonnegative and have one value per
point or cell. Cell values are validated before averaging to points. Area
values become `sqrt(area/pi)` after averaging; radius values are used directly.
This averaging can smooth discontinuities at branches. Empty pieces are valid.

Polydata goes directly to Tube. Other dataset types use Extract Surface first.
Set **NumberOfSides** (default 24) and **Capping** (enabled by default), then
click **Apply**. The tube uses absolute scalar radii without normalization.
Composite inputs require **Merge Blocks**. Tubes follow line cells, not
surface edges; caps do not construct watertight junctions at branches.

If coordinates are in metres, radius must be in metres and area in square
metres. Zero radius or area collapses the tube locally.

## Python usage and migration

```python
from paraview.simple import LoadPlugin
LoadPlugin("/path/to/immersx/scripts/paraview_variable_radius_tube.py", remote=True)
from paraview.simple import VariableRadiusTube

tubes = VariableRadiusTube(Input=reader)
tubes.InputArray = ['POINTS', 'Radius']
tubes.Interpretation = 'Automatic'
tubes.NumberOfSides = 24
tubes.Capping = 1
tubes.UpdatePipeline()
```

For cell area, use `tubes.InputArray = ['CELLS', 'area']` and
`tubes.Interpretation = 'Area'`. Leaving InputArray unset preserves the
legacy `AreaArray` fallback when no radius exists.

This plugin replaces `scripts/paraview_area_tube.py` and its **Area Tube**
filter. Reload the new plugin and replace `AreaTube` with `VariableRadiusTube`
in Python scripts. This documentation keeps its published page path.
