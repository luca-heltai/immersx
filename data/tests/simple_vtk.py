#!/Users/heltai/anaconda3/bin/python3

import numpy as np
import pyvista as pv

# using pyvista, generate a simple 1d unstructured grid, made of VTK_LINE
# objects, with 10 vertices and 9 edges, distributed from 0 to 1. Then add 1
# scalar point data field containing the x of each vertex, one vector field
# containing the x,y,z coordinates of the vertices, one scalar cell data field,
# containing the x coordinate of the center of the edge, and one vector cell
# data field containng the coordinates of the center of the edge

# Create 10 points along x from 0 to 1, y from 1 to 2, and z from 3 to 4
points = np.zeros((10, 3))
points[:, 0] = np.linspace(0, 1, 10)
points[:, 1] = np.linspace(1, 2, 10)
points[:, 2] = np.linspace(2, 3, 10)

# Create 9 VTK_LINE cells connecting consecutive points
cells = []
for i in range(9):
    cells.extend([2, i, i + 1])  # 2 = number of points per line

cells = np.array(cells)

# Create the unstructured grid
grid = pv.UnstructuredGrid(cells, np.full(9, pv.CellType.LINE), points)

# Point data: scalar field (x), vector field (xyz)
grid.point_data["x"] = points[:, 0]
grid.point_data["xyz"] = points

# Cell data: scalar field (center x), vector field (center xyz)
cell_centers = 0.5 * (points[:-1] + points[1:])
grid.cell_data["center_x"] = cell_centers[:, 0]
grid.cell_data["center_xyz"] = cell_centers

# Optionally, save or plot
grid.save("simple_1d_grid.vtk", binary=False)
# grid.plot(show_edges=True)
