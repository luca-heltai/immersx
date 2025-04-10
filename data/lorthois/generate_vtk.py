import pyvista as pv
import numpy as np

# Load additional data from split files

# VERTEX { float[3] VertexCoordinates } @1
# EDGE { int[2] EdgeConnectivity } @2
# EDGE { int NumEdgePoints } @3
# POINT { float[3] EdgePointCoordinates } @4
# POINT { float thickness } @5

# EDGE { float Hematocrit } @6
# EDGE { float fictional_edge } @7
# EDGE { float ArtVenCap_Label } @8
# EDGE { float thickness } @9
# EDGE { float length } @10
# EDGE { float Flowrate } @11
# EDGE { float concentration } @12

# VERTEX { float Pressure } @13

vertex_coordinates = np.loadtxt(
    'Network1_Split_1.txt', skiprows=1, usecols=(0, 1, 2))
edge_connectivity = np.loadtxt(
    'Network1_Split_2.txt', skiprows=1, usecols=(0, 1), dtype=int)
num_edge_points = np.loadtxt('Network1_Split_3.txt', skiprows=1, dtype=int)
edge_point_coordinates = np.loadtxt(
    'Network1_Split_4.txt', skiprows=1, usecols=(0, 1, 2))
point_thickness = np.loadtxt('Network1_Split_5.txt', skiprows=1)
hematocrit = np.loadtxt('Network1_Split_6.txt', skiprows=1)
fictional_edge = np.loadtxt('Network1_Split_7.txt', skiprows=1)
art_ven_cap_label = np.loadtxt('Network1_Split_8.txt', skiprows=1)
edge_thickness = np.loadtxt('Network1_Split_9.txt', skiprows=1)
length = np.loadtxt('Network1_Split_10.txt', skiprows=1)
flowrate = np.loadtxt('Network1_Split_11.txt', skiprows=1)
concentration = np.loadtxt('Network1_Split_12.txt', skiprows=1)
pressure = np.loadtxt('Network1_Split_13.txt', skiprows=1)


# Read edge data
# edges = np.loadtxt(edge_file, skiprows=1, usecols=(0, 1), dtype=int)
# lines = np.zeros((edges.shape[0], 3), dtype=int)
# lines[:, 0] = 2  # Number of points in the line
# lines[:, 1:] = edges

lines = np.zeros(
    (len(num_edge_points) + len(edge_point_coordinates)), dtype=int)

ids = np.cumsum(num_edge_points)
j = 0
for i in range(len(num_edge_points)):
    lines[j] = num_edge_points[i]
    start = j+1
    end = num_edge_points[i]+j+1

    start_p = ids[i-1] if i > 0 else 0
    end_p = ids[i]
    lines[start:end] = np.arange(start_p, end_p)
    j = end

min_p = edge_point_coordinates.min(axis=0)
max_p = edge_point_coordinates.max(axis=0)

# Rescale all coordinates to be between 0.1 and 0.9
edge_point_coordinates = (edge_point_coordinates - min_p) / \
    (max_p - min_p) * 0.8 + 0.1

# Rescale the thickness
point_thickness = point_thickness / point_thickness.max() * 0.05


# Create a PyVista PolyData object
polydata = pv.PolyData(edge_point_coordinates, lines=lines)

# Add labels to the grid, on each cell
polydata.cell_data['labels'] = art_ven_cap_label
polydata.cell_data['thickness'] = edge_thickness
polydata.point_data['point_thickness'] = point_thickness
polydata.cell_data['length'] = length
polydata.cell_data['fictional'] = fictional_edge
polydata.cell_data['concentration'] = concentration
polydata.cell_data['flowrate'] = flowrate
polydata.cell_data['hematocrit'] = hematocrit
# polydata.cell_data['pressure'] = pressure

# Save to VTK file
pv.UnstructuredGrid(polydata).save('Network2.vtk', binary=False)
polydata.save('Network1.vtk', binary=False)
grid = pv.UnstructuredGrid(polydata)

polydata2 = pv.merge([polydata], merge_points=True)

print(polydata)
print(polydata2)
print(grid)

# print(lines.shape, vertices.shape, np.min(edges),
# np.max(edges), np.min(vertices), np.max(vertices))

# px, py, pz, dx, dy, dz, r, id
# fourier modes
