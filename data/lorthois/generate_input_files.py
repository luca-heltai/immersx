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
# lines[:, 1:] = edges

lines = np.zeros(
    (len(num_edge_points) + len(edge_point_coordinates)), dtype=int)

n_inclusions = int(len(lines)/2)

line_ids = np.zeros(len(lines), dtype=int)
line_concentration = np.zeros_like(lines)

ids = np.cumsum(num_edge_points)
j = 0
for i in range(len(num_edge_points)):
    if fictional_edge[i] == 1:
        break
    lines[j] = num_edge_points[i]
    line_ids[j] = i  # Store the edge id for the line
    start = j+1
    end = num_edge_points[i]+j+1

    start_p = ids[i-1] if i > 0 else 0
    end_p = ids[i]
    lines[start:end] = np.arange(start_p, end_p)
    # Store the edge id for the points in the line
    line_ids[start:end] = i
    line_concentration[start:end] = concentration[i]
    j = end

# Drop all fictitional edges
lines = lines[:j]
line_ids = line_ids[:j]
line_concentration = line_concentration[:j]

# Reshape the lines and line_ids
lines = lines.reshape((-1, 2))
line_ids = line_ids.reshape((-1, 2))[:, 0]
line_concentration = line_concentration.reshape((-1, 2))[:, 0]

n_inclusions = lines.shape[0]

min_p = edge_point_coordinates.min(axis=0)
max_p = edge_point_coordinates.max(axis=0)

# Rescale all coordinates to be between 0.1 and 0.9
edge_point_coordinates = (edge_point_coordinates - min_p) / \
    (max_p - min_p) * 0.8 + 0.1

# Rescale the thickness
point_thickness = point_thickness / point_thickness.max() * 0.05

# Calculate directions outside the loop
directions = np.zeros((n_inclusions, 3), dtype=float)
radii = np.zeros(n_inclusions, dtype=float)
line_centers = np.zeros((n_inclusions, 3), dtype=float)
# line_pressure = np.zeros(n_inclusions, dtype=float)
for i in range(n_inclusions):
    start_p = lines[i, 0]
    end_p = lines[i, 1]
    start_point = edge_point_coordinates[start_p]
    end_point = edge_point_coordinates[end_p]
    line_centers[i] = .5*(start_point + end_point)
    directions[i] = end_point - start_point
    radii[i] = .5*(point_thickness[start_p] + point_thickness[end_p])
    # line_pressure[i] = .5*(pressure[start_p] + pressure[end_p])

# To create inclusions file, i need to write
# cx,cy,cz,dx,dy,dz,R,vesselID

out_data = np.c_[line_centers, directions, radii, line_ids]
# Save to file
np.savetxt('inclusions.txt', out_data,
           fmt='%.5f %.5f %.5f %.5f %.5f %.5f %.5f %d')

np.savetxt('concentration.txt', line_concentration, fmt='%.5f')
# np.savetxt('pressure.txt', line_pressure, fmt='%.5f')
