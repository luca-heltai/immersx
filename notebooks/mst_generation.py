#!/users/heltai/anaconda3/bin/python3
import numpy as np
import argparse  # Added argparse
import pyvista as pv
from pyvista import CellType


class Node:
    def __init__(self, parent, pos, index, path_distance=0.0):
        self.parent = parent
        self.pos = pos
        self.index = index
        self.children = []
        self.path_distance = path_distance
        self.radius = 0.0  # Added radius attribute

        if parent is not None:
            parent.children.append(self)


def mstree(points, balancing_factor=0.3, initial_radius=1.0, radius_decay_factor=0.2):  # Updated default parameters
    length = len(points)
    dimensions = len(points[0])

    closed_list = {}

    root_point = points[0]

    root_node = Node(None, root_point, 0)
    root_node.radius = initial_radius  # Set initial radius for root node
    closed_list[0] = root_node

    # Init open points list
    open_list = [x for x in range(1, length)]

    # Init distance to root_point
    distances_squared = np.sum(np.square(points - root_point), axis=1)
    distances = np.empty(length)
    for i in range(length - 1):
        distances[i] = np.sqrt(distances_squared[i])

    closest_point_in_tree = np.zeros(length, dtype=int)

    distances = np.sqrt(distances_squared)

    open_distance_list = distances.copy()[1:]

    while len(open_distance_list) > 0:
        minimum_index = np.argmin(open_distance_list)
        minimum = open_distance_list[minimum_index]
        point_index = open_list.pop(minimum_index)

        # Get closest point and append new node to it
        closest_point_index = closest_point_in_tree[point_index]

        location = points[point_index]

        parent_node = closed_list[closest_point_index]
        actual_distance = np.sqrt(
            np.sum(np.square(location - parent_node.pos)))
        path_distance = actual_distance + parent_node.path_distance
        node = Node(parent_node, location, point_index, path_distance)
        node.radius = initial_radius * \
            np.exp(-radius_decay_factor *
                   node.path_distance)  # Calculate and set radius

        # Add to closed list
        closed_list[point_index] = node
        # Remove from open list
        open_distance_list = np.delete(open_distance_list, minimum_index)

        open_points = points[open_list]
        weighted_distance = np.sqrt(np.sum(np.square(np.subtract(
            open_points, location)), axis=1)) + balancing_factor * path_distance
        open_distance_list_indeces = np.argmin(np.column_stack(
            (open_distance_list, weighted_distance)), axis=1)
        open_distance_list = np.minimum(open_distance_list, weighted_distance)
        changed_values = np.zeros(len(closest_point_in_tree), dtype=bool)
        changed_values.put(open_list, open_distance_list_indeces)
        closest_point_in_tree = np.where(
            changed_values == 1, point_index, closest_point_in_tree)

    return root_node


def tree_to_list(root_node):
    """Orders the nodes into a list recursivly using depth-first-search"""
    ls = [root_node]
    for child in root_node.children:
        ls.extend(tree_to_list(child))
    return ls


def build_tree_mesh(root_node, skin=False):
    nodes = tree_to_list(root_node)

    vertices = np.array([node.pos for node in nodes])

    edges = []
    for i, node in enumerate(nodes):
        if node.parent is not None:
            edges.append([i, nodes.index(node.parent)])

    edges = np.array(edges)

    return (vertices, edges)


def write_vtk(filename, vertices, edges, dict={}):
    """ Write a VTK file """

    # Build VTK cell array and cell types for VTK_LINE
    n_edges = edges.shape[0]
    cells = np.hstack([np.full((n_edges, 1), 2, dtype=int),
                      edges]).astype(np.int64).flatten()
    celltypes = np.full(n_edges, CellType.LINE, dtype=np.uint8)

    # Create UnstructuredGrid
    ugrid = pv.UnstructuredGrid(cells, celltypes, vertices)

    # Check if we have point or cell data
    for key, data in dict.items():
        if len(data) == len(vertices):
            ugrid.point_data[key] = data
        elif len(data) == len(edges):
            ugrid.cell_data[key] = data

    # Save to VTK file
    ugrid.save(filename, binary=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a Minimum Spanning Tree and save it as VTK.")
    parser.add_argument("--num-points", type=int, default=20,
                        help="Number of random points to generate.")
    parser.add_argument("--num-dimensions", type=int, default=3,
                        help="Number of dimensions for the points.")
    parser.add_argument("--balancing-factor", type=float,
                        default=0.3, help="Balancing factor for mstree.")  # Updated default
    parser.add_argument("--initial-radius", type=float,
                        default=1.0, help="Initial radius for the root node.")
    parser.add_argument("--radius-decay-factor", type=float,
                        default=0.2, help="Radius decay factor for mstree.")  # Updated default
    parser.add_argument("--min-coord", type=float, default=0.0,
                        help="Minimum coordinate value for random points.")
    parser.add_argument("--max-coord", type=float, default=1.0,
                        help="Maximum coordinate value for random points.")
    parser.add_argument("--output-file", type=str,
                        required=True, help="Path to the output VTK file.")

    args = parser.parse_args()

    points = args.min_coord + (args.max_coord - args.min_coord) * \
        np.random.rand(args.num_points, args.num_dimensions)

    tree = mstree(points,
                  balancing_factor=args.balancing_factor,
                  initial_radius=args.initial_radius,
                  radius_decay_factor=args.radius_decay_factor)

    (vertices, edges) = build_tree_mesh(tree)

    data_to_write = {}
    all_nodes = tree_to_list(tree)

    path_distance = [node.path_distance for node in all_nodes]
    data_to_write["path_distance"] = path_distance

    radii = [node.radius for node in all_nodes]
    data_to_write["radius"] = radii

    write_vtk(args.output_file, vertices, edges, data_to_write)
    print(f"Generated MST and saved to {args.output_file}")
