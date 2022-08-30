import bpy
import numpy as np
from numpy import *

import bmesh
from mathutils import Vector


def read_verts(mesh):
    mverts_co = np.zeros((len(mesh.vertices) * 3), dtype=float)
    mesh.vertices.foreach_get("co", mverts_co)
    return np.reshape(mverts_co, (len(mesh.vertices), 3))


def read_edges(mesh):
    fastedges = np.zeros((len(mesh.edges) * 2), dtype=np.int)  # [0.0, 0.0] * len(mesh.edges)
    mesh.edges.foreach_get("vertices", fastedges)
    return np.reshape(fastedges, (len(mesh.edges), 2))


def read_norms(mesh):
    mverts_no = np.zeros((len(mesh.vertices) * 3), dtype=float)
    mesh.vertices.foreach_get("normal", mverts_no)
    return np.reshape(mverts_no, (len(mesh.vertices), 3))


def write_vtk(filename, vertices, edges, lengths):
    """ Write a VTK file """
    f = open(filename, "w")

    f.write("# vtk DataFile Version 3.0\n")
    f.write("Fibers output - vtk\n")
    f.write("ASCII\n")
    f.write("DATASET UNSTRUCTURED_GRID\n")
    f.write("POINTS " + str(len(vertices)) + " DOUBLE\n")

    for v in vertices:
        f.write(str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + "\n")

    f.write("CELLS " + str(len(edges)) + " " + str(3 * len(edges)) + "\n")
    for e in edges:
        f.write("2 " + str(int(e[0])) + " " + str(int(e[1])) + "\n")

    f.write("CELL_TYPES " + str(len(edges)) + "\n")
    for e in edges:
        f.write("3\n")

    f.write("CELL_DATA " + str(len(edges)) + "\n")
    f.write("SCALARS length float \n")
    f.write("LOOKUP_TABLE default \n")
    for l in lengths:
        f.write(str(l) + "\n")
    f.close()

def color_vertex(obj, vert, color):
    """Paints a single vertex where vert is the index of the vertex
    and color is a tuple with the RGB values."""

    mesh = obj.data
    scn = bpy.context.scene

    #check if our mesh already has Vertex Colors, and if not add some... (first we need to make sure it's the active object)
    scn.objects.active = obj
    obj.select = True
    if mesh.vertex_colors:
        vcol_layer = mesh.vertex_colors.active
    else:
        vcol_layer = mesh.vertex_colors.new()

    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            loop_vert_index = mesh.loops[loop_index].vertex_index
            if vert == loop_vert_index:
                vcol_layer.data[loop_index].color = color

def write_vessels(mesh_name):
    mesh = bpy.data.meshes[mesh_name]

    vtk_name = mesh_name + ".vtk"

    # Read vertices and edge connectivity
    vertices = read_verts(mesh)
    edges = read_edges(mesh)

    # Segments: v0-v1
    segments = vertices[edges[:,0]]-vertices[edges[:,1]]
    lengths = sqrt(sum(segments**2, axis=-1))

    sel = lengths != 0

    segments = segments[sel,:]
    lengths = lengths[sel]
    edges = edges[sel,:]

    # Start by saving
    write_vtk(vtk_name, vertices, edges, lengths)

    # Hypersingularity file
    singularities = (vertices[edges[:, 0]] + vertices[edges[:, 1]]) / 2
    singularities = np.c_[singularities, segments]
    np.savetxt(mesh_name + "_hs.gpl", singularities)

    # Unit directions
    directions = segments/lengths[:,newaxis]
    sel = sum(directions, axis=1)<0
    directions[sel, :] *= -1
    segments[sel, :] *= -1

    # Compute phi and theta
    rxy = sqrt(sum((directions**2)[:,0:2], axis=1))
    theta = arctan2(directions[:,2], rxy)
    phi = arctan2(directions[:,1], directions[:,0])


    # Total length of vessels
    print("Total/min/max length: ", sum(lengths), min(lengths), max(lengths))

    # Verify they are correct
    check = c_[cos(theta)*cos(phi), cos(theta)*sin(phi), sin(theta)]
    print("Check (this should be zero)", np.linalg.norm(check-directions))

    # Extract principal directions
    C = segments.T.dot(segments)
    lam, V = np.linalg.eig(C)
    print(lam, V)
    J = np.linalg.det(C)

    print(J)

def create_ellipsoid(mesh_name, R=0.5, location=[0,0,0], N=101, M=56):

    mesh = bpy.data.meshes[mesh_name]

    # Read vertices and edge connectivity
    vertices = read_verts(mesh)
    edges = read_edges(mesh)

    # Segments: v0-v1
    segments = vertices[edges[:,0]]-vertices[edges[:,1]]
    lengths = sqrt(sum(segments**2, axis=-1))

    sel = lengths != 0

    segments = segments[sel,:]
    lengths = lengths[sel]
    edges = edges[sel,:]

    # Extract principal directions
    C = segments.T.dot(segments)
    lam, V = np.linalg.eig(C)

    m = prod(lam)**(1/3)
    id = np.argsort(lam)

    J = np.linalg.det(C)

    lam /= m


    vx = V[:, id[0]]
    vy = V[:, id[1]]
    vz = V[:, id[2]]


    a = lam[id[0]]
    b = lam[id[1]]
    c = lam[id[2]]

    theta = np.linspace(0, 2 * pi, N)
    phi = np.linspace(0, pi, M)

    Theta, Phi = np.meshgrid(theta, phi)

    x = sin(Theta) * sin(Phi)
    y = cos(Theta) * sin(Phi)
    z = cos(Phi)

    X = np.einsum('ij, k', x, vx)
    Y = np.einsum('ij, k', y, vy)
    Z = np.einsum('ij, k', z, vz)

    XYZ = R * (X * a + Y * b + Z * c)

    print("Total length "+mesh_name+": ", sum(lengths)*3e-3, "Vol: ", prod(lam))
    print("tau1: ", vx, "tau2: ", vy, "tau3: ", vz, "l1: ", a, "l2: ", b, "l3: ", c)

    bm = bmesh.new()

    verts_mesh = [bm.verts.new(Vector(p)) for p in XYZ[0, :]]
    verts_mesh.append(verts_mesh[0])

    for ring in range(1, XYZ.shape[0]):
        verts_mesh_face = [bm.verts.new(Vector(p)) for p in XYZ[ring, :]]
        verts_mesh_face.append(verts_mesh_face[0])

        for i in range(len(verts_mesh) - 1):
            face = bm.faces.new((
                verts_mesh[i], verts_mesh_face[i],
                verts_mesh_face[i + 1], verts_mesh[i + 1]
            ))
        verts_mesh = verts_mesh_face

    # Update vertex indices
    bm.verts.index_update()

    color_layer = bm.loops.layers.color.new("color" + mesh_name)
    # make a random color table for each vert
    n_separations = 9

    for face in bm.faces:
        for loop in face.loops:
            i = loop.vert.index
            fr = .5 + .5 * sin(theta[i % N] * n_separations)
            color = (fr, fr, fr)#.5 + .5 * sin(theta[i % N] * n_separations))
            loop[color_layer] = color

    # create mesh link it to scene
    mesh = bpy.data.meshes.new("ellipsoid_" + mesh_name)
    bm.to_mesh(mesh)
    obj = bpy.data.objects.new("ellipsoid_" + mesh_name, mesh)
    scene = bpy.context.scene
    scene.objects.link(obj)
    scene.objects.active = obj
    obj.select = True
    obj.location = Vector(location)

    # create material
    mat_name = "Material"+mesh_name
    materials = bpy.data.materials
    mat = materials.get(mat_name) or materials.new(mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    shader = nodes.get("ShaderNodeAttribute") or nodes.new("ShaderNodeAttribute")
    shader.attribute_name = "color"+mesh_name

    diffuse = nodes.get('Diffuse BSDF')
    mat.node_tree.links.new(shader.outputs[0], diffuse.inputs[0])

    # link the material to the newly created object
    if obj.data.materials:
        # assign to 1st material slot
        obj.data.materials[0] = mat
    else:
        # no slots
        obj.data.materials.append(mat)

def cylinders(vertices, edges, resolution, radius):
    """
    Create a cylinder, with at least the given resolution, and the given radius
    """
    inclusions = []
    for edge in edges:
        origin = vertices[edge[0]]
        final  = vertices[edge[1]]
        n_nodes = max(int(ceil(np.linalg.norm(final-origin)/resolution)),2)
        print(n_nodes)
        verts = linspace(origin, final, n_nodes)
        directions = diff(verts, axis=0)
        centers = (verts[1:,:]+verts[:-1,:])/2
        radii = ones(centers.shape[0],)*radius
        inclusions.append(c_[centers, directions, radii])
    return concatenate(inclusions)

D = bpy.data

mesh_name = 'CornerTree'

x = -3
for mesh_name in ['cylinders']:
    
    mesh = bpy.data.meshes[mesh_name]

    # Read vertices and edge connectivity
    vertices = read_verts(mesh)
    edges = read_edges(mesh)
    
    n_nodes = 64
    radius = .2
    resolution = 2*pi*radius/n_nodes
    
    inclusions = cylinders(vertices, edges, resolution, radius)
    
    np.savetxt("../build/cylinder_3d.gpl", inclusions)

    # write_vessels(mesh_name)
    # create_ellipsoid(mesh_name, 1, [x, 1, 1])
    x = x+3