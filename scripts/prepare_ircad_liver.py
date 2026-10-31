#!/usr/bin/env python3
# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Download IRCAD surface meshes and prepare a liver and two rooted 1D trees.

Requires VTK, NumPy and VMTK. Derived patient data stay in the output directory.
The case-12 recipe selects the largest intrahepatic tree of each vascular system.
"""
import argparse
from collections import deque
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile


SOURCE_SHA256 = {
    "liver": "aa74e6e699c60434b6472d12f8139ced62af7ee1986a817b56fc26b3d831a28f",
    "artery": "e7cf47e2831b605bea0b1d2e2a7439c936e7f4ba29eab6de312d6c3e8ac2c8c5",
    "venacava": "6f61d1559ccc40a28e89fad5327d21ff798a9283d0695afcd621a7e960f5f5c7",
}


def components(adjacency):
    unseen = set(adjacency)
    result = []
    while unseen:
        found = {min(unseen)}
        pending = list(found)
        for node in pending:
            for neighbor in sorted(adjacency[node]):
                if neighbor not in found:
                    found.add(neighbor)
                    pending.append(neighbor)
        unseen -= found
        result.append(found)
    return sorted(result, key=lambda nodes: (-len(nodes), min(nodes)))


def prepare(args):
    import vtk
    import numpy as np
    from vmtk import vmtkscripts

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw"
    raw.mkdir()
    url = ("https://cloud.ircad.fr/index.php/s/JN3z7EynBiwYyjy/download"
           "?path=%2F3Dircadb1.12&files=MESHES_VTK.zip")
    content = args.mesh_archive.read_bytes() if args.mesh_archive else urllib.request.urlopen(url, timeout=120).read()
    sources = {}
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for name in ["liver", "artery", "venacava"]:
            data = archive.read(f"MESHES_VTK/{name}.vtk")
            sources[name] = hashlib.sha256(data).hexdigest()
            if sources[name] != SOURCE_SHA256[name]:
                raise RuntimeError(f"Unexpected source checksum for {name}.vtk")
            (raw / f"{name}.vtk").write_bytes(data)
    manifest = dict(case="3Dircadb1.12", source_url=url,
                    source_sha256=sources, source_units="mm", output_units="m",
                    license="CC-BY-NC-ND-4.0",
                    source_page="https://www.ircad.fr/research-and-development/data-sets/liver-segmentation-3d-ircadb-01/",
                    numpy_version=np.__version__,
                    centerlines="VMTK approximate network extraction, advancement ratio 1.05",
                    vtk_version=vtk.vtkVersion.GetVTKVersion(), components={})
    surfaces = {}
    for name in ["liver", "artery", "venacava"]:
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(str(raw / f"{name}.vtk"))
        reader.Update()
        connected = vtk.vtkPolyDataConnectivityFilter()
        connected.SetInputData(reader.GetOutput())
        connected.SetExtractionModeToLargestRegion()
        connected.Update()
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputConnection(connected.GetOutputPort())
        transform = vtk.vtkTransform()
        transform.Scale(.001, .001, .001)
        scaled = vtk.vtkTransformPolyDataFilter()
        scaled.SetInputConnection(cleaner.GetOutputPort())
        scaled.SetTransform(transform)
        scaled.Update()
        surface = vtk.vtkPolyData()
        surface.DeepCopy(scaled.GetOutput())
        removed_double_faces = 0
        if name == "liver":
            # IRCAD case 12 has an oppositely oriented pair of identical
            # triangles attached along one edge. Cancel this zero-volume fin.
            faces = {}
            for i in range(surface.GetNumberOfCells()):
                cell = surface.GetCell(i)
                key = tuple(sorted(cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())))
                faces.setdefault(key, []).append(i)
            surface.BuildLinks()
            for duplicate in faces.values():
                if len(duplicate) == 1:
                    continue
                if len(duplicate) != 2:
                    raise RuntimeError("Ambiguous duplicate liver surface faces")
                a, b = [tuple(surface.GetCell(i).GetPointId(j) for j in range(3)) for i in duplicate]
                reverse = b[::-1]
                if not any(a == reverse[j:] + reverse[:j] for j in range(3)):
                    raise RuntimeError("Duplicate liver faces have the same orientation")
                for i in duplicate:
                    surface.DeleteCell(i)
                    removed_double_faces += 1
            surface.RemoveDeletedCells()
            cleaner = vtk.vtkCleanPolyData()
            cleaner.SetInputData(surface)
            cleaner.Update()
            surface = vtk.vtkPolyData()
            surface.DeepCopy(cleaner.GetOutput())
            edges = vtk.vtkFeatureEdges()
            edges.SetInputData(surface)
            edges.FeatureEdgesOff()
            edges.ManifoldEdgesOff()
            edges.BoundaryEdgesOn()
            edges.NonManifoldEdgesOn()
            edges.Update()
            if edges.GetOutput().GetNumberOfCells():
                raise RuntimeError("Liver surface still has open or nonmanifold edges")
        surfaces[name] = surface
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(str(output / f"{name}.stl"))
        writer.SetInputData(surface)
        writer.SetFileTypeToBinary()
        if writer.Write() != 1:
            raise RuntimeError(f"Cannot write {name} surface")
        manifest["components"][name] = dict(
            original_points=reader.GetOutput().GetNumberOfPoints(),
            retained_surface_points=surface.GetNumberOfPoints(),
            removed_opposite_duplicate_faces=removed_double_faces)

    for source, name in [("artery", "arterial"), ("venacava", "venous")]:
        surface = vtk.vtkPolyData()
        surface.DeepCopy(surfaces[source])
        edges = vtk.vtkFeatureEdges()
        edges.SetInputData(surface)
        edges.BoundaryEdgesOn()
        edges.FeatureEdgesOff()
        edges.NonManifoldEdgesOff()
        edges.ManifoldEdgesOff()
        edges.Update()
        if edges.GetOutput().GetNumberOfCells() == 0:
            # VMTK network extraction needs an opening. This artificial opening
            # is only for extraction; the retained STL is never modified.
            surface.BuildLinks()
            surface.DeleteCell(0)
            surface.RemoveDeletedCells()
        extraction = vmtkscripts.vmtkNetworkExtraction()
        extraction.Surface = surface
        extraction.AdvancementRatio = 1.05
        extraction.Execute()
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(extraction.Network)
        cleaner.SetToleranceIsAbsolute(True)
        cleaner.SetAbsoluteTolerance(1.e-8)
        cleaner.Update()
        network = cleaner.GetOutput()
        radius = network.GetPointData().GetArray("Radius")
        if radius is None:
            raise RuntimeError(f"No radius field extracted for {source}")
        enclosed = vtk.vtkSelectEnclosedPoints()
        enclosed.SetInputData(network)
        enclosed.SetSurfaceData(surfaces["liver"])
        enclosed.SetTolerance(1.e-6)
        enclosed.Update()
        inside = {i for i in range(network.GetNumberOfPoints()) if enclosed.IsInside(i)}
        adjacency = {}
        for i in range(network.GetNumberOfCells()):
            cell = network.GetCell(i)
            for j in range(cell.GetNumberOfPoints() - 1):
                a, b = cell.GetPointId(j), cell.GetPointId(j + 1)
                if a != b and a in inside and b in inside:
                    adjacency.setdefault(a, set()).add(b)
                    adjacency.setdefault(b, set()).add(a)
        groups = components(adjacency)
        if not groups:
            raise RuntimeError(f"No intrahepatic {name} tree found")
        keep = set(groups[0])
        adjacency = {i: adjacency[i] & keep for i in keep}
        if sum(map(len, adjacency.values())) != 2 * (len(keep) - 1):
            raise RuntimeError(f"Largest intrahepatic {name} component contains cycles")
        leaves = [i for i in sorted(keep) if len(adjacency[i]) == 1]
        root = max(leaves, key=lambda i: (radius.GetTuple1(i), -i))
        distance = vtk.vtkImplicitPolyDataDistance()
        distance.SetInput(surfaces["liver"])
        clearance = {i: abs(distance.EvaluateFunction(network.GetPoint(i))) for i in leaves}
        eligible = [i for i in leaves if i != root and
                    .0002 <= radius.GetTuple1(i) <= .0015 and
                    clearance[i] > max(.004, 3 * radius.GetTuple1(i))]
        selected = sorted(eligible, key=lambda i: (-clearance[i], i))[:2]
        if not selected:
            raise RuntimeError(f"No interior {name} growth attachment found")
        # Dense centreline samples are numerical points, not separate physical
        # junctions. Make each selected terminal's incoming interval longer
        # than its diameter. Older nonincident intervals must still pass the
        # growth wall check; this operation does not exempt them.
        removed_tail_points = 0
        maximum_tail_deviation = 0.
        for tip in selected:
            removed = []
            parent = next(iter(adjacency[tip]))
            while len(adjacency[parent]) == 2 and np.linalg.norm(
                    np.array(network.GetPoint(tip)) - network.GetPoint(parent)) < max(
                        .004, 4 * max(radius.GetTuple1(tip), radius.GetTuple1(parent))):
                ancestor = next(i for i in adjacency[parent] if i != tip)
                removed.append(parent)
                adjacency[tip].remove(parent)
                adjacency[tip].add(ancestor)
                adjacency[ancestor].remove(parent)
                adjacency[ancestor].add(tip)
                del adjacency[parent]
                keep.remove(parent)
                parent = ancestor
            start = np.array(network.GetPoint(tip))
            direction = np.array(network.GetPoint(parent)) - start
            for i in removed:
                relative = np.array(network.GetPoint(i)) - start
                t = np.clip(relative @ direction / (direction @ direction), 0., 1.)
                maximum_tail_deviation = max(maximum_tail_deviation,
                                              float(np.linalg.norm(relative - t * direction)))
            removed_tail_points += len(removed)
        ids = {old: new for new, old in enumerate(sorted(keep))}
        points = vtk.vtkPoints()
        points.SetDataTypeToDouble()
        for i in sorted(keep):
            points.InsertNextPoint(network.GetPoint(i))
        grid = vtk.vtkUnstructuredGrid()
        grid.SetPoints(points)
        pending = deque([root])
        seen = {root}
        while pending:
            a = pending.popleft()
            for b in sorted(adjacency[a]):
                if b in seen:
                    continue
                seen.add(b)
                pending.append(b)
                edge = vtk.vtkIdList()
                edge.InsertNextId(ids[a])
                edge.InsertNextId(ids[b])
                grid.InsertNextCell(vtk.VTK_LINE, edge)
        for field, values in [("Radius", [radius.GetTuple1(i) for i in sorted(keep)]),
                              ("Root", [int(i == root) for i in sorted(keep)]),
                              ("Grow", [int(i in selected) for i in sorted(keep)])]:
            array = vtk.vtkDoubleArray()
            array.SetName(field)
            for value in values:
                array.InsertNextValue(value)
            grid.GetPointData().AddArray(array)
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(str(output / f"{name}.vtk"))
        writer.SetInputData(grid)
        writer.SetFileTypeToASCII()
        if writer.Write() != 1:
            raise RuntimeError(f"Cannot write {name} line grid")
        manifest["components"][source].update(
            inside_components=[len(group) for group in groups],
            retained_tree_nodes=len(keep), retained_tree_edges=len(keep)-1,
            removed_tail_points=removed_tail_points,
            maximum_tail_deviation_m=maximum_tail_deviation,
            root=ids[root], growth_nodes=[ids[i] for i in selected],
            growth_radii=[radius.GetTuple1(i) for i in selected])
    (output / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output / "provenance.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New output directory")
    parser.add_argument("--mesh-archive", type=Path, help="Reuse a downloaded MESHES_VTK.zip")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
