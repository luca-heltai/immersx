# Build a volume domain from a surface

`Domain<3,3>` accepts a volume mesh or a closed surface mesh through `Grid
generator`. It first attempts to read a volume. If that fails, it reads a
`Triangulation<2,3>` and generates the enclosed volume using the Gmsh API.
STL and MSH use deal.II's Gmsh reader. Other formats use the same surface
readers as an ordinary embedded Domain, including VTK when available. Additional
quadrilateral surface formats such as OBJ are detected by Assimp when deal.II
provides it, subject to the element types supported by its Assimp reader.

```text
subsection Domain
  set Grid generator = liver.stl
  set Triangulation type = fullydistributed
  set Grid scale = 0.001
  subsection Surface mesh
    set Input = auto
    set Target edge length = 10
    set Feature angle = 40
    set Volume cells = tetrahedra
  end
end
```

Here the input coordinates and target edge length are in millimetres. `Grid
scale` converts the resulting volume to metres after meshing. Initial
refinement also happens after meshing. Set `Input = surface` to require surface
meshing explicitly, or `Input = volume` to disable surface detection. These
parameters are declared only for `Domain<3,3>`.

The surface must contain one connected, closed manifold shell. Open edges,
nonmanifold edges and separate shells are rejected. Repair the input and choose
the intended organ component before passing it to Domain. Surface boundary and
material labels are not transferred through remeshing; generated cells and
boundaries use zero ids. Input VTK field transfer is unavailable when generating
a new volume from a surface.

Gmsh classifies the triangulated surface into patches using `Feature angle`,
constructs a parametrization and a closed surface loop, and meshes its interior.
`Target edge length` sets both Gmsh mesh-size bounds; curvature and patch
boundaries can still require smaller edges. This requires deal.II built with
the Gmsh API. Domain initializes a Gmsh session when needed and preserves
an existing caller-owned session.
The implementation follows [Gmsh tutorial t13](https://gmsh.info/doc/texinfo/#t13).
It uses the process-global Gmsh model, as does deal.II's API reader, and must
not run concurrently with another Gmsh operation in the same process.

For an all-hexahedral mesh, use `Volume cells = hexahedra`. Gmsh first generates
tetrahedra, then its native all-hexahedral subdivision divides each
tetrahedron into four conforming hexahedra. This also works with deal.II 9.7.
The resulting unstructured mesh has more cells. Hexahedra support both
Domain triangulation types. Tetrahedra require `fullydistributed`.

Existing volume meshes retain their cells, ids and supported input fields;
the surface meshing settings do not remesh them. Embedded `Domain<2,3>` inputs
remain surface meshes.
