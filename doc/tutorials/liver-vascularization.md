# Vascularize a liver from a surface and two 1D trees

This tutorial starts with a closed liver surface and two rooted 1D vessel
meshes. `Domain<3,3>` meshes the enclosed tissue with the Gmsh API. The
`liver_vascular` application then extends the supplied arterial and venous
leaves toward common tissue territories. The supplied nodes, vessels and
radii remain fixed during growth.

```{mermaid}
flowchart LR
  S[Closed liver surface] --> M[Gmsh volume mesh]
  A[Arterial 1D tree] --> G[Paired vascular growth]
  V[Venous 1D tree] --> G
  M --> G
  G --> O[Vascularized tissue and physical archive]
```

The worked example uses **3D-IRCADb-1, case 12**. The
[IRCAD download](https://www.ircad.fr/research-and-development/data-sets/liver-segmentation-3d-ircadb-01/)
contains organ and vessel **surface** meshes, rather than ready-made
centerlines. A preparation script extracts approximate centerlines and writes
ordinary 1D VTK meshes. If you already have suitable 1D meshes, start at
[Describe the input trees](#describe-the-input-trees).

This is a geometric growth example. The pressures, flow and extension settings
below are illustrative, not a patient calibration. The example uses hepatic
arterial supply and a retained hepatic venous tree. It does not model the
additional portal-venous inflow or establish physiological liver perfusion.

## Build the application

Use deal.II with its Gmsh API, VTK and simplex support. The standalone growth
application does not require the optional `1dsolver` library.

```bash
cmake -S . -B /tmp/immersx-liver-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_COUPLED_PROBLEMS=OFF
cmake --build /tmp/immersx-liver-build --target liver_vascular -j 2
```

On gpu04, activate the installed Spack environment first:

```bash
source /software/spackgpu-v2/share/spack/setup-env.sh
spack env activate dealii
```

When configuring a new build there, select the MPI compiler wrapper with
`-DCMAKE_CXX_COMPILER=mpicxx` and set
`-DDEAL_II_DIR=/software/dealii/dealii-inst` if CMake does not discover it.
A Debug build names the executable `liver_vascular_debug`.

## Prepare the IRCAD inputs

The preparation step needs Python, VTK, NumPy and VMTK. The tested environment
used Python 3.10, VMTK 1.5.0, VTK 9.2.6 and NumPy 2.2.6. For example, prepare
an isolated environment with conda or micromamba:

```bash
micromamba create -y -n ircad-prep -c conda-forge \
  python=3.10 vmtk=1.5.0 vtk=9.2.6 numpy=2.2.6
micromamba run -n ircad-prep python scripts/prepare_ircad_liver.py \
  /tmp/immersx-liver-build/data/ircad-case12
```

The output directory must be new. The script downloads only the case's
`MESHES_VTK.zip`, checks the three source-file SHA-256 hashes, and keeps the
original files in `raw/`. To reuse an existing copy of that archive, add
`--mesh-archive /absolute/path/to/MESHES_VTK.zip`.

The script writes:

| File | Purpose |
| --- | --- |
| `liver.stl` | Closed tissue surface, in metres |
| `arterial.vtk` | Rooted arterial 1D mesh with radius and growth-leaf fields |
| `venous.vtk` | Rooted venous 1D mesh with the same fields |
| `artery.stl`, `venacava.stl` | Retained vessel surfaces for inspection |
| `provenance.json` | Source hashes, units, versions and preparation measurements |

The source coordinates are in millimetres. All prepared coordinates and radii
are converted to metres. The script selects the largest liver component and
cancels two oppositely oriented copies of the same triangle that form a
zero-volume fin. It rejects any remaining open or nonmanifold liver edges.
The raw source files are unchanged.

VMTK's approximate network extraction identifies tubular paths and radii.
The script keeps the largest connected intrahepatic component of each vessel
system, checks that it is a tree, and selects its widest leaf as the root.
Disconnected components are listed in `provenance.json`; they are not added as
obstacles in this example. Inspect these choices before adapting the recipe to
another anatomy.

Two interior leaves per system are marked for extension. Dense samples near
these leaves can put several numerical intervals within one vessel diameter.
Treating each interval as a separate physical vessel then blocks a new branch
against an older interval of the same tube. The preparation script removes
only degree-two samples from the selected tails until the incoming interval
is sufficiently long. It preserves the leaf and bifurcation positions and
records the maximum geometric deviation. For the tested inputs, this removes
eight arterial samples and seven venous samples, with maximum deviations of
about 0.817 mm and 1.341 mm respectively.

The resulting input has **176 arterial and 1,084 venous line cells**. These are
the fixed inputs to the growth calculation, not a claim of exact centerline
reconstruction from the medical images.

The IRCAD data use the CC BY-NC-ND 4.0 licence. Keep downloaded and derived
patient data outside the repository and consult the source terms before
sharing them. Cite Luc Soler and coauthors, *3D image reconstruction for
comparison of algorithm database: A patient specific anatomical and medical
image database*, IRCAD technical report, 2010.

(describe-the-input-trees)=
## Describe the input trees

Each input is a legacy VTK line mesh with three scalar point-data arrays:

| Array | Meaning |
| --- | --- |
| `Radius` | Positive vessel radius in metres |
| `Root` | One nonzero leaf: the arterial inlet or venous outlet |
| `Grow` | Nonzero on the non-root leaves to extend |

The mesh must be connected and acyclic. Line-cell direction need not already
follow the flow: the application orients edges from the root toward tissue.
Venous geometric orientation is also root-to-tissue; its physical flow is in
the opposite direction. Radii interpolate between the two endpoints of each
line cell. Viscosity comes from the parameter file.

The application assigns physical node ids from the arterial point indices,
then the venous point indices offset by the number of arterial points. These
are physical graph identities, not solver block numbers. For the supplied
prepared case, the arterial root is node **83** and the venous root is node
**177**. The reference pressure entries below refer to these roots. If you
change the input meshes, update the pressure keys to match their `Root` fields.

## Mesh the tissue and grow both systems

CMake generates the runnable file
`/tmp/immersx-liver-build/data/vascularization/liver.prm` from the checked-in
template. Its central settings are:

```text
subsection Liver
  set Arterial grid = /tmp/immersx-liver-build/data/ircad-case12/arterial.vtk
  set Venous grid = /tmp/immersx-liver-build/data/ircad-case12/venous.vtk
  set Output directory = /tmp/immersx-liver-build/test_output/liver-vascularization
  subsection Domain
    set Grid generator = /tmp/immersx-liver-build/data/ircad-case12/liver.stl
    set Triangulation type = fullydistributed
    subsection Surface mesh
      set Input = auto
      set Target edge length = .012
      set Feature angle = 60
      set Volume cells = tetrahedra
    end
  end
  subsection Vascularization
    subsection Growth
      set Seed length = .003
      set Minimum vessel length = .0002
      set Candidate vessels = 16
      set Local optimization steps = 5
    end
    subsection Sampling
      set Terminal count = 1000
      set Random seed = 23
    end
    subsection Physiology
      set Dynamic viscosity = .004
      set Total new-tissue reference flow = 2e-9
      set Reference pressures = 83:12000,177:2000
      set Minimum radius = .00005
    end
    subsection Venous network
      set Enabled = true
      set Terminal separation = .003
    end
  end
end
```

Use the generated file for the complete tested controls, including the
candidate, optimization and geometry budgets. All lengths here are in metres,
pressures in pascals and flow in cubic metres per second.

`Domain<3,3>` first distinguishes an existing volume from a surface. A surface
is read into `Triangulation<2,3>`; reading it into `Triangulation<3,3>` does not
create a volume. The separate meshing step classifies the surface into Gmsh
patches, parametrizes them, creates a closed surface loop and generates the
volume. It then transfers the tetrahedra to deal.II before partitioning.
See [surface domains](../how-to/surface-domains) for the supported input readers
and every surface-meshing parameter.

For this surface, a feature angle of 60 degrees allows Gmsh to build valid
patch parametrizations; 40 degrees failed in the tested installation. With a
12 mm target edge length the tested mesh had **13,351 tetrahedra** and a volume
of about **1.515 litres**. The cleaned surface encloses about 1.530 litres, so
the coarse remeshing changes the enclosed volume by about 0.94%.

The arterial generator starts at the marked input leaves and assigns new
common tissue territories. The venous generator subsequently extends the
marked venous leaves to these territories, with distinct endpoints separated
by 3 mm. Each accepted pair receives a positive microvascular resistance.
Both growth stages check their new vessel envelopes against the tissue
boundary and both vessel systems. Input geometry remains fixed.

The separation must accommodate the endpoint radii. The initial 1 mm trial
failed venous startup on this case; 3 mm passed the same geometry checks.
Increasing a finite search budget does not make a geometrically blocked
connection admissible.

The recipe requests 1,000 common tissue territories, each with one new
arterial and one new venous terminal. For a short trial, set `Terminal count`
to `32` and pass `--pairs 32` to the checker below.

Run the full recipe in Release on 32 MPI processes, as in the verified
1,000-pair calculation:

```bash
mpirun -np 32 /tmp/immersx-liver-build/liver_vascular \
  /tmp/immersx-liver-build/data/vascularization/liver.prm
```

For a short two-process trial, set `Terminal count = 32` and choose a new
output directory in a copy of the parameter file, then run:

```bash
mpirun -np 2 /tmp/immersx-liver-build/liver_vascular /absolute/path/to/liver-mpi.prm
```

The application resolves relative paths against its parameter file. It can be
launched from an unrelated directory. Each run requires a new output directory;
an unsuccessful result is not archived as an accepted vascular network.

For hexahedra, change `Volume cells` to `hexahedra`. Gmsh still generates the
tetrahedral volume first, then deal.II divides each tetrahedron into four
conforming hexahedra. The tested liver consequently had **53,404 hexahedra**
with the same enclosed volume. This is an unstructured hexahedral mesh, not a
structured organ-aligned grid. Regenerate and verify the vascular result on
the selected background; changing its cells can change sampled growth paths.

## Inspect and verify the result

Open `liver_0.pvtu` and `vessels_0.pvtu` in ParaView. Give the liver a low opacity
and colour the vessel field `System`: zero is arterial and one is venous.
`Generated` distinguishes fixed imported segments (zero) from new segments
(one). `Radius` is the radius at the numerical interval midpoint, repeated at its
patch endpoints. Use it as the point scalar for a varying-radius Tube filter.

The output also contains:

- `network/`: a native distributed physical archive, including territories and
  terminal pairs;
- `nodes-*.csv` and `vessels-*.csv`: each owned physical record once, with stable
  integer identifiers, coordinates, radii and provenance;
- `terminals-*.csv` and `pairs-*.csv`: common territories, demand and bed
  resistance;
- `summary.csv`: global background, vessel and pair counts and vessel volume
  fraction.

Check the physical result independently of the generator:

```bash
micromamba run -n ircad-prep python scripts/check_liver_vascular.py \
  /tmp/immersx-liver-build/data/ircad-case12 \
  /tmp/immersx-liver-build/test_output/liver-vascularization \
  --pairs 1000 --separation .003 \
  --report /tmp/immersx-liver-build/test_output/liver-vascularization/verification.json
```

The checker compares all supplied nodes, ports, edges and endpoint radii with
the exported physical records. It requires growth in both systems, unique
arterial–venous pairings, equal paired demand, positive bed resistances and the
requested separation. It also samples each new vessel against the exported
volume. This sample check supplements the generator's continuous geometry
certification; it does not replace it.

The verified 32-rank Release run produced **1,000 tissue pairs and 5,256
physical vessels**: the 1,260 fixed input segments plus 1,998 new arterial and
1,998 new venous segments. The independent check preserved all input nodes,
ports, edges and radii and found a positive sampled wall margin for every new
vessel. Numerical subdivision of these physical vessels is a separate step
and can produce more 1D cells.

The dense recipe uses 16 candidate vessels per insertion; eight candidates
exhausted the finite sampling budget on this case. MPI partitions change the
sampled growth paths, so another process count can need different search
controls. The smaller 32-pair case was also verified on two and four ranks.

A successful growth calculation demonstrates a paired geometric network on
this tissue mesh. It does not establish physiological accuracy, resolve
capillaries, or guarantee that every point of the liver is adequately served.
Use the existing [coverage and calibration controls](../how-to/vascular-generation)
when imposing a service distance or a volume objective.
