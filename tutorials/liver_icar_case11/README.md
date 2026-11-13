# Coupled IRCAD11 liver simulation

This tutorial records the coupled liver setup for 3D-IRCADb-01 case 11. It follows the structure of the [IRCAD12 solver pilot](liver-icar-coupled), but uses case 11 assets and records their current limits. Case 11 is not a calibrated patient model. The vessel network is an anatomically seeded, model-grown approximation; it is not a complete segmented vascular tree.

## Inputs and provenance

Download the [case11 reproducibility snapshot](https://github.com/luca-heltai/immersx/releases/download/liver-h-scaled-2026-11-10/ircad-case11-reproducibility-2026-11-11-v2.zip) linked from [issue #393](https://github.com/luca-heltai/immersx/issues/393). It contains scripts, parameter files, manifests, audit reports and the two figures below; it does **not** contain the original or derived anatomical meshes, native network binaries or run outputs. Keep the ZIP SHA256 (`3c45d3088ee3648fcfc6cda10c84827fe186c6b0e1c9df3e87b4444570cd6461`), provenance and IRCAD attribution with the extracted files.

The reproducibility snapshot's `download_case11.py` retrieves all 11 original VTKs and checks their CRC/SHA256. Preparation converts their coordinates from millimetres to metres, extracts the case-specific anatomy, builds the modeled arterial feeder, meshes the liver and grows the dense network. The command sequence and required VMTK/VTK/Gmsh environments are recorded in issue #393. The parameter templates in `data/liver_icar_case11/` expect the verified and prepared files under `assets/`:

```text
assets/
  inputs/inlet_ha.csv
  meshes/small.msh
  meshes/medium.msh
  network/manifest
  network/part-*.txt
```

The current coupled inputs include one physical HA/PV/HV archive. Both supplied profiles load that same archive and discretize it on different tissue meshes. Thus they do not yet implement the intended three-network refinement study. The small mesh has 4,804 tetrahedra and the medium mesh has 71,272 tetrahedra. The planned fine mesh has 1,052,000 tetrahedra, but its independently grown vascular archive is not complete. Do not use the IRCAD12 networks as substitutes.

The parameter templates are relocatable equivalents of the supplied small and medium profiles. Set the input asset root and output root in a private copy if they are not at the paths expanded by the build. The `tiny` profile reuses the small mesh and network with P1 solid elements. It is a setup/debug profile, not a validated 100k-DoF case.

![Case 11 vascular growth before and after](https://github.com/luca-heltai/immersx/releases/download/liver-h-scaled-2026-11-10/ircad-case11-dense-before-after.png)

![Case 11 liver volume and section](https://github.com/luca-heltai/immersx/releases/download/liver-h-scaled-2026-11-10/ircad-case11-dense-volume-section.png)

Download and verify the snapshot, then prepare the anatomy with the environments recorded in issue #393:

```sh
curl -L \
  https://github.com/luca-heltai/immersx/releases/download/liver-h-scaled-2026-11-10/ircad-case11-reproducibility-2026-11-11-v2.zip \
  -o ircad-case11-reproducibility.zip
echo '3c45d3088ee3648fcfc6cda10c84827fe186c6b0e1c9df3e87b4444570cd6461  ircad-case11-reproducibility.zip' | sha256sum -c -
mkdir case11-repro
unzip ircad-case11-reproducibility.zip -d case11-repro
cd case11-repro
python3 download_case11.py
/path/to/vmtk/python prepare_case11.py inputs --mesh-archive MESHES_VTK.zip --with-portal
/path/to/vtk-numpy/python create_arterial_feeder.py
/path/to/vtk/python select_growth_leaves.py
/path/to/gmsh-vtk/python generate_meshes.py
```

Run the accepted coarse growth and the independent audits as recorded in the snapshot:

```sh
/path/to/build/liver_vascular generate-dense-no-geometry-distinct.prm
python3 check_topology_hydraulics.py inputs generation-dense-4804-no-geometry-distinct \
  --triplets 4804 --separation .004 --portal-separation .003 \
  --portal-flow-fraction .69576933619234205 --viscosity .00371
python3 check_coverage.py generation-dense-4804-no-geometry-distinct
python3 prepare_parameters.py
```

Place the prepared `meshes/`, `network/` and `inputs/inlet_ha.csv` under the `assets/` layout above, or edit a private parameter copy to point to their actual paths. The build expands `@TEST_DATA_DIR@` and `@TEST_OUTPUT_DIR@` in the checked-in `.prm.in` templates.

## Anatomy and model assumptions

The source VTK coordinates are millimetres; the preparation step scales them by `.001` to metres. Eleven source VTKs are checksum-verified. VMTK found no usable intrahepatic arterial points in this case (1,247 extracted points on 105 lines, zero inside the liver). With explicit authorization, the pipeline adds a modeled 15 mm arterial feeder with 2.2 mm radius. Case 11's selected image-extracted trees contain 888 hepatic venous nodes and 301 portal nodes; other disconnected components are excluded and recorded in the provenance report.

The accepted dense graph has 4,804 HA/PV/HV terminal triplets and 30,011 physical segments. It was grown with geometric admissibility disabled; topology, archive import and hydraulic checks pass, but geometric containment is not certified. The report finds 197 portal and 271 venous terminal locations outside the background volume. The independent whole-cell `2H` distance certificate covers all coarse background cells, but measures proximity rather than containment or perfusion. Do not describe the graph as a validated anatomical vasculature.

| Background | Target edge length | Tetrahedra | Volume |
| --- | ---: | ---: | ---: |
| small | 12 mm | 4,804 | 1.37208 L |
| medium | 4.5 mm | 71,272 | 1.38616 L |
| large | 1.8 mm | 1,052,000 | 1.38826 L |

The hydraulic reference flow is `2e-9 m^3/s` (0.12 mL/min), with portal fraction 0.6957693362. These are demonstrative model values, not patient calibration. The arterial input is rescaled from the MOX tutorial signal to this target. Mechanical material properties and exterior Robin support are also modeling assumptions.

The effective lift scale is the local reference radius `sqrt(A0/pi)`, derived from the archived physical radii and taper and frozen in time. Although the parameter template contains `MetricFlowX vessel wall lift/Thickness = .001`, the application explicitly passes that local radius to `lift`; the configured 1 mm value does not define the effective lift thickness for this coupled case.

## Build and prepare a run

Build the coupled executable against compatible deal.II, MetricFlowX, SUNDIALS and MPI installations. The MetricFlowX dependency must include the corrections recorded in [PR #87](https://github.com/luca-heltai/metric-flow-x/pull/87).

```bash
cmake -S . -B /absolute/path/to/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_COUPLED_PROBLEMS=OFF \
  -DMetricFlowX_DIR=/absolute/path/to/metric-flow-x/lib/cmake/MetricFlowX
cmake --build /absolute/path/to/build --target metric_flow_x_elastodynamics -j 4
```

Copy the downloaded bundle to the build-tree data location expected by the templates, or edit their `@TEST_DATA_DIR@` paths in a private parameter copy. Keep outputs outside the source tree. To inspect the effective input before launching the solver:

```bash
mpirun -np 16 /absolute/path/to/build/apps/metric_flow_x_elastodynamics \
  /absolute/path/to/build/data/liver_icar_case11/small.prm
```

Choose a fresh output directory in the parameter file for each run. The supplied small and medium profiles use the deliberately permissive tolerances from the case11 campaign: IDA absolute tolerance `1e-1`, relative tolerance `1`, an initial absolute and relative tolerance factor of `1e8` ramped down by 7.4 s, and loose component tolerances for area, velocity, displacement and wall pressure. These settings let the solver return output for diagnosis; they do not give accurate trajectories. Small and medium runs completed with return code 0 in the campaign, but they reused the same network and are not physically validated. A separate small diagnostic using MUMPS-BLR-FGMRES reached 7.27 s over a short 0.07 s interval; it is not a full beat. `mumps-blr-fgmres` is not a selectable IDA solver type in current `master`; the diagnostic used an isolated experiment source snapshot and is not runnable from this branch yet.

## Profiles and solver settings

| Profile | Tissue mesh | Solid / flow FE | Linear solver | Evidence and limits |
| --- | ---: | --- | --- | --- |
| `tiny` | small, 4,804 cells | P1 / DG2 | MUMPS | Setup/debug profile only; not a validated 100k-DoF run |
| `small` | small, 4,804 cells | P2 / DG2 | MUMPS | Completed in the campaign with very loose tolerances; not physically validated |
| `small-blr-diagnostic` | small, 4,804 cells | P2 / DG2 | MUMPS-BLR-FGMRES | 16 ranks, FGMRES limit 100; reached 7.27 s in a short diagnostic interval |
| `medium` | medium, 71,272 cells | P2 / DG2 | Iterative | Uses replicated-flow inverse limit 2,048 and materialized Schur limit 512; no validated case11 transient |
| `fine` | 1,052,000 cells | P2 / DG2 | Iterative | Network generation remains incomplete; not runnable as a matched case |

The `small` profile uses direct MUMPS. The additional `small-blr-diagnostic` run used MUMPS with block low-rank compression and FGMRES, 16 MPI ranks, an FGMRES limit of 100 iterations and workspace relaxation 20,000. This configuration came from an isolated experiment source snapshot; `IDAAdapter` on `master` only offers `iterative`, `umfpack` and `mumps`. Support for the coupled IDA path is under review in [PR #416](https://github.com/luca-heltai/immersx/pull/416). The `medium` profile uses an iterative solver, limits the replicated flow inverse to 2,048 DoFs and materializes the Schur operator only up to 512 DoFs. All profiles preserve the large tolerance ramp from the campaign; do not interpret process completion as physical validation. The checked-in parameter templates reproduce the supplied small and medium settings. The BLR-FGMRES overlay is recorded in issue #393's run artifacts.

The case11 exploratory campaign also produced runs that completed with return code 0 under fixed steps and very loose tolerances. That exit status only confirms that the executable finished. It does not establish physical validity. In particular, runs that disable geometric constraints or accept non-finite/negative physical states must not be presented as successful simulations.

## Vascular refinement and target sizes

The intended tutorial targets are approximately 100k, 1M and 10M coupled DoFs, with three separately grown HA/PV/HV networks matched to the small, medium and fine tissue meshes. The available archive does not yet meet that contract: the supplied small and medium profiles share one network, and the fine generation has not produced a complete archive. Therefore this page does not claim those three target sizes or a network-refinement result. Once each role-specific archive is complete, update each profile's `Archived network/Directory` to its own archive and record the measured coupled DoFs and vessel-interval to background-cell ratios.

The small and medium tissue meshes are independent Gmsh remeshes, so their discrete domains differ slightly. Report their actual volumes and mesh statistics with any comparison. Retain all per-run parameter files, `run.json`, logs, output VTK files and network manifests together so the solver choice and network identity remain auditable.

## Interpretation

This is a reproducibility guide for the current case11 inputs and solver experiments. It does not assert convergence, physiological calibration, geometric admissibility of the dense network, or a completed 100k/1M/10M comparison. For anatomy preparation and the separate vascular growth workflow, see [Vascularize a liver](liver-vascularization).
