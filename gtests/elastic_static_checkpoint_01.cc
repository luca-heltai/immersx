// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <deal.II/base/function_parser.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastic_static.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

#include "../apps/metric_flow_x_elastostatic_checkpoint.h"
#include "test_paths.h"

using namespace dealii;
using namespace ImmersX;

namespace
{
  std::vector<std::string>
  mesh_signature(const ElasticStaticProblem<3> &problem)
  {
    std::vector<std::string> local, global;
    for (const auto &cell : problem.triangulation().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          std::ostringstream entry;
          entry << std::setprecision(17) << cell->id().to_string() << ':'
                << cell->material_id() << ':' << cell->manifold_id();
          for (const auto vertex : cell->vertex_indices())
            for (unsigned int d = 0; d < 3; ++d)
              entry << ':' << cell->vertex(vertex)[d];
          for (const auto face : cell->face_indices())
            {
              entry << ':' << cell->face(face)->manifold_id();
              if (cell->face(face)->at_boundary())
                entry << ':' << cell->face(face)->boundary_id();
              else
                entry << ":interior";
            }
          local.push_back(entry.str());
        }
    for (const auto &entries :
         Utilities::MPI::all_gather(MPI_COMM_WORLD, local))
      global.insert(global.end(), entries.begin(), entries.end());
    std::sort(global.begin(), global.end());
    return global;
  }
  std::vector<std::string>
  coefficient_signature(const ElasticStaticProblem<3> &problem)
  {
    ElasticStaticProblem<3>::VectorType relevant;
    relevant.reinit(problem.dof_handler().locally_owned_dofs(),
                    DoFTools::extract_locally_relevant_dofs(
                      problem.dof_handler()),
                    MPI_COMM_WORLD);
    relevant = problem.solution();
    relevant.update_ghost_values();
    std::vector<std::string> local, global;
    Vector<double> values(problem.dof_handler().get_fe().n_dofs_per_cell());
    for (const auto &cell : problem.dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_values(relevant, values);
          std::ostringstream entry;
          entry << cell->id().to_string() << std::hexfloat;
          for (const auto value : values)
            entry << ':' << value;
          local.push_back(entry.str());
        }
    for (const auto &entries :
         Utilities::MPI::all_gather(MPI_COMM_WORLD, local))
      global.insert(global.end(), entries.begin(), entries.end());
    std::sort(global.begin(), global.end());
    return global;
  }

  void
  check_partitioned_checkpoint(const unsigned int cells)
  {
    ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
    const auto directory = TestPaths::output_directory(
      "elastic-static-checkpoint-partition-" + std::to_string(cells));
    std::filesystem::create_directories(directory);
    const auto checkpoint =
      (std::filesystem::path(directory) / "solid").string();
    ParameterAcceptor::clear();
    ElasticStaticParameters<3> original_parameters("/Partition original/");
    ElasticStaticParameters<3> restored_parameters("/Partition restored/");
    const auto                 configuration = [](const std::string &section) {
      return "subsection " + section + R"(
        set FE degree = 2
        set Dirichlet boundary ids = 0,1,2,3,4,5
        subsection Domain
          set Triangulation type = fullydistributed
          set Initial refinement = 0
        end
        subsection Functions
          subsection Right hand side
            set Function expression = -6*(1+t);-6*(1+t);-6*(1+t)
            set Variable names = x,y,z,t
          end
          subsection Dirichlet boundary conditions
            set Function expression = (1+t)*(1+x*x);(1+t)*(2+y*y);(1+t)*(3+z*z)
            set Variable names = x,y,z,t
          end
        end
      end
      )";
    };
    initialize_parameters_from_string(configuration("Partition original") +
                                      configuration("Partition restored"));
    ElasticStaticProblem<3> original(original_parameters);
    original.use_fixed_stiffness();
    original.make_grid([&](auto &domain) {
      Triangulation<3> serial;
      GridGenerator::subdivided_hyper_rectangle(serial,
                                                std::vector<unsigned int>{cells,
                                                                          1,
                                                                          1},
                                                Point<3>(),
                                                Point<3>(1., 1., 1.),
                                                true);
      unsigned int index = 0;
      for (const auto &cell : serial.active_cell_iterators())
        {
          cell->set_subdomain_id(cells == 1 || index + 1 < cells ? 0 : 1);
          cell->set_material_id(cell->center()[0] < .5 ? 11 : 17);
          ++index;
        }
      const auto description = TriangulationDescription::Utilities::
        create_description_from_triangulation(serial, MPI_COMM_WORLD);
      auto &mesh = dynamic_cast<Domain<3>::FullyDistributedTriangulation &>(
        domain.triangulation());
      mesh.create_triangulation(description);
    });
    original.setup_fe();
    const auto ownership = Utilities::MPI::all_gather(
      MPI_COMM_WORLD, original.triangulation().n_locally_owned_active_cells());
    ASSERT_EQ(ownership.size(), 2u);
    EXPECT_EQ(ownership[0], cells == 1 ? 1u : cells - 1);
    EXPECT_EQ(ownership[1], cells == 1 ? 0u : 1u);
    const double time = .37;
    original.update_time(time);
    FunctionParser<3> exact(3);
    exact.initialize("x,y,z,t",
                     "(1+t)*(1+x*x);(1+t)*(2+y*y);(1+t)*(3+z*z)",
                     {},
                     true);
    exact.set_time(time);
    auto accepted = original.solution();
    VectorTools::interpolate(original.mapping(),
                             original.dof_handler(),
                             exact,
                             accepted);
    original.set_solution(accepted);
    EXPECT_GT(original.solution().l2_norm(), 0.);
    const auto geometry     = mesh_signature(original);
    const auto coefficients = coefficient_signature(original);
    original.save_checkpoint(checkpoint);
    EXPECT_EQ(original.stiffness_assemblies(), 1u);

    ElasticStaticProblem<3> restored(restored_parameters);
    restored.use_fixed_stiffness();
    restored.setup_from_checkpoint(checkpoint);
    EXPECT_EQ(Utilities::MPI::all_gather(
                MPI_COMM_WORLD,
                restored.triangulation().n_locally_owned_active_cells()),
              ownership);
    EXPECT_EQ(mesh_signature(restored), geometry);
    EXPECT_EQ(coefficient_signature(restored), coefficients);
    restored.update_time(time);
    EXPECT_EQ(coefficient_signature(restored), coefficients);
    EXPECT_EQ(restored.stiffness_assemblies(), 1u);
  }
} // namespace

TEST(ElasticStaticCheckpoint, BOTH_NativeMeshAndAcceptedDisplacementRoundTrip)
{
  const auto directory = TestPaths::output_directory(
    "elastic-static-checkpoint-" +
    std::to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)));
  std::filesystem::create_directories(directory);
  const auto input_mesh =
    (std::filesystem::path(directory) / "input.ucd").string();
  const auto checkpoint = (std::filesystem::path(directory) / "solid").string();
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      Triangulation<3> mesh;
      GridGenerator::hyper_cube(mesh, 0., 1., true);
      mesh.refine_global(1);
      for (const auto &cell : mesh.active_cell_iterators())
        cell->set_material_id(cell->center()[0] < .5 ? 11 : 17);
      GridOut output;
      // Keep this fixture independent of Gmsh entity conversion.
      output.set_flags(GridOutFlags::Ucd(false, true, true));
      std::ofstream file(input_mesh);
      output.write_ucd(mesh, file);
    }
  MPI_Barrier(MPI_COMM_WORLD);

  ParameterAcceptor::clear();
  ElasticStaticParameters<3> original_parameters("/Original/");
  ElasticStaticParameters<3> restored_parameters("/Restored/");
  const auto                 configuration = [](const std::string &section) {
    return "subsection " + section + R"(
      set FE degree = 2
      set Dirichlet boundary ids = 0,1,2,3,4,5
      subsection Domain
        set Triangulation type = fullydistributed
        set Initial refinement = 1
      end
      subsection Functions
        subsection Right hand side
          set Function expression = -6*(1+t);0;0
          set Variable names = x,y,z,t
        end
        subsection Dirichlet boundary conditions
          set Function expression = (1+t)*(1+x*x);0;0
          set Variable names = x,y,z,t
        end
      end
      subsection Solver
        subsection Control
          set Tolerance = 1.e-13
          set Reduction = 1.e-13
        end
      end
    end
  )";
  };
  initialize_parameters_from_string(configuration("Original") +
                                    configuration("Restored"));
  original_parameters.domain_parameters.name_of_grid = input_mesh;
  ElasticStaticProblem<3> original(original_parameters);
  original.use_fixed_stiffness();
  original.setup();
  unsigned int material_11 = 0, material_17 = 0;
  for (const auto &cell : original.triangulation().active_cell_iterators())
    if (cell->is_locally_owned())
      {
        material_11 += cell->material_id() == 11;
        material_17 += cell->material_id() == 17;
      }
  EXPECT_GT(Utilities::MPI::sum(material_11, MPI_COMM_WORLD), 0u);
  EXPECT_GT(Utilities::MPI::sum(material_17, MPI_COMM_WORLD), 0u);
  const double time = .37;
  original.update_time(time);
  original.solve();
  const auto signature    = mesh_signature(original);
  const auto coefficients = coefficient_signature(original);
  original.save_checkpoint(checkpoint);

  // Restoring must not consult or regenerate the original source mesh.
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::filesystem::remove(input_mesh);
  MPI_Barrier(MPI_COMM_WORLD);
  restored_parameters.domain_parameters.name_of_grid = input_mesh;
  ElasticStaticProblem<3> restored(restored_parameters);
  restored.use_fixed_stiffness();
  restored.setup_from_checkpoint(checkpoint);
  EXPECT_EQ(coefficient_signature(restored), coefficients);
  restored.update_time(time);
  EXPECT_EQ(coefficient_signature(restored), coefficients);
  EXPECT_EQ(original.stiffness_assemblies(), 1u);
  EXPECT_EQ(restored.stiffness_assemblies(), 1u);
  EXPECT_EQ(mesh_signature(restored), signature);
  EXPECT_EQ(restored.dof_handler().get_fe().get_name(),
            original.dof_handler().get_fe().get_name());

  FunctionParser<3> exact(3);
  exact.initialize("x,y,z,t", "(1+t)*(1+x*x);0;0", {}, true);
  exact.set_time(time);
  ElasticStaticProblem<3>::VectorType difference;
  difference.reinit(restored.solution());
  VectorTools::interpolate(restored.mapping(),
                           restored.dof_handler(),
                           exact,
                           difference);
  difference -= restored.solution();
  EXPECT_LT(difference.l2_norm(), 1.e-9);
  auto saved = restored.solution();
  restored.solve();
  saved -= restored.solution();
  EXPECT_LT(saved.l2_norm(), 1.e-9);
}


TEST(ElasticStaticCheckpoint,
     BOTH_CylindricalMeshAndAcceptedDisplacementRoundTrip)
{
  namespace Checkpoint = elastostatic_checkpoint_detail;
  const auto directory = TestPaths::output_directory(
    "elastic-static-cylinder-checkpoint-" +
    std::to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)));
  std::filesystem::create_directories(directory);
  const auto checkpoint = (std::filesystem::path(directory) / "solid").string();

  ParameterAcceptor::clear();
  ElasticStaticParameters<3> original_parameters("/Cylinder original/");
  ElasticStaticParameters<3> restored_parameters("/Cylinder restored/");
  const auto                 configuration = [](const std::string &section) {
    return "subsection " + section + R"(
      set FE degree = 2
      set Dirichlet boundary ids = 0,1,2
      subsection Domain
        set Triangulation type = fullydistributed
        set Grid generator = subdivided_cylinder
        set Grid generator arguments = 2:0.25:0.5
        set Initial refinement = 1
      end
      subsection Functions
        subsection Right hand side
          set Function expression = -6*(1+t);0;0
          set Variable names = x,y,z,t
        end
        subsection Dirichlet boundary conditions
          set Function expression = (1+t)*(1+x*x);0;0
          set Variable names = x,y,z,t
        end
      end
      subsection Solver
        subsection Control
          set Tolerance = 1.e-13
          set Reduction = 1.e-13
        end
      end
    end
  )";
  };
  initialize_parameters_from_string(configuration("Cylinder original") +
                                    configuration("Cylinder restored"));
  ElasticStaticProblem<3> original(original_parameters);
  original.use_fixed_stiffness();
  original.setup();
  const double time = .37;
  original.update_time(time);
  original.solve();
  const auto signature    = mesh_signature(original);
  const auto coefficients = coefficient_signature(original);
  const auto manifolds =
    Checkpoint::describe_solid_manifolds(original.triangulation(),
                                         MPI_COMM_WORLD);
  original.save_checkpoint(checkpoint);

  ElasticStaticProblem<3> restored(restored_parameters);
  restored.use_fixed_stiffness();
  restored.setup_from_checkpoint(checkpoint, [&](auto &mesh) {
    Checkpoint::restore_solid_manifolds(mesh, manifolds);
  });
  EXPECT_EQ(coefficient_signature(restored), coefficients);
  restored.update_time(time);
  EXPECT_EQ(coefficient_signature(restored), coefficients);
  EXPECT_EQ(original.stiffness_assemblies(), 1u);
  EXPECT_EQ(restored.stiffness_assemblies(), 1u);
  EXPECT_EQ(mesh_signature(restored), signature);
  FunctionParser<3> exact(3);
  exact.initialize("x,y,z,t", "(1+t)*(1+x*x);0;0", {}, true);
  exact.set_time(time);
  ElasticStaticProblem<3>::VectorType difference;
  difference.reinit(restored.solution());
  VectorTools::interpolate(restored.mapping(),
                           restored.dof_handler(),
                           exact,
                           difference);
  difference -= restored.solution();
  EXPECT_LT(difference.l2_norm(), 1.e-9);
}

TEST(ElasticStaticCheckpoint, BOTH_ManifoldConstructorStateAndInactiveIds)
{
  namespace Checkpoint = elastostatic_checkpoint_detail;
  Triangulation<3> mesh;
  GridGenerator::subdivided_cylinder(mesh, 2, .25, .5);
  mesh.refine_global(1);
  Tensor<1, 3> direction, periodicity;
  direction[0] = 1.;
  for (unsigned int d = 0; d < 3; ++d)
    periodicity[d] = 4.;
  const Point<3> point_on_axis(.125, 0., 0.);
  mesh.set_manifold(0,
                    CylindricalManifold<3>(direction, point_on_axis, 4.e-11));
  mesh.set_manifold(7, FlatManifold<3>(periodicity, 3.e-11));
  // The active-only ID list must not lose geometry on inactive parents.
  for (const auto &cell : mesh.cell_iterators())
    if (cell->has_children())
      cell->set_manifold_id(7);
  const auto manifolds =
    Checkpoint::describe_solid_manifolds(mesh, MPI_COMM_WORLD);
  EXPECT_TRUE(std::any_of(manifolds.begin(),
                          manifolds.end(),
                          [](const auto &value) { return value.id == 7; }));

  Triangulation<3> restored;
  Checkpoint::restore_solid_manifolds(restored, manifolds);
  const auto &cylinder =
    dynamic_cast<const CylindricalManifold<3> &>(restored.get_manifold(0));
  EXPECT_EQ(cylinder.get_direction(), direction);
  EXPECT_EQ(cylinder.get_point_on_axis(), point_on_axis);
  EXPECT_EQ(cylinder.get_tolerance(), 4.e-11);
  const auto &flat =
    dynamic_cast<const FlatManifold<3> &>(restored.get_manifold(7));
  EXPECT_EQ(flat.get_periodicity(), periodicity);
  EXPECT_EQ(flat.get_tolerance(), 3.e-11);

  mesh.set_manifold(7, SphericalManifold<3>());
  EXPECT_THROW(Checkpoint::describe_solid_manifolds(mesh, MPI_COMM_WORLD),
               dealii::ExceptionBase);
}


TEST(ElasticStaticCheckpoint, MPI_NativeUnequalOwnershipRoundTrip)
{
  check_partitioned_checkpoint(3);
}

TEST(ElasticStaticCheckpoint, MPI_NativeEmptyRankRoundTrip)
{
  check_partitioned_checkpoint(1);
}

TEST(ElasticStaticCheckpoint, BOTH_RejectIncompleteAndIncompatibleMetadata)
{
  namespace Checkpoint = elastostatic_checkpoint_detail;
  const auto directory = TestPaths::output_directory(
    "elastic-static-invalid-checkpoint-" +
    std::to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)));
  const auto write_manifest = [&](const std::string          &name,
                                  const Checkpoint::Metadata &metadata) {
    const auto root = std::filesystem::path(directory) / name;
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::filesystem::create_directories(root);
        {
          std::ofstream file(root / "coupled.bin", std::ios::binary);
          boost::archive::binary_oarchive archive(file);
          archive << metadata;
        }
        Checkpoint::Manifest files{
          {"coupled.bin", std::filesystem::file_size(root / "coupled.bin")}};
        std::ofstream file(root / "COMPLETE", std::ios::binary);
        boost::archive::binary_oarchive archive(file);
        archive << files;
      }
    MPI_Barrier(MPI_COMM_WORLD);
    return root.string();
  };
  const auto incomplete =
    (std::filesystem::path(directory) / "checkpoint.incomplete").string();
  EXPECT_THROW(Checkpoint::read_metadata(incomplete, MPI_COMM_WORLD),
               dealii::ExceptionBase);
  Checkpoint::Metadata metadata;
  metadata.ranks        = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  metadata.time         = .37;
  metadata.parameters   = "saved parameters";
  metadata.reference_fe = "saved reference FE";
  metadata.lambda_fe    = "saved multiplier FE";
  const auto valid      = write_manifest("valid", metadata);
  const auto accepted   = Checkpoint::read_metadata(valid, MPI_COMM_WORLD);
  EXPECT_EQ(accepted.version, metadata.version);
  EXPECT_EQ(accepted.ranks, metadata.ranks);
  EXPECT_EQ(accepted.time, metadata.time);
  ++metadata.version;
  const auto incompatible = write_manifest("incompatible", metadata);
  EXPECT_THROW(Checkpoint::read_metadata(incompatible, MPI_COMM_WORLD),
               dealii::ExceptionBase);
  --metadata.version;
  ++metadata.ranks;
  const auto wrong_ranks = write_manifest("wrong-ranks", metadata);
  EXPECT_THROW(Checkpoint::read_metadata(wrong_ranks, MPI_COMM_WORLD),
               dealii::ExceptionBase);
}
