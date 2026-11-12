// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <deal.II/base/function_parser.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

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
  const auto signature = mesh_signature(original);
  original.save_checkpoint(checkpoint);

  // Restoring must not consult or regenerate the original source mesh.
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::filesystem::remove(input_mesh);
  MPI_Barrier(MPI_COMM_WORLD);
  restored_parameters.domain_parameters.name_of_grid = input_mesh;
  ElasticStaticProblem<3> restored(restored_parameters);
  restored.setup_from_checkpoint(checkpoint);
  restored.update_time(time);
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
  original.setup();
  const double time = .37;
  original.update_time(time);
  original.solve();
  const auto signature = mesh_signature(original);
  const auto manifolds =
    Checkpoint::describe_solid_manifolds(original.triangulation(),
                                         MPI_COMM_WORLD);
  original.save_checkpoint(checkpoint);

  ElasticStaticProblem<3> restored(restored_parameters);
  restored.setup_from_checkpoint(checkpoint, [&](auto &mesh) {
    Checkpoint::restore_solid_manifolds(mesh, manifolds);
  });
  restored.update_time(time);
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
