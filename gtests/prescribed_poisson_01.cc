// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/data_out.h>

#include <gtest/gtest.h>
#include <immersx/core/constraint.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/io/utils.h>
#include <immersx/physics/poisson_residual.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "test_paths.h"

using namespace ImmersX;
using namespace dealii;

namespace
{
  bool
  output_has_field(const std::filesystem::path &directory,
                   const std::string           &name)
  {
    for (const auto &entry : std::filesystem::directory_iterator(directory))
      if (entry.path().extension() == ".vtu" ||
          entry.path().extension() == ".pvtu")
        {
          std::ifstream     input(entry.path());
          const std::string contents((std::istreambuf_iterator<char>(input)),
                                     std::istreambuf_iterator<char>());
          if (contents.find("Name=\"" + name + "\"") != std::string::npos)
            return true;
        }
    return false;
  }

  void
  output_multiplier(const std::filesystem::path  &directory,
                    const std::string            &name,
                    const DoFHandler<1, 2>       &dof_handler,
                    const ImmersXLA::MPI::Vector &values)
  {
    std::filesystem::create_directories(directory);
    DataOut<1, 2> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(values, name, DataOut<1, 2>::type_dof_data);
    data_out.build_patches();
    data_out.write_vtu_in_parallel((directory / (name + ".vtu")).string(),
                                   MPI_COMM_WORLD);
  }
} // namespace



TEST(PrescribedPoisson, MPI_UnifiedConstraintReplacement) // NOLINT
{
  ParameterAcceptor::clear();

  PoissonParameters<2> bulk_parameters("/Bulk Poisson/");

  initialize_parameters_from_string(R"(
    subsection Bulk Poisson
      set FE degree                   = 1
      set Dirichlet boundary ids      = 0
      subsection Domain
        set Initial refinement          = 2
        set Grid generator           = hyper_cube
        set Grid generator arguments = -1: 1: false
      end
      subsection Right hand side
        set Function expression = 0
        set Variable names      = x,y,t
      end
      subsection Dirichlet boundary conditions
        set Function expression = 0
        set Variable names      = x,y,t
      end
      subsection Solver
        subsection Control
          set Max steps  = 200
          set Reduction  = 1.e-12
          set Tolerance  = 1.e-14
          set Log result = false
        end
      end
    end
  )");

  const auto output_directory =
    TestPaths::output_directory("application-roadmap/p2-g2");
  bulk_parameters.output_directory = output_directory;
  bulk_parameters.output_name      = "p2-g2";
  PoissonSolver<2> bulk_problem(bulk_parameters);
  bulk_problem.make_grid();
  bulk_problem.setup_fe();
  bulk_problem.setup_system();
  bulk_problem.assemble_system();

  // This is a representation-only line mesh.  It is not a second Problem and
  // it has no PDE operator or state.  Its coefficients are the prescribed
  // target data for the immersed constraint.
  Triangulation<1, 2>      serial_line;
  std::vector<Point<2>>    line_vertices{Point<2>(-0.75, 0.13),
                                      Point<2>(0.75, 0.13)};
  std::vector<CellData<1>> line_cells(1);
  line_cells[0].vertices[0] = 0;
  line_cells[0].vertices[1] = 1;
  SubCellData line_subcells;
  serial_line.create_triangulation(line_vertices, line_cells, line_subcells);

  parallel::fullydistributed::Triangulation<1, 2> line_tria(MPI_COMM_WORLD);
  line_tria.copy_triangulation(serial_line);

  FE_DGQ<1, 2>     line_fe(0);
  DoFHandler<1, 2> line_dh(line_tria);
  line_dh.distribute_dofs(line_fe);
  const auto line_owned    = line_dh.locally_owned_dofs();
  const auto line_relevant = DoFTools::extract_locally_relevant_dofs(line_dh);
  AffineConstraints<double> line_constraints;
  line_constraints.reinit(line_owned, line_relevant);
  line_constraints.close();

  using FieldVector  = ImmersXLA::MPI::Vector;
  using GlobalVector = ImmersXLA::MPI::BlockVector;
  using Adapter      = LinearAdapter<FieldVector, GlobalVector>;
  LinearSolverParameters adapter_parameters;
  Adapter                adapter(adapter_parameters, MPI_COMM_WORLD);
  const auto             bulk      = adapter.add(bulk_problem, "bulk");
  const auto             bulk_view = fe_space(bulk_problem.dof_handler(),
                                  StaticMappingQ1<2>::mapping,
                                  bulk_problem.constraints(),
                                  bulk_problem.locally_relevant_dofs());
  const auto             line_view = fe_space(line_dh,
                                  StaticMappingQ1<1, 2>::mapping,
                                  line_constraints,
                                  line_relevant);
  const auto             bulk_field =
    bulk_view.field(bulk.fields().solution, "bulk_solution");
  const auto  lambda = line_view.field("lambda");
  FieldVector prescribed(line_owned, MPI_COMM_WORLD);
  prescribed = 1.;
  prescribed.compress(VectorOperation::insert);
  const auto coupling =
    adapter.add(make_constraint(weak_term(value(bulk_field), test(lambda)),
                                prescribed),
                "prescribed");

  auto state = adapter.make_state();
  adapter.solve(state);
  GlobalVector residual;
  adapter.evaluate_residual(state, residual);
  const auto &bulk_residual = adapter.field(residual, bulk.fields().solution);
  const auto &constraint_residual =
    adapter.field(residual, coupling.fields().multiplier);

  EXPECT_TRUE(bulk_problem.solution_is_finite());
  EXPECT_TRUE(std::isfinite(
    adapter.field(state, coupling.fields().multiplier).l2_norm()));
  EXPECT_GT(adapter.field(state, bulk.fields().solution).l2_norm(), 1.e-12);
  EXPECT_LT(bulk_residual.l2_norm(), 1.e-8);
  EXPECT_LT(constraint_residual.l2_norm(), 1.e-8);

  bulk_problem.set_solution(adapter.field(state, bulk.fields().solution));
  bulk_problem.output_results();
  output_multiplier(output_directory,
                    "multiplier",
                    line_dh,
                    adapter.field(state, coupling.fields().multiplier));
  EXPECT_TRUE(output_has_field(output_directory, "solution"));
  EXPECT_TRUE(output_has_field(output_directory, "multiplier"));
}


TEST(PrescribedPoisson,
     MPI_UnifiedConstraintWithIndependentMultiplier) // NOLINT
{
  ParameterAcceptor::clear();

  PoissonParameters<2> bulk_parameters("/Unified Bulk Poisson/");
  initialize_parameters_from_string(R"(
    subsection Unified Bulk Poisson
      set FE degree                   = 1
      set Dirichlet boundary ids      = 0
      subsection Domain
        set Initial refinement          = 2
        set Grid generator           = hyper_cube
        set Grid generator arguments = -1: 1: false
      end
      subsection Right hand side
        set Function expression = 1
        set Variable names      = x,y,t
      end
      subsection Dirichlet boundary conditions
        set Function expression = 0
        set Variable names      = x,y,t
      end
    end
  )");

  PoissonSolver<2> bulk_problem(bulk_parameters);
  bulk_problem.make_grid();
  bulk_problem.setup_fe();
  bulk_problem.setup_system();
  bulk_problem.assemble_system();

  Triangulation<1, 2>      serial_line;
  std::vector<Point<2>>    line_vertices{Point<2>(-0.75, 0.13),
                                      Point<2>(0.75, 0.13)};
  std::vector<CellData<1>> line_cells(1);
  line_cells[0].vertices[0] = 0;
  line_cells[0].vertices[1] = 1;
  SubCellData line_subcells;
  serial_line.create_triangulation(line_vertices, line_cells, line_subcells);

  parallel::fullydistributed::Triangulation<1, 2> line_tria(MPI_COMM_WORLD);
  line_tria.copy_triangulation(serial_line);
  FE_DGQ<1, 2>     line_fe(0);
  DoFHandler<1, 2> line_dh(line_tria);
  line_dh.distribute_dofs(line_fe);
  const auto line_owned    = line_dh.locally_owned_dofs();
  const auto line_relevant = DoFTools::extract_locally_relevant_dofs(line_dh);
  AffineConstraints<double> line_constraints;
  line_constraints.reinit(line_owned, line_relevant);
  line_constraints.close();

  using FieldVector  = ImmersXLA::MPI::Vector;
  using GlobalVector = ImmersXLA::MPI::BlockVector;
  using Adapter      = LinearAdapter<FieldVector, GlobalVector>;
  LinearSolverParameters adapter_parameters;
  Adapter                adapter(adapter_parameters, MPI_COMM_WORLD);
  const auto             bulk      = adapter.add(bulk_problem, "bulk");
  const auto             bulk_view = fe_space(bulk_problem.dof_handler(),
                                  StaticMappingQ1<2>::mapping,
                                  bulk_problem.constraints(),
                                  bulk_problem.locally_relevant_dofs());
  const auto             line_view = fe_space(line_dh,
                                  StaticMappingQ1<1, 2>::mapping,
                                  line_constraints,
                                  line_relevant);
  const auto             bulk_field =
    bulk_view.field(bulk.fields().solution, "bulk_solution");
  const auto lambda = line_view.field("lambda");

  FieldVector prescribed(line_owned, MPI_COMM_WORLD);
  prescribed = 1.;
  prescribed.compress(VectorOperation::insert);
  const auto coupling =
    adapter.add(make_constraint(weak_term(value(bulk_field), test(lambda)),
                                prescribed),
                "prescribed");

  auto state = adapter.make_state();
  adapter.solve(state);
  GlobalVector residual;
  adapter.evaluate_residual(state, residual);

  EXPECT_TRUE(std::isfinite(
    adapter.field(state, coupling.fields().multiplier).l2_norm()));
  EXPECT_GT(adapter.field(state, bulk.fields().solution).l2_norm(), 1.e-12);
  EXPECT_LT(residual.l2_norm(), 1.e-7);
}
