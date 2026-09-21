// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/function_lib.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/core/constraint.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/core/observable_lift.h>
#include <immersx/core/weak_term.h>
#include <immersx/coupling/tensor_product_lift.h>
#include <immersx/io/utils.h>
#ifdef DEAL_II_WITH_SUNDIALS
#  include <immersx/core/sundials_ida_adapter.h>
#  include <immersx/physics/elastodynamics.h>
#  include <immersx/physics/elastodynamics_semidiscrete.h>
#endif
#include <immersx/physics/elastic_static.h>
#include <immersx/physics/poisson.h>
#include <immersx/physics/poisson_residual.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "test_paths.h"

using namespace dealii;
using namespace ImmersX;

namespace
{
  using FieldVector  = ImmersXLA::MPI::Vector;
  using GlobalVector = ImmersXLA::MPI::BlockVector;
  using Adapter      = LinearAdapter<FieldVector, GlobalVector>;

  class AffineDisplacement : public Function<2>
  {
  public:
    AffineDisplacement()
      : Function<2>(2)
    {}

    void
    vector_value(const Point<2> &point, Vector<double> &values) const override
    {
      values    = 0.;
      values[0] = 0.05 * point[0];
    }
  };

  struct StraightTubeSurface
  {
    const ParticleCouplingParameters<3> &
    particle_coupling_parameters() const
    {
      return parameters;
    }

    Tensor<1, 3>
    normal(const Point<3> &point) const
    {
      Tensor<1, 3> result;
      (void)point;
      result[1] = 1.;
      return result;
    }

    ParticleCouplingParameters<3> parameters{"/Application roadmap tube/"};
  };

  void
  initialize_poisson_parameters(const std::string    &subsection,
                                PoissonParameters<2> &parameters)
  {
    parameters.output_directory =
      TestPaths::output_directory("application-roadmap/" + subsection);
    parameters.output_name        = subsection;
    parameters.fe_degree          = 1;
    parameters.initial_refinement = 1;
    parameters.dirichlet_ids      = {0};
    parameters.name_of_grid       = "hyper_cube";
    parameters.arguments_for_grid = "-1: 1: false";
    parameters.triangulation_type = "distributed";
  }

  void
  setup(PoissonSolver<2> &problem)
  {
    problem.make_grid();
    problem.setup_fe();
    problem.setup_system();
    problem.assemble_system();
  }

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

  template <int dim, int spacedim, typename VectorType>
  void
  output_multiplier(const std::filesystem::path     &directory,
                    const std::string               &name,
                    const DoFHandler<dim, spacedim> &dof_handler,
                    const VectorType                &values)
  {
    std::filesystem::create_directories(directory);
    DataOut<dim, spacedim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(values, name);
    data_out.build_patches();
    data_out.write_vtu_in_parallel((directory / (name + ".vtu")).string(),
                                   MPI_COMM_WORLD);
  }
} // namespace

TEST(ApplicationRoadmap, P1G0UsesFrozenForcingWithoutMultiplier)
{
  ParameterAcceptor::clear();
  PoissonParameters<2> parameters("/P1/");
  initialize_parameters_from_string(R"(
    subsection P1
      subsection Right hand side
        set Function expression = 0
        set Variable names = x,y,t
      end
      subsection Dirichlet boundary conditions
        set Function expression = 0
        set Variable names = x,y,t
      end
    end
  )");
  initialize_poisson_parameters("p1-g0", parameters);

  PoissonSolver<2> problem(parameters);
  setup(problem);

  LinearSolverParameters adapter_parameters;
  Adapter                adapter(adapter_parameters, MPI_COMM_WORLD);
  const auto             fields           = adapter.add(problem, "poisson");
  const auto             space            = fe_space(problem.dof_handler(),
                              StaticMappingQ1<2>::mapping,
                              problem.constraints(),
                              problem.locally_relevant_dofs());
  const auto             prescribed_field = space.field("prescribed_source");
  FieldVector prescribed(problem.locally_owned_dofs(), MPI_COMM_WORLD);
  prescribed = 1.;
  prescribed.compress(VectorOperation::insert);

  adapter.add(weak_term(frozen(prescribed_field, prescribed),
                        test(
                          space.field(fields.fields().solution, "solution"))),
              "prescribed-forcing");

  auto state = adapter.make_state();
  adapter.solve(state);
  problem.set_solution(adapter.field(state, fields.fields().solution));
  problem.output_results();

  EXPECT_EQ(adapter.saddle_points().size(), 0u);
  EXPECT_TRUE(problem.solution_is_finite());
  EXPECT_TRUE(output_has_field(parameters.output_directory, "solution"));
}

TEST(ApplicationRoadmap, P2G0UsesPrescribedConstraintAndReaction)
{
  ParameterAcceptor::clear();
  PoissonParameters<2> parameters("/P2/");
  initialize_parameters_from_string(R"(
    subsection P2
      subsection Right hand side
        set Function expression = 0
        set Variable names = x,y,t
      end
      subsection Dirichlet boundary conditions
        set Function expression = 0
        set Variable names = x,y,t
      end
    end
  )");
  initialize_poisson_parameters("p2-g0", parameters);

  PoissonSolver<2> problem(parameters);
  setup(problem);

  LinearSolverParameters adapter_parameters;
  Adapter                adapter(adapter_parameters, MPI_COMM_WORLD);
  const auto             fields = adapter.add(problem, "poisson");
  const auto             space  = fe_space(problem.dof_handler(),
                              StaticMappingQ1<2>::mapping,
                              problem.constraints(),
                              problem.locally_relevant_dofs());
  FE_Q<2>                multiplier_fe(1);
  DoFHandler<2>          multiplier_dh(problem.triangulation());
  multiplier_dh.distribute_dofs(multiplier_fe);
  AffineConstraints<double> multiplier_constraints;
  multiplier_constraints.close();
  const auto  multiplier_space = fe_space(multiplier_dh,
                                         StaticMappingQ1<2>::mapping,
                                         multiplier_constraints);
  const auto  solution = space.field(fields.fields().solution, "solution");
  const auto  lambda   = multiplier_space.field("lambda");
  FieldVector prescribed(multiplier_dh.locally_owned_dofs(), MPI_COMM_WORLD);
  prescribed = 0.25;
  prescribed.compress(VectorOperation::insert);

  const auto coupling =
    adapter.add(make_constraint(weak_term(value(solution), test(lambda)),
                                prescribed),
                "prescribed-constraint");

  auto state = adapter.make_state();
  adapter.solve(state);
  problem.set_solution(adapter.field(state, fields.fields().solution));
  problem.output_results();
  output_multiplier(parameters.output_directory,
                    "multiplier",
                    multiplier_dh,
                    adapter.field(state, coupling.fields().multiplier));

  EXPECT_EQ(adapter.saddle_points().size(), 1u);
  EXPECT_TRUE(problem.solution_is_finite());
  EXPECT_TRUE(std::isfinite(
    adapter.field(state, coupling.fields().multiplier).l2_norm()));
  EXPECT_TRUE(output_has_field(parameters.output_directory, "multiplier"));
}

TEST(ApplicationRoadmap, P1G1UsesFrozenForcingOnAnIndependentMesh)
{
  ParameterAcceptor::clear();
  PoissonParameters<2> parameters("/P1 G1/");
  initialize_parameters_from_string(R"(
    subsection P1 G1
      subsection Right hand side
        set Function expression = 0
        set Variable names = x,y,t
      end
      subsection Dirichlet boundary conditions
        set Function expression = 0
        set Variable names = x,y,t
      end
    end
  )");
  initialize_poisson_parameters("p1-g1", parameters);

  PoissonSolver<2> problem(parameters);
  setup(problem);

  parallel::distributed::Triangulation<2> source_tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(source_tria, -0.5, 1.5);
  source_tria.refine_global(1);
  FE_Q<2>       source_fe(1);
  DoFHandler<2> source_dh(source_tria);
  source_dh.distribute_dofs(source_fe);
  const auto source_owned = source_dh.locally_owned_dofs();
  const auto source_relevant =
    DoFTools::extract_locally_relevant_dofs(source_dh);
  AffineConstraints<double> source_constraints;
  source_constraints.reinit(source_owned, source_relevant);
  source_constraints.close();

  LinearSolverParameters adapter_parameters;
  Adapter                adapter(adapter_parameters, MPI_COMM_WORLD);
  const auto             fields       = adapter.add(problem, "poisson");
  const auto             target_space = fe_space(problem.dof_handler(),
                                     StaticMappingQ1<2>::mapping,
                                     problem.constraints(),
                                     problem.locally_relevant_dofs());
  const auto             source_space = fe_space(source_dh,
                                     StaticMappingQ1<2>::mapping,
                                     source_constraints,
                                     source_relevant);
  const auto prescribed_field         = source_space.field("prescribed_source");
  const auto solution =
    target_space.field(fields.fields().solution, "solution");
  FieldVector prescribed(source_owned, MPI_COMM_WORLD);
  prescribed = 1.;
  prescribed.compress(VectorOperation::insert);
  adapter.add(weak_term(frozen(prescribed_field, prescribed), test(solution)),
              "prescribed-forcing");

  auto state = adapter.make_state();
  adapter.solve(state);
  problem.set_solution(adapter.field(state, fields.fields().solution));
  problem.output_results();

  EXPECT_TRUE(problem.solution_is_finite());
  EXPECT_GT(problem.solution().l2_norm(), 1.e-12);
  EXPECT_EQ(adapter.saddle_points().size(), 0u);
}

TEST(ApplicationRoadmap, P2G1UsesPrescribedConstraintOnAnIndependentMesh)
{
  ParameterAcceptor::clear();
  PoissonParameters<2> parameters("/P2 G1/");
  initialize_parameters_from_string(R"(
    subsection P2 G1
      subsection Right hand side
        set Function expression = 0
        set Variable names = x,y,t
      end
      subsection Dirichlet boundary conditions
        set Function expression = 0
        set Variable names = x,y,t
      end
    end
  )");
  initialize_poisson_parameters("p2-g1", parameters);

  PoissonSolver<2> problem(parameters);
  setup(problem);

  parallel::distributed::Triangulation<2> multiplier_tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(multiplier_tria, -0.9, 0.9);
  multiplier_tria.refine_global(1);
  FE_Q<2>       multiplier_fe(1);
  DoFHandler<2> multiplier_dh(multiplier_tria);
  multiplier_dh.distribute_dofs(multiplier_fe);
  const auto multiplier_owned = multiplier_dh.locally_owned_dofs();
  const auto multiplier_relevant =
    DoFTools::extract_locally_relevant_dofs(multiplier_dh);
  AffineConstraints<double> multiplier_constraints;
  multiplier_constraints.reinit(multiplier_owned, multiplier_relevant);
  multiplier_constraints.close();

  LinearSolverParameters adapter_parameters;
  Adapter                adapter(adapter_parameters, MPI_COMM_WORLD);
  const auto             fields           = adapter.add(problem, "poisson");
  const auto             target_space     = fe_space(problem.dof_handler(),
                                     StaticMappingQ1<2>::mapping,
                                     problem.constraints(),
                                     problem.locally_relevant_dofs());
  const auto             multiplier_space = fe_space(multiplier_dh,
                                         StaticMappingQ1<2>::mapping,
                                         multiplier_constraints,
                                         multiplier_relevant);
  const auto             solution =
    target_space.field(fields.fields().solution, "solution");
  const auto  lambda = multiplier_space.field("lambda");
  FieldVector prescribed(multiplier_owned, MPI_COMM_WORLD);
  prescribed = 0.25;
  prescribed.compress(VectorOperation::insert);
  const auto coupling =
    adapter.add(make_constraint(weak_term(value(solution), test(lambda)),
                                prescribed),
                "prescribed-constraint");

  auto state = adapter.make_state();
  adapter.solve(state);
  problem.set_solution(adapter.field(state, fields.fields().solution));
  problem.output_results();
  output_multiplier(parameters.output_directory,
                    "multiplier",
                    multiplier_dh,
                    adapter.field(state, coupling.fields().multiplier));

  EXPECT_EQ(adapter.saddle_points().size(), 1u);
  EXPECT_TRUE(problem.solution_is_finite());
  EXPECT_TRUE(std::isfinite(
    adapter.field(state, coupling.fields().multiplier).l2_norm()));
  EXPECT_TRUE(output_has_field(parameters.output_directory, "multiplier"));
}

void
check_p1g2_embedded_source()
{
  ParameterAcceptor::clear();
  PoissonParameters<2> parameters("/P1 G2/");
  initialize_parameters_from_string(R"(
    subsection P1 G2
      subsection Right hand side
        set Function expression = 0
        set Variable names = x,y,t
      end
      subsection Dirichlet boundary conditions
        set Function expression = 0
        set Variable names = x,y,t
      end
    end
  )");
  initialize_poisson_parameters("p1-g2", parameters);

  PoissonSolver<2> problem(parameters);
  setup(problem);

  Triangulation<1, 2>      serial_line;
  std::vector<Point<2>>    line_vertices{Point<2>(-0.75, 0.0),
                                      Point<2>(0.75, 0.0)};
  std::vector<CellData<1>> line_cells(1);
  line_cells[0].vertices[0] = 0;
  line_cells[0].vertices[1] = 1;
  SubCellData line_subcells;
  serial_line.create_triangulation(line_vertices, line_cells, line_subcells);
  serial_line.refine_global(2);
  parallel::fullydistributed::Triangulation<1, 2> line_tria(MPI_COMM_WORLD);
  line_tria.copy_triangulation(serial_line);
  FE_Q<1, 2>       source_fe(1);
  DoFHandler<1, 2> source_dh(line_tria);
  source_dh.distribute_dofs(source_fe);
  const auto source_owned = source_dh.locally_owned_dofs();
  const auto source_relevant =
    DoFTools::extract_locally_relevant_dofs(source_dh);
  AffineConstraints<double> source_constraints;
  source_constraints.reinit(source_owned, source_relevant);
  source_constraints.close();

  LinearSolverParameters adapter_parameters;
  Adapter                adapter(adapter_parameters, MPI_COMM_WORLD);
  const auto             fields       = adapter.add(problem, "bulk-poisson");
  const auto             source_space = fe_space(source_dh,
                                     StaticMappingQ1<1, 2>::mapping,
                                     source_constraints,
                                     source_relevant);
  const auto             target_space = fe_space(problem.dof_handler(),
                                     StaticMappingQ1<2>::mapping,
                                     problem.constraints(),
                                     problem.locally_relevant_dofs());
  const auto             source       = source_space.field("prescribed_source");
  const auto             solution =
    target_space.field(fields.fields().solution, "solution");
  FieldVector prescribed(source_owned, MPI_COMM_WORLD);
  prescribed = 1.;
  prescribed.compress(VectorOperation::insert);
  adapter.add(weak_term(frozen(source, prescribed), test(solution)),
              "prescribed-line-forcing");

  auto state = adapter.make_state();
  adapter.solve(state);
  problem.set_solution(adapter.field(state, fields.fields().solution));
  problem.output_results();

  EXPECT_EQ(adapter.saddle_points().size(), 0u);
  EXPECT_TRUE(problem.solution_is_finite());
  EXPECT_GT(problem.solution().l2_norm(), 1.e-12);
}

TEST(ApplicationRoadmap, BOTH_P1G2UsesFrozenEmbeddedSourceForBulkForcing)
{
  check_p1g2_embedded_source();
}

TEST(ApplicationRoadmap, P1G3UsesOneWayLiftedTubeLoading)
{
  ParameterAcceptor::clear();
  ElasticStaticParameters<3> parameters("/P1 G3 solid/");
  parameters.output_directory =
    TestPaths::output_directory("application-roadmap/p1-g3");
  parameters.output_name        = "p1-g3";
  parameters.fe_degree          = 1;
  parameters.initial_refinement = 1;
  parameters.dirichlet_ids      = {0, 2, 4};
  parameters.name_of_grid       = "hyper_cube";
  parameters.arguments_for_grid = "-1: 1: false";
  parameters.triangulation_type = "distributed";

  ElasticStaticProblem<3> problem(parameters);
  problem.setup();

  Triangulation<1, 3>      serial_line;
  std::vector<Point<3>>    line_vertices{Point<3>(-0.75, 0.0, 0.0),
                                      Point<3>(0.75, 0.0, 0.0)};
  std::vector<CellData<1>> line_cells(1);
  line_cells[0].vertices[0] = 0;
  line_cells[0].vertices[1] = 1;
  SubCellData line_subcells;
  serial_line.create_triangulation(line_vertices, line_cells, line_subcells);
  parallel::fullydistributed::Triangulation<1, 3> line_tria(MPI_COMM_WORLD);
  line_tria.copy_triangulation(serial_line);
  FE_Q<1, 3>       line_fe(1);
  DoFHandler<1, 3> line_dh(line_tria);
  line_dh.distribute_dofs(line_fe);
  const auto line_owned    = line_dh.locally_owned_dofs();
  const auto line_relevant = DoFTools::extract_locally_relevant_dofs(line_dh);
  AffineConstraints<double> line_constraints;
  line_constraints.reinit(line_owned, line_relevant);
  line_constraints.close();

  LinearSolverParameters adapter_parameters;
  Adapter                adapter(adapter_parameters, MPI_COMM_WORLD);
  const auto             fields      = adapter.add(problem, "solid");
  const auto             solid_space = fe_space(problem.dof_handler(),
                                    StaticMappingQ1<3>::mapping,
                                    problem.constraints(),
                                    problem.locally_relevant_dofs());
  const auto             line_space  = fe_space(line_dh,
                                   StaticMappingQ1<1, 3>::mapping,
                                   line_constraints,
                                   line_relevant);
  const auto  displacement = solid_space.field(fields.fields().displacement,
                                              "displacement",
                                              FEValuesExtractors::Vector(0));
  const auto  source       = line_space.field("prescribed_pressure");
  FieldVector prescribed(line_owned, MPI_COMM_WORLD);
  prescribed = 1.;
  prescribed.compress(VectorOperation::insert);

  TensorProductLift<1, 2, 3, 1> lift_descriptor("/P1 G3 lift/");
  lift_descriptor.section.selected_coefficients = {0u};
  lift_descriptor.section.inclusion_degree      = 1;
  lift_descriptor.section.refinement_level      = 0;
  lift_descriptor.section.n_q_points            = 2;
  lift_descriptor.representative_n_q_points     = 2;
  const auto lifted_source = lift(frozen(source, prescribed), lift_descriptor);
  StraightTubeSurface surface;
  const auto          load = lifted_source * normal(surface);
  adapter.add(weak_term(load, test(displacement)), "prescribed-tube-load");

  auto state = adapter.make_state();
  adapter.solve(state);
  problem.set_solution(adapter.field(state, fields.fields().displacement));
  problem.output_results(0);

  EXPECT_TRUE(std::isfinite(problem.solution().l2_norm()));
  EXPECT_GT(problem.solution().l2_norm(), 1.e-12);
  EXPECT_TRUE(output_has_field(parameters.output_directory, "displacement"));
  EXPECT_TRUE(adapter.saddle_points().empty());
}

TEST(ApplicationRoadmap, P2G3UsesPrescribedLiftedLineMotion)
{
  ParameterAcceptor::clear();
  ElasticStaticParameters<3> parameters("/P2 G3 solid/");
  parameters.output_directory =
    TestPaths::output_directory("application-roadmap/p2-g3");
  parameters.output_name        = "p2-g3";
  parameters.fe_degree          = 1;
  parameters.initial_refinement = 1;
  parameters.dirichlet_ids      = {0, 2, 4};
  parameters.name_of_grid       = "hyper_cube";
  parameters.arguments_for_grid = "-1: 1: false";
  parameters.triangulation_type = "distributed";

  ElasticStaticProblem<3> problem(parameters);
  problem.setup();

  Triangulation<1, 3>      serial_line;
  std::vector<Point<3>>    line_vertices{Point<3>(-0.75, 0.0, 0.0),
                                      Point<3>(0.75, 0.0, 0.0)};
  std::vector<CellData<1>> line_cells(1);
  line_cells[0].vertices[0] = 0;
  line_cells[0].vertices[1] = 1;
  SubCellData line_subcells;
  serial_line.create_triangulation(line_vertices, line_cells, line_subcells);
  parallel::fullydistributed::Triangulation<1, 3> line_tria(MPI_COMM_WORLD);
  line_tria.copy_triangulation(serial_line);
  FE_Q<1, 3>       line_fe(1);
  DoFHandler<1, 3> line_dh(line_tria);
  line_dh.distribute_dofs(line_fe);
  const auto line_owned    = line_dh.locally_owned_dofs();
  const auto line_relevant = DoFTools::extract_locally_relevant_dofs(line_dh);
  AffineConstraints<double> line_constraints;
  line_constraints.reinit(line_owned, line_relevant);
  line_constraints.close();

  LinearSolverParameters adapter_parameters;
  Adapter                adapter(adapter_parameters, MPI_COMM_WORLD);
  const auto             fields      = adapter.add(problem, "solid");
  const auto             solid_space = fe_space(problem.dof_handler(),
                                    StaticMappingQ1<3>::mapping,
                                    problem.constraints(),
                                    problem.locally_relevant_dofs());
  const auto             line_space  = fe_space(line_dh,
                                   StaticMappingQ1<1, 3>::mapping,
                                   line_constraints,
                                   line_relevant);
  const auto displacement = solid_space.field(fields.fields().displacement,
                                              "displacement",
                                              FEValuesExtractors::Vector(0));
  const auto lambda       = line_space.field("lambda");

  TensorProductLift<1, 2, 3, 3> lift_descriptor("/P2 G3 lift/");
  lift_descriptor.section.selected_coefficients = {0u};
  lift_descriptor.section.inclusion_degree      = 1;
  lift_descriptor.section.refinement_level      = 0;
  lift_descriptor.section.n_q_points            = 2;
  lift_descriptor.representative_n_q_points     = 2;
  const auto  lifted_lambda = lift(value(lambda), lift_descriptor);
  FieldVector prescribed(line_owned, MPI_COMM_WORLD);
  prescribed = 0.1;
  prescribed.compress(VectorOperation::insert);
  const auto coupling =
    adapter.add(make_constraint(weak_term(value(displacement),
                                          test(lifted_lambda)),
                                prescribed),
                "prescribed-lifted-motion");

  auto state = adapter.make_state();
  adapter.solve(state);
  problem.set_solution(adapter.field(state, fields.fields().displacement));
  problem.output_results(0);
  output_multiplier(parameters.output_directory,
                    "multiplier",
                    line_dh,
                    adapter.field(state, coupling.fields().multiplier));

  EXPECT_EQ(adapter.saddle_points().size(), 1u);
  EXPECT_TRUE(std::isfinite(problem.solution().l2_norm()));
  EXPECT_TRUE(std::isfinite(
    adapter.field(state, coupling.fields().multiplier).l2_norm()));
  EXPECT_TRUE(output_has_field(parameters.output_directory, "displacement"));
  EXPECT_TRUE(output_has_field(parameters.output_directory, "multiplier"));
}

#ifdef DEAL_II_WITH_SUNDIALS
void
check_p3g2_mixed_dimensional_fiber(const std::string &output_prefix)
{
  ParameterAcceptor::clear();

  TimeIntervalParameters time_parameters("/P3 G2 time/");
  FixedStepParameters    fixed_step_parameters;
  IDAParameters          ida_parameters;
  time_parameters.initial_time                 = 0.;
  time_parameters.final_time                   = 1.e-4;
  fixed_step_parameters.time_step              = 1.e-4;
  fixed_step_parameters.number_of_steps        = 1;
  time_parameters.output_time_interval         = 1.e-4;
  ida_parameters.initial_step_size             = 1.e-4;
  ida_parameters.absolute_tolerance            = 1.e-8;
  ida_parameters.relative_tolerance            = 1.e-8;
  ida_parameters.maximum_order                 = 1;
  ida_parameters.maximum_non_linear_iterations = 20;

  ElastodynamicsParameters<2>    matrix_parameters("/P3 G2 matrix/",
                                                &time_parameters,
                                                &fixed_step_parameters,
                                                &ida_parameters);
  ElastodynamicsParameters<1, 2> fiber_parameters("/P3 G2 fiber/",
                                                  &time_parameters,
                                                  &fixed_step_parameters,
                                                  &ida_parameters);

  matrix_parameters.output_directory = TestPaths::output_directory(
    "application-roadmap/" + output_prefix + "/matrix");
  fiber_parameters.output_directory = TestPaths::output_directory(
    "application-roadmap/" + output_prefix + "/fiber");
  matrix_parameters.output_name        = "matrix";
  fiber_parameters.output_name         = "fiber";
  matrix_parameters.initial_refinement = 1;
  fiber_parameters.initial_refinement  = 1;
  matrix_parameters.dirichlet_ids.clear();
  fiber_parameters.dirichlet_ids.clear();
  matrix_parameters.name_of_grid       = "hyper_cube";
  matrix_parameters.arguments_for_grid = "-1: 1: false";
  fiber_parameters.name_of_grid        = "hyper_cube";
  fiber_parameters.arguments_for_grid  = "-0.6: 0.6: false";

  ElastodynamicsSolver<2>    matrix_problem(matrix_parameters);
  ElastodynamicsSolver<1, 2> fiber_problem(fiber_parameters);
  initialize_parameters();
  ParameterAcceptor::parse_all_parameters();
  const auto setup_problem = [](auto &problem) {
    problem.make_grid();
    problem.setup_fe();
    problem.setup_system();
    problem.assemble_operators();
    problem.set_initial_conditions();
  };
  setup_problem(matrix_problem);
  setup_problem(fiber_problem);

  auto matrix_displacement = matrix_problem.displacement();
  auto fiber_displacement  = fiber_problem.displacement();
  VectorTools::interpolate(matrix_problem.dof_handler(),
                           AffineDisplacement(),
                           matrix_displacement);
  VectorTools::interpolate(fiber_problem.dof_handler(),
                           AffineDisplacement(),
                           fiber_displacement);
  matrix_problem.set_displacement(matrix_displacement);
  fiber_problem.set_displacement(fiber_displacement);

  using FieldVector  = ImmersXLA::MPI::Vector;
  using GlobalVector = ImmersXLA::MPI::BlockVector;
  using Adapter      = IDAAdapter<FieldVector, GlobalVector>;
  Adapter    adapter(time_parameters, ida_parameters, MPI_COMM_WORLD);
  const auto matrix_fields = adapter.add(matrix_problem, "matrix");
  const auto fiber_fields  = adapter.add(fiber_problem, "fiber");

  const auto matrix_space = fe_space(matrix_problem.dof_handler(),
                                     matrix_problem.mapping(),
                                     matrix_problem.velocity_constraints(),
                                     matrix_problem.locally_relevant_dofs());
  const auto fiber_space  = fe_space(fiber_problem.dof_handler(),
                                    fiber_problem.mapping(),
                                    fiber_problem.velocity_constraints(),
                                    fiber_problem.locally_relevant_dofs());
  const auto matrix_velocity =
    matrix_space.field(matrix_fields.fields().velocity,
                       "matrix_velocity",
                       FEValuesExtractors::Vector(0));
  const auto fiber_velocity = fiber_space.field(fiber_fields.fields().velocity,
                                                "fiber_velocity",
                                                FEValuesExtractors::Vector(0));
  const auto multiplier     = fiber_space.field("fiber_velocity_multiplier",
                                            FEValuesExtractors::Vector(0));
  const auto coupling =
    adapter.add(make_constraint(
                  weak_term(value(matrix_velocity), test(multiplier)) -
                  weak_term(value(fiber_velocity), test(multiplier))),
                "fiber-coupling");

  auto state     = adapter.make_state();
  auto state_dot = adapter.make_state();
  adapter.field(state, matrix_fields.fields().displacement) =
    matrix_problem.displacement();
  adapter.field(state, matrix_fields.fields().velocity) =
    matrix_problem.velocity();
  adapter.field(state, fiber_fields.fields().displacement) =
    fiber_problem.displacement();
  adapter.field(state, fiber_fields.fields().velocity) =
    fiber_problem.velocity();
  adapter.field(state, coupling.fields().multiplier) = 0.;
  adapter.field(state_dot, matrix_fields.fields().displacement) =
    matrix_problem.velocity();
  adapter.field(state_dot, fiber_fields.fields().displacement) =
    fiber_problem.velocity();
  FieldVector matrix_acceleration;
  FieldVector fiber_acceleration;
  matrix_problem.initial_acceleration(matrix_acceleration);
  fiber_problem.initial_acceleration(fiber_acceleration);
  adapter.field(state_dot, matrix_fields.fields().velocity) =
    matrix_acceleration;
  adapter.field(state_dot, fiber_fields.fields().velocity) = fiber_acceleration;
  adapter.field(state_dot, coupling.fields().multiplier)   = 0.;

  adapter.solve(state, state_dot);
  matrix_problem.accept_state(
    adapter.field(state, matrix_fields.fields().displacement),
    adapter.field(state, matrix_fields.fields().velocity),
    time_parameters.final_time,
    1);
  fiber_problem.accept_state(
    adapter.field(state, fiber_fields.fields().displacement),
    adapter.field(state, fiber_fields.fields().velocity),
    time_parameters.final_time,
    1);
  matrix_problem.output_results();
  fiber_problem.output_results();
  const auto interaction_directory = TestPaths::output_directory(
    "application-roadmap/" + output_prefix + "/interaction");
  output_multiplier(interaction_directory,
                    "multiplier",
                    fiber_problem.dof_handler(),
                    adapter.field(state, coupling.fields().multiplier));

  EXPECT_EQ(adapter.saddle_points().size(), 1u);
  EXPECT_EQ(adapter.saddle_points().front().participants.size(), 2u);
  EXPECT_TRUE(matrix_problem.state_is_finite());
  EXPECT_TRUE(fiber_problem.state_is_finite());
  EXPECT_TRUE(std::isfinite(
    adapter.field(state, coupling.fields().multiplier).l2_norm()));
  EXPECT_GT(adapter.field(state, coupling.fields().multiplier).l2_norm(),
            1.e-12);
  EXPECT_GT(matrix_problem.displacement().l2_norm(), 1.e-12);
  EXPECT_GT(fiber_problem.displacement().l2_norm(), 1.e-12);
  EXPECT_TRUE(
    output_has_field(matrix_parameters.output_directory, "displacement"));
  EXPECT_TRUE(
    output_has_field(fiber_parameters.output_directory, "displacement"));
  EXPECT_TRUE(output_has_field(interaction_directory, "multiplier_0"));
}

TEST(ApplicationRoadmap, BOTH_P3G2UsesMixedDimensionalFiberElastodynamics)
{
  check_p3g2_mixed_dimensional_fiber("p3-g2");
}
#endif

TEST(ApplicationRoadmap, P3G0UsesTwoLiveProblemsAndParticipantReactions)
{
  ParameterAcceptor::clear();
  PoissonParameters<2> first_parameters("/First/");
  PoissonParameters<2> second_parameters("/Second/");
  initialize_parameters_from_string(R"(
    subsection First
      subsection Right hand side
        set Function expression = 0
        set Variable names = x,y,t
      end
      subsection Dirichlet boundary conditions
        set Function expression = 0
        set Variable names = x,y,t
      end
    end
    subsection Second
      subsection Right hand side
        set Function expression = 1
        set Variable names = x,y,t
      end
      subsection Dirichlet boundary conditions
        set Function expression = 0
        set Variable names = x,y,t
      end
    end
  )");
  initialize_poisson_parameters("p3-g0-first", first_parameters);
  initialize_poisson_parameters("p3-g0-second", second_parameters);

  PoissonSolver<2> first(first_parameters);
  PoissonSolver<2> second(second_parameters);
  setup(first);
  setup(second);

  LinearSolverParameters adapter_parameters;
  Adapter                adapter(adapter_parameters, MPI_COMM_WORLD);
  const auto             first_fields  = adapter.add(first, "first");
  const auto             second_fields = adapter.add(second, "second");
  const auto             first_space   = fe_space(first.dof_handler(),
                                    StaticMappingQ1<2>::mapping,
                                    first.constraints(),
                                    first.locally_relevant_dofs());
  const auto             second_space  = fe_space(second.dof_handler(),
                                     StaticMappingQ1<2>::mapping,
                                     second.constraints(),
                                     second.locally_relevant_dofs());
  FE_Q<2>                multiplier_fe(1);
  DoFHandler<2>          multiplier_dh(second.triangulation());
  multiplier_dh.distribute_dofs(multiplier_fe);
  AffineConstraints<double> multiplier_constraints;
  multiplier_constraints.close();
  const auto multiplier_space = fe_space(multiplier_dh,
                                         StaticMappingQ1<2>::mapping,
                                         multiplier_constraints);
  const auto first_solution =
    first_space.field(first_fields.fields().solution, "first_solution");
  const auto second_solution =
    second_space.field(second_fields.fields().solution, "second_solution");
  const auto lambda = multiplier_space.field("lambda");
  const auto coupling =
    adapter.add(make_constraint(
                  weak_term(value(first_solution), test(lambda)) -
                  weak_term(value(second_solution), test(lambda))),
                "tied");

  auto state = adapter.make_state();
  adapter.solve(state);
  first.set_solution(adapter.field(state, first_fields.fields().solution));
  second.set_solution(adapter.field(state, second_fields.fields().solution));
  first.output_results();
  second.output_results();
  output_multiplier(second_parameters.output_directory,
                    "multiplier",
                    multiplier_dh,
                    adapter.field(state, coupling.fields().multiplier));

  EXPECT_EQ(adapter.saddle_points().size(), 1u);
  EXPECT_EQ(adapter.saddle_points().front().participants.size(), 2u);
  EXPECT_TRUE(first.solution_is_finite());
  EXPECT_TRUE(second.solution_is_finite());
  EXPECT_TRUE(std::isfinite(
    adapter.field(state, coupling.fields().multiplier).l2_norm()));
  EXPECT_TRUE(
    output_has_field(second_parameters.output_directory, "multiplier"));
}

void
check_p3g1_tied_elasticity(const std::string &output_prefix)
{
  ParameterAcceptor::clear();
  ElasticStaticParameters<2> first_parameters("/Elastic first/");
  ElasticStaticParameters<2> second_parameters("/Elastic second/");

  const auto initialize_elastic_parameters = [](auto              &parameters,
                                                const std::string &subsection) {
    parameters.output_directory =
      TestPaths::output_directory("application-roadmap/" + subsection);
    parameters.output_name        = subsection;
    parameters.fe_degree          = 1;
    parameters.initial_refinement = 2;
    parameters.dirichlet_ids      = {0, 1, 2, 3};
    parameters.name_of_grid       = "hyper_cube";
    parameters.arguments_for_grid = "-1: 1: false";
    parameters.triangulation_type = "distributed";
  };

  initialize_elastic_parameters(first_parameters, output_prefix + "-first");
  initialize_elastic_parameters(second_parameters, output_prefix + "-second");
  second_parameters.arguments_for_grid = "-0.5: 1.5: false";

  ElasticStaticProblem<2> first(first_parameters);
  ElasticStaticProblem<2> second(second_parameters);
  first.setup();
  second.setup();
  first.set_forcing(
    Functions::ConstantFunction<2>(std::vector<double>{1., 0.}));

  LinearSolverParameters adapter_parameters;
  Adapter                adapter(adapter_parameters, MPI_COMM_WORLD);
  const auto             first_fields  = adapter.add(first, "first");
  const auto             second_fields = adapter.add(second, "second");
  const auto             first_space   = fe_space(first.dof_handler(),
                                    StaticMappingQ1<2>::mapping,
                                    first.constraints(),
                                    first.locally_relevant_dofs());
  const auto             second_space  = fe_space(second.dof_handler(),
                                     StaticMappingQ1<2>::mapping,
                                     second.constraints(),
                                     second.locally_relevant_dofs());
  FESystem<2>            multiplier_fe(FE_Q<2>(1), 2);
  DoFHandler<2>          multiplier_dh(second.triangulation());
  multiplier_dh.distribute_dofs(multiplier_fe);
  AffineConstraints<double> multiplier_constraints;
  multiplier_constraints.close();
  const auto multiplier_space = fe_space(multiplier_dh,
                                         StaticMappingQ1<2>::mapping,
                                         multiplier_constraints);
  const auto first_displacement =
    first_space.field(first_fields.fields().displacement,
                      "first_displacement",
                      FEValuesExtractors::Vector(0));
  const auto second_displacement =
    second_space.field(second_fields.fields().displacement,
                       "second_displacement",
                       FEValuesExtractors::Vector(0));
  const auto lambda =
    multiplier_space.field("lambda", FEValuesExtractors::Vector(0));
  const auto coupling =
    adapter.add(make_constraint(
                  weak_term(value(first_displacement), test(lambda)) -
                  weak_term(value(second_displacement), test(lambda))),
                "tied-elasticity");

  auto state = adapter.make_state();
  adapter.solve(state);
  first.set_solution(adapter.field(state, first_fields.fields().displacement));
  second.set_solution(
    adapter.field(state, second_fields.fields().displacement));
  first.output_results(0);
  second.output_results(0);
  output_multiplier(second_parameters.output_directory,
                    "multiplier",
                    multiplier_dh,
                    adapter.field(state, coupling.fields().multiplier));

  ASSERT_EQ(adapter.saddle_points().size(), 1u);
  EXPECT_EQ(adapter.saddle_points().front().participants.size(), 2u);
  EXPECT_TRUE(std::isfinite(first.solution().l2_norm()));
  EXPECT_TRUE(std::isfinite(second.solution().l2_norm()));
  EXPECT_GT(first.solution().l2_norm(), 1.e-12);
  EXPECT_GT(second.solution().l2_norm(), 1.e-12);
  EXPECT_TRUE(std::isfinite(
    adapter.field(state, coupling.fields().multiplier).l2_norm()));
  EXPECT_GT(adapter.field(state, coupling.fields().multiplier).l2_norm(),
            1.e-12);
  EXPECT_TRUE(
    output_has_field(first_parameters.output_directory, "displacement"));
  EXPECT_TRUE(
    output_has_field(second_parameters.output_directory, "displacement"));
  EXPECT_TRUE(
    output_has_field(second_parameters.output_directory, "multiplier_0"));
}

TEST(ApplicationRoadmap, BOTH_P3G1UsesNonmatchingTiedElasticity)
{
  check_p3g1_tied_elasticity("p3-g1");
}
