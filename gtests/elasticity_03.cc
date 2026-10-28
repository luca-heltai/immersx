#include <deal.II/base/function_parser.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/coupling/inclusions.h>
#include <immersx/physics/elasticity.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace ImmersX;
#include <immersx/io/utils.h>

#include "test_paths.h"

using namespace dealii;

namespace
{
  constexpr int dim = 2;
  using Problem     = ElasticityProblem<dim>;

  // Unit square [-1,1]^2 with colorized boundary ids (0: x=-1, 1: x=1,
  // 2: y=-1, 3: y=1), mu = 2 and lambda = 3 so that swapped Lame
  // coefficients change every expected value, and no boundary conditions.
  // Tests append the sections they need; later entries override earlier ones.
  const std::string default_parameters = R"(
    subsection Immersed Problem
      set Initial refinement     = 2
      set Dirichlet boundary ids =
      subsection Grid generation
        set Domain type              = generate
        set Grid generator           = hyper_cube
        set Grid generator arguments = -1: 1: true
      end
      subsection Material properties
        subsection default
          set Lame mu     = 2
          set Lame lambda = 3
        end
      end
    end
    subsection Solvers
      subsection Augmented Lagrange
        set Max steps = 1000
        set Reduction = 1.e-12
        set Tolerance = 1.e-13
      end
      subsection Displacement
        set Max steps = 1000
        set Reduction = 1.e-12
        set Tolerance = 1.e-13
      end
      subsection Reduced mass
        set Max steps = 1000
        set Reduction = 1.e-12
        set Tolerance = 1.e-13
      end
    end
  )";

  void
  parse_parameters(ElasticityProblemParameters<dim> &par,
                   const std::string                &extra_parameters = "")
  {
    par.output_directory = TestPaths::output_directory("elasticity-03");
    initialize_parameters_from_string(default_parameters + extra_parameters);
  }

  // Run the setup part of run_static() without assembling or solving.
  void
  setup_system(Problem &problem, const bool refine_one_cell = false)
  {
    problem.make_grid();
    if (refine_one_cell)
      {
        // Refine the lower-left cell only to create hanging nodes.
        for (const auto &cell : problem.tria->active_cell_iterators())
          if (cell->is_locally_owned() && cell->center()[0] < -0.5 &&
              cell->center()[1] < -0.5)
            cell->set_refine_flag();
        problem.tria->execute_coarsening_and_refinement();
      }
    problem.setup_fe();
    problem.inclusions.setup_inclusions_particles(*problem.tria);
    problem.setup_dofs();
  }

  LA::MPI::Vector
  interpolate(const Problem &problem, const Function<dim> &function)
  {
    LA::MPI::Vector vector(problem.owned_dofs[0], problem.mpi_communicator);
    VectorTools::interpolate(problem.dh, function, vector);
    return vector;
  }

  LA::MPI::Vector
  interpolate(const Problem &problem, const std::string &expression)
  {
    FunctionParser<dim> function(dim);
    function.initialize("x,y", expression, {});
    return interpolate(problem, function);
  }

  LA::MPI::Vector
  rigid_body_mode(const Problem &problem, const unsigned int mode)
  {
    return interpolate(problem, RigidBodyMotion<dim>(mode));
  }

  double
  max_difference(const LA::MPI::Vector &a, const LA::MPI::Vector &b)
  {
    LA::MPI::Vector difference(a);
    difference -= b;
    return difference.linfty_norm();
  }

  // Square matrices only; rectangular products need an explicit range vector.
  LA::MPI::Vector
  vmult(const LA::MPI::SparseMatrix &matrix, const LA::MPI::Vector &src)
  {
    LA::MPI::Vector dst(src);
    matrix.vmult(dst, src);
    return dst;
  }

  // Set every locally owned entry that `constraints` constrains to zero.
  void
  clear_constrained_entries(const AffineConstraints<double> &constraints,
                            LA::MPI::Vector                 &vector)
  {
    for (const auto i : vector.locally_owned_elements())
      if (constraints.is_constrained(i))
        vector[i] = 0.0;
    vector.compress(VectorOperation::insert);
  }

  std::string
  per_run_output_name(const std::string &name)
  {
    return name + "_np" +
           std::to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
  }
} // namespace



TEST(ElasticityProblem, RigidBodyMotion2DModes)
{
  const Point<2> p(0.3, -0.7);

  const RigidBodyMotion<2> translation_x(0);
  const RigidBodyMotion<2> translation_y(1);
  const RigidBodyMotion<2> rotation(2);

  EXPECT_EQ(translation_x.n_components, 2u);
  EXPECT_DOUBLE_EQ(translation_x.value(p, 0), 1.0);
  EXPECT_DOUBLE_EQ(translation_x.value(p, 1), 0.0);
  EXPECT_DOUBLE_EQ(translation_y.value(p, 0), 0.0);
  EXPECT_DOUBLE_EQ(translation_y.value(p, 1), 1.0);
  EXPECT_DOUBLE_EQ(rotation.value(p, 0), -p[1]);
  EXPECT_DOUBLE_EQ(rotation.value(p, 1), p[0]);
}



TEST(ElasticityProblem, RigidBodyMotion3DModes)
{
  const Point<3> p(0.2, -0.5, 0.9);

  for (unsigned int mode = 0; mode < 3; ++mode)
    {
      const RigidBodyMotion<3> translation(mode);
      for (unsigned int c = 0; c < 3; ++c)
        EXPECT_DOUBLE_EQ(translation.value(p, c), (c == mode) ? 1.0 : 0.0);
    }

  const std::vector<Tensor<1, 3>> expected_rotations = {
    Tensor<1, 3>({0., p[2], -p[1]}),
    Tensor<1, 3>({-p[2], 0., p[0]}),
    Tensor<1, 3>({p[1], -p[0], 0.})};
  for (unsigned int r = 0; r < 3; ++r)
    {
      const RigidBodyMotion<3> rotation(3 + r);
      double                   tangential = 0.0;
      for (unsigned int c = 0; c < 3; ++c)
        {
          EXPECT_DOUBLE_EQ(rotation.value(p, c), expected_rotations[r][c]);
          tangential += rotation.value(p, c) * p[c];
        }
      // Infinitesimal rotations about the origin are orthogonal to p.
      EXPECT_NEAR(tangential, 0.0, 1e-15);
    }
}



TEST(ElasticityProblem, BOTH_StiffnessIsSymmetricWithRigidBodyKernel)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<dim> par;
  Problem                          problem(par);
  parse_parameters(par);
  setup_system(problem);
  problem.assemble_elasticity_system();

  const auto &A = problem.stiffness_matrix;
  for (unsigned int mode = 0; mode < 3; ++mode)
    EXPECT_LT(vmult(A, rigid_body_mode(problem, mode)).linfty_norm(), 1e-12)
      << "rigid body mode " << mode;

  const auto x = interpolate(problem, "sin(x) + y*y; x*y - cos(y)");
  const auto y = interpolate(problem, "x*x*y; exp(x) + y");
  EXPECT_NEAR(vmult(A, x) * y, x * vmult(A, y), 1e-12 * A.frobenius_norm());
  EXPECT_GT(vmult(A, x) * x, 0.0);
}


TEST(ElasticityProblem, BOTH_PointCouplingQueriesReflectInclusions)
{
  {
    ParameterAcceptor::clear();
    ElasticityProblemParameters<dim> par;
    Problem                          problem(par);
    parse_parameters(par);
    setup_system(problem);

    EXPECT_FALSE(problem.uses_tensor_product_coupling());
    EXPECT_FALSE(problem.has_immersed_coupling());
    EXPECT_EQ(problem.n_multiplier_dofs(), 0u);
    EXPECT_EQ(problem.solution.block(1).size(), 0u);
  }
  {
    ParameterAcceptor::clear();
    ElasticityProblemParameters<dim> par;
    Problem                          problem(par);
    parse_parameters(par, R"(
      subsection Immersed Problem
        subsection Immersed inclusions
          set Inclusions                     = 0, 0, 0.5; 0.4, 0.4, 0.2
          set Number of fourier coefficients = 3
        end
      end
    )");
    setup_system(problem);

    // Three Fourier modes times two vector components per inclusion.
    EXPECT_FALSE(problem.uses_tensor_product_coupling());
    EXPECT_TRUE(problem.has_immersed_coupling());
    EXPECT_EQ(problem.n_multiplier_components_per_coupling_dof(), 6u);
    EXPECT_EQ(problem.n_multiplier_dofs(), 12u);
    EXPECT_EQ(problem.solution.block(1).size(), 12u);
  }
}



TEST(ElasticityProblem, BOTH_PointCouplingAveragesDisplacementOverInclusion)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<dim> par;
  Problem                          problem(par);
  parse_parameters(par, R"(
    subsection Immersed Problem
      subsection Immersed inclusions
        set Inclusions                     = 0, 0, 0.5
        set Number of fourier coefficients = 1
        set Inclusions refinement          = 64
        subsection Boundary data
          set Function expression = 0.1; -0.2
        end
      end
    end
  )");
  setup_system(problem);
  problem.assemble_coupling();

  const auto  &Bt = problem.coupling_matrix;
  const double section_measure =
    problem.inclusions.get_section_measure(0);
  EXPECT_NEAR(section_measure, numbers::PI, 1e-14);

  // B u is the mean of u over the inclusion boundary for the constant modes.
  const std::vector<std::pair<unsigned int, std::array<double, 2>>> cases = {
    {0, {{1.0, 0.0}}}, {1, {{0.0, 1.0}}}, {2, {{0.0, 0.0}}}};
  for (const auto &[mode, expected] : cases)
    {
      LA::MPI::Vector mean(problem.solution.block(1));
      Bt.Tvmult(mean, rigid_body_mode(problem, mode));
      for (const auto i : mean.locally_owned_elements())
        EXPECT_NEAR(mean[i], expected[i], 1e-12) << "rigid body mode " << mode;
    }

  const auto &g = problem.system_rhs.block(1);
  for (const auto i : g.locally_owned_elements())
    {
      EXPECT_NEAR(problem.inclusion_matrix.diag_element(i),
                  section_measure,
                  1e-12);
      EXPECT_NEAR(g[i], (i == 0) ? 0.1 : -0.2, 1e-12);
    }
}



TEST(ElasticityProblem, BOTH_SaddlePointSolutionSatisfiesBothEquations)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<dim> par;
  Problem                          problem(par);
  parse_parameters(par, R"(
    subsection Immersed Problem
      set Dirichlet boundary ids = 0, 1, 2, 3
      subsection Immersed inclusions
        set Inclusions                     = 0, 0, 0.5
        set Number of fourier coefficients = 1
        set Inclusions refinement          = 64
        subsection Boundary data
          set Function expression = 0.1; -0.2
        end
      end
    end
    subsection Functions
      subsection Right hand side
        set Function expression = 1; x
      end
    end
  )");
  setup_system(problem);
  problem.assemble_elasticity_system();
  problem.assemble_coupling();
  problem.solve_static();

  const auto &u      = problem.solution.block(0);
  const auto &lambda = problem.solution.block(1);
  const auto &f      = problem.system_rhs.block(0);
  const auto &g      = problem.system_rhs.block(1);

  // A u + B^T lambda = f.
  LA::MPI::Vector Bt_lambda(u);
  problem.coupling_matrix.vmult(Bt_lambda, lambda);
  LA::MPI::Vector momentum_residual = vmult(problem.stiffness_matrix, u);
  momentum_residual += Bt_lambda;
  momentum_residual -= f;
  EXPECT_LT(momentum_residual.l2_norm(), 1e-8 * f.l2_norm());

  // B u = g.
  LA::MPI::Vector constraint_residual(lambda);
  problem.coupling_matrix.Tvmult(constraint_residual, u);
  constraint_residual -= g;
  EXPECT_LT(constraint_residual.l2_norm(), 1e-8 * g.l2_norm());
  EXPECT_GT(lambda.l2_norm(), 0.0);
}


TEST(ElasticityProblem, BOTH_RefinementTransfersAffineSolutionExactly)
{
  for (const std::string strategy : {"global", "fixed_number"})
    {
      ParameterAcceptor::clear();
      ElasticityProblemParameters<dim> par;
      Problem                          problem(par);
      parse_parameters(par, R"(
        subsection Immersed Problem
          subsection Refinement and remeshing
            set Strategy            = )" + strategy + R"(
            set Refinement fraction = 0.3
          end
        end
      )");
      setup_system(problem);

      const std::string affine = "1 + x + 2*y; -0.5 + 3*x - y";
      const auto        n_cells_before = problem.tria->n_global_active_cells();
      problem.solution.block(0)        = interpolate(problem, affine);
      problem.locally_relevant_solution = problem.solution;

      problem.refine_and_transfer();

      if (strategy == "global")
        EXPECT_EQ(problem.tria->n_global_active_cells(), 4 * n_cells_before);
      else
        EXPECT_GT(problem.tria->n_global_active_cells(), n_cells_before);
      EXPECT_EQ(problem.solution.block(0).size(), problem.dh.n_dofs());
      EXPECT_LT(max_difference(problem.solution.block(0),
                               interpolate(problem, affine)),
                1e-12)
        << strategy;
    }
}



TEST(ElasticityProblem, BOTH_InclusionsRefinementStrategyStopsCycles)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<dim> par;
  Problem                          problem(par);
  parse_parameters(par, R"(
    subsection Immersed Problem
      subsection Refinement and remeshing
        set Strategy                    = inclusions
        set Number of refinement cycles = 3
      end
    end
  )");
  setup_system(problem);

  const auto n_cells_before = problem.tria->n_global_active_cells();
  problem.refine_and_transfer();
  EXPECT_EQ(problem.tria->n_global_active_cells(), n_cells_before);
  EXPECT_EQ(problem.cycle, 3u);
}



TEST(ElasticityProblem, BOTH_FullyDistributedBackendRejectsRefinement)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<dim> par;
  Problem                          problem(par);
  parse_parameters(par, R"(
    subsection Immersed Problem
      subsection Grid generation
        set Triangulation type = fullydistributed
      end
      subsection Refinement and remeshing
        set Strategy = global
      end
    end
  )");
  EXPECT_FALSE(problem.uses_fully_distributed_triangulation());
  setup_system(problem);
  EXPECT_TRUE(problem.uses_fully_distributed_triangulation());
  EXPECT_EQ(problem.tria->n_global_active_cells(), 16u);

  problem.locally_relevant_solution = problem.solution;
  EXPECT_THROW(problem.refine_and_transfer(), ExceptionBase);
}


TEST(ElasticityProblem, BOTH_OutputSolutionFileNameEncodesCycleAndTimeStep)
{
  {
    ParameterAcceptor::clear();
    ElasticityProblemParameters<dim> par;
    Problem                          problem(par);
    const std::string                name = per_run_output_name("static");
    parse_parameters(par,
                     "subsection Immersed Problem\n"
                     "  set Output name = " +
                       name +
                       "\n"
                       "end\n");
    setup_system(problem);
    problem.locally_relevant_solution = problem.solution;
    problem.cycle                     = 3;

    const auto filename = problem.output_solution();
    EXPECT_EQ(filename, name + "_3.vtu");
    EXPECT_TRUE(
      std::filesystem::exists(par.output_directory + "/" + filename));
  }
  {
    ParameterAcceptor::clear();
    ElasticityProblemParameters<dim> par;
    Problem                          problem(par);
    const std::string                name = per_run_output_name("dynamic");
    parse_parameters(par,
                     "subsection Immersed Problem\n"
                     "  set Output name = " +
                       name +
                       "\n"
                       "  subsection Time parameters\n"
                       "    set Final time = 1\n"
                       "  end\n"
                       "  subsection Material properties\n"
                       "    subsection default\n"
                       "      set Density = 1\n"
                       "    end\n"
                       "  end\n"
                       "end\n");
    ASSERT_EQ(par.time_mode, TimeMode::Dynamic);
    setup_system(problem);
    problem.locally_relevant_solution = problem.solution;
    problem.cycle                     = 1;
    problem.time_step                 = 7;

    const auto filename = problem.output_solution();
    EXPECT_EQ(filename, name + "_1_007.vtu");
    EXPECT_TRUE(
      std::filesystem::exists(par.output_directory + "/" + filename));
  }
}

namespace
{
  // Initialize inclusions with the given directions and vessel ids. The
  // initialization runs check_vessels(), which checks the directions.
  void
  initialize_inclusions_with_directions(
    const std::vector<Tensor<1, 3>> &directions,
    const std::vector<unsigned int> &vessel_ids)
  {
    Inclusions<3> inclusions;
    inclusions.set_n_q_points(4);
    inclusions.set_n_coefficients(1);
    // cx, cy, cz, dx, dy, dz, r, vessel id
    for (unsigned int i = 0; i < directions.size(); ++i)
      inclusions.inclusions.push_back({0.1 * i,
                                       0,
                                       0,
                                       directions[i][0],
                                       directions[i][1],
                                       directions[i][2],
                                       0.05,
                                       double(vessel_ids[i])});
    inclusions.initialize();
  }
} // namespace

TEST(Inclusion3, CheckVesselDirectionsAcceptsParallelDirections)
{
  EXPECT_NO_THROW(initialize_inclusions_with_directions(
    {Tensor<1, 3>({0, 0, 1}), Tensor<1, 3>({0, 0, 2})}, {0, 0}));
}

TEST(Inclusion3, CheckVesselDirectionsAcceptsDifferentVessels)
{
  EXPECT_NO_THROW(initialize_inclusions_with_directions(
    {Tensor<1, 3>({0, 0, 1}), Tensor<1, 3>({0, 0.1, 0.2})}, {0, 1}));
}

TEST(Inclusion3, CheckVesselDirectionsRejectsNonParallelDirections)
{
  EXPECT_ANY_THROW(initialize_inclusions_with_directions(
    {Tensor<1, 3>({0, 0, 1}), Tensor<1, 3>({0, 0.1, 0.2})}, {0, 0}));
}

TEST(Inclusion3, CheckVesselDirectionsRejectsOppositeDirections)
{
  EXPECT_ANY_THROW(initialize_inclusions_with_directions(
    {Tensor<1, 3>({0, 0, 1}), Tensor<1, 3>({0, 0, -1})}, {0, 0}));
}
