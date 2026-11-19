#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/vessel_elasticity.h>
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <regex>
#include <sstream>

#include "test_paths.h"

// Newmark steps of VesselElasticityProblem with the wall displacement imposed
// by the multiplier, as used by coupled_elasticity_viscous. The geometry is
// the straight cylinder of radius 0.2 along x in the cube [-1, 1]^3 of
// coupled_elasticity_03.cc. Every test has a nonzero body force and
// time-dependent Dirichlet data.
#ifdef DEAL_II_WITH_VTK

using namespace dealii;
using namespace ImmersX;

namespace
{
  using Overrides = std::vector<std::pair<std::string, std::string>>;

  // Write `content` below the test output directory and return the file name.
  std::string
  input_file(const std::string &name, const std::string &content)
  {
    const auto filename = TestPaths::output_path(
      "gtests/data/coupled_elasticity_newmark_01/" + name);
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::filesystem::create_directories(filename.parent_path());
        auto temporary = filename;
        temporary += "." + std::to_string(::getpid());
        std::ofstream(temporary) << content;
        std::filesystem::rename(temporary, filename);
      }
    MPI_Barrier(MPI_COMM_WORLD);
    return filename.string();
  }

  // One reduced cell from x = 1 to x = -1 with vessel id 0 and displacement
  // 0.1 at both vertices.
  std::string
  reduced_grid()
  {
    return input_file("single_cell.vtk",
                      "# vtk DataFile Version 3.0\n"
                      "reduced grid\n"
                      "ASCII\n"
                      "DATASET UNSTRUCTURED_GRID\n"
                      "POINTS 2 double\n"
                      "1 0 0\n"
                      "-1 0 0\n"
                      "CELLS 1 3\n"
                      "2 0 1\n"
                      "CELL_TYPES 1\n"
                      "3\n"
                      "POINT_DATA 2\n"
                      "SCALARS displacement float\n"
                      "LOOKUP_TABLE default\n"
                      "0.1\n"
                      "0.1\n"
                      "CELL_DATA 1\n"
                      "SCALARS vessel_id int\n"
                      "LOOKUP_TABLE default\n"
                      "0\n");
  }

  // 50 point inclusions of length 0.04 and radius 0.2 covering x in [-1, 1],
  // all in vessel 0. The tests cluster them with segments: without
  // clustering, 100 multipliers on this coarse mesh leave B with a kernel, so
  // lambda and the pressure are not unique.
  std::string
  inclusions_file()
  {
    std::ostringstream text;
    for (unsigned int i = 0; i < 50; ++i)
      text << -0.98 + 0.04 * i << " 0 0 0.04 0 0 0.2 0\n";
    return input_file("inclusions.txt", text.str());
  }

  // Parameters of the cylinder with point or tensor-product coupling. A
  // dynamic run goes from t = 0 to t = 100 with time step 0.1. Each
  // "set <key> = ..." entry listed in `overrides` is replaced.
  std::string
  parameters(const bool         point_coupling,
             const bool         dynamic,
             const Overrides   &overrides = {},
             const std::string &dirichlet = "0; 0; 0.01*t")
  {
    const std::string name = std::string(point_coupling ? "point" : "tensor") +
                             (dynamic ? "_dynamic" : "_static");
    std::string text = R"(
      set dimension               = 3
      set space dimension         = 3
      set reduced dimension       = 1
      set cross section dimension = 2
      subsection Functions
        subsection Right hand side
          set Function expression = 0; 0.05; 0.1
        end
        subsection Dirichlet boundary conditions
          set Function expression = @DIRICHLET@
        end
      end
      subsection Immersed Problem
        set Coupling type                      = @COUPLING@
        set Dirichlet boundary ids             = 2,3,4,5
        set FE degree                          = 1
        set Initial refinement                 = 2
        set Normal flux boundary ids           = 0,1
        set Output directory                   = @TEST_OUTPUT_DIR@/gtests/data/coupled_elasticity_newmark_01_@NAME@
        set Output name                        = coupled_elasticity_newmark_01_@NAME@
        set Output pressure                    = false
        set Output results also before solving = false
        set Pressure coupling                  = false
        set Remove interior stress from pressure = true
        subsection Grid generation
          set Domain type              = generate
          set Grid generator           = hyper_cube
          set Grid generator arguments = -1:1:true
          set Triangulation type       = distributed
        end
        subsection Immersed inclusions
          set Bounding boxes extraction level   = 0
          set Cluster inclusions with segments  = true
          set Inclusions file                   = @INCLUSIONS_FILE@
          set Inclusions refinement             = 100
          set Number of fourier coefficients    = 2
          set Reference inclusion data          = @REFERENCE_DATA@
          set Selection of Fourier coefficients = @SELECTION@
          subsection Boundary data
            set Function expression = 0; 0; 0
          end
        end
        subsection Material properties
          subsection default
            set Density        = @DENSITY@
            set Lame lambda    = 1
            set Lame mu        = 1
            set Rayleigh alpha = 0
            set Rayleigh beta  = 0
            set Viscosity eta  = 0
          end
        end
        subsection Refinement and remeshing
          set Maximum number of cells     = 2000000
          set Number of refinement cycles = 1
          set Refinement fraction         = 1
          set Strategy                    = global
        end
        subsection Time parameters
          set Initial time  = 0
          set Final time    = @FINAL_TIME@
          set Time step     = 0.1
          set Newmark beta  = 0.25
          set Newmark gamma = 0.5
        end
      end
      subsection Tensor product coupling
        subsection Cross section
          set Geometry type            = hyper_ball
          set Maximum basis degree     = 1
          set Refinement level         = 1
          set Selected modes           = 3, 7
        end
        subsection Particle coupling
          set RTree extraction level = 1
        end
        subsection Representative domain
          set Finite element degree            = 1
          set Input file fields                = displacement, vessel_id
          set Number of quadrature points      = 3
          set Number of quadrature repetitions = 8
          set Quadrature type                  = gauss
          set Reduced grid name                = @GRID@
          set Reduced right hand side          = 0.1; 0.1
          set Thickness                        = 0.2
        end
      end
      subsection Solvers
        subsection Augmented Lagrange
          set Max steps = 1000
          set Reduction = 1.e-12
          set Tolerance = 1.e-14
        end
        subsection Displacement
          set Max steps = 1000
          set Reduction = 1.e-12
          set Tolerance = 1.e-14
        end
        subsection Reduced mass
          set Max steps = 1000
          set Reduction = 1.e-12
          set Tolerance = 1.e-14
        end
      end
    )";

    const auto replace = [&text](const std::string &token,
                                 const std::string &value) {
      for (std::size_t pos = 0;
           (pos = text.find(token, pos)) != std::string::npos;
           pos += value.size())
        text.replace(pos, token.size(), value);
    };
    replace("@COUPLING@", point_coupling ? "Point" : "TensorProduct");
    replace("@NAME@", name);
    replace("@INCLUSIONS_FILE@", point_coupling ? inclusions_file() : "");
    replace("@REFERENCE_DATA@", point_coupling ? "0,0,0,0.1,0,0,0,0.1,0" : "");
    replace("@SELECTION@", point_coupling ? "3,7" : "");
    replace("@GRID@", point_coupling ? "" : reduced_grid());
    replace("@DENSITY@", dynamic ? "1" : "0");
    replace("@FINAL_TIME@", dynamic ? "100" : "0");
    replace("@DIRICHLET@", dirichlet);

    for (const auto &[key, value] : overrides)
      {
        const std::regex entry("set " + key + R"(\s*=[^\n]*)");
        EXPECT_TRUE(std::regex_search(text, entry)) << key;
        text = std::regex_replace(text, entry, "set " + key + " = " + value);
      }
    return TestPaths::expand_configured_paths(text);
  }

  // Declare the parameters, read them from `text`, and pass the problem to
  // `test`. The problem is built before parsing so that its Inclusions member
  // reads its parameters too.
  void
  with_problem(const std::string                                       &text,
               const std::function<void(VesselElasticityProblem<3> &)> &test)
  {
    ParameterAcceptor::clear();
    ElasticityProblemParameters<3> par;
    VesselElasticityProblem<3>     problem(par);
    initialize_parameters();
    ParameterAcceptor::prm.parse_input_from_string(text);
    ParameterAcceptor::parse_all_parameters();
    // The time mode is inferred from the time and material parameters, which
    // a single parsing pass reads after the consistency check.
    par.check_model_consistency();
    test(problem);
  }

  // Mean wall pressure of the only vessel, on every rank.
  double
  vessel_pressure(const VesselElasticityProblem<3> &problem)
  {
    const Vector<double> pressure(problem.pressure_at_immersed_surface);
    return pressure[0];
  }

  struct State
  {
    LA::MPI::Vector displacement;
    double          pressure;
  };

  // Displacement and vessel pressure of a static solve at `time`.
  State
  static_state(const bool         point_coupling,
               const double       time,
               const double       data,
               const std::string &dirichlet = "0; 0; 0.01*t")
  {
    State state;
    with_problem(parameters(point_coupling, false, {}, dirichlet),
                 [&](auto &problem) {
                   EXPECT_EQ(problem.par.time_mode, TimeMode::Static);
                   problem.run_timestep0();
                   if (point_coupling)
                     problem.update_vessel_radius_data({data});
                   problem.run_timestep(time);
                   state.displacement = problem.solution.block(0);
                   state.pressure     = vessel_pressure(problem);
                 });
    return state;
  }

  double
  relative_difference(const LA::MPI::Vector &u, const LA::MPI::Vector &v)
  {
    auto difference = u;
    difference -= v;
    return difference.l2_norm() / v.l2_norm();
  }

  // Relative residual of the wall constraint B u - O lambda = g.
  double
  constraint_residual(const VesselElasticityProblem<3> &problem)
  {
    const auto     &g = problem.system_rhs.block(1);
    LA::MPI::Vector residual(g), outside(g);
    problem.multiplier_load_matrix().Tvmult(residual,
                                            problem.solution.block(0));
    problem.outside_mass_matrix.vmult(outside, problem.solution.block(1));
    residual -= outside;
    residual -= g;
    return residual.l2_norm() / g.l2_norm();
  }

  // The first dynamic step starts at the static equilibrium of the wall data,
  // the body force and the Dirichlet data at that time.
  void
  check_equilibrium_start(const bool point_coupling)
  {
    const double time      = 0.5;
    const auto   reference = static_state(point_coupling, time, 0.1);
    with_problem(parameters(point_coupling, true), [&](auto &problem) {
      ASSERT_EQ(problem.par.time_mode, TimeMode::Dynamic);
      problem.run_timestep0();
      problem.run_timestep(time);
      EXPECT_DOUBLE_EQ(problem.current_time, time);
      EXPECT_LT(relative_difference(problem.solution.block(0),
                                    reference.displacement),
                1.e-8);
      EXPECT_NEAR(vessel_pressure(problem),
                  reference.pressure,
                  1.e-8 * std::abs(reference.pressure));
      EXPECT_EQ(problem.velocity.block(0).l2_norm(), 0.);
      EXPECT_EQ(problem.acceleration.block(0).l2_norm(), 0.);
      EXPECT_LT(constraint_residual(problem), 1.e-6);
    });
  }

  // With constant wall data, body force and Dirichlet data, Newmark steps
  // from the equilibrium stay at the equilibrium. This checks the signs and
  // scaling of the predictor, the lifting and the constraint row.
  void
  check_steady_state(const bool point_coupling, const Overrides &overrides)
  {
    with_problem(
      parameters(point_coupling, true, overrides, "0; 0; 0.01"),
      [&](auto &problem) {
        problem.run_timestep0();
        problem.run_timestep(0.);
        const LA::MPI::Vector u0       = problem.solution.block(0);
        const LA::MPI::Vector lambda0  = problem.solution.block(1);
        const double          pressure = vessel_pressure(problem);
        for (const double time : {0.1, 0.2, 0.3})
          {
            problem.run_timestep(time);
            EXPECT_LT(relative_difference(problem.solution.block(0), u0),
                      1.e-8);
            EXPECT_LT(relative_difference(problem.solution.block(1), lambda0),
                      1.e-8);
            EXPECT_NEAR(vessel_pressure(problem),
                        pressure,
                        1.e-8 * std::abs(pressure));
            EXPECT_LT(problem.velocity.block(0).l2_norm(),
                      1.e-8 * u0.l2_norm());
            EXPECT_LT(constraint_residual(problem), 1.e-6);
          }
      });
  }
} // namespace

TEST(CoupledElasticityNewmark, BOTH_StartsAtStaticEquilibriumPoint)
{
  check_equilibrium_start(true);
}

TEST(CoupledElasticityNewmark, BOTH_StartsAtStaticEquilibriumTensorProduct)
{
  check_equilibrium_start(false);
}

// Refinement around the inclusions creates hanging nodes.
TEST(CoupledElasticityNewmark, BOTH_ConstantDataKeepsEquilibriumPoint)
{
  check_steady_state(true, {{"Strategy", "inclusions"}});
}

TEST(CoupledElasticityNewmark, BOTH_ConstantDataKeepsEquilibriumTensorProduct)
{
  check_steady_state(false, {});
}

// After a change of the wall data, the damped tissue relaxes to the static
// solution of the new data. The wall data hold at every step. With Rayleigh
// damping D = 4 M + 0.5 A and rho = 1 every mode of M u'' + D u' + A u = 0 is
// overdamped. The jump of the wall data also excites stiff modes,
// whose Newmark accelerations decay only by the numerical dissipation, by
// about 0.43 per step with gamma = 0.9 and beta = 0.49. The multiplier
// balances rho M a, so it converges with them.
TEST(CoupledElasticityNewmark, BOTH_ViscousRelaxationReachesStaticSolutionPoint)
{
  const double final_time = 8.;
  const auto   reference  = static_state(true, final_time, 0.05, "0; 0; 0.01");
  with_problem(parameters(true,
                          true,
                          {{"Rayleigh alpha", "4"},
                           {"Rayleigh beta", "0.5"},
                           {"Time step", "0.2"},
                           {"Newmark beta", "0.49"},
                           {"Newmark gamma", "0.9"}},
                          "0; 0; 0.01"),
               [&](auto &problem) {
                 problem.run_timestep0();
                 problem.run_timestep(0.);
                 const double initial_error =
                   relative_difference(problem.solution.block(0),
                                       reference.displacement);
                 problem.update_vessel_radius_data({0.05});
                 for (unsigned int step = 1; step <= 40; ++step)
                   {
                     problem.run_timestep(0.2 * step);
                     ASSERT_LT(constraint_residual(problem), 1.e-6)
                       << "step " << step;
                   }
                 EXPECT_NEAR(problem.current_time, final_time, 1.e-12);
                 EXPECT_GT(initial_error, 0.1);
                 // A serial Debug run gave 6.4e-9 for the displacement; the
                 // pressure agreed to the six digits printed.
                 EXPECT_LT(relative_difference(problem.solution.block(0),
                                               reference.displacement),
                           1.e-6);
                 EXPECT_NEAR(vessel_pressure(problem),
                             reference.pressure,
                             1.e-6 * std::abs(reference.pressure));
               });
}

// Small, smoothly growing wall data with a small time step and a body force.
// Each step solves the saddle point problem for the increment beta dt^2 a
// with the momentum row scaled by 1 / (beta dt^2), while the constraint row
// holds the small residual g - B u_pred. Without the same scaling of the
// constraint row, FGMRES stopped after one iteration in the first steps, with
// relative constraint residuals up to 4.7e-3 (serial Release run); with it,
// the residuals stay below 2e-5. On the cube of
// doc/tutorials/metric-flow-x-elastodynamics-mms.md the unmet constraint made
// the multiplier oscillate from step to step with an amplitude growing like
// 1 / dt^2.
TEST(CoupledElasticityNewmark,
     BOTH_SmallTimeStepKeepsWallConstraintTensorProduct)
{
  with_problem(parameters(false,
                          true,
                          {{"Time step", "0.001"},
                           {"Reduced right hand side", "1e-5*t^3; 1e-5*t^3"}},
                          "0; 0; 0"),
               [&](auto &problem) {
                 problem.run_timestep0();
                 problem.run_timestep(0.);
                 for (unsigned int step = 1; step <= 20; ++step)
                   {
                     problem.run_timestep(0.001 * step);
                     EXPECT_LT(constraint_residual(problem), 1.e-4)
                       << "step " << step;
                   }
               });
}

// A coupled Newmark step must advance the tissue by exactly one time step.
TEST(CoupledElasticityNewmark, BOTH_RejectsStepOfWrongSizePoint)
{
  with_problem(parameters(true, true), [](auto &problem) {
    problem.run_timestep0();
    problem.run_timestep(0.);
    EXPECT_THROW(problem.run_timestep(0.2), std::exception);
  });
}

#endif
