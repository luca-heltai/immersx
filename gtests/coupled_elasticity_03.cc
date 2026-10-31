#include <gtest/gtest.h>
#include <immersx/physics/coupled_elasticity.h>

using namespace ImmersX;
#include <immersx/io/utils.h>
#include <immersx/physics/elasticity_problem_parameters.h>
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <regex>
#include <sstream>

#include "test_paths.h"

// Tests of CoupledElasticityProblem::update_vessel_radius_data() and
// split_pressure_over_inclusions() on a straight cylinder along x embedded in
// a cube. The setup is the one of
// coupled_elasticity_02.cc. The reduced grids and inclusion files are
// generated in the test output directory.
#ifdef DEAL_II_WITH_VTK

using namespace dealii;

namespace
{
  using Overrides = std::vector<std::pair<std::string, std::string>>;

  // Write `content` below the test output directory and return the file name.
  // Rank 0 writes a temporary file and renames it, so that test processes
  // running at the same time never read a partially written file.
  std::string
  input_file(const std::string &name, const std::string &content)
  {
    const auto filename =
      TestPaths::output_path("gtests/data/coupled_elasticity_03/" + name);
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

  // Reduced grid on the x axis with one cell between consecutive `vertices`.
  // Cell i has vessel id vessel_ids[i], and every vertex has displacement 0.1.
  std::string
  reduced_grid(const std::string               &name,
               const std::vector<double>       &vertices,
               const std::vector<unsigned int> &vessel_ids)
  {
    AssertDimension(vertices.size(), vessel_ids.size() + 1);
    const auto         n_cells = vessel_ids.size();
    std::ostringstream vtk;
    vtk << "# vtk DataFile Version 3.0\n"
        << "reduced grid\n"
        << "ASCII\n"
        << "DATASET UNSTRUCTURED_GRID\n"
        << "POINTS " << vertices.size() << " double\n";
    for (const double x : vertices)
      vtk << x << " 0 0\n";
    vtk << "CELLS " << n_cells << " " << 3 * n_cells << "\n";
    for (unsigned int i = 0; i < n_cells; ++i)
      vtk << "2 " << i << " " << i + 1 << "\n";
    vtk << "CELL_TYPES " << n_cells << "\n";
    for (unsigned int i = 0; i < n_cells; ++i)
      vtk << "3\n";
    vtk << "POINT_DATA " << vertices.size() << "\n"
        << "SCALARS displacement float\n"
        << "LOOKUP_TABLE default\n";
    for (unsigned int i = 0; i < vertices.size(); ++i)
      vtk << "0.1\n";
    vtk << "CELL_DATA " << n_cells << "\n"
        << "SCALARS vessel_id int\n"
        << "LOOKUP_TABLE default\n";
    for (const auto id : vessel_ids)
      vtk << id << "\n";
    return input_file(name + ".vtk", vtk.str());
  }

  // One cell from x = 1 to x = -1 with vessel id 0.
  std::string
  single_cell_grid()
  {
    return reduced_grid("single_cell", {1., -1.}, {0});
  }

  // `n_inclusions` point inclusions of equal length covering x in [-1, 1]
  // (for 50: centered at -0.98, -0.94, ..., 0.98), split into `n_vessels`
  // consecutive vessels of equal size. Each line is cx, cy, cz, dx, dy, dz, r,
  // vessel id.
  std::string
  inclusions_file(const unsigned int n_vessels,
                  const unsigned int n_inclusions = 50)
  {
    const double       length = 2. / n_inclusions;
    std::ostringstream text;
    for (unsigned int i = 0; i < n_inclusions; ++i)
      text << -1. + (i + 0.5) * length << " 0 0 " << length << " 0 0 0.2 "
           << i * n_vessels / n_inclusions << "\n";
    return input_file("inclusions_" + std::to_string(n_vessels) + "_" +
                        std::to_string(n_inclusions) + ".txt",
                      text.str());
  }

  // The inclusions of inclusions_file(), with inclusions 0-48 in vessel 0 and
  // inclusion 49 alone in vessel 1.
  std::string
  inclusions_file_with_single_inclusion_vessel()
  {
    std::ostringstream text;
    for (unsigned int i = 0; i < 50; ++i)
      text << -0.98 + 0.04 * i << " 0 0 0.04 0 0 0.2 " << (i == 49 ? 1 : 0)
           << "\n";
    return input_file("inclusions_single_inclusion_vessel.txt", text.str());
  }

  // Parameters of the cylinder with point coupling, using the inclusions of
  // inclusions_file(1), or with tensor-product coupling. Each "set <key> = ..."
  // entry listed in `overrides` is replaced by "set <key> = <value>". The inner
  // displacement solve needs more than the default 100 iterations on locally
  // refined meshes and on two ranks.
  std::string
  parameters(const bool         point_coupling,
             const std::string &reduced_grid,
             const std::string &input_file_fields,
             const Overrides   &overrides = {})
  {
    const std::string name = point_coupling ? "point" : "tensor";
    std::string       text = R"(
      set dimension               = 3
      set space dimension         = 3
      set reduced dimension       = 1
      set cross section dimension = 2
      subsection Immersed Problem
        set Coupling type                      = @COUPLING@
        set Dirichlet boundary ids             = 2,3,4,5
        set FE degree                          = 1
        set Initial refinement                 = 3
        set Normal flux boundary ids           = 0,1
        set Output directory                   = @TEST_OUTPUT_DIR@/gtests/data/coupled_elasticity_03_@NAME@
        set Output name                        = coupled_elasticity_03_@NAME@
        set Output pressure                    = true
        set Output results also before solving = false
        set Pressure coupling                  = false
        set Remove interior stress from pressure = true
        set Weak Dirichlet penalty coefficient = 1000
        subsection Grid generation
          set Domain type              = generate
          set Grid generator           = hyper_cube
          set Grid generator arguments = -1:1:true
          set Triangulation type       = distributed
          set Grid scale               = 1
        end
        subsection Immersed inclusions
          set Bounding boxes extraction level   = 0
          set Cluster inclusions with segments  = @CLUSTER@
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
            set Density        = 0
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
      end
      subsection Tensor product coupling
        subsection Cross section
          set Inclusion type           = hyper_ball
          set Maximum inclusion degree = 1
          set Refinement level         = 1
          set Selected indices         = 3, 7
        end
        subsection Particle coupling
          set RTree extraction level = 1
        end
        subsection Representative domain
          set Finite element degree            = 1
          set Input file fields                = @FIELDS@
          set Number of quadrature points      = 3
          set Number of quadrature repetitions = @REPETITIONS@
          set Quadrature type                  = gauss
          set Reduced grid name                = @GRID@
          set Reduced right hand side          = 0.1; 0.1
          set Thickness                        = 0.2
        end
      end
      subsection Solvers
        subsection Displacement
          set Max steps = 1000
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
    replace("@CLUSTER@", point_coupling ? "true" : "false");
    replace("@INCLUSIONS_FILE@", point_coupling ? inclusions_file(1) : "");
    replace("@REFERENCE_DATA@", point_coupling ? "0,0,0,0.1,0,0,0,0.1,0" : "");
    replace("@SELECTION@", point_coupling ? "3,7" : "");
    replace("@REPETITIONS@", point_coupling ? "4" : "8");
    replace("@FIELDS@", input_file_fields);
    replace("@GRID@", reduced_grid);

    for (const auto &[key, value] : overrides)
      {
        const std::regex entry("set " + key + R"(\s*=[^\n]*)");
        EXPECT_TRUE(std::regex_search(text, entry)) << key;
        text = std::regex_replace(text, entry, "set " + key + " = " + value);
      }
    return TestPaths::expand_configured_paths(text);
  }

  // Point inclusions along x: inclusions 0-24 belong to vessel 0 and
  // inclusions 25-49 to vessel 1. Inclusions are not clustered, so each one
  // owns its own multiplier dofs.
  std::string
  point_parameters(const Overrides &overrides = {})
  {
    Overrides all = {{"Inclusions file", inclusions_file(2)},
                     {"Cluster inclusions with segments", "false"}};
    all.insert(all.end(), overrides.begin(), overrides.end());
    return parameters(true, single_cell_grid(), "displacement", all);
  }

  // Declare the parameters, read them from `text`, and pass the problem to
  // `test`. The problem is built before parsing, as in app_coupled_elasticity,
  // so that its Inclusions member reads its parameters too.
  void
  with_problem(const std::string                                        &text,
               const std::function<void(CoupledElasticityProblem<3> &)> &test)
  {
    ParameterAcceptor::clear();
    ElasticityProblemParameters<3> par;
    CoupledElasticityProblem<3>    problem(par);
    initialize_parameters();
    ParameterAcceptor::prm.parse_input_from_string(text);
    ParameterAcceptor::parse_all_parameters();
    test(problem);
  }

  // All entries of a distributed vector, on every rank.
  std::vector<double>
  gather(const TrilinosWrappers::MPI::Vector &v)
  {
    std::vector<double> local(v.size(), 0.);
    for (const auto i : v.locally_owned_elements())
      local[i] = v[i];
    std::vector<double> values(v.size(), 0.);
    Utilities::MPI::sum(local, MPI_COMM_WORLD, values);
    return values;
  }

  // Value at inclusion k of n of the samples placed at equal distances from
  // the first to the last inclusion of a vessel, interpolated linearly. A
  // single sample is a constant value.
  double
  interpolated_sample(const std::vector<double> &samples,
                      const unsigned int         k,
                      const unsigned int         n)
  {
    if (samples.size() == 1)
      return samples[0];
    const double       x = k * (samples.size() - 1.) / (n - 1.);
    const unsigned int j =
      std::min<unsigned int>(std::floor(x), samples.size() - 2);
    const double w = x - j;
    return (1 - w) * samples[j] + w * samples[j + 1];
  }

  // Set the pressure of inclusion i to pressure(i), for the 50 inclusions of
  // point_parameters().
  void
  set_pressure_by_inclusion(CoupledElasticityProblem<3>           &problem,
                            const std::function<double(unsigned)> &pressure)
  {
    auto &by_inclusion = problem.pressure_at_immersed_surface_by_inclusion;
    by_inclusion.reinit(Utilities::MPI::create_evenly_distributed_partitioning(
                          MPI_COMM_WORLD, 50),
                        MPI_COMM_WORLD);
    for (const auto i : by_inclusion.locally_owned_elements())
      by_inclusion[i] = pressure(i);
    by_inclusion.compress(VectorOperation::insert);
  }
} // namespace

// -----------------------------------------------------------------------------
// update_vessel_radius_data(new_data)
// -----------------------------------------------------------------------------

// One value per vessel is assigned to all inclusions of that vessel.
TEST(CoupledElasticity, BOTH_UpdateVesselRadiusDataPerVesselPoint)
{
  with_problem(point_parameters(), [](auto &problem) {
    problem.run_timestep0();
    ASSERT_NO_THROW(problem.update_vessel_radius_data({0.3, 0.7}));
    for (unsigned int i = 0; i < 50; ++i)
      EXPECT_DOUBLE_EQ(problem.inclusions.inclusions_data[i][3],
                       i < 25 ? 0.3 : 0.7)
        << "inclusion " << i;
  });
}

// One value per inclusion is assigned to that inclusion.
TEST(CoupledElasticity, BOTH_UpdateVesselRadiusDataPerInclusionPoint)
{
  with_problem(point_parameters(), [](auto &problem) {
    problem.run_timestep0();
    std::vector<double> new_data(50);
    for (unsigned int i = 0; i < new_data.size(); ++i)
      new_data[i] = 0.01 * (i + 1);
    ASSERT_NO_THROW(problem.update_vessel_radius_data(new_data));
    for (unsigned int i = 0; i < 50; ++i)
      EXPECT_DOUBLE_EQ(problem.inclusions.inclusions_data[i][3], new_data[i])
        << "inclusion " << i;
  });
}

// The value is a radial displacement: it is written to the cos(theta) mode of
// the first component and to the sin(theta) mode of the second component, the
// indices 3 and 7 of the selection, and every other coefficient is zero.
TEST(CoupledElasticity, BOTH_UpdateVesselRadiusDataWritesRadialModePoint)
{
  with_problem(point_parameters(), [](auto &problem) {
    problem.run_timestep0();
    ASSERT_NO_THROW(problem.update_vessel_radius_data({0.3, 0.7}));
    for (unsigned int i = 0; i < 50; ++i)
      {
        const auto  &data     = problem.inclusions.inclusions_data[i];
        const double expected = i < 25 ? 0.3 : 0.7;
        ASSERT_GT(data.size(), 7u) << "inclusion " << i;
        for (unsigned int k = 0; k < data.size(); ++k)
          EXPECT_DOUBLE_EQ(data[k], (k == 3 || k == 7) ? expected : 0.)
            << "inclusion " << i << ", coefficient " << k;
      }
  });
}

// An empty vector leaves the data unchanged.
TEST(CoupledElasticity, BOTH_UpdateVesselRadiusDataEmptyPoint)
{
  with_problem(point_parameters(), [](auto &problem) {
    problem.run_timestep0();
    const auto before = problem.inclusions.inclusions_data;
    ASSERT_NO_THROW(problem.update_vessel_radius_data({}));
    EXPECT_EQ(problem.inclusions.inclusions_data, before);
  });
}

// A size that is neither the number of vessels nor the number of inclusions
// is rejected, and the data is left unchanged.
TEST(CoupledElasticity, BOTH_UpdateVesselRadiusDataWrongSizePoint)
{
  with_problem(point_parameters(), [](auto &problem) {
    problem.run_timestep0();
    const auto before = problem.inclusions.inclusions_data;
    EXPECT_ANY_THROW(problem.update_vessel_radius_data({0.1, 0.2, 0.3}));
    EXPECT_EQ(problem.inclusions.inclusions_data, before);
  });
}

// The updated data reaches the solve. The problem is linear in the boundary
// data, so doubling the radial displacement from 0.1 to 0.2 doubles the
// pressure of each vessel. From cycle 1 on, run_timestep() only reassembles
// the coupling right-hand side, which is where the new data is read.
TEST(CoupledElasticity, BOTH_UpdateVesselRadiusDataChangesPressurePoint)
{
  with_problem(point_parameters(), [](auto &problem) {
    problem.run_timestep0();
    problem.run_timestep();
    const auto before = gather(problem.pressure_at_immersed_surface);

    ASSERT_NO_THROW(problem.update_vessel_radius_data({0.2, 0.2}));
    problem.run_timestep();
    const auto after = gather(problem.pressure_at_immersed_surface);

    ASSERT_EQ(before.size(), 2u);
    ASSERT_EQ(after.size(), 2u);
    for (unsigned int v = 0; v < 2; ++v)
      {
        ASSERT_NE(before[v], 0.0) << "vessel " << v;
        EXPECT_NEAR(after[v], 2 * before[v], 1e-6 * std::abs(before[v]))
          << "vessel " << v;
      }
  });
}

// -----------------------------------------------------------------------------
// compute_pressure_at_immersed_surface()
// -----------------------------------------------------------------------------

namespace
{
  // Compute only the multiplier part of the pressure, without the interior
  // stress, so that the expected values depend on the multiplier alone.
  const std::pair<std::string, std::string> multiplier_only = {
    "Remove interior stress from pressure",
    "false"};

  // Set multiplier dof i to 1 + i.
  void
  set_increasing_multiplier(CoupledElasticityProblem<3> &problem)
  {
    auto           &lambda = problem.locally_relevant_solution.block(1);
    LA::MPI::Vector owned(lambda.locally_owned_elements(), MPI_COMM_WORLD);
    for (const auto i : owned.locally_owned_elements())
      owned[i] = 1. + i;
    owned.compress(VectorOperation::insert);
    lambda = owned;
  }

  // Set the multiplier to 1 + i, compute the pressure, and compare it with
  // the formulas of doc/concepts/vessel-pressure.md: inclusion k uses the
  // multiplier block b(k) (its own, or its segment's), only the coefficients
  // 3 and 7 count, and
  //   p_k = (lambda_{b(k),3} + lambda_{b(k),7}) / (2 |S_k|),
  //   p_v = sum_{k in v} (lambda_{b(k),3} + lambda_{b(k),7}) / 2 / |Gamma_v|,
  // with |Gamma_v| the sum of |S_k| over the inclusions of v.
  void
  check_pressure_from_multiplier(CoupledElasticityProblem<3> &problem)
  {
    set_increasing_multiplier(problem);
    problem.compute_pressure_at_immersed_surface();

    const auto &inclusions = problem.inclusions;
    const auto  n          = inclusions.n_dofs_per_inclusion();
    const auto  selection  = inclusions.get_selected_coefficients();

    std::vector<unsigned int> block_of(inclusions.n_inclusions(),
                                       numbers::invalid_unsigned_int);
    if (inclusions.cluster_with_segments)
      {
        for (unsigned int s = 0; s < inclusions.n_global_segments(); ++s)
          for (const auto k : inclusions.get_inclusions_in_a_segment(s))
            block_of[k] = s;
      }
    else
      for (unsigned int k = 0; k < block_of.size(); ++k)
        block_of[k] = k;

    std::vector<double> by_inclusion(inclusions.n_inclusions(), 0.);
    std::vector<double> force(inclusions.get_n_vessels(), 0.);
    std::vector<double> area(inclusions.get_n_vessels(), 0.);
    for (unsigned int k = 0; k < block_of.size(); ++k)
      {
        ASSERT_NE(block_of[k], numbers::invalid_unsigned_int)
          << "inclusion " << k;
        double normal_force = 0;
        for (unsigned int j = 0; j < n; ++j)
          if (selection[j] == 3 || selection[j] == 7)
            normal_force += (1. + block_of[k] * n + j) / 2.;
        const auto measure = inclusions.get_section_measure(k);
        by_inclusion[k]    = normal_force / measure;
        force[inclusions.get_vesselID(k)] += normal_force;
        area[inclusions.get_vesselID(k)] += measure;
      }

    const auto computed_by_inclusion =
      gather(problem.pressure_at_immersed_surface_by_inclusion);
    const auto computed_by_vessel = gather(problem.pressure_at_immersed_surface);
    ASSERT_EQ(computed_by_inclusion.size(), by_inclusion.size());
    ASSERT_EQ(computed_by_vessel.size(), force.size());
    for (unsigned int k = 0; k < by_inclusion.size(); ++k)
      EXPECT_NEAR(computed_by_inclusion[k],
                  by_inclusion[k],
                  1e-12 * std::abs(by_inclusion[k]))
        << "inclusion " << k;
    for (unsigned int v = 0; v < force.size(); ++v)
      EXPECT_NEAR(computed_by_vessel[v],
                  force[v] / area[v],
                  1e-12 * std::abs(force[v] / area[v]))
        << "vessel " << v;
  }

  // Pressure of each vessel after one solve of the point-coupled problem
  // with `n_inclusions` inclusions on the cylinder, split into two vessels.
  std::vector<double>
  solved_vessel_pressure(const unsigned int n_inclusions, const bool cluster)
  {
    std::vector<double> pressure;
    with_problem(point_parameters(
                   {{"Inclusions file", inclusions_file(2, n_inclusions)},
                    {"Cluster inclusions with segments",
                     cluster ? "true" : "false"}}),
                 [&](auto &problem) {
                   problem.run_timestep0();
                   problem.run_timestep();
                   pressure = gather(problem.pressure_at_immersed_surface);
                 });
    return pressure;
  }
} // namespace

// Without clustering, inclusion k owns multiplier block k.
TEST(CoupledElasticity, BOTH_PressureFromMultiplierPoint)
{
  with_problem(point_parameters({multiplier_only}), [](auto &problem) {
    problem.run_timestep0();
    ASSERT_EQ(problem.inclusions.n_dofs(), 50u * 2u);
    check_pressure_from_multiplier(problem);
  });
}

// Coefficients other than 3 and 7 do not exert a normal force on the wall,
// so they do not enter the pressure. Here they are the constant mode in x and
// cos(theta) in y.
TEST(CoupledElasticity, BOTH_PressureIgnoresTangentialCoefficientsPoint)
{
  with_problem(point_parameters(
                 {multiplier_only,
                  {"Number of fourier coefficients", "4"},
                  {"Selection of Fourier coefficients", "0, 3, 4, 7"}}),
               [](auto &problem) {
                 problem.run_timestep0();
                 ASSERT_EQ(problem.inclusions.n_dofs(), 50u * 4u);
                 check_pressure_from_multiplier(problem);
               });
}

// Without any of the coefficients 3 and 7 the multiplier exerts no normal
// force, and the pressure is zero.
TEST(CoupledElasticity, BOTH_PressureWithoutNormalCoefficientsIsZeroPoint)
{
  with_problem(point_parameters(
                 {multiplier_only,
                  {"Number of fourier coefficients", "2"},
                  {"Selection of Fourier coefficients", "0, 4"}}),
               [](auto &problem) {
                 problem.run_timestep0();
                 set_increasing_multiplier(problem);
                 problem.compute_pressure_at_immersed_surface();
                 for (const auto p :
                      gather(problem.pressure_at_immersed_surface))
                   EXPECT_EQ(p, 0.);
                 for (const auto p :
                      gather(problem.pressure_at_immersed_surface_by_inclusion))
                   EXPECT_EQ(p, 0.);
               });
}

// With "Cluster inclusions with segments" the multiplier is numbered by
// segment: every inclusion of a segment receives the segment's block.
TEST(CoupledElasticity, BOTH_PressureFromMultiplierClusteredPoint)
{
  with_problem(point_parameters(
                 {multiplier_only, {"Cluster inclusions with segments", "true"}}),
               [](auto &problem) {
                 problem.run_timestep0();
                 const auto n_segments = problem.inclusions.n_global_segments();
                 ASSERT_GT(n_segments, 0u);
                 ASSERT_LT(n_segments, 50u);
                 ASSERT_EQ(problem.inclusions.n_dofs(), n_segments * 2u);
                 check_pressure_from_multiplier(problem);
               });
}

// The pressure is a mean normal stress, so it does not depend on how finely
// the vessel is split into inclusions. With segments the constraint of a
// segment only changes by the quadrature of its wall, and the pressure agrees
// closely.
TEST(CoupledElasticity, BOTH_PressureIndependentOfInclusionSpacingClusteredPoint)
{
  const auto coarse = solved_vessel_pressure(50, true);
  const auto fine   = solved_vessel_pressure(100, true);
  ASSERT_EQ(coarse.size(), 2u);
  ASSERT_EQ(fine.size(), 2u);
  for (unsigned int v = 0; v < 2; ++v)
    {
      ASSERT_NE(coarse[v], 0.) << "vessel " << v;
      EXPECT_NEAR(fine[v], coarse[v], 1e-2 * std::abs(coarse[v]))
        << "vessel " << v;
    }
}

// Without segments every inclusion adds its own constraint, so more
// inclusions per bulk cell change the discrete problem. The pressure still
// has the same size: the multiplier of each inclusion scales with its wall
// area.
TEST(CoupledElasticity, BOTH_PressureIndependentOfInclusionSpacingPoint)
{
  const auto coarse = solved_vessel_pressure(50, false);
  const auto fine   = solved_vessel_pressure(100, false);
  ASSERT_EQ(coarse.size(), 2u);
  ASSERT_EQ(fine.size(), 2u);
  for (unsigned int v = 0; v < 2; ++v)
    {
      ASSERT_NE(coarse[v], 0.) << "vessel " << v;
      EXPECT_NEAR(fine[v], coarse[v], 0.1 * std::abs(coarse[v]))
        << "vessel " << v;
    }
}

// Tensor-product coupling on four reduced cells, two per vessel, with the
// selected modes 3 and 7. The multiplier components are set to the linear
// functions lambda_3(x) = 1 + x and lambda_7(x) = 2 + 3x. The pressure at
// each support point is k (lambda_3 + lambda_7) with k = 1/2, since the modes
// are cos(theta) e_x and sin(theta) e_y, up to the polygonal approximation of
// the reference circle (k = 0.475 on the 8-sided polygon of Refinement level
// 1), and the pressure of each
// vessel is the mean of k (lambda_3 + lambda_7) over its cells.
TEST(CoupledElasticity, BOTH_PressureFromMultiplierTensorProduct)
{
  const std::vector<double> vertices = {-1., -0.5, 0., 0.5, 1.};
  with_problem(
    parameters(false,
               reduced_grid("four_cells", vertices, {0, 0, 1, 1}),
               "displacement, vessel_id",
               {multiplier_only}),
    [&](auto &problem) {
      problem.run_timestep0();
      ASSERT_EQ(problem.n_vessels(), 2u);

      const auto &reduced_dh =
        problem.tensor_product_coupling->get_dof_handler();
      ASSERT_EQ(reduced_dh.get_fe().n_components(), 2u);

      const auto lambda_3 = [](const double x) { return 1. + x; };
      const auto lambda_7 = [](const double x) { return 2. + 3. * x; };
      {
        auto           &lambda = problem.locally_relevant_solution.block(1);
        LA::MPI::Vector owned(lambda.locally_owned_elements(), MPI_COMM_WORLD);
        VectorTools::interpolate(
          reduced_dh,
          FunctionFromFunctionObjects<3>(
            {[&](const Point<3> &p) { return lambda_3(p[0]); },
             [&](const Point<3> &p) { return lambda_7(p[0]); }}),
          owned);
        lambda = owned;
      }

      problem.compute_pressure_at_immersed_surface();

      // One pressure per support point, i.e., per vertex for linear elements.
      // Compare the sorted values, since the numbering of the support points
      // follows the reduced DoFHandler.
      auto by_point = gather(problem.pressure_at_immersed_surface_by_inclusion);
      ASSERT_EQ(by_point.size(), vertices.size());
      std::sort(by_point.begin(), by_point.end());
      const double k = by_point[0] / (lambda_3(-1.) + lambda_7(-1.));
      EXPECT_NEAR(k, 0.5, 3e-2);
      for (unsigned int n = 0; n < vertices.size(); ++n)
        EXPECT_NEAR(by_point[n],
                    k * (lambda_3(vertices[n]) + lambda_7(vertices[n])),
                    1e-12)
          << "support point " << n;

      // The mean of a linear function over a vessel is its value at the
      // middle of the vessel: x = -0.5 for vessel 0 and x = 0.5 for vessel 1.
      const auto by_vessel = gather(problem.pressure_at_immersed_surface);
      ASSERT_EQ(by_vessel.size(), 2u);
      EXPECT_NEAR(by_vessel[0], k * (lambda_3(-0.5) + lambda_7(-0.5)), 1e-12);
      EXPECT_NEAR(by_vessel[1], k * (lambda_3(0.5) + lambda_7(0.5)), 1e-12);
    });
}

namespace
{
  // Solve, then set the multiplier to zero, so that the pressure computed
  // afterwards is the interior stress alone.
  void
  solve_and_zero_multiplier(CoupledElasticityProblem<3> &problem)
  {
    problem.run_timestep0();
    problem.run_timestep();
    auto &lambda = problem.locally_relevant_solution.block(1);
    lambda       = 0;
    problem.compute_pressure_at_immersed_surface();
  }

  // Interior stress 2 (lambda + mu) u_n / r of the point inclusions of
  // parameters(): lambda = mu = 1, r = 0.2, and the reference data 0.1 in
  // the coefficients 3 and 7, i.e., the wall displacement 0.1 (cos(theta),
  // sin(theta)) with mean normal value u_n = 0.1.
  constexpr double point_interior_stress = 2. * (1. + 1.) * 0.1 / 0.2;
} // namespace

// The material inside the vessels contributes the normal stress
// 2 (lambda + mu) u_n / r to the multiplier. With the multiplier set to zero,
// the pressure is this stress alone, on every inclusion and every vessel.
TEST(CoupledElasticity, BOTH_PressureRemovesInteriorStressPoint)
{
  with_problem(point_parameters(), [](auto &problem) {
    solve_and_zero_multiplier(problem);
    for (const auto p :
         gather(problem.pressure_at_immersed_surface_by_inclusion))
      EXPECT_NEAR(p, point_interior_stress, 1e-12);
    for (const auto p : gather(problem.pressure_at_immersed_surface))
      EXPECT_NEAR(p, point_interior_stress, 1e-12);
  });
}

// With segments the data of a segment is the sum over its inclusions, and
// each inclusion receives the mean. The data are the same on all inclusions,
// so the interior stress is the same as without segments.
TEST(CoupledElasticity, BOTH_PressureRemovesInteriorStressClusteredPoint)
{
  with_problem(point_parameters(
                 {{"Cluster inclusions with segments", "true"}}),
               [](auto &problem) {
                 solve_and_zero_multiplier(problem);
                 for (const auto p : gather(
                        problem.pressure_at_immersed_surface_by_inclusion))
                   EXPECT_NEAR(p, point_interior_stress, 1e-12);
                 for (const auto p :
                      gather(problem.pressure_at_immersed_surface))
                   EXPECT_NEAR(p, point_interior_stress, 1e-12);
               });
}

// Tensor-product coupling with the data 0.1 for the modes 3 and 7. The data
// follow the point-coupling convention, so they impose the wall displacement
// 0.1 (cos(theta), sin(theta)) with mean normal value u_n = 0.1, and the
// interior stress is the same as with point coupling, the same at every
// support point and on every vessel. The cross section is refined three
// times: on the coarser polygon of the other tests, u_n is 5% smaller.
TEST(CoupledElasticity, BOTH_PressureRemovesInteriorStressTensorProduct)
{
  with_problem(parameters(false,
                          reduced_grid("four_cells", {-1., -0.5, 0., 0.5, 1.},
                                       {0, 0, 1, 1}),
                          "displacement, vessel_id",
                          {{"Refinement level", "3"}}),
               [](auto &problem) {
                 solve_and_zero_multiplier(problem);
                 const auto by_point =
                   gather(problem.pressure_at_immersed_surface_by_inclusion);
                 ASSERT_EQ(by_point.size(), 5u);
                 EXPECT_NEAR(by_point[0],
                             point_interior_stress,
                             1e-2 * point_interior_stress);
                 for (const auto p : by_point)
                   EXPECT_NEAR(p, by_point[0], 1e-8 * by_point[0]);
                 const auto by_vessel =
                   gather(problem.pressure_at_immersed_surface);
                 ASSERT_EQ(by_vessel.size(), 2u);
                 for (const auto p : by_vessel)
                   EXPECT_NEAR(p, by_point[0], 1e-8 * by_point[0]);
               });
}

// -----------------------------------------------------------------------------
// update_vessel_radius_data(new_data, cells_per_vessel)
// -----------------------------------------------------------------------------

// Two cells per vessel give the values at the two ends of the vessel. The
// inclusions in between get the linear interpolation of these values.
TEST(CoupledElasticity, BOTH_UpdateVesselRadiusDataByCellsPoint)
{
  with_problem(point_parameters(), [](auto &problem) {
    problem.run_timestep0();
    const std::vector<double> ends = {0.1, 0.3, 0.5, 0.9};
    ASSERT_NO_THROW(problem.update_vessel_radius_data(ends, {2, 2}));
    for (unsigned int vessel = 0; vessel < 2; ++vessel)
      for (unsigned int k = 0; k < 25; ++k)
        {
          const double first    = ends[2 * vessel];
          const double last     = ends[2 * vessel + 1];
          const double expected = first + (last - first) * k / 24.;
          EXPECT_NEAR(problem.inclusions.inclusions_data[25 * vessel + k][3],
                      expected,
                      1e-12)
            << "vessel " << vessel << ", inclusion " << k;
        }
  });
}

// One cell per vessel gives the same value to all inclusions of the vessel,
// the last one included.
TEST(CoupledElasticity, BOTH_UpdateVesselRadiusDataByOneCellPoint)
{
  with_problem(point_parameters(), [](auto &problem) {
    problem.run_timestep0();
    ASSERT_NO_THROW(problem.update_vessel_radius_data({0.2, 0.4}, {1, 1}));
    for (unsigned int i = 0; i < 50; ++i)
      EXPECT_DOUBLE_EQ(problem.inclusions.inclusions_data[i][3],
                       i < 25 ? 0.2 : 0.4)
        << "inclusion " << i;
  });
}

// Vessels with different numbers of cells. Vessel 0 has three samples, so the
// middle sample lies at inclusion 12 and the interpolation has a kink there.
// Vessel 1 has four samples.
TEST(CoupledElasticity, BOTH_UpdateVesselRadiusDataByCellsInterpolatesPoint)
{
  with_problem(point_parameters(), [](auto &problem) {
    problem.run_timestep0();
    const std::vector<std::vector<double>> samples = {{0.1, 0.5, 0.2},
                                                      {0.3, 0.6, 0.4, 0.8}};
    std::vector<double>                    new_data;
    for (const auto &s : samples)
      new_data.insert(new_data.end(), s.begin(), s.end());
    ASSERT_NO_THROW(problem.update_vessel_radius_data(new_data, {3, 4}));
    for (unsigned int vessel = 0; vessel < 2; ++vessel)
      for (unsigned int k = 0; k < 25; ++k)
        EXPECT_NEAR(problem.inclusions.inclusions_data[25 * vessel + k][3],
                    interpolated_sample(samples[vessel], k, 25),
                    1e-12)
          << "vessel " << vessel << ", inclusion " << k;
  });
}

// -----------------------------------------------------------------------------
// split_pressure_over_inclusions()
// -----------------------------------------------------------------------------

// Inclusion i has pressure i. Each vessel has 25 inclusions and 49 cells, so
// cell c of vessel v lies at inclusion 25 v + c / 2 and gets that pressure by
// linear interpolation.
TEST(CoupledElasticity, BOTH_SplitPressureOverInclusionsPoint)
{
  with_problem(point_parameters(), [](auto &problem) {
    problem.run_timestep0();
    auto &by_inclusion = problem.pressure_at_immersed_surface_by_inclusion;
    by_inclusion.reinit(Utilities::MPI::create_evenly_distributed_partitioning(
                          MPI_COMM_WORLD, 50),
                        MPI_COMM_WORLD);
    for (const auto i : by_inclusion.locally_owned_elements())
      by_inclusion[i] = i;
    by_inclusion.compress(VectorOperation::insert);

    const auto split =
      problem.split_pressure_over_inclusions({49, 49}, Vector<double>(50));
    ASSERT_EQ(split.size(), 2u);
    for (unsigned int vessel = 0; vessel < 2; ++vessel)
      {
        ASSERT_EQ(split[vessel].size(), 49u);
        for (unsigned int c = 0; c < 49; ++c)
          EXPECT_NEAR(split[vessel][c], 25. * vessel + c / 2., 1e-12)
            << "vessel " << vessel << ", cell " << c;
      }
  });
}

// With one cell per inclusion, cell c of vessel v gets the pressure of
// inclusion 25 v + c. The pressure is not linear in the inclusion index, so
// this also checks that no interpolation happens between inclusions.
TEST(CoupledElasticity,
     BOTH_SplitPressureOverInclusionsOneCellPerInclusionPoint)
{
  with_problem(point_parameters(), [](auto &problem) {
    problem.run_timestep0();
    set_pressure_by_inclusion(problem, [](unsigned i) { return 1. * i * i; });

    const auto split =
      problem.split_pressure_over_inclusions({25, 25}, Vector<double>(50));
    ASSERT_EQ(split.size(), 2u);
    for (unsigned int vessel = 0; vessel < 2; ++vessel)
      {
        ASSERT_EQ(split[vessel].size(), 25u);
        for (unsigned int c = 0; c < 25; ++c)
          {
            const double i = 25. * vessel + c;
            EXPECT_NEAR(split[vessel][c], i * i, 1e-12)
              << "vessel " << vessel << ", cell " << c;
          }
      }
  });
}

// One cell per vessel gets the mean pressure of the inclusions of that
// vessel: 12 for inclusions 0-24 and 37 for inclusions 25-49.
TEST(CoupledElasticity, BOTH_SplitPressureOverInclusionsOneCellPoint)
{
  with_problem(point_parameters(), [](auto &problem) {
    problem.run_timestep0();
    set_pressure_by_inclusion(problem, [](unsigned i) { return 1. * i; });

    const auto split =
      problem.split_pressure_over_inclusions({1, 1}, Vector<double>(50));
    ASSERT_EQ(split.size(), 2u);
    ASSERT_EQ(split[0].size(), 1u);
    ASSERT_EQ(split[1].size(), 1u);
    EXPECT_NEAR(split[0][0], 12., 1e-12);
    EXPECT_NEAR(split[1][0], 37., 1e-12);
  });
}

// Vessels with different numbers of cells, and a pressure that is not linear
// in the inclusion index: cell c of a vessel with n cells lies at position
// c (25 - 1) / (n - 1) along the inclusions of the vessel, and its pressure is
// the linear interpolation of the two neighboring inclusions. The first and
// last cells get the pressure of the first and last inclusions.
TEST(CoupledElasticity, BOTH_SplitPressureOverInclusionsInterpolatesPoint)
{
  with_problem(point_parameters(), [](auto &problem) {
    problem.run_timestep0();
    const auto pressure = [](unsigned i) { return 1. * i * i; };
    set_pressure_by_inclusion(problem, pressure);

    const std::vector<int> n_cells = {3, 7};
    const auto             split =
      problem.split_pressure_over_inclusions(n_cells, Vector<double>(50));
    ASSERT_EQ(split.size(), 2u);
    for (unsigned int vessel = 0; vessel < 2; ++vessel)
      {
        std::vector<double> vessel_pressure(25);
        for (unsigned int k = 0; k < 25; ++k)
          vessel_pressure[k] = pressure(25 * vessel + k);

        ASSERT_EQ(split[vessel].size(), static_cast<unsigned>(n_cells[vessel]));
        for (int c = 0; c < n_cells[vessel]; ++c)
          EXPECT_NEAR(split[vessel][c],
                      interpolated_sample(vessel_pressure, c, n_cells[vessel]),
                      1e-12)
            << "vessel " << vessel << ", cell " << c;
        EXPECT_DOUBLE_EQ(split[vessel].front(), vessel_pressure.front());
        EXPECT_DOUBLE_EQ(split[vessel].back(), vessel_pressure.back());
      }
  });
}

// Vessel 1 has a single inclusion, so all its cells get the pressure of that
// inclusion, whatever the number of cells. Vessel 0 is split as usual.
TEST(CoupledElasticity, BOTH_SplitPressureOverInclusionsSingleInclusionPoint)
{
  with_problem(
    point_parameters(
      {{"Inclusions file", inclusions_file_with_single_inclusion_vessel()}}),
    [](auto &problem) {
      problem.run_timestep0();
      ASSERT_EQ(problem.n_vessels(), 2u);
      const auto pressure = [](unsigned i) { return 1. * i * i; };
      set_pressure_by_inclusion(problem, pressure);

      std::vector<double> vessel_pressure(49);
      for (unsigned int k = 0; k < 49; ++k)
        vessel_pressure[k] = pressure(k);

      const auto split =
        problem.split_pressure_over_inclusions({3, 4}, Vector<double>(50));
      ASSERT_EQ(split.size(), 2u);
      ASSERT_EQ(split[0].size(), 3u);
      for (unsigned int c = 0; c < 3; ++c)
        EXPECT_NEAR(split[0][c],
                    interpolated_sample(vessel_pressure, c, 3),
                    1e-12)
          << "cell " << c;
      ASSERT_EQ(split[1].size(), 4u);
      for (unsigned int c = 0; c < 4; ++c)
        EXPECT_DOUBLE_EQ(split[1][c], pressure(49)) << "cell " << c;

      // One cell in each vessel: the mean pressure of vessel 0, and the
      // pressure of the only inclusion of vessel 1.
      const auto one_cell =
        problem.split_pressure_over_inclusions({1, 1}, Vector<double>(50));
      ASSERT_EQ(one_cell.size(), 2u);
      ASSERT_EQ(one_cell[0].size(), 1u);
      ASSERT_EQ(one_cell[1].size(), 1u);
      double mean = 0;
      for (const auto p : vessel_pressure)
        mean += p;
      mean /= vessel_pressure.size();
      EXPECT_NEAR(one_cell[0][0], mean, 1e-10);
      EXPECT_DOUBLE_EQ(one_cell[1][0], pressure(49));
    });
}

// The number of cells must be given for each vessel.
TEST(CoupledElasticity,
     BOTH_SplitPressureOverInclusionsWrongNumberOfVesselsPoint)
{
  with_problem(point_parameters(), [](auto &problem) {
    problem.run_timestep0();
    set_pressure_by_inclusion(problem, [](unsigned i) { return 1. * i; });
    EXPECT_ANY_THROW(
      problem.split_pressure_over_inclusions({49}, Vector<double>(50)));
  });
}

// A vessel without cells is rejected.
TEST(CoupledElasticity, BOTH_SplitPressureOverInclusionsZeroCellsPoint)
{
  with_problem(point_parameters(), [](auto &problem) {
    problem.run_timestep0();
    set_pressure_by_inclusion(problem, [](unsigned i) { return 1. * i; });
    EXPECT_ANY_THROW(
      problem.split_pressure_over_inclusions({49, 0}, Vector<double>(50)));
  });
}

#endif
