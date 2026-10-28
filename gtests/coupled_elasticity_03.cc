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

  // 50 point inclusions along x, centered at -0.98, -0.94, ..., 0.98 and split
  // into `n_vessels` consecutive vessels of equal size. Each line is cx, cy,
  // cz, dx, dy, dz, r, vessel id.
  std::string
  inclusions_file(const unsigned int n_vessels)
  {
    std::ostringstream text;
    for (unsigned int i = 0; i < 50; ++i)
      text << -0.98 + 0.04 * i << " 0 0 0.04 0 0 0.2 " << i * n_vessels / 50
           << "\n";
    return input_file("inclusions_" + std::to_string(n_vessels) + ".txt",
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
