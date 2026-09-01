// ---------------------------------------------------------------------
// Test that the example applications in `apps/` run with generated parameter
// files. This extends test coverage without relying on the launch directory.
// ---------------------------------------------------------------------
#include <deal.II/base/mpi.h>

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "test_paths.h"

// Quote a path for the POSIX shell used by std::system().
static std::string
shell_quote(const std::string &value)
{
  std::string quoted = "'";
  for (const char character : value)
    if (character == '\'')
      quoted += "'\\''";
    else
      quoted += character;
  quoted += "'";
  return quoted;
}

// Debug builds postfix executables with `_debug` (e.g., elasticity_debug).
static std::string
executable_name(const char *exe)
{
#ifdef DEBUG
  return std::string(exe) + "_debug";
#else
  return std::string(exe);
#endif
}

static void
prepare_parameter_outputs(const std::filesystem::path &parameter_file)
{
  std::ifstream input(parameter_file);
  ASSERT_TRUE(input.good()) << "Could not open generated parameter file '"
                            << parameter_file.string() << "'.";

  std::string line;
  while (std::getline(input, line))
    {
      const auto output_key = line.find("set Output directory");
      const auto error_key  = line.find("set Error file name");
      const auto key = output_key != std::string::npos ? output_key : error_key;
      if (key == std::string::npos)
        continue;

      const auto equals = line.find('=', key);
      if (equals == std::string::npos)
        continue;

      auto       value = line.substr(equals + 1);
      const auto first = value.find_first_not_of(" \t");
      const auto last  = value.find_last_not_of(" \t\r");
      if (first == std::string::npos)
        continue;
      value = value.substr(first, last - first + 1);

      const std::filesystem::path path(value);
      const auto                  directory =
        output_key != std::string::npos ? path : path.parent_path();
      if (directory.empty())
        continue;

      std::error_code error;
      std::filesystem::create_directories(directory, error);
      ASSERT_FALSE(error) << "Could not prepare parameter output directory '"
                          << directory.string() << "': " << error.message();
    }
}

// Run an executable from a private build-tree directory and expect exit
// status 0. Generated parameter files contain absolute input and output paths,
// so this working directory is only scratch space for relative files.
static void
run_application(const char                  *exe,
                const std::filesystem::path &parameter_file,
                const std::filesystem::path &run_directory)
{
  std::error_code error;
  std::filesystem::create_directories(run_directory, error);
  ASSERT_FALSE(error) << "Could not prepare application test directory '"
                      << run_directory.string() << "': " << error.message();
  prepare_parameter_outputs(parameter_file);

  const auto executable = ImmersX::TestPaths::binary_path(executable_name(exe));
  const std::string cmd = "cd " + shell_quote(run_directory.string()) + " && " +
                          shell_quote(executable.string()) + " " +
                          shell_quote(parameter_file.string());
  const int ret = std::system(cmd.c_str());
  EXPECT_EQ(ret, 0) << "Command failed: " << cmd;
}

// Helper to check we are on a single MPI rank. The main test driver already
// filters MPI tests, but we keep an explicit guard for clarity.
static bool
is_single_rank()
{
  return dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1;
}

// Run one deliberately small canonical parameter file. The full parameter
// matrix is covered by dedicated application and solver tests; this test also
// verifies that the input used by the published tutorials remains executable.
static void
run_app_with_parameter(const char *exe, const char *relative_parameter_path)
{
  auto parameter_file =
    ImmersX::TestPaths::binary_path(relative_parameter_path);
  if (!std::filesystem::is_regular_file(parameter_file))
    parameter_file = ImmersX::TestPaths::source_path(relative_parameter_path);
  if (!std::filesystem::is_regular_file(parameter_file))
    GTEST_SKIP() << "Representative parameter file is unavailable: "
                 << parameter_file.string();

  const auto run_directory =
    ImmersX::TestPaths::output_path("app_executables") / executable_name(exe) /
    parameter_file.stem();
  run_application(exe, parameter_file, run_directory);
}

TEST(AppExecutables, Elasticity)
{
  if (!is_single_rank())
    GTEST_SKIP() << "MPI test – skipped on multi‑rank";
  run_app_with_parameter(
    "elasticity",
    "gtests/parameters/elasticity_dynamic_purely_elastic_strong.prm");
}

TEST(AppExecutables, Laplacian)
{
  if (!is_single_rank())
    GTEST_SKIP() << "MPI test – skipped on multi‑rank";
  run_app_with_parameter("laplacian", "gtests/parameters/laplacian_simple.prm");
}

TEST(AppExecutables, CoupledElasticity)
{
  if (!is_single_rank())
    GTEST_SKIP() << "MPI test – skipped on multi‑rank";
  GTEST_SKIP() << "No coupled_elasticity application is configured in CI.";
}

TEST(AppExecutables, PseudoCoupling1D)
{
  if (!is_single_rank())
    GTEST_SKIP() << "MPI test – skipped on multi‑rank";
  GTEST_SKIP() << "No pseudocoupling1D application is configured in CI.";
}

TEST(AppExecutables, ReducedPoisson)
{
  if (!is_single_rank())
    GTEST_SKIP() << "MPI test – skipped on multi‑rank";
  run_app_with_parameter("reduced_poisson",
                         "tutorials/reduced_poisson/single_cylinder_3d.prm");
}

TEST(AppExecutables, NavierStokes)
{
  if (!is_single_rank())
    GTEST_SKIP() << "MPI test – skipped on multi‑rank";
  run_app_with_parameter("navier_stokes",
                         "tutorials/navier_stokes/transient_2d.prm");
}

TEST(AppExecutables, CoupledPoissonElasticity)
{
  if (!is_single_rank())
    GTEST_SKIP() << "MPI test – skipped on multi‑rank";

  run_app_with_parameter(
    "coupled_poisson_elasticity",
    "tutorials/coupled_poisson_elasticity/coupled_poisson_elasticity.prm");

  const auto diagnostics = ImmersX::TestPaths::output_path(
                             "tutorial-output/coupled-poisson-elasticity") /
                           "coupled_poisson_elasticity_diagnostics.txt";
  std::ifstream input(diagnostics);
  ASSERT_TRUE(input.good()) << diagnostics;

  std::string line;
  double      residual       = 1.;
  double      pressure_error = 1.;
  double      traction_error = 1.;
  while (std::getline(input, line))
    {
      const auto equals = line.find('=');
      if (equals == std::string::npos)
        continue;
      const auto key   = line.substr(0, equals);
      const auto value = std::stod(line.substr(equals + 1));
      if (key == "coupled_residual ")
        residual = value;
      else if (key == "pressure_scale_error ")
        pressure_error = value;
      else if (key == "traction_balance_error ")
        traction_error = value;
    }

  EXPECT_LT(residual, 1.e-9);
  EXPECT_LT(pressure_error, 1.e-14);
  EXPECT_LT(traction_error, 1.e-9);

  const auto output_directory = ImmersX::TestPaths::output_path(
    "tutorial-output/coupled-poisson-elasticity");
  ASSERT_TRUE(std::filesystem::exists(output_directory));
  EXPECT_TRUE(std::filesystem::exists(output_directory / "solution.pvd"));
  EXPECT_TRUE(
    std::filesystem::exists(output_directory / "coupled_elasticity.pvd"));

  bool has_poisson_output    = false;
  bool has_elasticity_output = false;
  bool has_poisson_field     = false;
  bool has_elasticity_field  = false;
  for (const auto &entry :
       std::filesystem::directory_iterator(output_directory))
    {
      if (entry.path().extension() != ".vtu" &&
          entry.path().extension() != ".pvtu")
        continue;

      const auto filename          = entry.path().filename().string();
      const bool is_poisson_output = filename.rfind("solution_", 0) == 0;
      const bool is_elasticity_output =
        filename.rfind("coupled_elasticity_", 0) == 0;
      has_poisson_output |= is_poisson_output;
      has_elasticity_output |= is_elasticity_output;

      std::ifstream     output(entry.path());
      const std::string contents((std::istreambuf_iterator<char>(output)),
                                 std::istreambuf_iterator<char>());
      has_poisson_field |=
        is_poisson_output &&
        contents.find("Name=\"solution\"") != std::string::npos;
      has_elasticity_field |=
        is_elasticity_output &&
        contents.find("Name=\"displacement\"") != std::string::npos;
    }
  EXPECT_TRUE(has_poisson_output);
  EXPECT_TRUE(has_elasticity_output);
  EXPECT_TRUE(has_poisson_field);
  EXPECT_TRUE(has_elasticity_field);
}

TEST(AppExecutables, ElasticStatic)
{
  if (!is_single_rank())
    GTEST_SKIP() << "MPI test – skipped on multi‑rank";
  run_app_with_parameter("elastic_static",
                         "tutorials/elastic_static/elastic_static.prm");

  const auto output_directory =
    ImmersX::TestPaths::output_path("tutorial-output/elastic-static");
  EXPECT_TRUE(std::filesystem::exists(output_directory));
  EXPECT_TRUE(std::filesystem::exists(output_directory / "elastic_static.pvd"));

  bool has_material_id = false;
  for (const auto &entry :
       std::filesystem::directory_iterator(output_directory))
    if (entry.path().extension() == ".vtu" ||
        entry.path().extension() == ".pvtu")
      {
        std::ifstream     output(entry.path());
        const std::string contents((std::istreambuf_iterator<char>(output)),
                                   std::istreambuf_iterator<char>());
        has_material_id |= contents.find("material_id") != std::string::npos;
      }
  EXPECT_TRUE(has_material_id);
}

TEST(AppExecutables, TutorialPoisson)
{
  if (!is_single_rank())
    GTEST_SKIP() << "MPI test – skipped on multi‑rank";
  run_app_with_parameter("poisson", "tutorials/poisson/poisson_2d.prm");

  const auto output_directory =
    ImmersX::TestPaths::output_path("tutorial-output/poisson-2d");
  EXPECT_TRUE(std::filesystem::exists(output_directory / "poisson_2d.pvd"));
  bool has_solution = false;
  for (const auto &entry :
       std::filesystem::directory_iterator(output_directory))
    if (entry.path().extension() == ".vtu" ||
        entry.path().extension() == ".pvtu")
      {
        std::ifstream     output(entry.path());
        const std::string contents((std::istreambuf_iterator<char>(output)),
                                   std::istreambuf_iterator<char>());
        has_solution |= contents.find("Name=\"solution\"") != std::string::npos;
      }
  EXPECT_TRUE(has_solution);
}

TEST(AppExecutables, CoupledPoisson)
{
  if (!is_single_rank())
    GTEST_SKIP() << "MPI test – skipped on multi‑rank";

  run_app_with_parameter("coupled_poisson",
                         "tutorials/coupled_poisson/coupled_poisson.prm");

  const auto output_directory =
    ImmersX::TestPaths::output_path("tutorial-output/coupled-poisson");
  ASSERT_TRUE(std::filesystem::exists(output_directory));
  EXPECT_TRUE(std::filesystem::exists(output_directory / "bulk_solution.pvd"));
  EXPECT_TRUE(
    std::filesystem::exists(output_directory / "embedded_solution.pvd"));
  EXPECT_TRUE(
    std::filesystem::exists(output_directory / "coupled_multiplier.pvd"));

  bool has_bulk       = false;
  bool has_embedded   = false;
  bool has_multiplier = false;
  for (const auto &entry :
       std::filesystem::directory_iterator(output_directory))
    if (entry.path().extension() == ".vtu" ||
        entry.path().extension() == ".pvtu")
      {
        std::ifstream     output(entry.path());
        const std::string contents((std::istreambuf_iterator<char>(output)),
                                   std::istreambuf_iterator<char>());
        has_bulk |= contents.find("Name=\"solution\"") != std::string::npos &&
                    entry.path().filename().string().find("bulk_solution") !=
                      std::string::npos;
        has_embedded |=
          contents.find("Name=\"solution\"") != std::string::npos &&
          entry.path().filename().string().find("embedded_solution") !=
            std::string::npos;
        has_multiplier |=
          contents.find("Name=\"lagrange_multiplier\"") != std::string::npos;
      }

  EXPECT_TRUE(has_bulk);
  EXPECT_TRUE(has_embedded);
  EXPECT_TRUE(has_multiplier);
}

TEST(AppExecutables, TutorialElastodynamics)
{
  if (!is_single_rank())
    GTEST_SKIP() << "MPI test – skipped on multi‑rank";
  run_app_with_parameter("elastodynamics",
                         "tutorials/elastodynamics/strong_dirichlet.prm");
}

TEST(AppExecutables, TutorialFiberReinforcedElastodynamics)
{
  if (!is_single_rank())
    GTEST_SKIP() << "MPI test – skipped on multi‑rank";
  run_app_with_parameter(
    "fiber_reinforced_elastodynamics",
    "tutorials/fiber_reinforced_elastodynamics/parameters.prm");
}
