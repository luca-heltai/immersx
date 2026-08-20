// ---------------------------------------------------------------------
// Test that the example applications in `apps/` run with generated parameter
// files. This extends test coverage without relying on the launch directory.
// ---------------------------------------------------------------------
#include <deal.II/base/mpi.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

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
// so this working directory is only scratch space for legacy relative files.
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

// Find generated parameter files whose filename starts with the given
// application name.
static std::vector<std::filesystem::path>
find_parameter_files(const std::string &app_name)
{
  std::vector<std::filesystem::path> result;
  const auto      dir = ImmersX::TestPaths::binary_path("gtests/parameters");
  std::error_code error;
  for (const auto &entry : std::filesystem::directory_iterator(dir, error))
    {
      if (error)
        break;
      const auto &path = entry.path();
      const auto  name = path.filename().string();
      if (!entry.is_regular_file() || path.extension() != ".prm")
        continue;
      if (name.rfind(app_name, 0) == 0)
        result.push_back(path);
    }

  std::sort(result.begin(), result.end());
  return result;
}

// Run the given executable once for each matching generated parameter file.
// If none are found, skip the test.
static void
run_app_with_discovered_params(const char *exe)
{
  const std::string app_name = exe;
  const auto        files    = find_parameter_files(app_name);
  if (files.empty())
    GTEST_SKIP() << "No parameter files found for " << app_name;

  for (const auto &parameter_file : files)
    {
      const auto run_directory =
        ImmersX::TestPaths::output_path("app_executables") /
        executable_name(exe) / parameter_file.stem();
      run_application(exe, parameter_file, run_directory);
    }
}

TEST(AppExecutables, Elasticity)
{
  if (!is_single_rank())
    GTEST_SKIP() << "MPI test – skipped on multi‑rank";
  run_app_with_discovered_params("elasticity");
}

TEST(AppExecutables, Laplacian)
{
  if (!is_single_rank())
    GTEST_SKIP() << "MPI test – skipped on multi‑rank";
  run_app_with_discovered_params("laplacian");
}

TEST(AppExecutables, CoupledElasticity)
{
  if (!is_single_rank())
    GTEST_SKIP() << "MPI test – skipped on multi‑rank";
  run_app_with_discovered_params("coupled_elasticity");
}

TEST(AppExecutables, PseudoCoupling1D)
{
  if (!is_single_rank())
    GTEST_SKIP() << "MPI test – skipped on multi‑rank";
  run_app_with_discovered_params("pseudocoupling1D");
}

TEST(AppExecutables, ReducedPoisson)
{
  if (!is_single_rank())
    GTEST_SKIP() << "MPI test – skipped on multi‑rank";
  run_app_with_discovered_params("reduced_poisson");
}

TEST(AppExecutables, NavierStokes)
{
  if (!is_single_rank())
    GTEST_SKIP() << "MPI test – skipped on multi‑rank";
  run_app_with_discovered_params("navier_stokes");
}
