// ---------------------------------------------------------------------
// Test that the example applications in `apps/` run with minimal parameter
// files. This extends test coverage without heavy computation.
// ---------------------------------------------------------------------
#include <deal.II/base/mpi.h>

#include <dirent.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

static std::string
source_path(const char *relative_path)
{
  return std::string(APP_SOURCE_DIR) + "/" + relative_path;
}

// Quote a path for the POSIX shell used by std::system().  The paths passed to
// this test are generated from the source tree and build tree, but quoting
// them also keeps the test working when either directory contains spaces.
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

// Run an executable with an arbitrary argument string and expect exit status 0.
static void
run_application(const char *exe, const std::string &args)
{
  // Keep all application-generated files below the build tree.  The parameter
  // files use paths such as ../data/..., which resolve to the source data
  // directory through this build-local link when the test runs below.
  const std::string test_directory =
    std::string(APP_BINARY_DIR) + "/test_directory";
  const std::string data_link = std::string(APP_BINARY_DIR) + "/data";
  const std::string setup_cmd =
    "mkdir -p " + shell_quote(test_directory) + " && " + "ln -sfn " +
    shell_quote(source_path("data")) + " " + shell_quote(data_link);
  const int setup_ret = std::system(setup_cmd.c_str());
  ASSERT_EQ(setup_ret, 0) << "Could not prepare application test directory: "
                          << setup_cmd;

  const std::string executable =
    std::string(APP_BINARY_DIR) + "/" + executable_name(exe);
  const std::string cmd = "cd " + shell_quote(test_directory) + " && " +
                          shell_quote(executable) + " " + args;
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

// Find parameter files in `gtests/parameters` whose filename starts with
// the given application name. Returns full path (prefixed with SOURCE_DIR).
static std::vector<std::string>
find_parameter_files(const std::string &app_name)
{
  std::vector<std::string> result;
  const std::string        dir = source_path("gtests/parameters");
  DIR                     *d   = opendir(dir.c_str());
  if (!d)
    return result;

  struct dirent *ent;
  while ((ent = readdir(d)) != nullptr)
    {
      const std::string name = ent->d_name;
      if (name.size() == 0)
        continue;
      if (name[0] == '.')
        continue;
      // match prefix
      if (name.rfind(app_name, 0) == 0)
        result.push_back(dir + "/" + name);
    }
  closedir(d);

  std::sort(result.begin(), result.end());
  return result;
}

// Run the given executable once for each matching parameter file in
// `gtests/parameters`. If none found, skip the test.
static void
run_app_with_discovered_params(const char *exe)
{
  const std::string app_name = exe;
  const auto        files    = find_parameter_files(app_name);
  if (files.empty())
    GTEST_SKIP() << "No parameter files found for " << app_name;

  for (const auto &f : files)
    {
      std::string args = shell_quote(f);
      // Special case: coupled_elasticity needs an extra 1D input file
      if (app_name == "coupled_elasticity")
        args += " " + shell_quote(source_path("prms/input_1d.dat"));
      run_application(exe, args);
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
