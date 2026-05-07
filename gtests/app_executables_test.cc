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
  return std::string(SOURCE_DIR) + "/" + relative_path;
}

// Run an executable with an arbitrary argument string and expect exit status 0.
static void
run_application(const char *exe, const std::string &args)
{
  std::string cmd = std::string("./") + exe + " " + args;
  const int   ret = std::system(cmd.c_str());
  // POSIX: low 8 bits contain the exit status.
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
      std::string args = f;
      // Special case: coupled_elasticity needs an extra 1D input file
      if (app_name == "coupled_elasticity")
        args += " " + source_path("prms/input_1d.dat");
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
