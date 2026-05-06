// ---------------------------------------------------------------------
// Test that the example applications in `apps/` run with minimal parameter
// files. This extends test coverage without heavy computation.
// ---------------------------------------------------------------------
#include <deal.II/base/mpi.h>

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

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

TEST(AppExecutables, Elasticity)
{
  if (!is_single_rank())
    GTEST_SKIP() << "MPI test – skipped on multi‑rank";
  run_application("elasticity",
                  source_path("tutorials/elasticity/"
                              "damped_kv_dispersion.prm"));
}

TEST(AppExecutables, Laplacian)
{
  if (!is_single_rank())
    GTEST_SKIP() << "MPI test – skipped on multi‑rank";
  run_application("laplacian", source_path("gtests/app_laplacian.prm"));
}

TEST(AppExecutables, CoupledElasticity)
{
  if (!is_single_rank())
    GTEST_SKIP() << "MPI test – skipped on multi‑rank";
  // Provide a 3D parameter file and a minimal 1D input file.
  run_application("coupled_elasticity",
                  source_path("tutorials/elasticity/"
                              "damped_kv_dispersion.prm") +
                    " " + source_path("prms/input_1d.dat"));
}

TEST(AppExecutables, PseudoCoupling1D)
{
  if (!is_single_rank())
    GTEST_SKIP() << "MPI test – skipped on multi‑rank";
  run_application("pseudocoupling1D",
                  source_path("tutorials/reduced_poisson/"
                              "single_cylinder_3d.prm"));
}

TEST(AppExecutables, ReducedPoisson)
{
  if (!is_single_rank())
    GTEST_SKIP() << "MPI test – skipped on multi‑rank";
  run_application("reduced_poisson",
                  source_path("tutorials/reduced_poisson/"
                              "single_cylinder_3d.prm"));
}
