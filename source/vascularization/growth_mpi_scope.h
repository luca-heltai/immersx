// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_growth_mpi_scope_h
#define immersx_internal_growth_mpi_scope_h
#include <cstdint>

#if defined(IMMERSX_VASCULAR_MPI_SCOPE_IMPLEMENTATION)
#  define IMMERSX_VASCULAR_MPI_WEAK
#  define IMMERSX_VASCULAR_MPI_SCOPE_SUPPORTED 1
#elif defined(__APPLE__) && (defined(__GNUC__) || defined(__clang__))
#  define IMMERSX_VASCULAR_MPI_WEAK __attribute__((weak_import))
#  define IMMERSX_VASCULAR_MPI_SCOPE_SUPPORTED 1
#elif defined(__GNUC__) || defined(__clang__)
#  define IMMERSX_VASCULAR_MPI_WEAK __attribute__((weak))
#  define IMMERSX_VASCULAR_MPI_SCOPE_SUPPORTED 1
#else
#  define IMMERSX_VASCULAR_MPI_WEAK
#  define IMMERSX_VASCULAR_MPI_SCOPE_SUPPORTED 0
#endif
extern "C" std::uint64_t
immersx_vascular_mpi_new_run() noexcept IMMERSX_VASCULAR_MPI_WEAK;
extern "C" std::uintptr_t
immersx_vascular_mpi_begin(std::uint64_t run,
                           unsigned int  phase,
                           unsigned int  system,
                           std::uint64_t requested,
                           std::uint64_t leaves) noexcept
  IMMERSX_VASCULAR_MPI_WEAK;
extern "C" void
immersx_vascular_mpi_end(std::uintptr_t previous) noexcept
  IMMERSX_VASCULAR_MPI_WEAK;
namespace ImmersX::Vascularization::internal
{
  enum class GrowthMpiPhase : unsigned int
  {
    initialization,
    batch,
    finalization,
    count
  };
#ifndef IMMERSX_VASCULAR_MPI_SCOPE_IMPLEMENTATION
  class GrowthMpiRun;
  // Optional benchmark hooks have no MPI calls or dependency on the profiler.
  // End initialization before the batch loop; nested finalization restores its
  // enclosing batch. finish() is also safe on early returns and repeated calls.
  class GrowthMpiScope
  {
  public:
    GrowthMpiScope(const GrowthMpiScope &) = delete;
    GrowthMpiScope &
    operator=(const GrowthMpiScope &) = delete;
    ~GrowthMpiScope()
    {
      finish();
    }
    void
    finish() noexcept
    {
#  if IMMERSX_VASCULAR_MPI_SCOPE_SUPPORTED
      if (active)
        immersx_vascular_mpi_end(previous);
#  endif
      active = false;
    }

  private:
    friend class GrowthMpiRun;
    GrowthMpiScope(std::uint64_t  run,
                   GrowthMpiPhase phase,
                   unsigned int   system,
                   std::uint64_t  requested,
                   std::uint64_t  leaves) noexcept
    {
#  if IMMERSX_VASCULAR_MPI_SCOPE_SUPPORTED
      if (run != 0 && immersx_vascular_mpi_begin && immersx_vascular_mpi_end)
        {
          previous = immersx_vascular_mpi_begin(
            run, static_cast<unsigned int>(phase), system, requested, leaves);
          active = true;
        }
#  else
      (void)run;
      (void)phase;
      (void)system;
      (void)requested;
      (void)leaves;
#  endif
    }
    std::uintptr_t previous = 0;
    bool           active   = false;
  };
  class GrowthMpiRun
  {
  public:
    GrowthMpiRun(unsigned int system, std::uint64_t requested) noexcept
      : system(system)
      , requested(requested)
    {
#  if IMMERSX_VASCULAR_MPI_SCOPE_SUPPORTED
      if (immersx_vascular_mpi_new_run && immersx_vascular_mpi_begin &&
          immersx_vascular_mpi_end)
        run = immersx_vascular_mpi_new_run();
#  endif
    }
    GrowthMpiScope
    phase(GrowthMpiPhase phase, std::uint64_t leaves = 0) const noexcept
    {
      return GrowthMpiScope(run, phase, system, requested, leaves);
    }

  private:
    std::uint64_t run = 0;
    unsigned int  system;
    std::uint64_t requested;
  };
#endif
} // namespace ImmersX::Vascularization::internal
#undef IMMERSX_VASCULAR_MPI_WEAK
#undef IMMERSX_VASCULAR_MPI_SCOPE_SUPPORTED
#endif
