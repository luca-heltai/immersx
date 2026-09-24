// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II library.
//
// ---------------------------------------------------------------------

#ifndef immersx_coral_reduced_poisson_h
#define immersx_coral_reduced_poisson_h

#include <memory>
#include <string>

namespace ImmersX::Coral
{
  /** Graph-facing workflow for a point-backed 3D ReducedPoisson problem.
   *
   * The point is supplied by the workflow so the installed example does not
   * depend on a source-tree VTK asset. All assembly and solution work remains
   * in the existing ReducedPoisson implementation.
   */
  class ReducedPoisson3D
  {
  public:
    explicit ReducedPoisson3D(const std::string &parameter_file);

    ~ReducedPoisson3D();

    ReducedPoisson3D(const ReducedPoisson3D &) = delete;
    ReducedPoisson3D &
    operator=(const ReducedPoisson3D &)   = delete;
    ReducedPoisson3D(ReducedPoisson3D &&) = delete;
    ReducedPoisson3D &
    operator=(ReducedPoisson3D &&) = delete;

    void
    run();

    unsigned int
    n_reduced_dofs() const;

    double
    coupling_matrix_frobenius_norm() const;

    double
    bulk_solution_l2_norm() const;

    double
    multiplier_solution_l2_norm() const;

    bool
    state_is_finite() const;

  private:
    class Impl;
    std::unique_ptr<Impl> implementation;
  };
} // namespace ImmersX::Coral

#endif // immersx_coral_reduced_poisson_h
