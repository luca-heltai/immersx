// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II library.
//
// ---------------------------------------------------------------------

#ifndef immersx_coral_coupled_poisson_h
#define immersx_coral_coupled_poisson_h

#include <memory>
#include <string>

namespace ImmersX::Coral
{
  /**
   * Graph-facing façade for the existing 2D bulk/embedded Poisson workflow.
   *
   * The implementation owns the two Poisson Problems, their semantic fields,
   * the multiplier FE space, and the LinearAdapter. It deliberately delegates
   * assembly and solution to those existing ImmersX components.
   */
  class CoupledPoisson2D
  {
  public:
    explicit CoupledPoisson2D(const std::string &parameter_file);

    ~CoupledPoisson2D();

    CoupledPoisson2D(const CoupledPoisson2D &) = delete;
    CoupledPoisson2D &
    operator=(const CoupledPoisson2D &)   = delete;
    CoupledPoisson2D(CoupledPoisson2D &&) = delete;
    CoupledPoisson2D &
    operator=(CoupledPoisson2D &&) = delete;

    void
    run();

    double
    residual_norm() const;

  private:
    class Impl;
    std::unique_ptr<Impl> implementation;
  };
} // namespace ImmersX::Coral

#endif // immersx_coral_coupled_poisson_h
