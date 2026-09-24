// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II library.
//
// ---------------------------------------------------------------------

#ifndef immersx_coral_coupled_poisson_elasticity_h
#define immersx_coral_coupled_poisson_elasticity_h

#include <memory>
#include <string>

namespace ImmersX::Coral
{
  /** Graph-facing façade for the existing 1D-pressure/3D-elasticity solve. */
  class CoupledPoissonElasticity3D
  {
  public:
    explicit CoupledPoissonElasticity3D(const std::string &parameter_file);

    ~CoupledPoissonElasticity3D();

    CoupledPoissonElasticity3D(const CoupledPoissonElasticity3D &) = delete;
    CoupledPoissonElasticity3D &
    operator=(const CoupledPoissonElasticity3D &)             = delete;
    CoupledPoissonElasticity3D(CoupledPoissonElasticity3D &&) = delete;
    CoupledPoissonElasticity3D &
    operator=(CoupledPoissonElasticity3D &&) = delete;

    void
    run();

    double
    residual_norm() const;

    double
    pressure_scale_error() const;

    double
    traction_balance_error() const;

  private:
    class Impl;
    std::unique_ptr<Impl> implementation;
  };
} // namespace ImmersX::Coral

#endif // immersx_coral_coupled_poisson_elasticity_h
