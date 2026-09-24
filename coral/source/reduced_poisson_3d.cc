// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <immersx/coral/reduced_poisson.h>
#include <immersx/physics/reduced_poisson.h>

#include <cmath>

namespace ImmersX::Coral
{
  class ReducedPoisson3D::Impl
  {
  public:
    using Parameters = ImmersX::ReducedPoissonParameters<3, 0, 3>;
    using Problem    = ImmersX::ReducedPoisson<3, 3, 0, 3>;

    explicit Impl(const std::string &parameter_file)
      : parameters()
    {
      dealii::ParameterAcceptor::initialize(parameter_file);

      dealii::Point<3> point;
      point[0] = 0.5;
      point[1] = 0.5;
      point[2] = 0.5;
      parameters.reduced_coupling_parameters.tensor_product_space_parameters
        .point_cloud.points = {point};

      problem = std::make_unique<Problem>(parameters);
    }

    void
    run()
    {
      problem->run();
    }

    unsigned int
    n_reduced_dofs() const
    {
      return problem->n_reduced_dofs();
    }

    double
    coupling_matrix_frobenius_norm() const
    {
      return problem->coupling_matrix_frobenius_norm();
    }

    double
    bulk_solution_l2_norm() const
    {
      return problem->bulk_solution_l2_norm();
    }

    double
    multiplier_solution_l2_norm() const
    {
      return problem->multiplier_solution_l2_norm();
    }

    bool
    state_is_finite() const
    {
      return problem->bulk_solution_is_finite() &&
             problem->multiplier_solution_is_finite();
    }

  private:
    Parameters               parameters;
    std::unique_ptr<Problem> problem;
  };

  ReducedPoisson3D::ReducedPoisson3D(const std::string &parameter_file)
    : implementation(std::make_unique<Impl>(parameter_file))
  {}

  ReducedPoisson3D::~ReducedPoisson3D() = default;

  void
  ReducedPoisson3D::run()
  {
    implementation->run();
  }

  unsigned int
  ReducedPoisson3D::n_reduced_dofs() const
  {
    return implementation->n_reduced_dofs();
  }

  double
  ReducedPoisson3D::coupling_matrix_frobenius_norm() const
  {
    return implementation->coupling_matrix_frobenius_norm();
  }

  double
  ReducedPoisson3D::bulk_solution_l2_norm() const
  {
    return implementation->bulk_solution_l2_norm();
  }

  double
  ReducedPoisson3D::multiplier_solution_l2_norm() const
  {
    return implementation->multiplier_solution_l2_norm();
  }

  bool
  ReducedPoisson3D::state_is_finite() const
  {
    return implementation->state_is_finite();
  }
} // namespace ImmersX::Coral
