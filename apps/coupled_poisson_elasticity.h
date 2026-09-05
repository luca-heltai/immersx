// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_coupled_poisson_elasticity_h
#define immersx_coupled_poisson_elasticity_h

#include <deal.II/base/mpi.h>

#include <immersx/core/observable.h>
#include <immersx/core/observable_lift.h>
#include <immersx/core/problem_handle.h>
#include <immersx/core/weak_term.h>
#include <immersx/physics/elastic_static.h>
#include <immersx/physics/poisson_residual.h>

#include <string>
#include <utility>

namespace CoupledPoissonElasticity
{
  struct Pressure
  {
    double factor = 2.;
  };

  using PressureLift = ImmersX::TensorProductLift<1, 2, 3, 1>;

  /** Fixed cylindrical support for the lifted pressure. */
  class CylinderSurface
  {
  public:
    using Point = dealii::Point<3>;

    CylinderSurface(
      const double       radius     = 0.2,
      const Point        center     = Point(0.5, 0.5, 0.),
      const unsigned int n_theta    = 8,
      const std::string &subsection = "/Pressure traction/Particle coupling/")
      : center_(center)
      , particle_parameters_(subsection)
    {
      AssertThrow(radius > 0.,
                  dealii::ExcMessage("Cylinder radius must be positive."));
      AssertThrow(n_theta > 0,
                  dealii::ExcMessage(
                    "Cylinder must have a positive angular resolution."));
    }

    dealii::Tensor<1, 3>
    normal(const Point &point) const
    {
      dealii::Tensor<1, 3> result;
      result[0] = point[0] - center_[0];
      result[1] = point[1] - center_[1];
      result[2] = 0.;
      return result / result.norm();
    }

    const ImmersX::ParticleCouplingParameters<3> &
    particle_coupling_parameters() const
    {
      return particle_parameters_;
    }

  private:
    Point                                  center_;
    ImmersX::ParticleCouplingParameters<3> particle_parameters_;
  };

  template <typename Quantity>
  inline typename Quantity::value_type
  sample_pressure(const Quantity                        &quantity,
                  const ImmersX::ImmersXLA::MPI::Vector &solution)
  {
    class SingleFieldState
      : public ImmersX::StateAccessor<ImmersX::ImmersXLA::MPI::Vector>
    {
    public:
      SingleFieldState(const ImmersX::FieldId                 field,
                       const ImmersX::ImmersXLA::MPI::Vector &values)
        : field_(field)
        , values_(&values)
      {}

      const ImmersX::ImmersXLA::MPI::Vector &
      field(const ImmersX::FieldId field, const double) const override
      {
        AssertThrow(field == field_,
                    dealii::ExcMessage(
                      "The pressure state accessor received an unknown "
                      "field."));
        return *values_;
      }

    private:
      ImmersX::FieldId                       field_;
      const ImmersX::ImmersXLA::MPI::Vector *values_;
    } state(quantity.dependencies().front(), solution);
    const ImmersX::EvaluationContext<ImmersX::ImmersXLA::MPI::Vector> context(
      0., state);
    return quantity.evaluate(context);
  }

  template <typename Quantity, typename TargetField>
  inline double
  traction_balance(const Quantity                            &quantity,
                   const CylinderSurface                     &surface,
                   const TargetField                         &target,
                   const ImmersX::ElasticStaticProblem<3, 3> &problem,
                   const ImmersX::ImmersXLA::MPI::Vector     &pressure_state,
                   const ImmersX::ImmersXLA::MPI::Vector     &elastic_state)
  {
    using ElasticVector = ImmersX::ElasticStaticProblem<3, 3>::VectorType;

    const auto load = quantity * ImmersX::normal(surface);
    const auto expected =
      ImmersX::detail::make_normal_load_operator<Quantity,
                                                 CylinderSurface,
                                                 TargetField,
                                                 ElasticVector>(load, target);
    const auto    pressure = sample_pressure(quantity, pressure_state);
    ElasticVector expected_load;
    expected.reinit_range_vector(expected_load, false);
    expected.vmult(expected_load, pressure);

    ElasticVector actual_load;
    actual_load.reinit(problem.forcing());
    actual_load = problem.forcing();
    ElasticVector stiffness_load;
    stiffness_load.reinit(actual_load);
    problem.stiffness_operator().vmult(stiffness_load, elastic_state);
    actual_load -= stiffness_load;

    expected_load -= actual_load;
    return expected_load.l2_norm();
  }

} // namespace CoupledPoissonElasticity

#endif // immersx_coupled_poisson_elasticity_h
