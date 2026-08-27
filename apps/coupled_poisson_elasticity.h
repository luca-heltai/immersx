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
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/vector.h>

#include <immersx/core/expression_representation.h>
#include <immersx/core/lifting.h>
#include <immersx/core/load_interaction.h>
#include <immersx/core/problem_handle.h>
#include <immersx/coupling/particle_coupling.h>
#include <immersx/physics/elastic_static.h>
#include <immersx/physics/poisson_residual.h>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

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

    CylinderSurface(const double       radius  = 0.2,
                    const Point        center  = Point(0.5, 0.5, 0.),
                    const unsigned int n_theta = 8)
      : geometry_({0., 1.},
                  ImmersX::RepresentationDomain(2, 3, "cylinder"),
                  [](const Point &point) { return point[2]; })
    {
      const dealii::QGauss<1> axial_quadrature(2);
      const double            pi = std::acos(-1.);

      for (unsigned int axial = 0; axial < axial_quadrature.size(); ++axial)
        for (unsigned int angular = 0; angular < n_theta; ++angular)
          {
            const double s = axial_quadrature.point(axial)[0];
            const double theta =
              2. * pi * static_cast<double>(angular) / n_theta;
            Point                point(center[0] + radius * std::cos(theta),
                        center[1] + radius * std::sin(theta),
                        s);
            dealii::Tensor<1, 3> normal;
            normal[0] = std::cos(theta);
            normal[1] = std::sin(theta);
            normal[2] = 0.;

            points_.push_back(point);
            normals_.push_back(normal);
            weights_.push_back(radius * (2. * pi / n_theta) *
                               axial_quadrature.weight(axial));
          }

      target_prototype_.reinit(points_.size());
      request_ = ImmersX::EvaluationRequest(points_);
    }

    const ImmersX::ParametricGeometryMap &
    geometry() const
    {
      return geometry_;
    }

    const dealii::Vector<double> &
    target_prototype() const
    {
      return target_prototype_;
    }

    const ImmersX::EvaluationRequest &
    request() const
    {
      return request_;
    }

    const std::vector<Point> &
    points() const
    {
      return points_;
    }

    const std::vector<dealii::Tensor<1, 3>> &
    normals() const
    {
      return normals_;
    }

    const std::vector<double> &
    weights() const
    {
      return weights_;
    }

  private:
    ImmersX::ParametricGeometryMap    geometry_;
    std::vector<Point>                points_;
    std::vector<dealii::Tensor<1, 3>> normals_;
    std::vector<double>               weights_;
    dealii::Vector<double>            target_prototype_;
    ImmersX::EvaluationRequest        request_;
  };

  /** One-way pressure-to-elasticity traction on a fixed cylinder. */
  class Traction
  {
  public:
    explicit Traction(const std::string &subsection = "/Pressure traction/")
      : particle_parameters_(subsection + "Particle coupling/")
    {}

    void
    attach(const ImmersX::ElasticStaticProblem<3, 3> &problem)
    {
      problem_ = &problem;
    }

    const ImmersX::ElasticStaticProblem<3, 3> &
    problem() const
    {
      AssertThrow(problem_ != nullptr,
                  dealii::ExcMessage("Traction is not attached to a problem."));
      return *problem_;
    }

    const ImmersX::ParticleCouplingParameters<3> &
    particle_coupling_parameters() const
    {
      return particle_parameters_;
    }

  private:
    ImmersX::ParticleCouplingParameters<3>     particle_parameters_;
    const ImmersX::ElasticStaticProblem<3, 3> *problem_ = nullptr;
  };

  template <typename Adapter, typename Fields>
  auto
  make_representation(const ImmersX::ProblemHandle<Adapter, Fields> &problem,
                      const Pressure                                &pressure)
  {
    const auto &poisson = *problem.fields().problem;
    const ImmersX::FiniteElementRepresentation<1, 3> source(
      poisson.triangulation(),
      poisson.dof_handler(),
      poisson.locally_owned_dofs(),
      poisson.locally_relevant_dofs(),
      poisson.constraints());
    const auto active = ImmersX::state_field(source, problem.fields().solution);
    return ImmersX::make_fe_expression(source,
                                       {ImmersX::value(active, "A")},
                                       "factor*A",
                                       {{"factor", pressure.factor}});
  }

  template <typename Quantity>
  auto
  make_lift(const Quantity &quantity, const CylinderSurface &surface)
  {
    using Lifted =
      ImmersX::LiftingRepresentation<Quantity,
                                     ImmersX::ParametricGeometryMap,
                                     dealii::Vector<double>>;
    return Lifted(quantity,
                  surface.geometry(),
                  surface.target_prototype(),
                  surface.request());
  }

  template <typename Quantity, typename ElasticVector>
  dealii::LinearOperator<ElasticVector, typename Quantity::value_type>
  make_traction_operator(const Quantity &quantity, const Traction &traction)
  {
    using QuantityVector = typename Quantity::value_type;
    using Operator = dealii::LinearOperator<ElasticVector, QuantityVector>;

    const auto                &problem        = traction.problem();
    const auto                &dof_handler    = problem.dof_handler();
    const auto                &finite_element = dof_handler.get_fe();
    const auto                &owned_dofs     = problem.locally_owned_dofs();
    const dealii::MappingQ1<3> mapping;

    const auto source_points = quantity.locally_owned_quadrature_points(
      dealii::Quadrature<Quantity::support_dimension>());
    const auto point_indices = [&quantity, n = source_points.size()] {
      std::vector<dealii::types::global_dof_index> result;
      result.reserve(n);
      for (std::size_t q = 0; q < n; ++q)
        result.push_back(quantity.point_index(q));
      return result;
    }();
    auto distribution =
      std::make_shared<ImmersX::DistributedLiftedQuadrature<3>>(
        traction.particle_coupling_parameters());
    distribution->initialize(problem.triangulation(), mapping, source_points);

    using Entry  = std::pair<dealii::types::global_dof_index, double>;
    auto entries = std::make_shared<
      std::map<dealii::types::particle_index, std::vector<Entry>>>();
    const dealii::FEValuesExtractors::Vector displacement(0);
    for (const auto &particle :
         distribution->particle_coupling().get_particles())
      {
        const auto &cell = particle.get_surrounding_cell();
        const typename dealii::DoFHandler<3>::cell_iterator dh_cell(
          *cell, &dof_handler);
        std::vector<dealii::types::global_dof_index> dof_indices(
          finite_element.n_dofs_per_cell());
        dh_cell->get_dof_indices(dof_indices);
        const dealii::Quadrature<3> point_quadrature(
          std::vector<dealii::Point<3>>{particle.get_reference_location()});
        dealii::FEValues<3> fe_values(mapping,
                                      finite_element,
                                      point_quadrature,
                                      dealii::update_values);
        fe_values.reinit(dh_cell);
        const auto &stencil = distribution->stencil(particle.get_id());
        const auto  offset =
          particle.get_location() - stencil.representative_point;
        const auto normal = offset / offset.norm();
        for (unsigned int i = 0; i < dof_indices.size(); ++i)
          if (owned_dofs.is_element(dof_indices[i]))
            {
              const auto component =
                finite_element.system_to_component_index(i).first;
              (*entries)[particle.get_id()].emplace_back(
                dof_indices[i],
                stencil.physical_weight *
                  (component < 3 ?
                     (normal * fe_values[displacement].value(i, 0)) :
                     0.));
            }
      }

    const auto *prototype = &problem.solution();

    Operator result;
    result.reinit_range_vector = [prototype](ElasticVector &vector,
                                             const bool     omit) {
      vector.reinit(*prototype, omit);
    };
    result.reinit_domain_vector =
      [owned        = quantity.locally_owned_points(),
       relevant     = quantity.locally_relevant_points(),
       communicator = quantity.mpi_communicator()](QuantityVector &vector,
                                                   const bool      omit) {
        vector.reinit(owned, relevant, communicator);
        if (!omit)
          vector = 0.;
      };
    const auto values_on_target =
      [distribution, point_indices](const QuantityVector &source) {
        dealii::Vector<double> local_values(point_indices.size());
        for (std::size_t q = 0; q < point_indices.size(); ++q)
          local_values[q] = source[point_indices[q]];
        return distribution->values_on_target(local_values);
      };
    result.vmult = [entries, values_on_target](ElasticVector &destination,
                                               const QuantityVector &source) {
      const auto values = values_on_target(source);
      destination       = 0.;
      for (const auto &[id, point_entries] : *entries)
        for (const auto &[row, value] : point_entries)
          destination[row] += value * values.at(id);
    };
    result.vmult_add = [entries,
                        values_on_target](ElasticVector        &destination,
                                          const QuantityVector &source) {
      const auto values = values_on_target(source);
      for (const auto &[id, point_entries] : *entries)
        for (const auto &[row, value] : point_entries)
          destination[row] += value * values.at(id);
    };
    result.Tvmult =
      [entries, distribution, point_indices](QuantityVector      &destination,
                                             const ElasticVector &source) {
        std::map<dealii::types::particle_index, double> values;
        for (const auto &[id, point_entries] : *entries)
          for (const auto &[row, value] : point_entries)
            values[id] += value * source[row];
        dealii::Vector<double> local_values(point_indices.size());
        local_values = 0.;
        distribution->add_transpose_to_source(values, local_values);
        destination = 0.;
        for (std::size_t q = 0; q < point_indices.size(); ++q)
          destination[point_indices[q]] += local_values[q];
      };
    result.Tvmult_add =
      [entries, distribution, point_indices](QuantityVector      &destination,
                                             const ElasticVector &source) {
        std::map<dealii::types::particle_index, double> values;
        for (const auto &[id, point_entries] : *entries)
          for (const auto &[row, value] : point_entries)
            values[id] += value * source[row];
        dealii::Vector<double> local_values(point_indices.size());
        local_values = 0.;
        distribution->add_transpose_to_source(values, local_values);
        for (std::size_t q = 0; q < point_indices.size(); ++q)
          destination[point_indices[q]] += local_values[q];
      };
    return result;
  }

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

  template <typename Quantity>
  inline double
  traction_balance(const Quantity                        &quantity,
                   const Traction                        &traction,
                   const ImmersX::ImmersXLA::MPI::Vector &pressure_state,
                   const ImmersX::ImmersXLA::MPI::Vector &elastic_state)
  {
    using ElasticVector = ImmersX::ElasticStaticProblem<3, 3>::VectorType;

    const auto expected =
      make_traction_operator<Quantity, ElasticVector>(quantity, traction);
    const auto    pressure = sample_pressure(quantity, pressure_state);
    ElasticVector expected_load;
    expected.reinit_range_vector(expected_load, false);
    expected.vmult(expected_load, pressure);

    ElasticVector actual_load;
    actual_load.reinit(traction.problem().forcing());
    actual_load = traction.problem().forcing();
    ElasticVector stiffness_load;
    stiffness_load.reinit(actual_load);
    traction.problem().stiffness_operator().vmult(stiffness_load,
                                                  elastic_state);
    actual_load -= stiffness_load;

    expected_load -= actual_load;
    return expected_load.l2_norm();
  }

  template <typename Quantity, typename Adapter, typename Fields>
  auto
  make_interaction(const Quantity                                &quantity,
                   const ImmersX::ProblemHandle<Adapter, Fields> &target,
                   const Traction                                &traction)
  {
    using ElasticVector = ImmersX::ElasticStaticProblem<3, 3>::VectorType;
    using Coupling =
      ImmersX::CouplingOperator<typename Quantity::value_type, ElasticVector>;

    const auto operator_view =
      make_traction_operator<Quantity, ElasticVector>(quantity, traction);
    const ImmersX::CouplingSpace<ElasticVector> target_space(
      traction.problem().solution());
    const Coupling coupling(operator_view, target_space);

    return ImmersX::RepresentationLoadInteraction<Quantity, ElasticVector>(
      quantity, target.fields().displacement, coupling);
  }

} // namespace CoupledPoissonElasticity

#endif // immersx_coupled_poisson_elasticity_h
