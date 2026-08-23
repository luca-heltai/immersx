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

#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/vector.h>

#include <immersx/core/lifting.h>
#include <immersx/core/load_interaction.h>
#include <immersx/core/problem_handle.h>
#include <immersx/physics/elastic_static.h>
#include <immersx/physics/poisson_residual.h>

#include <cmath>
#include <utility>
#include <vector>

namespace CoupledPoissonElasticity
{
  struct Pressure
  {
    unsigned int source_size = 2;
    double       factor      = 2.;
  };

  /** The pressure observable p=2u, with a serial value space for lifting. */
  class PressureRepresentation
  {
  public:
    using SourceVectorType    = ImmersX::ImmersXLA::MPI::Vector;
    using value_type          = dealii::Vector<double>;
    using state_type          = SourceVectorType;
    using quantity_space_type = ImmersX::QuantitySpace<value_type>;
    using Operator = ImmersX::RepresentationOperator<value_type, state_type>;

    PressureRepresentation(const ImmersX::FieldId source,
                           const unsigned int     source_size,
                           const double           factor)
      : source_(source)
      , source_size_(source_size)
      , factor_(factor)
    {}

    ImmersX::FieldId
    source() const
    {
      return source_;
    }

    const ImmersX::RepresentationDomain &
    domain() const
    {
      static const ImmersX::RepresentationDomain domain(1, 3, "line");
      return domain;
    }

    quantity_space_type
    quantity_space() const
    {
      return quantity_space_type(domain());
    }

    template <typename Geometry>
    decltype(auto)
    lift(const Geometry &geometry) const
    {
      return ImmersX::detail::invoke_lift(*this, geometry, 0);
    }

    value_type
    evaluate(const ImmersX::EvaluationContext<state_type> &context,
             const ImmersX::EvaluationRequest & = {}) const
    {
      const auto values = gather(context.state(source_));
      value_type result(source_size_);
      for (unsigned int i = 0; i < source_size_; ++i)
        result[i] = factor_ * values[i];
      return result;
    }

    Operator
    linearize(const ImmersX::EvaluationContext<state_type> &context,
              const ImmersX::EvaluationRequest & = {}) const
    {
      const auto *reference   = &context.state(source_);
      const auto  source_size = source_size_;
      const auto  factor      = factor_;

      Operator result;
      result.reinit_range_vector = [source_size](value_type &vector,
                                                 const bool  omit) {
        vector.reinit(source_size);
        if (!omit)
          vector = 0.;
      };
      result.reinit_domain_vector = [reference](state_type &vector,
                                                const bool  omit) {
        vector.reinit(*reference, omit);
      };
      result.vmult = [source_size, factor](value_type       &destination,
                                           const state_type &source) {
        const auto values = gather(source);
        destination.reinit(source_size);
        for (unsigned int i = 0; i < source_size; ++i)
          destination[i] = factor * values[i];
      };
      result.vmult_add = [source_size, factor](value_type       &destination,
                                               const state_type &source) {
        const auto values = gather(source);
        for (unsigned int i = 0; i < source_size; ++i)
          destination[i] += factor * values[i];
      };
      result.Tvmult = [factor](state_type       &destination,
                               const value_type &source) {
        destination = 0.;
        for (const auto index : destination.locally_owned_elements())
          if (index < source.size())
            destination[index] = factor * source[index];
      };
      result.Tvmult_add = [factor](state_type       &destination,
                                   const value_type &source) {
        for (const auto index : destination.locally_owned_elements())
          if (index < source.size())
            destination[index] += factor * source[index];
      };
      return result;
    }

  private:
    static std::vector<double>
    gather(const state_type &source)
    {
      std::vector<double> local(source.size(), 0.);
      for (const auto index : source.locally_owned_elements())
        local[index] = source[index];

      std::vector<double> global(source.size(), 0.);
      MPI_Allreduce(local.data(),
                    global.data(),
                    static_cast<int>(source.size()),
                    MPI_DOUBLE,
                    MPI_SUM,
                    source.get_mpi_communicator());
      return global;
    }

    ImmersX::FieldId source_;
    unsigned int     source_size_;
    double           factor_;
  };

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
    Traction(const ImmersX::ElasticStaticProblem<3, 3> &problem,
             const CylinderSurface                     &surface)
      : problem_(&problem)
      , surface_(&surface)
    {}

    const ImmersX::ElasticStaticProblem<3, 3> &
    problem() const
    {
      return *problem_;
    }

    const CylinderSurface &
    surface() const
    {
      return *surface_;
    }

  private:
    const ImmersX::ElasticStaticProblem<3, 3> *problem_;
    const CylinderSurface                     *surface_;
  };

  template <typename Adapter, typename Fields>
  auto
  make_representation(const ImmersX::ProblemHandle<Adapter, Fields> &problem,
                      const Pressure                                &pressure)
  {
    return PressureRepresentation(problem.fields().solution,
                                  pressure.source_size,
                                  pressure.factor);
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

  template <typename ElasticVector>
  dealii::LinearOperator<ElasticVector, dealii::Vector<double>>
  make_traction_operator(const Traction &traction)
  {
    using Operator =
      dealii::LinearOperator<ElasticVector, dealii::Vector<double>>;

    const auto &problem        = traction.problem();
    const auto &surface        = traction.surface();
    const auto &dof_handler    = problem.dof_handler();
    const auto &finite_element = dof_handler.get_fe();
    const auto &owned_dofs     = problem.locally_owned_dofs();

    std::vector<std::vector<std::pair<dealii::types::global_dof_index, double>>>
                                             entries(surface.points().size());
    const dealii::MappingQ1<3>               mapping;
    const dealii::FEValuesExtractors::Vector displacement(0);

    for (unsigned int q = 0; q < surface.points().size(); ++q)
      {
        const auto cell_and_unit_point =
          dealii::GridTools::find_active_cell_around_point(mapping,
                                                           dof_handler,
                                                           surface.points()[q]);
        const auto &cell = cell_and_unit_point.first;
        if (cell == dof_handler.end())
          continue;

        const std::vector<dealii::Point<3>> unit_points{
          cell_and_unit_point.second};
        const dealii::Quadrature<3> point_quadrature(unit_points);
        dealii::FEValues<3>         fe_values(mapping,
                                      finite_element,
                                      point_quadrature,
                                      dealii::update_values);
        fe_values.reinit(cell);

        std::vector<dealii::types::global_dof_index> dof_indices(
          finite_element.n_dofs_per_cell());
        cell->get_dof_indices(dof_indices);
        for (unsigned int i = 0; i < dof_indices.size(); ++i)
          if (owned_dofs.is_element(dof_indices[i]))
            entries[q].emplace_back(dof_indices[i],
                                    surface.weights()[q] *
                                      (surface.normals()[q] *
                                       fe_values[displacement].value(i, 0)));
      }

    const auto *prototype = &problem.solution();
    const auto  n_points  = surface.points().size();

    Operator result;
    result.reinit_range_vector = [prototype](ElasticVector &vector,
                                             const bool     omit) {
      vector.reinit(*prototype, omit);
    };
    result.reinit_domain_vector = [n_points](dealii::Vector<double> &vector,
                                             const bool              omit) {
      vector.reinit(n_points);
      if (!omit)
        vector = 0.;
    };
    result.vmult = [entries](ElasticVector                &destination,
                             const dealii::Vector<double> &source) {
      destination = 0.;
      for (unsigned int q = 0; q < entries.size(); ++q)
        for (const auto &[row, value] : entries[q])
          destination[row] += value * source[q];
    };
    result.vmult_add = [entries](ElasticVector                &destination,
                                 const dealii::Vector<double> &source) {
      for (unsigned int q = 0; q < entries.size(); ++q)
        for (const auto &[row, value] : entries[q])
          destination[row] += value * source[q];
    };
    result.Tvmult = [entries](dealii::Vector<double> &destination,
                              const ElasticVector    &source) {
      destination = 0.;
      for (unsigned int q = 0; q < entries.size(); ++q)
        for (const auto &[row, value] : entries[q])
          destination[q] += value * source[row];
    };
    result.Tvmult_add = [entries](dealii::Vector<double> &destination,
                                  const ElasticVector    &source) {
      for (unsigned int q = 0; q < entries.size(); ++q)
        for (const auto &[row, value] : entries[q])
          destination[q] += value * source[row];
    };
    return result;
  }

  inline dealii::Vector<double>
  sample_pressure(const CylinderSurface                 &surface,
                  const ImmersX::ImmersXLA::MPI::Vector &solution)
  {
    std::vector<double> local(solution.size(), 0.);
    for (const auto index : solution.locally_owned_elements())
      local[index] = solution[index];
    std::vector<double> global(solution.size(), 0.);
    MPI_Allreduce(local.data(),
                  global.data(),
                  static_cast<int>(solution.size()),
                  MPI_DOUBLE,
                  MPI_SUM,
                  solution.get_mpi_communicator());

    AssertThrow(solution.size() == 2,
                dealii::ExcMessage(
                  "The vertical slice expects a two-DoF line pressure."));
    dealii::Vector<double> result(surface.points().size());
    for (unsigned int q = 0; q < surface.points().size(); ++q)
      {
        const double s =
          surface.geometry().source_parameter(surface.points()[q]);
        result[q] = 2. * ((1. - s) * global[0] + s * global[1]);
      }
    return result;
  }

  inline double
  traction_balance(const Traction                        &traction,
                   const ImmersX::ImmersXLA::MPI::Vector &pressure_state,
                   const ImmersX::ImmersXLA::MPI::Vector &elastic_state)
  {
    using ElasticVector = ImmersX::ElasticStaticProblem<3, 3>::VectorType;

    const auto expected = make_traction_operator<ElasticVector>(traction);
    const auto pressure = sample_pressure(traction.surface(), pressure_state);
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
      ImmersX::CouplingOperator<dealii::Vector<double>, ElasticVector>;

    const auto operator_view = make_traction_operator<ElasticVector>(traction);
    const ImmersX::CouplingSpace<ElasticVector> target_space(
      traction.problem().solution());
    const Coupling coupling(operator_view, target_space);

    return ImmersX::RepresentationLoadInteraction<Quantity, ElasticVector>(
      quantity, target.fields().displacement, coupling);
  }

} // namespace CoupledPoissonElasticity

#endif // immersx_coupled_poisson_elasticity_h
