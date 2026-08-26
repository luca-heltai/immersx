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
#include <immersx/coupling/particle_coupling.h>
#include <immersx/physics/elastic_static.h>
#include <immersx/physics/poisson_residual.h>

#include <cmath>
#include <utility>
#include <vector>

namespace CoupledPoissonElasticity
{
  struct Pressure
  {
    double factor = 2.;
  };

  using PressureLift = ImmersX::TensorProductLift<1, 2, 3, 1>;

  /** The pressure observable p=2u evaluated in the source Poisson FE space. */
  class PressureRepresentation
  {
  public:
    using ProblemType         = ImmersX::PoissonSolver<1, 3>;
    using SourceVectorType    = ImmersX::ImmersXLA::MPI::Vector;
    using value_type          = dealii::Vector<double>;
    using state_type          = SourceVectorType;
    using quantity_space_type = ImmersX::QuantitySpace<value_type>;
    using Operator = ImmersX::RepresentationOperator<value_type, state_type>;
    using TriangulationType = dealii::parallel::TriangulationBase<1, 3>;
    using DoFHandlerType    = dealii::DoFHandler<1, 3>;
    using ExtractorType     = dealii::FEValuesExtractors::Scalar;
    using QuadraturePoint   = ImmersX::RepresentationQuadraturePoint<3, double>;

    PressureRepresentation(const ImmersX::FieldId source,
                           const ProblemType     &problem,
                           const double           factor)
      : source_(source)
      , problem_(&problem)
      , factor_(factor)
    {}

    static constexpr unsigned int support_dimension        = 1;
    static constexpr unsigned int ambient_dimension        = 3;
    static constexpr unsigned int representative_dimension = 1;

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
             const ImmersX::EvaluationRequest             &request = {}) const
    {
      AssertThrow(!request.points.empty(),
                  dealii::ExcMessage(
                    "Pressure FE evaluation requires representative points."));
      return evaluate_points(context.state(source_), request.points);
    }

    Operator
    linearize(const ImmersX::EvaluationContext<state_type> &context,
              const ImmersX::EvaluationRequest             &request = {}) const
    {
      const auto *reference = &context.state(source_);
      const auto  points    = request.points;
      AssertThrow(!points.empty(),
                  dealii::ExcMessage(
                    "Pressure FE linearization requires representative "
                    "points."));

      Operator result;
      result.reinit_range_vector = [n = points.size()](value_type &vector,
                                                       const bool  omit) {
        vector.reinit(n);
        if (!omit)
          vector = 0.;
      };
      result.reinit_domain_vector = [reference](state_type &vector,
                                                const bool  omit) {
        vector.reinit(*reference, omit);
      };
      const auto *problem = problem_;
      const auto  scale   = factor_;
      result.vmult = [problem, scale, points](value_type       &destination,
                                              const state_type &source) {
        destination = evaluate_points(*problem, scale, source, points);
      };
      result.vmult_add = [problem, scale, points](value_type       &destination,
                                                  const state_type &source) {
        const auto values = evaluate_points(*problem, scale, source, points);
        for (unsigned int i = 0; i < values.size(); ++i)
          destination[i] += values[i];
      };
      result.Tvmult = [problem, scale, points](state_type       &destination,
                                               const value_type &source) {
        apply_point_evaluation_transpose(
          *problem, scale, points, source, destination, false);
      };
      result.Tvmult_add = [problem, scale, points](state_type &destination,
                                                   const value_type &source) {
        apply_point_evaluation_transpose(
          *problem, scale, points, source, destination, true);
      };
      return result;
    }

    const TriangulationType &
    triangulation() const
    {
      return problem_->triangulation();
    }

    const DoFHandlerType &
    dof_handler() const
    {
      return problem_->dof_handler();
    }

    const dealii::FiniteElement<1, 3> &
    finite_element() const
    {
      return dof_handler().get_fe();
    }

    const dealii::Mapping<1, 3> &
    mapping() const
    {
      return dealii::StaticMappingQ1<1, 3>::mapping;
    }

    const dealii::IndexSet &
    locally_owned_dofs() const
    {
      return problem_->locally_owned_dofs();
    }

    const dealii::IndexSet &
    locally_relevant_dofs() const
    {
      return problem_->locally_relevant_dofs();
    }

    const dealii::AffineConstraints<double> &
    constraints() const
    {
      return problem_->constraints();
    }

    const ExtractorType &
    extractor() const
    {
      static const ExtractorType extractor(0);
      return extractor;
    }

    const ImmersX::RepresentationMetadata &
    metadata() const
    {
      static const ImmersX::RepresentationMetadata metadata;
      return metadata;
    }

    const std::vector<ImmersX::FieldId> &
    dependencies() const
    {
      return metadata().dependencies;
    }

    std::uint64_t
    geometry_version() const
    {
      return 0;
    }

    MPI_Comm
    mpi_communicator() const
    {
      return triangulation().get_mpi_communicator();
    }

    unsigned int
    n_dofs_per_cell() const
    {
      return finite_element().n_dofs_per_cell();
    }

    std::vector<QuadraturePoint>
    locally_owned_quadrature_points(
      const dealii::Quadrature<1> &quadrature) const
    {
      dealii::FEValues<1, 3>       fe_values(mapping(),
                                       finite_element(),
                                       quadrature,
                                       dealii::update_values |
                                         dealii::update_quadrature_points |
                                         dealii::update_JxW_values);
      std::vector<QuadraturePoint> result;
      result.reserve(triangulation().n_locally_owned_active_cells() *
                     quadrature.size());
      std::vector<dealii::types::global_dof_index> dof_indices(
        n_dofs_per_cell());
      for (const auto &cell : dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            cell->get_dof_indices(dof_indices);
            for (const auto q : fe_values.quadrature_point_indices())
              {
                QuadraturePoint point;
                point.point                 = fe_values.quadrature_point(q);
                point.tangent               = cell->vertex(1) - cell->vertex(0);
                point.weight                = fe_values.JxW(q);
                point.source_entity_id      = cell->global_active_cell_index();
                point.representative_qpoint = q;
                point.dof_indices           = dof_indices;
                point.basis_values.resize(n_dofs_per_cell());
                for (unsigned int i = 0; i < n_dofs_per_cell(); ++i)
                  point.basis_values[i] = fe_values[extractor()].value(i, q);
                result.emplace_back(std::move(point));
              }
          }
      return result;
    }

    dealii::Vector<double>
    evaluate_points(const state_type                    &source,
                    const std::vector<dealii::Point<3>> &points) const
    {
      return evaluate_points(*problem_, factor_, source, points);
    }

    static dealii::Vector<double>
    evaluate_points(const ProblemType                   &problem,
                    const double                         scale,
                    const state_type                    &source,
                    const std::vector<dealii::Point<3>> &points)
    {
      state_type relevant;
      relevant.reinit(problem.locally_owned_dofs(),
                      problem.locally_relevant_dofs(),
                      problem.triangulation().get_mpi_communicator());
      relevant = source;
      relevant.update_ghost_values();

      dealii::Vector<double> result(points.size());
      for (unsigned int q = 0; q < points.size(); ++q)
        {
          const auto data = local_shape_data(problem, points[q]);
          if (data.dof_indices.empty())
            continue;
          for (unsigned int i = 0; i < data.shape_values.size(); ++i)
            result[q] +=
              scale * data.shape_values[i] * relevant[data.dof_indices[i]];
        }
      return result;
    }

  private:
    /** Local FE data at one physical point on a locally owned cell. */
    struct LocalShapeData
    {
      std::vector<dealii::types::global_dof_index> dof_indices;
      std::vector<double>                          shape_values;
    };

    /**
     * Evaluate the scalar shape functions of the source FE space at one
     * physical point. The returned data is empty when the point does not lie
     * in a locally owned cell.
     */
    static LocalShapeData
    local_shape_data(const ProblemType &problem, const dealii::Point<3> &point)
    {
      LocalShapeData                result;
      const dealii::MappingQ1<1, 3> mapping;
      const auto                    cell_and_unit_point =
        dealii::GridTools::find_active_cell_around_point(mapping,
                                                         problem.dof_handler(),
                                                         point);
      const auto &cell = cell_and_unit_point.first;
      if (cell == problem.dof_handler().end() || !cell->is_locally_owned())
        return result;

      const dealii::Quadrature<1> point_quadrature(
        std::vector<dealii::Point<1>>{cell_and_unit_point.second});
      dealii::FEValues<1, 3> fe_values(mapping,
                                       problem.dof_handler().get_fe(),
                                       point_quadrature,
                                       dealii::update_values);
      fe_values.reinit(cell);
      const unsigned int n_dofs =
        problem.dof_handler().get_fe().n_dofs_per_cell();
      result.dof_indices.resize(n_dofs);
      cell->get_dof_indices(result.dof_indices);
      result.shape_values.resize(n_dofs);
      for (unsigned int i = 0; i < n_dofs; ++i)
        result.shape_values[i] =
          fe_values[dealii::FEValuesExtractors::Scalar(0)].value(i, 0);
      return result;
    }

    /**
     * Accumulate r = A^T z (or add it to destination) for the point
     * evaluation operator A: r_i = sum_j phi_i(x_j) z_j. Contributions are
     * gathered locally and compressed into the owned DoFs; no assumption is
     * made about point or DoF numbering.
     */
    static void
    apply_point_evaluation_transpose(
      const ProblemType                   &problem,
      const double                         scale,
      const std::vector<dealii::Point<3>> &points,
      const dealii::Vector<double>        &source,
      state_type                          &destination,
      const bool                           add)
    {
      // deal.II's own distributed vector permits accumulating into ghost
      // entries; the Trilinos wrapper does not. We gather locally and then
      // compress so that every owned DoF receives the sum of all local cell
      // contributions, including DoFs stored as ghosts on other ranks' cells.
      dealii::LinearAlgebra::distributed::Vector<double> contribution;
      contribution.reinit(problem.locally_owned_dofs(),
                          problem.locally_relevant_dofs(),
                          problem.triangulation().get_mpi_communicator());
      contribution = 0.;
      for (unsigned int q = 0; q < points.size(); ++q)
        {
          const auto data = local_shape_data(problem, points[q]);
          if (data.dof_indices.empty())
            continue;
          for (unsigned int i = 0; i < data.shape_values.size(); ++i)
            contribution[data.dof_indices[i]] +=
              scale * data.shape_values[i] * source[q];
        }
      contribution.compress(dealii::VectorOperation::add);
      if (add)
        for (const auto index : destination.locally_owned_elements())
          destination[index] += contribution[index];
      else
        for (const auto index : destination.locally_owned_elements())
          destination[index] = contribution[index];
    }

    ImmersX::FieldId   source_;
    const ProblemType *problem_;
    double             factor_;
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
    explicit Traction(const ImmersX::ElasticStaticProblem<3, 3> &problem)
      : problem_(&problem)
    {}

    const ImmersX::ElasticStaticProblem<3, 3> &
    problem() const
    {
      return *problem_;
    }

  private:
    const ImmersX::ElasticStaticProblem<3, 3> *problem_;
  };

  template <typename Adapter, typename Fields>
  auto
  make_representation(const ImmersX::ProblemHandle<Adapter, Fields> &problem,
                      const Pressure                                &pressure)
  {
    return PressureRepresentation(problem.fields().solution,
                                  *problem.fields().problem,
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

  template <typename Quantity, typename ElasticVector>
  dealii::LinearOperator<ElasticVector, dealii::Vector<double>>
  make_traction_operator(const Quantity &quantity, const Traction &traction)
  {
    using Operator =
      dealii::LinearOperator<ElasticVector, dealii::Vector<double>>;

    const auto                &problem        = traction.problem();
    const auto                &dof_handler    = problem.dof_handler();
    const auto                &finite_element = dof_handler.get_fe();
    const auto                &owned_dofs     = problem.locally_owned_dofs();
    const dealii::MappingQ1<3> mapping;

    const auto source_points = quantity.locally_owned_quadrature_points(
      dealii::Quadrature<Quantity::support_dimension>());
    static const ImmersX::ParticleCouplingParameters<3> particle_parameters(
      "/Coupled Poisson elasticity particle coupling/");
    auto distribution =
      std::make_shared<ImmersX::DistributedLiftedQuadrature<3>>(
        particle_parameters);
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
    const auto  n_points  = source_points.size();

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
    result.vmult = [entries,
                    distribution](ElasticVector                &destination,
                                  const dealii::Vector<double> &source) {
      const auto values = distribution->values_on_target(source);
      destination       = 0.;
      for (const auto &[id, point_entries] : *entries)
        for (const auto &[row, value] : point_entries)
          destination[row] += value * values.at(id);
    };
    result.vmult_add = [entries,
                        distribution](ElasticVector                &destination,
                                      const dealii::Vector<double> &source) {
      const auto values = distribution->values_on_target(source);
      for (const auto &[id, point_entries] : *entries)
        for (const auto &[row, value] : point_entries)
          destination[row] += value * values.at(id);
    };
    result.Tvmult = [entries, distribution](dealii::Vector<double> &destination,
                                            const ElasticVector    &source) {
      std::map<dealii::types::particle_index, double> values;
      for (const auto &[id, point_entries] : *entries)
        for (const auto &[row, value] : point_entries)
          values[id] += value * source[row];
      destination = 0.;
      distribution->add_transpose_to_source(values, destination);
    };
    result.Tvmult_add = [entries,
                         distribution](dealii::Vector<double> &destination,
                                       const ElasticVector    &source) {
      std::map<dealii::types::particle_index, double> values;
      for (const auto &[id, point_entries] : *entries)
        for (const auto &[row, value] : point_entries)
          values[id] += value * source[row];
      distribution->add_transpose_to_source(values, destination);
    };
    return result;
  }

  template <typename Quantity>
  inline dealii::Vector<double>
  sample_pressure(const Quantity                        &quantity,
                  const ImmersX::ImmersXLA::MPI::Vector &solution)
  {
    std::vector<dealii::Point<3>> points;
    points.reserve(quantity.lifted_points().size());
    for (const auto &point : quantity.lifted_points())
      points.push_back(point.representative_point);
    return quantity.source_representation().evaluate_points(solution, points);
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
      ImmersX::CouplingOperator<dealii::Vector<double>, ElasticVector>;

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
