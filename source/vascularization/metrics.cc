// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/rtree.h>

#include <immersx/compatibility/reference_cell.h>
#include <immersx/vascularization/metrics.h>

#include <cmath>
#include <memory>

namespace ImmersX::Vascularization
{
  void
  CoverageParameters::validate() const
  {
    AssertThrow(
      std::isfinite(service_radius) && service_radius >= 0. &&
        quadrature_order > 0 && quadrature_order <= 32,
      dealii::ExcMessage(
        "Coverage requires a finite nonnegative service radius and quadrature order between 1 and 32."));
  }

  TissueMetrics
  measure_tissue(const dealii::Triangulation<3> &background,
                 const dealii::Mapping<3>       &mapping,
                 const NetworkData              &network,
                 const DiscretizationResult     &discretization,
                 const CoverageParameters       &parameters,
                 MPI_Comm                        communicator)
  {
    using namespace dealii;
    namespace bgi = boost::geometry::index;
    AssertThrow(Utilities::MPI::n_mpi_processes(communicator) == 1,
                ExcMessage(
                  "The current tissue measurement requires one rank."));
    parameters.validate();
    AssertThrow(
      discretization.validation.valid() && validate_data(network, 1).valid(),
      ExcMessage(
        "Tissue measurement requires a valid network and discretization."));
    TissueMetrics result;
    const auto    reference_cells = background.get_reference_cells();
    AssertThrow(
      reference_cells.size() == 1 &&
        (reference_cells[0].is_simplex() || reference_cells[0].is_hyper_cube()),
      ExcMessage(
        "Tissue measurement requires a nonempty tetrahedral or hexahedral background."));
    result.background_volume = GridTools::volume(background, mapping);
    AssertThrow(
      std::isfinite(result.background_volume) && result.background_volume > 0.,
      ExcMessage("Tissue measurement requires a positive background volume."));
    result.inside_vessel_volume = discretization.inside_volume;
    result.volume_fraction =
      result.inside_vessel_volume / result.background_volume;
    result.coverage_evaluated = parameters.service_radius > 0.;
    if (!result.coverage_evaluated)
      return result;
    std::array<std::vector<Point<3>>, 2> terminals;
    for (const auto &[id, t] : network.terminals)
      {
        const auto &node = network.nodes.at(t.node);
        if (node.port != Port::none)
          continue;
        if (t.system == System::arterial)
          terminals[0].push_back(node.position);
        else if (t.system == System::venous)
          terminals[1].push_back(node.position);
      }
    const std::array<RTree<Point<3>>, 2> trees{
      {pack_rtree(terminals[0]), pack_rtree(terminals[1])}};
    const auto served = [&](unsigned int system, const Point<3> &point) {
      for (const auto &nearest :
           trees[system] | bgi::adaptors::queried(bgi::nearest(point, 1)))
        return nearest.distance(point) <= parameters.service_radius;
      return false;
    };
    // Retain each FE for the lifetime of its FEValues observer. Geometry-only
    // quadrature needs no DoFs and supports both tetrahedra and hexahedra.
    std::array<std::unique_ptr<FE_Nothing<3>>, 2> elements;
    std::array<std::unique_ptr<FEValues<3>>, 2>   values;
    for (const auto &cell : background.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          AssertThrow(
            cell->reference_cell().is_simplex() ||
              cell->reference_cell().is_hyper_cube(),
            ExcMessage("Tissue measurement requires tetrahedra or hexahedra."));
          const unsigned int type = cell->reference_cell().is_simplex() ? 0 : 1;
          if (!values[type])
            {
              elements[type] =
                std::make_unique<FE_Nothing<3>>(cell->reference_cell());
              values[type] = std::make_unique<FEValues<3>>(
                mapping,
                *elements[type],
                compatibility::gauss_quadrature<3>(cell->reference_cell(),
                                                   parameters.quadrature_order),
                update_quadrature_points | update_JxW_values);
            }
          auto &fe = *values[type];
          fe.reinit(cell);
          for (unsigned int q = 0; q < fe.n_quadrature_points; ++q)
            {
              const double weight = fe.JxW(q);
              AssertThrow(std::isfinite(weight) && weight > 0.,
                          ExcMessage("Invalid background quadrature volume."));
              result.coverage_quadrature_volume += weight;
              if (result.coverage_evaluated)
                {
                  const bool artery = served(0, fe.quadrature_point(q)),
                             vein   = served(1, fe.quadrature_point(q));
                  if (artery)
                    result.arterial_coverage += weight;
                  if (vein)
                    result.venous_coverage += weight;
                  if (artery && vein)
                    result.dual_coverage += weight;
                }
            }
        }
    AssertThrow(result.coverage_quadrature_volume > 0.,
                ExcMessage("Coverage quadrature has zero volume."));
    result.arterial_coverage /= result.coverage_quadrature_volume;
    result.venous_coverage /= result.coverage_quadrature_volume;
    result.dual_coverage /= result.coverage_quadrature_volume;
    return result;
  }
} // namespace ImmersX::Vascularization
