// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/numerics/rtree.h>

#include <boost/serialization/array.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

#include <immersx/compatibility/reference_cell.h>
#include <immersx/vascularization/mesh_scale.h>
#include <immersx/vascularization/metrics.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <set>

namespace ImmersX::Vascularization
{
  namespace
  {
    double
    background_volume(const dealii::Triangulation<3> &background,
                      const dealii::Mapping<3>       &mapping,
                      MPI_Comm                        comm)
    {
      using namespace dealii;
      const auto native_volume_supported =
        Utilities::MPI::min(static_cast<unsigned int>(
                              background.get_reference_cells().size() == 1),
                            comm);
      if (native_volume_supported)
        return GridTools::volume(background, mapping);
      // Fully distributed empty ranks carry artificial hypercubes. Their type
      // enters the global reference-cell list even for a purely tetrahedral
      // physical mesh, which GridTools::volume() cannot handle. Use its same
      // Gauss rule of order 2 on owned cells, ignoring artificial
      // ones.
      std::unique_ptr<FE_Nothing<3>> element;
      std::unique_ptr<FEValues<3>>   values;
      double                         local_volume = 0.;
      for (const auto &cell : background.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            if (!values)
              {
                element =
                  std::make_unique<FE_Nothing<3>>(cell->reference_cell());
                values = std::make_unique<FEValues<3>>(
                  mapping,
                  *element,
                  compatibility::gauss_quadrature<3>(cell->reference_cell(), 2),
                  update_JxW_values);
              }
            values->reinit(cell);
            for (unsigned int q = 0; q < values->n_quadrature_points; ++q)
              local_volume += values->JxW(q);
          }
      return Utilities::MPI::sum(local_volume, comm);
    }

    std::array<std::vector<dealii::Point<3>>, 2>
    service_centres(const dealii::Triangulation<3> &background,
                    const dealii::Mapping<3>       &mapping,
                    const NetworkData              &network,
                    double                          radius,
                    MPI_Comm                        comm)
    {
      using namespace dealii;
      namespace bgi            = boost::geometry::index;
      const auto          rank = Utilities::MPI::this_mpi_process(comm);
      GridTools::Cache<3> cache(background, mapping);
      const auto &tree = cache.get_locally_owned_cell_bounding_boxes_rtree();
      std::vector<BoundingBox<3>> local_boxes;
      if (!tree.empty())
        local_boxes = extract_rtree_level(tree, 1);
      const auto boxes = Utilities::MPI::all_gather(comm, local_boxes);
      std::vector<std::pair<BoundingBox<3>, unsigned int>> covering;
      for (unsigned int r = 0; r < boxes.size(); ++r)
        for (const auto &box : boxes[r])
          covering.emplace_back(box, r);
      const auto covering_tree = pack_rtree(covering);
      using Centre             = std::pair<System, Point<3>>;
      std::map<unsigned int, std::vector<Centre>> send;
      for (const auto &[id, terminal] : network.terminals)
        if (terminal.owner == rank)
          {
            const auto &node = network.nodes.at(terminal.node);
            if (node.port != Port::none)
              continue;
            Point<3> lo = node.position, hi = node.position;
            for (unsigned int d = 0; d < 3; ++d)
              {
                lo[d] -= radius;
                hi[d] += radius;
              }
            const BoundingBox<3>   envelope(std::make_pair(lo, hi));
            std::set<unsigned int> recipients;
            for (const auto &[box, source] :
                 covering_tree |
                   bgi::adaptors::queried(bgi::intersects(envelope)))
              recipients.insert(source);
            for (const auto destination : recipients)
              send[destination].push_back({terminal.system, node.position});
          }
      // A service ball may cross many partitions even if no vessel ghost is
      // present there. Route centres independently of physical vessel halos.
      std::array<std::vector<Point<3>>, 2> result;
      for (const auto &[source, centres] :
           Utilities::MPI::some_to_some(comm, send))
        for (const auto &[system, position] : centres)
          {
            if (system == System::arterial)
              result[0].push_back(position);
            else if (system == System::venous)
              result[1].push_back(position);
          }
      return result;
    }
  } // namespace

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
    return measure_tissue(background,
                          mapping,
                          network,
                          discretization,
                          parameters,
                          communicator,
                          MeshLengthScale{});
  }

  TissueMetrics
  measure_tissue(const dealii::Triangulation<3> &background,
                 const dealii::Mapping<3>       &mapping,
                 const NetworkData              &network,
                 const DiscretizationResult     &discretization,
                 const CoverageParameters       &parameters,
                 MPI_Comm                        communicator,
                 const MeshLengthScale          &scale)
  {
    using namespace dealii;
    namespace bgi          = boost::geometry::index;
    const auto size        = Utilities::MPI::n_mpi_processes(communicator);
    const auto require_all = [&](bool valid, const std::string &message) {
      AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(valid),
                                      communicator) == 1,
                  ExcMessage(message));
    };
    const std::array<double, 6> controls{
      {parameters.service_radius,
       static_cast<double>(parameters.quadrature_order),
       discretization.inside_volume,
       static_cast<double>(scale.policy),
       scale.mesh_minimum,
       scale.mesh_maximum}};
    const auto all_controls =
      Utilities::MPI::all_gather(communicator, controls);
    require_all(
      std::all_of(all_controls.begin(),
                  all_controls.end(),
                  [&](const auto &other) { return other == controls; }),
      "Coverage and length-scale controls and global inside volume must agree across ranks.");
    parameters.validate();
    scale.validate();
    int comparison = MPI_UNEQUAL;
    MPI_Comm_compare(background.get_mpi_communicator(),
                     communicator,
                     &comparison);
    const auto *mapping_q  = dynamic_cast<const MappingQ<3> *>(&mapping);
    const auto *mapping_fe = dynamic_cast<const MappingFE<3> *>(&mapping);
    require_all(
      discretization.validation.valid() &&
        validate_data(network, size).valid() &&
        std::isfinite(discretization.inside_volume) &&
        discretization.inside_volume >= 0. && background.n_levels() > 0 &&
        ((mapping_q && mapping_q->get_degree() == 1) ||
         (mapping_fe && mapping_fe->get_degree() == 1)) &&
        (comparison == MPI_IDENT || comparison == MPI_CONGRUENT),
      "Tissue measurement requires valid vascular data and discretization, a first-order mapping and congruent communicators.");
    TissueMetrics result;
    bool          compatible_background = true;
    unsigned int  families[2]{0, 0};
    for (const auto &cell : background.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          if (cell->reference_cell().is_simplex())
            families[0] = 1;
          else if (cell->reference_cell().is_hyper_cube())
            families[1] = 1;
          else
            compatible_background = false;
          compatible_background =
            mapping.is_compatible_with(cell->reference_cell()) &&
            compatible_background;
          if (scale.policy == LengthScalePolicy::local_mesh_size)
            compatible_background = compatible_background &&
                                    std::isfinite(cell->diameter()) &&
                                    cell->diameter() >= scale.mesh_minimum &&
                                    cell->diameter() <= scale.mesh_maximum;
        }
    Utilities::MPI::max(families, communicator, families);
    require_all(
      compatible_background && families[0] + families[1] == 1,
      "Tissue measurement requires one physical tetrahedral or hexahedral cell type compatible with the mapping.");
    result.background_volume =
      background_volume(background, mapping, communicator);
    AssertThrow(
      std::isfinite(result.background_volume) && result.background_volume > 0.,
      ExcMessage("Tissue measurement requires a positive background volume."));
    result.inside_vessel_volume = discretization.inside_volume;
    result.volume_fraction =
      result.inside_vessel_volume / result.background_volume;
    result.coverage_evaluated = parameters.service_radius > 0.;
    if (!result.coverage_evaluated)
      return result;
    // Check both extrema before rank-local quadrature can throw on underflow.
    scale.physical_length(parameters.service_radius, scale.minimum());
    const double reach =
      scale.physical_length(parameters.service_radius, scale.maximum());
    const auto terminals =
      service_centres(background, mapping, network, reach, communicator);
    const std::array<RTree<Point<3>>, 2> trees{
      {pack_rtree(terminals[0]), pack_rtree(terminals[1])}};
    const auto served =
      [&](unsigned int system, const Point<3> &point, double threshold) {
        for (const auto &nearest :
             trees[system] | bgi::adaptors::queried(bgi::nearest(point, 1)))
          return nearest.distance(point) <= threshold;
        return false;
      };
    // Retain each FE for the lifetime of its FEValues observer. Geometry-only
    // quadrature needs no DoFs and supports both tetrahedra and hexahedra.
    std::array<std::unique_ptr<FE_Nothing<3>>, 2> elements;
    std::array<std::unique_ptr<FEValues<3>>, 2>   values;
    bool                                          valid_weights = true;
    for (const auto &cell : background.active_cell_iterators())
      if (cell->is_locally_owned())
        {
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
          const double threshold =
            scale.physical_length(parameters.service_radius, cell->diameter());
          auto &fe = *values[type];
          fe.reinit(cell);
          for (unsigned int q = 0; q < fe.n_quadrature_points; ++q)
            {
              const double weight = fe.JxW(q);
              valid_weights =
                std::isfinite(weight) && weight > 0. && valid_weights;
              result.coverage_quadrature_volume += weight;
              if (result.coverage_evaluated)
                {
                  const bool artery =
                               served(0, fe.quadrature_point(q), threshold),
                             vein =
                               served(1, fe.quadrature_point(q), threshold);
                  if (artery)
                    result.arterial_coverage += weight;
                  if (vein)
                    result.venous_coverage += weight;
                  if (artery && vein)
                    result.dual_coverage += weight;
                }
            }
        }
    require_all(valid_weights, "Invalid background quadrature volume.");
    double integrals[4]{result.coverage_quadrature_volume,
                        result.arterial_coverage,
                        result.venous_coverage,
                        result.dual_coverage};
    Utilities::MPI::sum(integrals, communicator, integrals);
    result.coverage_quadrature_volume = integrals[0];
    result.arterial_coverage          = integrals[1];
    result.venous_coverage            = integrals[2];
    result.dual_coverage              = integrals[3];
    AssertThrow(std::isfinite(result.coverage_quadrature_volume) &&
                  result.coverage_quadrature_volume > 0.,
                ExcMessage(
                  "Coverage quadrature requires positive finite volume."));
    result.arterial_coverage /= result.coverage_quadrature_volume;
    result.venous_coverage /= result.coverage_quadrature_volume;
    result.dual_coverage /= result.coverage_quadrature_volume;
    return result;
  }
} // namespace ImmersX::Vascularization
