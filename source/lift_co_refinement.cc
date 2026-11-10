// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/mpi.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_tools_cache.h>

#include <boost/geometry/index/rtree.hpp>

#include <immersx/coupling/immersed_repartitioner.h>
#include <immersx/coupling/tensor_product_lift.h>
#include <immersx/grid/lift_co_refinement.h>
#include <immersx/grid/local_co_refinement_parameters.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>


namespace ImmersX
{
  namespace
  {
    LiftCoRefinementReport
    co_refine_lift_meshes(
      dealii::Triangulation<3>                              &solid,
      dealii::Triangulation<1, 3>                           &fluid,
      TensorProductLift<1, 2, 3, 3>                         &lift,
      const RefinementParameters<1>                         &parameters,
      const std::function<double(const dealii::CellId &,
                                 const dealii::Point<1> &)> &radius,
      const unsigned int maximum_section_level,
      const MPI_Comm     communicator)
    {
      using namespace dealii;
      namespace bgi             = boost::geometry::index;
      const auto [lower, upper] = parameters.diameter_bounds();
      AssertThrow(solid.all_reference_cells_are_hyper_cube(),
                  ExcMessage(
                    "Lift co-refinement currently requires solid hexahedra."));
      AssertThrow(
        (dynamic_cast<parallel::TriangulationBase<1, 3> *>(&fluid) == nullptr),
        ExcMessage(
          "Lift co-refinement requires a serial fluid preparation mesh."));
      const bool serial_solid =
        dynamic_cast<parallel::TriangulationBase<3> *>(&solid) == nullptr;
      const bool refine_solid = parameters.refinement_strategy == "space" ||
                                parameters.refinement_strategy == "both";
      const bool refine_lift = parameters.refinement_strategy == "embedded" ||
                               parameters.refinement_strategy == "both";
      LiftCoRefinementReport report;
      report.initial_solid_cells = solid.n_global_active_cells();
      report.initial_fluid_cells = fluid.n_global_active_cells();
      solid.refine_global(parameters.space_pre_refinement_cycles);
      fluid.refine_global(parameters.embedded_pre_refinement_cycles);
      const MappingQ1<3>    solid_mapping;
      GridTools::Cache<3>   cache(solid, solid_mapping);
      const MappingQ1<1, 3> fluid_mapping;
      const FE_Q<1, 3>      geometry_fe(1);

      struct Point
      {
        dealii::Point<3>      position;
        CellId                fluid_cell;
        std::array<double, 2> scale;
      };
      const auto build_points = [&]() {
        TensorProductLiftSupport<1, 2, 3, 3> support(lift.parameters());
        const auto         quadrature = support.representative_quadrature();
        FEValues<1, 3>     values(fluid_mapping,
                              geometry_fe,
                              quadrature,
                              update_quadrature_points | update_JxW_values |
                                update_jacobians);
        std::vector<Point> points;
        for (const auto &cell : fluid.active_cell_iterators())
          {
            values.reinit(cell);
            for (const auto q : values.quadrature_point_indices())
              {
                const double r = radius(cell->id(), quadrature.point(q));
                AssertThrow(
                  std::isfinite(r) && r > 0.,
                  ExcMessage("Lift co-refinement requires a positive radius."));
                const auto section =
                  support.transform(values.quadrature_point(q),
                                    values.jacobian(q),
                                    values.JxW(q),
                                    r,
                                    q);
                for (const auto &point : section)
                  {
                    // This is the physical length carried by one section
                    // integration point, independent of the axial rule.
                    const double circumferential = point.weight / values.JxW(q);
                    AssertThrow(
                      std::isfinite(circumferential) && circumferential > 0.,
                      ExcMessage(
                        "Lift co-refinement requires positive section weights."));
                    points.push_back({point.point,
                                      cell->id(),
                                      {{cell->diameter(), circumferential}}});
                  }
              }
          }
        AssertThrow(!points.empty(),
                    ExcMessage("The lift has no support points."));
        AssertThrow(points.size() <=
                      static_cast<std::size_t>(std::numeric_limits<int>::max()),
                    ExcMessage(
                      "Too many lift points for refinement synchronization."));
        return points;
      };

      const auto finish_pass = [&](auto &tria, const bool serial) {
        unsigned long long marked = 0;
        if (serial)
          {
            std::vector<unsigned int> flags(tria.n_active_cells(), 0);
            for (const auto &cell : tria.active_cell_iterators())
              flags[cell->active_cell_index()] = cell->refine_flag_set();
            AssertThrow(
              flags.size() <=
                static_cast<std::size_t>(std::numeric_limits<int>::max()),
              ExcMessage(
                "Too many serial cells for refinement synchronization."));
            const int error = MPI_Allreduce(MPI_IN_PLACE,
                                            flags.data(),
                                            static_cast<int>(flags.size()),
                                            MPI_UNSIGNED,
                                            MPI_MAX,
                                            communicator);
            AssertThrow(error == MPI_SUCCESS,
                        ExcMessage("Refinement flag synchronization failed."));
            for (const auto &cell : tria.active_cell_iterators())
              if (flags[cell->active_cell_index()] != 0)
                {
                  cell->set_refine_flag();
                  ++marked;
                }
          }
        else
          {
            for (const auto &cell : tria.active_cell_iterators())
              marked += cell->is_locally_owned() && cell->refine_flag_set();
            marked = Utilities::MPI::sum(marked, communicator);
          }
        if (marked == 0)
          return false;
        const auto before = tria.n_global_active_cells();
        tria.execute_coarsening_and_refinement();
        AssertThrow(
          tria.n_global_active_cells() > before,
          ExcMessage(
            "Lift co-refinement marked cells without making progress."));
        return true;
      };

      // Only exact point-containing cells take part. In particular, no radius
      // is added to a cell box when determining physical interaction.
      const auto evaluate = [&](const bool mark_lower,
                                const bool mark_upper,
                                const bool post_refine,
                                bool      &mark_section) {
        LiftCoRefinementReport    current;
        const auto                points = build_points();
        std::vector<unsigned int> found(points.size(), 0);
        const auto &tree = cache.get_locally_owned_cell_bounding_boxes_rtree();
        for (std::size_t p = 0; p < points.size(); ++p)
          {
            const auto          &point = points[p];
            const BoundingBox<3> query(
              std::make_pair(point.position, point.position));
            for (const auto &[box, cell] :
                 tree | bgi::adaptors::queried(bgi::intersects(query)))
              {
                dealii::Point<3> unit;
                try
                  {
                    unit =
                      solid_mapping.transform_real_to_unit_cell(cell,
                                                                point.position);
                  }
                catch (const Mapping<3>::ExcTransformationFailed &)
                  {
                    continue;
                  }
                if (!GeometryInfo<3>::is_inside_unit_cell(unit, 1.e-10))
                  continue;
                found[p] = 1;
                ++current.point_cell_pairs;
                const auto &[a, b] = box.get_boundary_points();
                const double h     = a.distance(b);
                for (unsigned int direction = 0; direction < 2; ++direction)
                  {
                    const double ratio = point.scale[direction] / h;
                    current.minimum_ratio[direction] =
                      std::min(current.minimum_ratio[direction], ratio);
                    current.maximum_ratio[direction] =
                      std::max(current.maximum_ratio[direction], ratio);
                    current.lower_violations[direction] += ratio <= lower;
                    current.upper_violations[direction] += ratio >= upper;
                    if (mark_lower && refine_solid && ratio <= lower &&
                        cell->level() < parameters.max_refinement_level)
                      cell->set_refine_flag();
                    if (mark_upper && refine_lift && ratio >= upper)
                      {
                        if (direction == 0)
                          {
                            const auto source =
                              fluid.create_cell_iterator(point.fluid_cell);
                            if (source->level() <
                                parameters.max_refinement_level)
                              source->set_refine_flag();
                          }
                        else if (direction == 1 &&
                                 lift.section.refinement_level <
                                   maximum_section_level)
                          mark_section = true;
                      }
                  }
                if (post_refine)
                  cell->set_refine_flag();
              }
          }
        const int error = MPI_Allreduce(MPI_IN_PLACE,
                                        found.data(),
                                        static_cast<int>(found.size()),
                                        MPI_UNSIGNED,
                                        MPI_MAX,
                                        communicator);
        AssertThrow(error == MPI_SUCCESS,
                    ExcMessage("Lift coverage synchronization failed."));
        for (const auto covered : found)
          current.outside_points += covered == 0;
        const auto count = [&](const unsigned long long n) {
          return serial_solid ? Utilities::MPI::max(n, communicator) :
                                Utilities::MPI::sum(n, communicator);
        };
        current.point_cell_pairs = count(current.point_cell_pairs);
        for (unsigned int d = 0; d < 2; ++d)
          {
            current.lower_violations[d] = count(current.lower_violations[d]);
            current.upper_violations[d] = count(current.upper_violations[d]);
            current.minimum_ratio[d] =
              Utilities::MPI::min(current.minimum_ratio[d], communicator);
            current.maximum_ratio[d] =
              Utilities::MPI::max(current.maximum_ratio[d], communicator);
          }
        return current;
      };

      const auto adjust = [&]() {
        bool changed = true;
        while (changed)
          {
            ++report.iterations;
            bool section = false;
            evaluate(true, false, false, section);
            const bool solid_changed = finish_pass(solid, serial_solid);
            report.solid_passes += solid_changed;
            // Query the refined solid before deciding what to refine in the
            // lift. Fluid refinement changes the physical support locations.
            evaluate(false, true, false, section);
            const bool fluid_changed = finish_pass(fluid, true);
            report.fluid_passes += fluid_changed;
            section = Utilities::MPI::max(section ? 1u : 0u, communicator) != 0;
            if (section)
              {
                ++lift.section.refinement_level;
                ++report.section_passes;
              }
            changed = solid_changed || fluid_changed || section;
          }
      };
      adjust();
      for (unsigned int i = 0; i < parameters.space_post_refinement_cycles; ++i)
        {
          bool section = false;
          evaluate(false, false, true, section);
          report.solid_passes += finish_pass(solid, serial_solid);
          adjust();
        }
      fluid.refine_global(parameters.embedded_post_refinement_cycles);
      adjust();
      bool       section      = false;
      const auto final        = evaluate(false, false, false, section);
      report.point_cell_pairs = final.point_cell_pairs;
      report.outside_points   = final.outside_points;
      report.lower_violations = final.lower_violations;
      report.upper_violations = final.upper_violations;
      report.minimum_ratio    = final.minimum_ratio;
      report.maximum_ratio    = final.maximum_ratio;
      report.solid_cells      = solid.n_global_active_cells();
      report.fluid_cells      = fluid.n_global_active_cells();
      report.section_level    = lift.section.refinement_level;
      return report;
    }
  } // namespace

  LiftCoRefinementReport
  co_refine_lifted_domains(
    Domain<3>                                             &solid,
    Domain<1, 3>                                          &fluid,
    TensorProductLift<1, 2, 3, 3>                         &lift,
    const RefinementParameters<1>                         &parameters,
    const std::function<double(const dealii::CellId &,
                               const dealii::Point<1> &)> &radius,
    const unsigned int maximum_section_level)
  {
    using namespace dealii;
    parameters.diameter_bounds();
    int       comparison = MPI_UNEQUAL;
    const int error =
      MPI_Comm_compare(solid.communicator(), fluid.communicator(), &comparison);
    AssertThrow(error == MPI_SUCCESS &&
                  (comparison == MPI_IDENT || comparison == MPI_CONGRUENT),
                ExcMessage(
                  "Lift co-refinement requires congruent communicators."));
    if (parameters.embedded_partitioning_policy == "immersed")
      AssertThrow(
        fluid.uses_fully_distributed_triangulation(),
        ExcMessage(
          "Immersed ownership requires a fullydistributed fluid Domain."));
    if (!solid.grid_is_prepared())
      solid.prepare_grid();
    if (!fluid.grid_is_prepared())
      fluid.prepare_grid(true);
    // Do not keep caches or FEValues alive when finalization destroys a serial
    // preparation mesh.
    auto report = co_refine_lift_meshes(solid.refinement_triangulation(),
                                        fluid.refinement_triangulation(),
                                        lift,
                                        parameters,
                                        radius,
                                        maximum_section_level,
                                        solid.communicator());
    solid.finalize_grid();
    if (parameters.embedded_partitioning_policy == "default")
      fluid.finalize_grid();
    else if (parameters.embedded_partitioning_policy == "immersed")
      {
        const ImmersedRepartitioner<1, 3> policy(solid.triangulation());
        fluid.finalize_grid(policy);
      }
    if (Utilities::MPI::this_mpi_process(solid.communicator()) == 0)
      std::cout << "Lift co-refinement: " << report.solid_cells
                << " solid cells, " << report.fluid_cells
                << " fluid cells, section level " << report.section_level
                << ", violations (axial/section lower, axial/section upper): "
                << report.lower_violations[0] << '/'
                << report.lower_violations[1] << ", "
                << report.upper_violations[0] << '/'
                << report.upper_violations[1] << ", outside points "
                << report.outside_points << '.' << std::endl;
    return report;
  }
} // namespace ImmersX
