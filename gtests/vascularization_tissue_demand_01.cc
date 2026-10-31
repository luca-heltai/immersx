// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/function_lib.h>
#include <deal.II/base/function_parser.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_description.h>

#include <gtest/gtest.h>

#include <limits>

#include "../source/vascularization/tissue_demand.h"
using namespace dealii;
using namespace ImmersX::Vascularization::internal;
namespace
{
  double
  integrate(const Triangulation<3>              &background,
            const Mapping<3>                    &mapping,
            decltype(ReferenceCells::Hexahedron) reference,
            const Function<3>                   &density,
            unsigned int                         copies)
  {
    FE_Nothing<3> element(reference);
    FEValues<3>   values(mapping,
                       element,
                       tissue_demand_quadrature(reference, 2, copies, 100000),
                       update_quadrature_points | update_JxW_values);
    double        local = 0.;
    for (const auto &cell : background.active_cell_iterators())
      if (cell->is_locally_owned())
        for (const auto &sample : sample_tissue_demand(cell, values, density))
          local += sample.mass;
    return Utilities::MPI::sum(local, MPI_COMM_WORLD);
  }
} // namespace
TEST(VascularTissueDemand,
     BOTH_PhysicalMassIsIndependentOfBackgroundAndIntegrationRefinement)
{
  parallel::distributed::Triangulation<3> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background);
  background.refine_global(1);
  FunctionParser<3> density;
  density.initialize("x,y,z", "1+3*x+2*y+z", {});
  for (unsigned int refinement = 0; refinement < 3; ++refinement)
    {
      for (const auto &cell : background.active_cell_iterators())
        if (cell->is_locally_owned() && cell->center()[0] < .5)
          cell->set_refine_flag();
      background.execute_coarsening_and_refinement();
      for (unsigned int copies : {1u, 2u, 4u})
        EXPECT_NEAR(integrate(background,
                              MappingQ1<3>(),
                              ReferenceCells::Hexahedron,
                              density,
                              copies),
                    4.,
                    2.e-12);
    }
}
TEST(VascularTissueDemand,
     BOTH_TetrahedronUsesPhysicalJxWAndEmptyRanksParticipate)
{
  Triangulation<3> serial;
  GridGenerator::reference_cell(serial, ReferenceCells::Tetrahedron);
  const auto description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      serial, MPI_COMM_WORLD);
  parallel::fullydistributed::Triangulation<3> background(MPI_COMM_WORLD);
  background.create_triangulation(description);
  const MappingFE<3> mapping(FE_SimplexP<3>(1));
  FunctionParser<3>  density;
  density.initialize("x,y,z", "1+3*x+2*y+z", {});
  for (unsigned int copies : {1u, 2u, 4u})
    EXPECT_NEAR(integrate(background,
                          mapping,
                          ReferenceCells::Tetrahedron,
                          density,
                          copies),
                5. / 12.,
                2.e-13);
}
TEST(VascularTissueDemand, QuadratureBudgetIsCheckedBeforeNativeSubdivision)
{
  for (const auto &reference :
       {ReferenceCells::Hexahedron, ReferenceCells::Tetrahedron})
    {
      EXPECT_THROW(tissue_demand_quadrature(reference, 2, 1u << 30, 100000),
                   ExceptionBase);
      EXPECT_THROW(tissue_demand_quadrature(reference, 2, 1, 0), ExceptionBase);
      EXPECT_THROW(tissue_demand_quadrature(reference, 2, 3, 100000),
                   ExceptionBase);
      EXPECT_THROW(tissue_demand_quadrature(reference, 0, 1, 100000),
                   ExceptionBase);
      EXPECT_THROW(tissue_demand_quadrature(reference, 9, 1, 100000),
                   ExceptionBase);
    }
}
TEST(VascularTissueDemand, NonpositiveOrUnrepresentablePhysicalMassFails)
{
  Triangulation<3> background;
  GridGenerator::hyper_cube(background);
  FE_Nothing<3> element;
  MappingQ1<3>  mapping;
  FEValues<3>   values(
    mapping,
    element,
    tissue_demand_quadrature(ReferenceCells::Hexahedron, 2, 1, 100),
    update_quadrature_points | update_JxW_values);
  for (double value : {0., -1., std::numeric_limits<double>::infinity()})
    EXPECT_THROW(sample_tissue_demand(background.begin_active(),
                                      values,
                                      Functions::ConstantFunction<3>(value)),
                 ExceptionBase);
}
