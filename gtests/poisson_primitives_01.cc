// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <gtest/gtest.h>
#include <immersx/core/boundary_conditions.h>
#include <immersx/core/domain.h>
#include <immersx/core/field_contributor.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/core/observable.h>
#include <immersx/core/weak_term.h>
#include <immersx/io/output_handler.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include "test_paths.h"

namespace
{
  using FieldVector  = ImmersX::ImmersXLA::MPI::Vector;
  using GlobalVector = ImmersX::ImmersXLA::MPI::BlockVector;
  using Adapter      = ImmersX::LinearAdapter<FieldVector, GlobalVector>;

  class SingleFieldState : public ImmersX::StateAccessor<FieldVector>
  {
  public:
    SingleFieldState(const ImmersX::FieldId field, const FieldVector &values)
      : field_(field)
      , values_(values)
    {}

    const FieldVector &
    field(const ImmersX::FieldId field, const double) const override
    {
      AssertThrow(field == field_, dealii::ExcMessage("Unexpected field."));
      return values_;
    }

  private:
    ImmersX::FieldId   field_;
    const FieldVector &values_;
  };
} // namespace

TEST(PoissonPrimitives, BOTH_SolveFromGenericPrimitives)
{
  using namespace ImmersX;

  dealii::ParameterAcceptor::clear();
  DomainParameters<2> domain_parameters("/Primitive domain/");
  domain_parameters.name_of_grid       = "hyper_cube";
  domain_parameters.arguments_for_grid = "-1: 1: false";
  domain_parameters.initial_refinement = 1;

  Domain<2> domain(domain_parameters);
  domain.make_grid();

  FiniteElementSpaceParameters<2> space_parameters("/Primitive space/");
  space_parameters.finite_element = "FE_Q<2>(1)";
  FiniteElementSpace<2> space(domain.triangulation(), space_parameters);
  set_constant_dirichlet_boundary_condition(space, 0, 1.);

  const auto view         = space.view();
  const auto unregistered = view.field("u");

  LinearAdapterParameters adapter_parameters("/Primitive adapter/");
  adapter_parameters.solver         = LinearSolver::iterative;
  adapter_parameters.preconditioner = LinearPreconditioner::none;
  adapter_parameters.tolerance      = 1.e-12;
  Adapter adapter(adapter_parameters, MPI_COMM_WORLD);

  const auto registration =
    adapter.add(algebraic_field(unregistered), "solution");
  const auto u = registration.fields();
  ASSERT_TRUE(u.is_registered());

  adapter.add(weak_term(gradient(u), gradient(test(u))), "laplace");

  auto state = adapter.make_state();
  adapter.solve(state);

  const auto &solution = adapter.field(state, u);
  ASSERT_TRUE(std::isfinite(solution.l2_norm()));
  double local_maximum_error = 0.;
  for (const auto index : solution.locally_owned_elements())
    local_maximum_error =
      std::max(local_maximum_error, std::abs(solution[index] - 1.));
  const double maximum_error =
    dealii::Utilities::MPI::max(local_maximum_error, MPI_COMM_WORLD);
  EXPECT_LT(maximum_error, 1.e-10);

  auto residual = adapter.make_state();
  adapter.evaluate_residual(state, residual);
  EXPECT_LT(residual.l2_norm(), 1.e-10);

  OutputHandlerParameters output_parameters("/Primitive output/");
  output_parameters.output_directory =
    TestPaths::output_directory("poisson-primitives");
  OutputHandler<2, 2, FieldVector> output(space.view(),
                                          output_parameters,
                                          "solution");
  output.add_field(u);
  SingleFieldState output_state(u.field_id(), solution);
  ASSERT_NO_THROW(output.write(output_state, 0.));
  MPI_Barrier(MPI_COMM_WORLD);

  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      const std::filesystem::path directory(output_parameters.output_directory);
      EXPECT_TRUE(std::filesystem::exists(directory / "solution.pvd"));
      std::string output_text;
      for (const auto &entry : std::filesystem::directory_iterator(directory))
        if (entry.path().extension() == ".vtu" ||
            entry.path().extension() == ".pvtu")
          {
            std::ifstream input(entry.path());
            output_text.append(std::istreambuf_iterator<char>(input), {});
          }
      EXPECT_NE(output_text.find("u"), std::string::npos);
    }
}
