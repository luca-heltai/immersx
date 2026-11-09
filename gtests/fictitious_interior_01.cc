// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>

#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/core/sundials_ida_adapter.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastodynamics_semidiscrete.h>
#include <immersx/physics/fictitious_interior.h>

#include <array>
#include <cmath>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

namespace
{
  using Strain = dealii::SymmetricTensor<2, 3>;

  ImmersX::ElastodynamicsMaterialCoefficients
  material()
  {
    ImmersX::ElastodynamicsMaterialCoefficients coefficients;
    coefficients.density       = 1.7;
    coefficients.lame_lambda   = 2.3;
    coefficients.lame_mu       = 1.4;
    coefficients.viscosity_eta = .1;
    coefficients.rayleigh_beta = .2;
    return coefficients;
  }

  Strain
  elastic_stress(const Strain                                      &strain,
                 const ImmersX::ElastodynamicsMaterialCoefficients &c)
  {
    return 2. * c.lame_mu * strain + c.lame_lambda * dealii::trace(strain) *
                                       dealii::unit_symmetric_tensor<3>();
  }

  Strain
  viscous_stress(const Strain                                      &strain,
                 const ImmersX::ElastodynamicsMaterialCoefficients &c)
  {
    return c.viscosity_eta * strain +
           c.rayleigh_beta * elastic_stress(strain, c);
  }

  Strain
  affine_strain(const double g, const double h)
  {
    Strain strain;
    strain[0][0] = strain[1][1] = g;
    strain[2][2]                = h;
    return strain;
  }

  double
  modal_quadratic(const dealii::FullMatrix<double> &matrix,
                  const double                      g,
                  const double                      h)
  {
    return matrix(0, 0) * g * g + (matrix(0, 1) + matrix(1, 0)) * g * h +
           matrix(1, 1) * h * h;
  }
} // namespace

TEST(FictitiousInterior, BOTH_AffineStressAndReciprocity)
{
  const auto   c       = material();
  const auto   elastic = ImmersX::FictitiousInteriorModes::elastic_hessian(c);
  const auto   viscous = ImmersX::FictitiousInteriorModes::viscous_hessian(c);
  const double g = .13, h = -.07, g_dot = .21, h_dot = .04;
  const auto   stress = elastic_stress(affine_strain(g, h), c) +
                      viscous_stress(affine_strain(g_dot, h_dot), c);
  const double radial_force = elastic(0, 0) * g + elastic(0, 1) * h +
                              viscous(0, 0) * g_dot + viscous(0, 1) * h_dot;
  const double axial_force = elastic(1, 0) * g + elastic(1, 1) * h +
                             viscous(1, 0) * g_dot + viscous(1, 1) * h_dot;
  EXPECT_NEAR(radial_force, stress[0][0] + stress[1][1], 1.e-14);
  EXPECT_NEAR(axial_force, stress[2][2], 1.e-14);
  EXPECT_DOUBLE_EQ(elastic(0, 1), elastic(1, 0));
  EXPECT_DOUBLE_EQ(viscous(0, 1), viscous(1, 0));
  EXPECT_NEAR(elastic(0, 0) * elastic(1, 1) - elastic(0, 1) * elastic(1, 0),
              4. * c.lame_mu * (3. * c.lame_lambda + 2. * c.lame_mu),
              1.e-13);
}

TEST(FictitiousInterior, BOTH_PlaneStrainReduction)
{
  const auto   c       = material();
  const auto   elastic = ImmersX::FictitiousInteriorModes::elastic_hessian(c);
  const auto   viscous = ImmersX::FictitiousInteriorModes::viscous_hessian(c);
  const double g = .17, g_dot = -.09;
  const auto   stress = elastic_stress(affine_strain(g, 0.), c) +
                      viscous_stress(affine_strain(g_dot, 0.), c);
  EXPECT_NEAR(.5 * (elastic(0, 0) * g + viscous(0, 0) * g_dot),
              stress[0][0],
              1.e-14);
  // Plane strain still has an axial reaction through the Lame cross term.
  EXPECT_NEAR(elastic(1, 0) * g + viscous(1, 0) * g_dot, stress[2][2], 1.e-14);
  EXPECT_GT(std::abs(stress[2][2]), .1);
}

TEST(FictitiousInterior, BOTH_ProjectedEnergyLeavesNonnegativeComplement)
{
  const auto                                      c = material();
  const std::array<dealii::FullMatrix<double>, 2> hessians{
    {ImmersX::FictitiousInteriorModes::elastic_hessian(c),
     ImmersX::FictitiousInteriorModes::viscous_hessian(c)}};
  const auto   comm  = MPI_COMM_WORLD;
  const auto   rank  = dealii::Utilities::MPI::this_mpi_process(comm);
  const auto   ranks = dealii::Utilities::MPI::n_mpi_processes(comm);
  std::mt19937 generator(48123);
  std::uniform_real_distribution<double> distribution(-1., 1.);
  dealii::Tensor<1, 3>                   tangent;
  tangent[0] = 1. / 3.;
  tangent[1] = 2. / 3.;
  tangent[2] = 2. / 3.;
  Strain axial;
  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = i; j < 3; ++j)
      axial[i][j] = tangent[i] * tangent[j];
  const auto radial = dealii::unit_symmetric_tensor<3>() - axial;

  for (unsigned int trial = 0; trial < 40; ++trial)
    {
      std::array<Strain, 9> strains;
      std::array<double, 9> weights;
      double                volume = 0., g_integral = 0., h_integral = 0.;
      for (unsigned int q = 0; q < strains.size(); ++q)
        {
          weights[q] = 1.1 + distribution(generator);
          for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = i; j < 3; ++j)
              strains[q][i][j] = distribution(generator);
          if (q % ranks == rank)
            {
              volume += weights[q];
              g_integral += .5 * weights[q] * (strains[q] * radial);
              h_integral += weights[q] * (strains[q] * axial);
            }
        }
      volume         = dealii::Utilities::MPI::sum(volume, comm);
      const double g = dealii::Utilities::MPI::sum(g_integral, comm) / volume;
      const double h = dealii::Utilities::MPI::sum(h_integral, comm) / volume;
      const auto   projected = g * radial + h * axial;
      for (unsigned int kind = 0; kind < 2; ++kind)
        {
          double total = 0., complement = 0.;
          for (unsigned int q = 0; q < strains.size(); ++q)
            if (q % ranks == rank)
              {
                const auto difference = strains[q] - projected;
                const auto stress = kind == 0 ? elastic_stress(strains[q], c) :
                                                viscous_stress(strains[q], c);
                const auto remaining = kind == 0 ?
                                         elastic_stress(difference, c) :
                                         viscous_stress(difference, c);
                total += weights[q] * (strains[q] * stress);
                complement += weights[q] * (difference * remaining);
              }
          total                = dealii::Utilities::MPI::sum(total, comm);
          complement           = dealii::Utilities::MPI::sum(complement, comm);
          const double removed = volume * modal_quadratic(hessians[kind], g, h);
          EXPECT_GE(total - removed, -1.e-12);
          EXPECT_NEAR(total - removed, complement, 2.e-12);
        }
    }
}

TEST(FictitiousInterior, BOTH_RadialOnlyRemovalCanLeaveNegativeEnergy)
{
  auto c        = material();
  c.lame_lambda = c.lame_mu = 1.;
  const double g = 1., h = -2. / 3.;
  const auto   strain      = affine_strain(g, h);
  const double full        = strain * elastic_stress(strain, c);
  const double radial_only = 4. * (c.lame_lambda + c.lame_mu) * g * g;
  EXPECT_NEAR(full - radial_only, -4. / 3., 1.e-14);
  EXPECT_LT(full - radial_only, 0.);
  const auto complete = ImmersX::FictitiousInteriorModes::elastic_hessian(c);
  EXPECT_NEAR(full - modal_quadratic(complete, g, h), 0., 1.e-14);
}

TEST(FictitiousInterior, BOTH_AxialVirtualWorkHasOppositeEndpointForces)
{
  const auto   c       = material();
  const auto   elastic = ImmersX::FictitiousInteriorModes::elastic_hessian(c);
  const auto   viscous = ImmersX::FictitiousInteriorModes::viscous_hessian(c);
  const double g = .13, h = -.07, g_dot = .21, h_dot = .04;
  const double q_z = elastic(1, 0) * g + elastic(1, 1) * h +
                     viscous(1, 0) * g_dot + viscous(1, 1) * h_dot;
  const double            area = .31, length = 1.7;
  const dealii::QGauss<1> quadrature(3);
  double                  weak_work = 0.;
  for (unsigned int q = 0; q < quadrature.size(); ++q)
    {
      const double s = length * quadrature.point(q)[0];
      // w_z(s) = 2 + s + s^3 exercises both ends and a varying test strain.
      weak_work +=
        area * q_z * (1. + 3. * s * s) * length * quadrature.weight(q);
    }
  const auto stress = elastic_stress(affine_strain(g, h), c) +
                      viscous_stress(affine_strain(g_dot, h_dot), c);
  const double left_force  = -area * stress[2][2];
  const double right_force = area * stress[2][2];
  EXPECT_NEAR(weak_work,
              left_force * 2. +
                right_force * (2. + length + std::pow(length, 3)),
              1.e-13);
  EXPECT_NEAR(left_force + right_force, 0., 1.e-14);
}


namespace
{
  class AffineDisplacement : public dealii::Function<3>
  {
  public:
    AffineDisplacement(const double g, const double h)
      : dealii::Function<3>(3)
      , g(g)
      , h(h)
    {}
    double
    value(const dealii::Point<3> &p,
          const unsigned int      component = 0) const override
    {
      return component == 2 ? h * p[2] : g * (p[component] - .5);
    }

  private:
    double g, h;
  };

  class FictitiousInteriorAssembly : public testing::Test
  {
  protected:
    using Problem = ImmersX::ElastodynamicsSolver<3>;
    using Vector  = typename Problem::VectorType;
    using Matrix  = typename Problem::MatrixType;
    void
    SetUp() override
    {
      dealii::ParameterAcceptor::clear();
      parameters = std::make_unique<ImmersX::ElastodynamicsParameters<3>>();
      ImmersX::initialize_parameters_from_string(R"(
        subsection Elastodynamics
          set Dirichlet boundary ids =
          subsection Domain
            set Grid generator = hyper_cube
            set Grid generator arguments = 0: 1: true
            set Initial refinement = 1
            set Triangulation type = distributed
          end
          subsection Material properties
            subsection default
              set Density = 1.7
              set Lame lambda = 2.3
              set Lame mu = 1.4
              set Viscosity eta = 0.1
              set Rayleigh beta = 0.2
            end
          end
          subsection Functions
            subsection Body force
              set Function expression = 1; 0.2; -0.3
            end
          end
        end
      )");
      problem = std::make_unique<Problem>(*parameters);
      problem->make_grid();
      problem->setup_fe();
      problem->setup_system();
      problem->assemble_operators();
      const ImmersX::FictitiousInteriorCylinder cylinder{
        dealii::Point<3>(.5, .5, 0.), dealii::Point<3>(.5, .5, 1.), .3};
      correction =
        std::make_unique<ImmersX::FictitiousInteriorCorrection<Vector, Matrix>>(
          *problem, *parameters, cylinder);
    }
    Vector
    affine(const double g, const double h) const
    {
      Vector vector;
      vector.reinit(problem->locally_owned_dofs(), MPI_COMM_WORLD);
      dealii::VectorTools::interpolate(problem->mapping(),
                                       problem->dof_handler(),
                                       AffineDisplacement(g, h),
                                       vector);
      return vector;
    }
    double
    work(const dealii::LinearOperator<Vector, Vector> &op,
         const Vector                                 &left,
         const Vector                                 &right) const
    {
      Vector product;
      product.reinit(problem->locally_owned_dofs(), MPI_COMM_WORLD);
      op.vmult(product, right);
      return left * product;
    }
    std::unique_ptr<ImmersX::ElastodynamicsParameters<3>> parameters;
    std::unique_ptr<Problem>                              problem;
    std::unique_ptr<ImmersX::FictitiousInteriorCorrection<Vector, Matrix>>
      correction;
  };
} // namespace

TEST_F(FictitiousInteriorAssembly, BOTH_AffineProjectionAndEnergyOracle)
{
  const double g = .13, h = -.07;
  const auto   u     = affine(g, h);
  const auto   modes = correction->modes(u);
  EXPECT_NEAR(modes[0], g, 1.e-14);
  EXPECT_NEAR(modes[1], h, 1.e-14);
  const auto   eps     = affine_strain(g, h);
  const double density = eps * elastic_stress(eps, material());
  EXPECT_NEAR(correction->removed_elastic_energy(u),
              .5 * correction->volume() * density,
              1.e-13);
  EXPECT_NEAR(work(correction->elastic_operator().view, u, u),
              correction->volume() * density,
              1.e-13);
  EXPECT_NEAR(
    work(ImmersX::matrix_operator<Vector>(problem->stiffness_matrix()).view,
         u,
         u) -
      work(correction->elastic_operator().view, u, u),
    (1. - correction->volume()) * density,
    1.e-12);
  EXPECT_NEAR(
    work(ImmersX::matrix_operator<Vector>(problem->mass_matrix()).view, u, u),
    material().density * (g * g / 6. + h * h / 3.),
    1.e-13);
  const auto velocity    = affine(.21, .04);
  const auto strain_rate = affine_strain(.21, .04);
  EXPECT_NEAR(correction->removed_viscous_dissipation(velocity),
              correction->volume() *
                (strain_rate * viscous_stress(strain_rate, material())),
              1.e-13);
  EXPECT_NEAR(correction->reference_volume(),
              dealii::numbers::PI * .09,
              1.e-14);
}

TEST_F(FictitiousInteriorAssembly, BOTH_RemainingOperatorsAndModalAdjoint)
{
  Vector     u = affine(0., 0.), w = affine(0., 0.);
  const auto k =
    ImmersX::matrix_operator<Vector>(problem->stiffness_matrix()).view;
  const auto d =
    ImmersX::matrix_operator<Vector>(problem->damping_matrix()).view;
  const std::array<ImmersX::MaterializedOperator<Vector, Matrix>, 2> operators{
    {correction->elastic_operator(), correction->viscous_operator()}};
  const auto ce = operators[0].view;
  const auto cv = operators[1].view;
  for (unsigned int trial = 0; trial < 12; ++trial)
    {
      for (const auto i : problem->locally_owned_dofs())
        {
          u[i] = std::sin(.71 * (i + 1) * (trial + 1));
          w[i] = std::cos(.33 * (i + 1) * (trial + 2));
        }
      EXPECT_GE(work(k, u, u) - work(ce, u, u), -1.e-11);
      EXPECT_GE(work(d, u, u) - work(cv, u, u), -1.e-11);
      EXPECT_NEAR(work(ce, w, u), work(ce, u, w), 1.e-12);
      EXPECT_NEAR(work(cv, w, u), work(cv, u, w), 1.e-12);
    }
  for (const auto &op : operators)
    {
      Vector     factored = affine(0., 0.), assembled = affine(0., 0.);
      const auto matrix = op.matrix();
      op.view.vmult(factored, u);
      matrix->vmult(assembled, u);
      factored -= assembled;
      EXPECT_NEAR(factored.l2_norm(), 0., 1.e-12);
      op.view.Tvmult(factored, w);
      matrix->Tvmult(assembled, w);
      factored -= assembled;
      EXPECT_NEAR(factored.l2_norm(), 0., 1.e-12);
    }
  // Returned operators own their moments even after the builder is destroyed.
  const auto detached = [&]() {
    const ImmersX::FictitiousInteriorCylinder cylinder{
      dealii::Point<3>(.5, .5, 0.), dealii::Point<3>(.5, .5, 1.), .3};
    const ImmersX::FictitiousInteriorCorrection<Vector, Matrix> temporary(
      *problem, *parameters, cylinder);
    return temporary.elastic_operator();
  }();
  EXPECT_NEAR(work(detached.view, w, u), work(ce, w, u), 1.e-12);
  // Axial test w=(0,0,z) has no radial virtual strain. Its work equals the
  // two opposite end forces with sampled area V_h/L, not a lateral pressure.
  const auto state      = affine(.13, -.07);
  const auto axial_test = affine(0., 1.);
  const auto stress     = elastic_stress(affine_strain(.13, -.07), material());
  EXPECT_NEAR(work(ce, axial_test, state),
              correction->volume() * stress[2][2],
              1.e-13);
}


TEST_F(FictitiousInteriorAssembly, BOTH_RemainingOperatorEigenvalues)
{
  const std::array<dealii::LinearOperator<Vector, Vector>, 2> remaining{
    {ImmersX::matrix_operator<Vector>(problem->stiffness_matrix()).view -
       correction->elastic_operator().view,
     ImmersX::matrix_operator<Vector>(problem->damping_matrix()).view -
       correction->viscous_operator().view}};
  const auto size = problem->locally_owned_dofs().size();
  for (const auto &op : remaining)
    {
      dealii::LAPACKFullMatrix<double> dense(size, size);
      Vector basis = affine(0., 0.), column = affine(0., 0.);
      for (unsigned int j = 0; j < size; ++j)
        {
          basis = 0.;
          if (problem->locally_owned_dofs().is_element(j))
            basis[j] = 1.;
          basis.compress(dealii::VectorOperation::insert);
          op.vmult(column, basis);
          std::vector<double> entries(size, 0.);
          for (const auto i : problem->locally_owned_dofs())
            entries[i] = column[i];
          dealii::Utilities::MPI::sum(
            dealii::ArrayView<const double>(entries.data(), entries.size()),
            MPI_COMM_WORLD,
            dealii::make_array_view(entries));
          for (unsigned int i = 0; i < size; ++i)
            dense(i, j) = entries[i];
        }
      for (unsigned int i = 0; i < size; ++i)
        for (unsigned int j = 0; j < size; ++j)
          ASSERT_NEAR(dense(i, j), dense(j, i), 1.e-13);
      dealii::Vector<double>     eigenvalues;
      dealii::FullMatrix<double> eigenvectors;
      dense.compute_eigenvalues_symmetric(
        -1.e6, 1.e6, 1.e-13, eigenvalues, eigenvectors);
      ASSERT_EQ(eigenvalues.size(), size);
      EXPECT_GE(eigenvalues[0], -1.e-11);
      EXPECT_NEAR(eigenvalues[5], 0., 1.e-11);
      EXPECT_GT(eigenvalues[6], 1.e-10);
      EXPECT_GT(eigenvalues[size - 1], .01);
    }
}


TEST_F(FictitiousInteriorAssembly, BOTH_RejectHangingNodeConstraints)
{
  Problem refined(*parameters);
  refined.make_grid();
  // Disable island removal so the single refined cell creates hanging nodes.
  // This owned fixture is changed before distributing DoFs.
  auto &mesh = const_cast<dealii::parallel::TriangulationBase<3> &>(
    refined.triangulation());
  mesh.set_mesh_smoothing(
    dealii::Triangulation<3>::limit_level_difference_at_vertices);
  unsigned int marked = 0;
  for (const auto &cell : refined.triangulation().active_cell_iterators())
    if (cell->is_locally_owned() && cell->center()[0] < .5 &&
        cell->center()[1] < .5 && cell->center()[2] < .5)
      {
        cell->set_refine_flag();
        ++marked;
      }
  ASSERT_EQ(dealii::Utilities::MPI::sum(marked, MPI_COMM_WORLD), 1u);
  mesh.execute_coarsening_and_refinement();
  ASSERT_EQ(refined.triangulation().n_global_active_cells(), 15u);
  refined.setup_fe();
  refined.setup_system();
  const auto constraint_count =
    dealii::Utilities::MPI::sum(refined.constraints().n_constraints(),
                                MPI_COMM_WORLD);
  ASSERT_GT(constraint_count, 0u);
  const ImmersX::FictitiousInteriorCylinder cylinder{
    dealii::Point<3>(.5, .5, 0.), dealii::Point<3>(.5, .5, 1.), .3};
  using Correction = ImmersX::FictitiousInteriorCorrection<Vector, Matrix>;
  EXPECT_THROW(Correction(refined, *parameters, cylinder),
               dealii::ExceptionBase);
}

#ifdef DEAL_II_WITH_SUNDIALS
TEST_F(FictitiousInteriorAssembly, BOTH_RetainedMassAndStateJacobian)
{
  using GlobalVector = ImmersX::LA::MPI::BlockVector;
  using Adapter      = ImmersX::IDAAdapter<Vector, GlobalVector>;
  ImmersX::TimeIntervalParameters time;
  ImmersX::IDAParameters          ida_parameters;
  Adapter                         adapter(time,
                  ida_parameters,
                  MPI_COMM_WORLD,
                  [](const dealii::LinearOperator<GlobalVector> &,
                     const GlobalVector &,
                     GlobalVector &,
                     const double) {});
  const auto                      solid = adapter.add(*problem, "solid");
  const auto                      uid   = solid.fields().displacement.id();
  const auto                      vid   = solid.fields().velocity.id();
  const auto                      ce    = correction->elastic_operator();
  const auto                      cv    = correction->viscous_operator();
  adapter.add(
    [&](auto &builder) {
      builder.term(vid, "interior-modes")
        .residual([=](const auto &context) {
          return -1. *
                 (ce.view * context.state(uid) + cv.view * context.state(vid));
        })
        .state(uid, -1. * ce)
        .state(vid, -1. * cv);
      return std::tuple<>();
    },
    "interior");
  auto state                     = adapter.make_state();
  auto derivative                = adapter.make_state();
  auto residual                  = adapter.make_state();
  adapter.field(state, uid)      = affine(.13, -.07);
  adapter.field(state, vid)      = affine(.21, .04);
  adapter.field(derivative, uid) = affine(.21, .04);
  adapter.field(derivative, vid) = affine(.31, -.12);
  adapter.solver().residual(0., state, derivative, residual);
  const auto baseline               = residual;
  const auto acceleration_increment = affine(.09, -.16);
  adapter.field(derivative, vid) += acceleration_increment;
  adapter.solver().residual(0., state, derivative, residual);
  auto observed = adapter.field(residual, vid);
  observed -= adapter.field(baseline, vid);
  Vector expected;
  expected.reinit(problem->locally_owned_dofs(), MPI_COMM_WORLD);
  problem->mass_matrix().vmult(expected, acceleration_increment);
  EXPECT_GT(expected.l2_norm(), 1.e-3);
  observed -= expected;
  EXPECT_NEAR(observed.l2_norm(), 0., 1.e-12);

  auto increment                = adapter.make_state();
  adapter.field(increment, uid) = affine(-.11, .03);
  adapter.field(increment, vid) = affine(.07, .09);
  auto         jacobian_action  = adapter.make_state();
  const double alpha            = 1.7;
  adapter.solver().setup_jacobian(0., state, derivative, alpha);
  adapter.current_jacobian().vmult(jacobian_action, increment);
  const double step = 1.e-6;
  adapter.solver().residual(0., state, derivative, residual);
  const auto base_residual = residual;
  state.add(step, increment);
  derivative.add(alpha * step, increment);
  adapter.solver().residual(0., state, derivative, residual);
  residual -= base_residual;
  residual *= 1. / step;
  residual -= jacobian_action;
  EXPECT_NEAR(residual.l2_norm(), 0., 2.e-9);

  Adapter    iterative(time, ida_parameters, MPI_COMM_WORLD);
  const auto iterative_solid  = iterative.add(*problem, "iterative-solid");
  const auto iterative_uid    = iterative_solid.fields().displacement.id();
  const auto iterative_vid    = iterative_solid.fields().velocity.id();
  const auto factored_elastic = ce.view;
  const auto factored_viscous = cv.view;
  iterative.add(
    [&](auto &builder) {
      builder.term(iterative_vid, "factored-interior")
        .residual([=](const auto &context) {
          return -1. * (factored_elastic * context.state(iterative_uid) +
                        factored_viscous * context.state(iterative_vid));
        })
        .state(iterative_uid, -1. * factored_elastic)
        .state(iterative_vid, -1. * factored_viscous);
      builder.second_order_preconditioner(
        iterative_uid,
        ImmersX::matrix_operator<Vector>(problem->mass_matrix()),
        ImmersX::matrix_operator<Vector>(problem->stiffness_matrix()),
        ImmersX::matrix_operator<Vector>(problem->damping_matrix()));
      return std::tuple<>();
    },
    "factored-interior");
  auto iterative_state                            = iterative.make_state();
  auto iterative_derivative                       = iterative.make_state();
  iterative.field(iterative_state, iterative_uid) = affine(.13, -.07);
  iterative.field(iterative_state, iterative_vid) = affine(.21, .04);
  iterative.solver().setup_jacobian(0.,
                                    iterative_state,
                                    iterative_derivative,
                                    alpha);
  EXPECT_TRUE(iterative.has_current_preconditioner());
  auto iterative_increment = iterative.make_state();
  auto iterative_action    = iterative.make_state();
  iterative.field(iterative_increment, iterative_uid) =
    adapter.field(increment, uid);
  iterative.field(iterative_increment, iterative_vid) =
    adapter.field(increment, vid);
  iterative.current_jacobian().vmult(iterative_action, iterative_increment);
  iterative_action -= jacobian_action;
  EXPECT_NEAR(iterative_action.l2_norm(), 0., 1.e-12);
}
#endif
