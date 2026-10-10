// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// The ImmersX application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the file LICENSE.md at the top level
// of the ImmersX distribution.
//
// ---------------------------------------------------------------------


#include <deal.II/base/logstream.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <Epetra_MultiVector.h>
#include <Teuchos_ParameterList.hpp>
#include <immersx/io/utils.h>
#include <immersx/physics/elasticity_freq.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>

namespace ImmersX
{
  namespace
  {

    // Class declaration
    template <int spacedim>
    class RigidBodyMotion : public Function<spacedim>
    {
    public:
      explicit RigidBodyMotion(const unsigned int type);

      double
      value(const Point<spacedim> &p,
            const unsigned int     component = 0) const override;

    private:
      const unsigned int type;
    };

    template <int spacedim>
    RigidBodyMotion<spacedim>::RigidBodyMotion(const unsigned int _type)
      : Function<spacedim>(spacedim)
      , type(_type)
    {
      Assert(spacedim == 2 || spacedim == 3, ExcNotImplemented());
      Assert((spacedim == 2 && type <= 2) || (spacedim == 3 && type <= 5),
             ExcNotImplemented());
    }

    template <int spacedim>
    double
    RigidBodyMotion<spacedim>::value(const Point<spacedim> &p,
                                     const unsigned int     component) const
    {
      if constexpr (spacedim == 2)
        {
          const std::array<double, 3> modes{
            {static_cast<double>(component == 0),
             static_cast<double>(component == 1),
             (component == 0) ? -p[1] :
             (component == 1) ? p[0] :
                                0.0}};
          return modes[type];
        }
      else
        {
          const std::array<double, 6> modes{
            {static_cast<double>(component == 0),
             static_cast<double>(component == 1),
             static_cast<double>(component == 2),
             (component == 0) ? 0.0 :
             (component == 1) ? p[2] :
                                -p[1],
             (component == 0) ? -p[2] :
             (component == 1) ? 0.0 :
                                p[0],
             (component == 0) ? p[1] :
             (component == 1) ? -p[0] :
                                0.0}};
          return modes[type];
        }
    }

    template <typename Operator>
    class SplitDisplacementAppl
    { // for surrogate matrix and AMG application only, sr,si  are source real
      // and imaginary di and dr represent destination vectors
    public:
      SplitDisplacementAppl(const Operator &op,
                            const IndexSet &owned,
                            MPI_Comm        comm)
        : op(op)
        , owned(owned)
      {
        sr.reinit(owned, comm);
        si.reinit(owned, comm);
        dr.reinit(owned, comm);
        di.reinit(owned, comm);
      }
      void
      vmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) const
      {
        for (const auto i : owned)
          {
            sr[i] = src[2 * i];
            si[i] = src[2 * i + 1];
          }
        sr.compress(VectorOperation::insert);
        si.compress(VectorOperation::insert);
        op.vmult(dr, sr);
        op.vmult(di, si);
        for (const auto i : owned)
          {
            dst[2 * i]     = dr[i];
            dst[2 * i + 1] = di[i];
          }
        dst.compress(VectorOperation::insert);
      }

    private:
      const Operator         &op;
      const IndexSet          owned;
      mutable LA::MPI::Vector sr, si, dr, di;
    };

    using ComplexCoefficient = std::complex<double>;

    ComplexCoefficient
    complex_dot(const LA::MPI::Vector &a,
                const LA::MPI::Vector &b,
                const IndexSet        &owned,
                const MPI_Comm         comm)
    { // conugated vectordot product for projection
      double real = 0.0, imag = 0.0;
      for (const auto i : owned)
        {
          real += a[2 * i] * b[2 * i] + a[2 * i + 1] * b[2 * i + 1];
          imag += a[2 * i] * b[2 * i + 1] - a[2 * i + 1] * b[2 * i];
        }
      return {Utilities::MPI::sum(real, comm), Utilities::MPI::sum(imag, comm)};
    }

    void
    complex_add(LA::MPI::Vector         &dst,
                const ComplexCoefficient factor,
                const LA::MPI::Vector   &src,
                const IndexSet          &owned)
    { // updates vector
      for (const auto i : owned)
        {
          const double real = src[2 * i], imag = src[2 * i + 1];
          dst[2 * i] += factor.real() * real - factor.imag() * imag;
          dst[2 * i + 1] += factor.imag() * real + factor.real() * imag;
        }
      dst.compress(VectorOperation::insert);
    }

    template <typename Operator, typename Preconditioner>
    void
    complex_fgmres(const Operator        &op,
                   LA::MPI::Vector       &x,
                   const LA::MPI::Vector &b,
                   const Preconditioner  &prec,
                   const IndexSet        &owned,
                   const MPI_Comm         comm,
                   SolverControl         &control,
                   const bool             allow_iteration_limit,
                   const bool             print_progress = false)
    {
      using C                        = ComplexCoefficient;
      constexpr unsigned int restart = 30; // limit search cycle to 30
                                           // directions
      const unsigned int limit  = control.max_steps();
      const double       target = control.tolerance();
      const bool         print_on_this_rank =
        print_progress && Utilities::MPI::this_mpi_process(comm) == 0;

      auto print_iteration = [&](const unsigned int step,
                                 const double       residual_estimate) {
        if (!print_on_this_rank)
          return;

        std::ostringstream line;
        line << "Outer iteration " << step
             << " | residual estimate: " << std::scientific
             << std::setprecision(6) << residual_estimate;

        std::cout << line.str() << std::endl;
      };
      LA::MPI::Vector residual, work;
      residual.reinit(b);
      work.reinit(b);

      auto true_residual = [&]() {
        op.vmult(residual, x);
        residual.sadd(-1.0, 1.0, b);
        const double norm = residual.l2_norm();
        AssertThrow(std::isfinite(norm),
                    ExcMessage("Non-finite complex FGMRES residual."));
        return norm;
      };
      unsigned int iterations = 0;
      double       beta       = true_residual();
      control.check(iterations, beta);


      if (beta <= target)
        return;
      while (iterations < limit)
        {
          const unsigned int length = std::min(restart, limit - iterations);

          std::vector<std::unique_ptr<LA::MPI::Vector>> basis, corrections;
          basis.reserve(length + 1);   // arnoldi vectors
          corrections.reserve(length); // preconditioned vectors for update

          auto first = std::make_unique<LA::MPI::Vector>();
          first->reinit(b);
          *first = residual;
          *first *= 1.0 / beta;
          basis.emplace_back(std::move(first)); // normalized

          std::vector<std::vector<C>> h(length + 1,
                                        std::vector<C>(length, C(0.0)));
          std::vector<double>         cosine(length, 0.0);
          std::vector<C> sine(length, C(0.0)), g(length + 1, C(0.0));
          g[0] = beta;

          unsigned int used = 0;
          bool         arnoldi_breakdown =
            false; // check if no usable new direction remains
          for (unsigned int j = 0; j < length; ++j)
            {
              auto z = std::make_unique<LA::MPI::Vector>();
              z->reinit(b);
              *z = 0.0;
              prec.vmult(*z, *basis[j]); // z_j=prec(v_j)

              AssertThrow(std::isfinite(z->l2_norm()),
                          ExcMessage(
                            "Non-finite complex FGMRES correction.")); // check

              op.vmult(work, *z);
              corrections.emplace_back(std::move(z));
              const double before = work.l2_norm();
              AssertThrow(std::isfinite(before),
                          ExcMessage("Non-finite complex Arnoldi vector."));
              // orthogonalize
              for (unsigned int pass = 0; pass < 2; ++pass)
                for (unsigned int i = 0; i <= j; ++i)
                  {
                    const C projection =
                      complex_dot(*basis[i], work, owned, comm);
                    h[i][j] += projection;
                    complex_add(work, -projection, *basis[i], owned);
                  }

              const double next_norm = work.l2_norm(); // next direction
              AssertThrow(std::isfinite(next_norm),
                          ExcMessage("Non-finite complex Arnoldi norm."));
              h[j + 1][j] = next_norm;
              arnoldi_breakdown =
                next_norm <=
                100.0 * std::numeric_limits<double>::epsilon() * before;
              if (!arnoldi_breakdown)
                {
                  auto next = std::make_unique<LA::MPI::Vector>();
                  next->reinit(b);
                  *next = work;
                  *next *= 1.0 / next_norm;
                  basis.emplace_back(std::move(next));
                }
              for (unsigned int i = 0; i < j; ++i)
                {
                  const C top = cosine[i] * h[i][j] + sine[i] * h[i + 1][j];
                  h[i + 1][j] =
                    -std::conj(sine[i]) * h[i][j] + cosine[i] * h[i + 1][j];
                  h[i][j] = top;
                }

              const C a     = h[j][j],
                      lower = h[j + 1][j]; // move ot upper triangular form
              const double abs_a = std::abs(a), abs_lower = std::abs(lower);
              const double rotation_norm = std::hypot(abs_a, abs_lower);
              AssertThrow(rotation_norm > 0.0 && std::isfinite(rotation_norm),
                          ExcMessage(
                            "Singular complex FGMRES least-squares step."));
              if (abs_a == 0.0)
                {
                  cosine[j] = 0.0;
                  sine[j]   = std::conj(lower) / abs_lower;
                  h[j][j]   = abs_lower;
                }
              else
                {
                  const C phase = a / abs_a;
                  cosine[j]     = abs_a / rotation_norm;
                  sine[j]       = phase * std::conj(lower) / rotation_norm;
                  h[j][j]       = phase * rotation_norm;
                }
              h[j + 1][j] = 0.0;

              const C old_g = g[j]; // also apply on rhs
              g[j]          = cosine[j] * old_g;
              g[j + 1]      = -std::conj(sine[j]) * old_g;
              ++iterations;
              used = j + 1;

              const double estimate = std::abs(g[j + 1]);

              control.check(iterations, estimate);
              print_iteration(iterations, estimate);
              if (estimate <= target || arnoldi_breakdown)
                break;
            }

          std::vector<C> coefficients(used, C(0.0));
          for (int i = static_cast<int>(used) - 1; i >= 0;
               --i) // back substitution
            {
              C rhs = g[i];
              for (unsigned int k = i + 1; k < used; ++k)
                rhs -= h[i][k] * coefficients[k];
              AssertThrow(std::abs(h[i][i]) > 0.0,
                          ExcMessage("Singular Arnoldi factor."));
              coefficients[i] = rhs / h[i][i];
            }

          for (unsigned int j = 0; j < used; ++j)
            complex_add(x,
                        coefficients[j],
                        *corrections[j],
                        owned); // update solution vector x
          beta = true_residual();
          control.check(iterations, beta);
          if (beta <= target)
            return;
          if (arnoldi_breakdown)
            throw SolverControl::NoConvergence(iterations, beta);
        }
      if (!allow_iteration_limit)
        throw SolverControl::NoConvergence(iterations, beta);
    }

    template <typename Operator, typename Preconditioner>
    class SurrogateInverse
    { // approx solution with surrogate equation
    public:
      SurrogateInverse(const Operator       &op,
                       const Preconditioner &prec,
                       const IndexSet       &owned,
                       MPI_Comm              comm)
        : op(op)
        , prec(prec)
        , owned(owned)
        , comm(comm)
      {}
      void
      vmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) const
      {
        dst = 0.0;

        const double norm = src.l2_norm();
        if (norm == 0.0)
          return;

        AssertThrow(std::isfinite(norm), ExcMessage("Non-finite inner RHS."));
        SolverControl inner_control(30, 0.1 * norm, false, false);

        complex_fgmres(op, dst, src, prec, owned, comm, inner_control, true);

        AssertThrow(std::isfinite(dst.l2_norm()),
                    ExcMessage("Non-finite inner correction."));
      }

    private:
      const Operator       &op;
      const Preconditioner &prec;
      const IndexSet        owned;
      const MPI_Comm        comm;
    };
    // reading BC and converting to complex numebr
    template <typename Map, typename Key>
    double
    phase_factor(const Map   &phases,
                 const Key    id,
                 const double fallback,
                 const bool   imaginary)
    {
      const auto   it      = phases.find(id);
      const double degrees = it == phases.end() ? fallback : it->second;
      const double radians = degrees * std::acos(-1.0) / 180.0;
      return imaginary ? std::sin(radians) : std::cos(radians);
    }
    template <int spacedim>
    class ScaledAmplitude : public Function<spacedim>
    {
    public:
      ScaledAmplitude(const Function<spacedim> &amplitude, const double factor)
        : Function<spacedim>(amplitude.n_components)
        , amplitude(amplitude)
        , factor(factor)
      {}
      double
      value(const Point<spacedim> &p,
            const unsigned int     component = 0) const override
      {
        return factor * amplitude.value(p, component);
      }

    private:
      const Function<spacedim> &amplitude;
      const double              factor;
    };
  } // namespace
  template <int dim, int spacedim>
  ElasticityFreqProblem<dim, spacedim>::ElasticityFreqProblem(
    const ElasticityFreqProblemParameters<dim, spacedim> &par)
    : par(par)
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
    , triangulation_storage(std::in_place_type<DistributedTriangulation>,
                            mpi_communicator,
                            typename Triangulation<spacedim>::MeshSmoothing(
                              Triangulation<spacedim>::smoothing_on_refinement |
                              Triangulation<spacedim>::smoothing_on_coarsening),
                            parallel::distributed::Triangulation<
                              spacedim>::construct_multigrid_hierarchy)
    , tria(&std::get<DistributedTriangulation>(triangulation_storage))
    , dh()
    , displacement(0)
  {}

  template <int dim, int spacedim>
  bool
  ElasticityFreqProblem<dim, spacedim>::uses_fully_distributed_triangulation()
    const
  {
    return std::holds_alternative<FullyDistributedTriangulation>(
      triangulation_storage);
  }

  template <int dim, int spacedim>
  void
  ElasticityFreqProblem<dim, spacedim>::make_grid()
  {
    const bool need_fully_distributed =
      (par.triangulation_type == "fullydistributed");

    if (need_fully_distributed && !uses_fully_distributed_triangulation())
      triangulation_storage.template emplace<FullyDistributedTriangulation>(
        mpi_communicator);
    else if (!need_fully_distributed && uses_fully_distributed_triangulation())
      triangulation_storage.template emplace<DistributedTriangulation>(
        mpi_communicator,
        typename Triangulation<spacedim>::MeshSmoothing(
          Triangulation<spacedim>::smoothing_on_refinement |
          Triangulation<spacedim>::smoothing_on_coarsening),
        parallel::distributed::Triangulation<
          spacedim>::construct_multigrid_hierarchy);

    tria = &std::visit(
      [](auto &selected_tria) -> parallel::TriangulationBase<spacedim> & {
        return selected_tria;
      },
      triangulation_storage);

    dh.reinit(*tria);

    if (!uses_fully_distributed_triangulation())
      {
        auto &distributed_tria =
          std::get<DistributedTriangulation>(triangulation_storage);

        if (par.domain_type == "generate")
          {
            try
              {
                GridGenerator::generate_from_name_and_arguments(
                  distributed_tria, par.name_of_grid, par.arguments_for_grid);
              }
            catch (...)
              {
                pcout << "Generating from name and argument failed."
                      << std::endl
                      << "Trying to read from file name." << std::endl;
                read_grid_and_cad_files(par.name_of_grid,
                                        par.arguments_for_grid,
                                        distributed_tria);
              }
          }
        else if (par.domain_type == "cylinder")
          {
            Assert(spacedim == 2, ExcInternalError());
            GridGenerator::hyper_ball(distributed_tria, Point<spacedim>(), 1.);
            std::cout << " ATTENTION: GRID: cirle of radius 1." << std::endl;
          }
        else if (par.domain_type == "cheese")
          {
            Assert(spacedim == 2, ExcInternalError());
            GridGenerator::cheese(distributed_tria,
                                  std::vector<unsigned int>(2, 2));
          }
        else if (par.domain_type == "file")
          {
            GridIn<spacedim> gi;
            gi.attach_triangulation(distributed_tria);
#ifdef DEAL_II_WITH_GMSH_API
            std::string infile(par.name_of_grid);
#else
            std::ifstream infile(par.name_of_grid);
            Assert(infile.good(), ExcIO());
#endif
            try
              {
                gi.read_msh(infile);
              }
            catch (...)
              {
                // Try standard readers if msh reader fails, in case the file is
                // not actually a file that msh understands
                gi.read(par.name_of_grid);
              }
          }

        if (par.grid_scale != 1.0)
          GridTools::scale(par.grid_scale, distributed_tria);

        distributed_tria.refine_global(par.initial_refinement);
        return;
      }

    Triangulation<spacedim> serial_tria(
      typename Triangulation<spacedim>::MeshSmoothing(
        Triangulation<spacedim>::smoothing_on_refinement |
        Triangulation<spacedim>::smoothing_on_coarsening));

    if (par.domain_type == "generate")
      {
        try
          {
            GridGenerator::generate_from_name_and_arguments(
              serial_tria, par.name_of_grid, par.arguments_for_grid);
          }
        catch (...)
          {
            pcout << "Generating from name and argument failed." << std::endl
                  << "Trying to read from file name." << std::endl;
            read_grid_and_cad_files(par.name_of_grid,
                                    par.arguments_for_grid,
                                    serial_tria);
          }
      }
    else if (par.domain_type == "cylinder")
      {
        Assert(spacedim == 2, ExcInternalError());
        GridGenerator::hyper_ball(serial_tria, Point<spacedim>(), 1.);
        std::cout << " ATTENTION: GRID: cirle of radius 1." << std::endl;
      }
    else if (par.domain_type == "cheese")
      {
        Assert(spacedim == 2, ExcInternalError());
        GridGenerator::cheese(serial_tria, std::vector<unsigned int>(2, 2));
      }
    else if (par.domain_type == "file")
      {
        GridIn<spacedim> gi;
        gi.attach_triangulation(serial_tria);
#ifdef DEAL_II_WITH_GMSH_API
        std::string infile(par.name_of_grid);
#else
        std::ifstream infile(par.name_of_grid);
        Assert(infile.good(), ExcIO());
#endif
        try
          {
            gi.read_msh(infile);
            // gi.read_vtk(infile);
          }
        catch (...)
          {
            // Try standard readers if msh reader fails, in case the file is not
            // actually a file that msh understands
            gi.read(par.name_of_grid);
          }
      }

    if (par.grid_scale != 1.0)
      GridTools::scale(par.grid_scale, serial_tria);

    serial_tria.refine_global(par.initial_refinement);
    auto &fully_distributed_tria =
      std::get<FullyDistributedTriangulation>(triangulation_storage);
    for (const auto manifold_id : serial_tria.get_manifold_ids())
      if (manifold_id != numbers::flat_manifold_id)
        fully_distributed_tria.set_manifold(
          manifold_id, serial_tria.get_manifold(manifold_id));

    fully_distributed_tria.copy_triangulation(serial_tria);

    pcout << "Number of active cells: " << tria->n_active_cells() << std::endl
          << "   Boundary ids: "
          << Patterns::Tools::to_string(tria->get_boundary_ids()) << std::endl;

    std::set<types::material_id> material_ids;
    for (const auto &cell : tria->active_cell_iterators())
      material_ids.insert(cell->material_id());
    pcout << "   Material ids: " << Patterns::Tools::to_string(material_ids)
          << std::endl;

    pcout << "   Grid volume: " << GridTools::volume(*tria) << std::endl
          << "   Dirichlet boundary ids: "
          << Patterns::Tools::to_string(par.dirichlet_ids) << std::endl
          << "   Weak Dirichlet boundary ids: "
          << Patterns::Tools::to_string(par.weak_dirichlet_ids) << std::endl
          << "   Neumann boundary ids: "
          << Patterns::Tools::to_string(par.neumann_ids) << std::endl;
  }

  template <int dim, int spacedim>
  void
  ElasticityFreqProblem<dim, spacedim>::setup_fe()
  {
    TimerOutput::Scope t(computing_timer, "Initial setup");
    fe = std::make_unique<FESystem<spacedim>>(FE_Q<spacedim>(par.fe_degree),
                                              spacedim);
    quadrature = std::make_unique<QGauss<spacedim>>(par.fe_degree + 1);
    face_quadrature_formula =
      std::make_unique<QGauss<spacedim - 1>>(par.fe_degree + 1);
  }

  template <int dim, int spacedim>
  void
  ElasticityFreqProblem<dim, spacedim>::print_parameters() const
  {
#ifdef USE_PETSC_LA
    pcout << "Running ElasticityFreqProblem<"
          << Utilities::dim_string(dim, spacedim) << "> using PETSc."
          << std::endl;
#else
    pcout << "Running ElasticityFreqProblem<"
          << Utilities::dim_string(dim, spacedim) << "> using Trilinos."
          << std::endl;
#endif
    pcout << "   Triangulation backend: " << par.triangulation_type
          << std::endl;
    par.prm.print_parameters(par.output_directory + "/" + par.output_name +
                               "_" + std::to_string(dim) +
                               std::to_string(spacedim) + ".prm",
                             ParameterHandler::Short);
    par.prm.print_parameters(par.output_directory + "/" + par.output_name +
                               "_full_" + std::to_string(dim) +
                               std::to_string(spacedim) + ".prm",
                             ParameterHandler::PRM);
#if DEAL_II_VERSION_GTE(9, 7, 0)
    par.prm.print_parameters(par.output_directory + "/" + par.output_name +
                               "_changed_" + std::to_string(dim) +
                               std::to_string(spacedim) + ".prm",
                             ParameterHandler::KeepOnlyChanged |
                               ParameterHandler::Short);
#endif
  }

  template <int dim, int spacedim>
  void
  ElasticityFreqProblem<dim, spacedim>::compute_internal_and_boundary_stress(
    bool openfilefirsttime) const
  {
    TimerOutput::Scope t(computing_timer, "Postprocessing: Computing stresses");

    std::map<types::boundary_id, Tensor<1, spacedim>> boundary_stress;
    std::map<types::boundary_id, double>              u_dot_n;

    auto                                 all_ids = tria->get_boundary_ids();
    std::map<types::boundary_id, double> perimeter;
    for (auto id : all_ids)
      // for (const auto id : par.dirichlet_ids)
      {
        boundary_stress[id] = 0.0;
        perimeter[id]       = 0.0;
        u_dot_n[id]         = 0.0;
      }
    double internal_area = 0.;
    //   // FEValues<spacedim>               fe_values(*fe,
    //   //                              *quadrature,
    //   //                              update_values | update_gradients |
    //   //                                update_quadrature_points | update_JxW_values);
    FEFaceValues<spacedim>           fe_face_values(*fe,
                                          *face_quadrature_formula,
                                          update_values | update_gradients |
                                            update_JxW_values |
                                            update_quadrature_points |
                                            update_normal_vectors);
    const FEValuesExtractors::Vector displacement(0);

    //   // const unsigned int                   dofs_per_cell =
    //   fe->n_dofs_per_cell();
    //   // const unsigned int                   n_q_points    =
    //   quadrature->size();
    //   // std::vector<types::global_dof_index>
    //   local_dof_indices(dofs_per_cell);
    //   // Tensor<2, spacedim>                  grad_phi_u;
    //   // double                               div_phi_u;
    Tensor<2, spacedim> identity;
    for (unsigned int ix = 0; ix < spacedim; ++ix)
      identity[ix][ix] = 1;

    //   // std::vector<std::vector<Tensor<1,spacedim>>>
    //   // solution_gradient(face_quadrature_formula->size(),
    //   // std::vector<Tensor<1,spacedim> >(spacedim+1));
    std::vector<Tensor<2, spacedim>> displacement_gradient(
      face_quadrature_formula->size());
    std::vector<double> displacement_divergence(
      face_quadrature_formula->size());
    std::vector<Tensor<1, spacedim>> displacement_values(
      face_quadrature_formula->size());

    for (const auto &cell : dh.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            const auto &mp = par.get_material_properties(cell->material_id());
            //           // if constexpr (spacedim == 2)
            //           //   {
            //           //     cell->get_dof_indices(local_dof_indices);
            //           //     fe_values.reinit(cell);
            //           //     for (unsigned int q = 0; q < n_q_points; ++q)
            //           //       {
            //           //         internal_area += fe_values.JxW(q);
            //           //         for (unsigned int k = 0; k < dofs_per_cell;
            //           ++k)
            //           //           {
            //           //             grad_phi_u =
            //           // fe_values[displacement].symmetric_gradient(k, q);
            //           //             div_phi_u =
            //           fe_values[displacement].divergence(k, q);
            //           //             internal_stress +=
            //           //               (2 * par.Lame_mu * grad_phi_u +
            //           //                par.Lame_lambda * div_phi_u *
            //           identity)
            //           *
            //           //               locally_relevant_solution.block(
            //           //                 0)[local_dof_indices[k]] *
            //           //               fe_values.JxW(q);
            //           //             average_displacement +=
            //           //               fe_values[displacement].value(k, q) *
            //           //               locally_relevant_solution.block(
            //           //                 0)[local_dof_indices[k]] *
            //           //               fe_values.JxW(q);
            //           //           }
            //           //       }
            //           //   }

            for (unsigned int f = 0; f < GeometryInfo<spacedim>::faces_per_cell;
                 ++f)
              // for (const auto &f : cell->face_iterators())
              // for (const auto f : GeometryInfo<spacedim>::face_indices())
              if (cell->face(f)->at_boundary())
                {
                  auto boundary_index = cell->face(f)->boundary_id();
                  fe_face_values.reinit(cell, f);

                  fe_face_values[displacement].get_function_gradients(
                    locally_relevant_solution.block(0), displacement_gradient);
                  fe_face_values[displacement].get_function_values(
                    locally_relevant_solution.block(0), displacement_values);
                  fe_face_values[displacement].get_function_divergences(
                    locally_relevant_solution.block(0),
                    displacement_divergence);

                  for (unsigned int q = 0;
                       q < fe_face_values.n_quadrature_points;
                       ++q)
                    {
                      perimeter[boundary_index] += fe_face_values.JxW(q);
                      boundary_stress[boundary_index] +=
                        (2 * mp.Lame_mu * displacement_gradient[q] +
                         mp.Lame_lambda * displacement_divergence[q] *
                           identity) *
                        fe_face_values.JxW(q) * fe_face_values.normal_vector(q);
                      u_dot_n[boundary_index] +=
                        (displacement_values[q] *
                         fe_face_values.normal_vector(q)) *
                        fe_face_values.JxW(q);
                    }
                }
          }
      }

    // if constexpr (spacedim == 2)
    //   {
    //     internal_stress = Utilities::MPI::sum(internal_stress,
    //     mpi_communicator); average_displacement =
    //       Utilities::MPI::sum(average_displacement, mpi_communicator);
    //     internal_area = Utilities::MPI::sum(internal_area, mpi_communicator);

    //     internal_stress /= internal_area;
    //     average_displacement /= internal_area;
    //   }
    for (auto id : all_ids) // par.dirichlet_ids) // all_ids)
      {
        boundary_stress[id] =
          Utilities::MPI::sum(boundary_stress[id], mpi_communicator);
        perimeter[id] = Utilities::MPI::sum(perimeter[id], mpi_communicator);
        Assert(perimeter[id] > 0, ExcInternalError());
        boundary_stress[id] /= perimeter[id];
      }

    const unsigned int output_index =
      (par.time_mode == ElasticityFreqTimeMode::Static) ? cycle : time_step;
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        const std::string filename(par.output_directory + "/" +
                                   par.output_name + "_forces_" + stress_part +
                                   "_cycle_" + std::to_string(cycle) + ".txt");
        std::ofstream     forces_file;
        if (openfilefirsttime)
          {
            forces_file.open(filename);
            if constexpr (spacedim == 2)
              {
                forces_file << "cycle area";
                // forces_file
                //   << " meanInternalStressxx meanInternalStressxy
                //   meanInternalStressyx meanInternalStressyy avg_u_x avg_u_y";
                for (auto id : all_ids)
                  forces_file // << " perimeter" << id
                    << " boundaryStressX_" << id << " boundaryStressY_" << id
                    << " uDotN_" << id;
                forces_file << std::endl;
              }
            else
              {
                forces_file << "cycle";
                for (auto id : all_ids)
                  forces_file // << " perimeter" << id
                    << " sigmanX_" << id << " sigmanY_" << id << " sigmanZ_"
                    << id << " uDotN_" << id;
                forces_file << std::endl;
              }
          }
        else
          forces_file.open(filename, std::ios_base::app);

        if constexpr (spacedim == 2)
          {
            forces_file << output_index << " " << internal_area << " ";
            for (auto id : all_ids)
              forces_file // << perimeter[id] << " "
                << boundary_stress[id] << " " << u_dot_n[id] << " ";
            forces_file << std::endl;
          }
        else // spacedim = 3
          {
            forces_file << output_index << " ";
            for (auto id : all_ids)
              forces_file // << perimeter[id] << " "
                << boundary_stress[id] << " " << u_dot_n[id] << " ";
            forces_file << std::endl;
          }
        forces_file.close();
      }

    return;
  }

  template <int dim, int spacedim>
  void
  ElasticityFreqProblem<dim, spacedim>::refine_and_transfer()
  {
    if (!uses_fully_distributed_triangulation())
      {
        TimerOutput::Scope t(computing_timer, "Refine");
        Vector<float>      error_per_cell(tria->n_active_cells());
        KellyErrorEstimator<spacedim>::estimate(
          dh,
          QGauss<spacedim - 1>(par.fe_degree + 1),
          {},
          locally_relevant_solution.block(0),
          error_per_cell);
        if (par.refinement_strategy == "fixed_fraction")
          {
            parallel::distributed::GridRefinement::
              refine_and_coarsen_fixed_fraction(
                std::get<DistributedTriangulation>(triangulation_storage),
                error_per_cell,
                par.refinement_fraction,
                par.coarsening_fraction);
          }
        else if (par.refinement_strategy == "fixed_number")
          {
            parallel::distributed::GridRefinement::
              refine_and_coarsen_fixed_number(
                std::get<DistributedTriangulation>(triangulation_storage),
                error_per_cell,
                par.refinement_fraction,
                par.coarsening_fraction,
                par.max_cells);
          }
        else if (par.refinement_strategy == "global")
          for (const auto &cell : tria->active_cell_iterators())
            cell->set_refine_flag();
        else if (par.refinement_strategy == "inclusions")
          {
            pcout
              << " Refinement around inclusions only implemented for coupled "
                 "elasticity"
              << std::endl;
            cycle = par.n_refinement_cycles;
          }

        execute_actual_refine_and_transfer();
        return;
      }

    AssertThrow(
      false,
      ExcMessage(
        "Adaptive refinement is not implemented with "
        "parallel::fullydistributed::Triangulation in ElasticityFreqProblem. "
        "The current setup supports serial mesh construction, serial refinement, "
        "and then copy_triangulation() before distribute_dofs()."));
  }

  template <int dim, int spacedim>
  void
  ElasticityFreqProblem<dim, spacedim>::execute_actual_refine_and_transfer()
  {
    if (!uses_fully_distributed_triangulation())
      {
        SolutionTransfer<spacedim, LA::MPI::Vector> transfer(dh);
        std::get<DistributedTriangulation>(triangulation_storage)
          .prepare_coarsening_and_refinement();
        transfer.prepare_for_coarsening_and_refinement(
          locally_relevant_solution.block(0));
        std::get<DistributedTriangulation>(triangulation_storage)
          .execute_coarsening_and_refinement();
        setup_dofs();
        transfer.interpolate(solution.block(0));
        constraints.distribute(solution.block(0));
        locally_relevant_solution.block(0) = solution.block(0);
        locally_relevant_solution.block(1) = solution.block(1);
        return;
      }

    AssertThrow(
      false,
      ExcMessage(
        "Adaptive refinement is not implemented with "
        "parallel::fullydistributed::Triangulation in ElasticityFreqProblem."));
  }


  template <int dim, int spacedim>
  void
  ElasticityFreqProblem<dim, spacedim>::setup_constraints()
  {
    par.set_boundary_condition_times(0.0);
    auto make = [&](const bool imaginary, AffineConstraints<double> &c) {
      c.clear();
      c.reinit(owned_dofs[0], relevant_dofs[0]);
      DoFTools::make_hanging_node_constraints(dh, c);
      for (const auto id : par.dirichlet_ids)
        {
          ScaledAmplitude<spacedim> amplitude(
            par.get_dirichlet_bc(id),
            phase_factor(
              par.dirichlet_phase_by_id, id, par.dirichlet_phase, imaginary));
          VectorTools::interpolate_boundary_values(dh, id, amplitude, c);
        }
      // two maps for owning ampltiude and pointing to object
      std::map<types::boundary_id, std::unique_ptr<ScaledAmplitude<spacedim>>>
                                                               storage;
      std::map<types::boundary_id, const Function<spacedim> *> functions;
      for (const auto id : par.normal_flux_ids)
        {
          storage[id] = std::make_unique<ScaledAmplitude<spacedim>>(
            par.get_neumann_bc(id),
            phase_factor(
              par.neumann_phase_by_id, id, par.neumann_phase, imaginary));
          functions[id] = storage[id].get();
        }
      if (!par.normal_flux_ids.empty())
        VectorTools::compute_nonzero_normal_flux_constraints(
          dh, 0, par.normal_flux_ids, functions, c);
      c.close();
    };
    make(false, constraints);
    make(true, imaginary_constraints);
  }

  template <int dim, int spacedim>
  void
  ElasticityFreqProblem<dim, spacedim>::setup_dofs()
  {
    TimerOutput::Scope t(computing_timer, "Setup frequency domain DoFs");
    dh.distribute_dofs(*fe);
    owned_dofs    = {dh.locally_owned_dofs(), IndexSet(0)};
    relevant_dofs = {DoFTools::extract_locally_relevant_dofs(dh), IndexSet(0)};
    setup_constraints();
    solution.reinit(owned_dofs, mpi_communicator);
    locally_relevant_solution.reinit(owned_dofs,
                                     relevant_dofs,
                                     mpi_communicator);
    pcout << "Frequency displacement domain DoFs: " << dh.n_dofs() << std::endl;
  }

  template <int dim, int spacedim>
  void
  ElasticityFreqProblem<dim, spacedim>::assemble_frequency_part(
    double                           omega,
    bool                             imaginary,
    bool                             build_matrix,
    const AffineConstraints<double> &fc,
    LA::MPI::SparseMatrix           &matrix,
    LA::MPI::Vector                 &rhs)
  {
    const bool positive_surrogate = matrix.m() == dh.n_dofs();
    par.set_rhs_times(0.0);
    par.set_boundary_condition_times(0.0);
    FEValues<spacedim>     fe_values(*fe,
                                 *quadrature,
                                 update_values | update_gradients |
                                   update_quadrature_points |
                                   update_JxW_values);
    FEFaceValues<spacedim> fe_face_values(*fe,
                                          *face_quadrature_formula,
                                          update_values |
                                            update_quadrature_points |
                                            update_JxW_values |
                                            update_gradients |
                                            update_normal_vectors);

    const unsigned int dofs_per_cell = fe->n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature->size();

    // Constant-less matrices
    FullMatrix<double> cell_value(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_grad(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_div(dofs_per_cell, dofs_per_cell);

    // Penalty matrices for weak Dirichlet conditions
    FullMatrix<double> cell_penalty_grad(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_penalty_div(dofs_per_cell, dofs_per_cell);

    // Parameter dependent matrices
    FullMatrix<double> cell_penalty_value(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_damping(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_stiffness(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_mass(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_newmark(dofs_per_cell, dofs_per_cell);

    Vector<double>              cell_rhs(dofs_per_cell);
    Vector<double>              cell_penalty_value_rhs(dofs_per_cell);
    Vector<double>              cell_penalty_grad_rhs(dofs_per_cell);
    Vector<double>              cell_penalty_div_rhs(dofs_per_cell);
    std::vector<Vector<double>> rhs_values(n_q_points,
                                           Vector<double>(spacedim));

    std::vector<Tensor<2, spacedim>> grad_phi_u(dofs_per_cell);
    std::vector<double>              div_phi_u(dofs_per_cell);
    std::vector<Tensor<1, spacedim>> phi_u(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          // Get material properties
          const auto &mp = par.get_material_properties(cell->material_id());

          cell_grad          = 0;
          cell_value         = 0;
          cell_div           = 0;
          cell_penalty_value = 0;
          cell_penalty_grad  = 0;
          cell_penalty_div   = 0;
          cell_mass          = 0;
          cell_damping       = 0;
          cell_newmark       = 0;
          cell_stiffness     = 0;

          cell_rhs               = 0;
          cell_penalty_grad_rhs  = 0;
          cell_penalty_div_rhs   = 0;
          cell_penalty_value_rhs = 0;

          fe_values.reinit(cell);
          par.get_rhs(cell->material_id())
            .vector_value_list(fe_values.get_quadrature_points(), rhs_values);
          const double source_factor = phase_factor(par.rhs_phase_by_id,
                                                    cell->material_id(),
                                                    par.rhs_phase,
                                                    imaginary);
          for (auto &values : rhs_values)
            values *= source_factor;

          // Assemble bulk contributions, no constants yet
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                  grad_phi_u[k] =
                    fe_values[displacement].symmetric_gradient(k, q);
                  div_phi_u[k] = fe_values[displacement].divergence(k, q);
                  phi_u[k]     = fe_values[displacement].value(k, q);
                }
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      cell_grad(i, j) +=
                        scalar_product(grad_phi_u[i], grad_phi_u[j]) *
                        fe_values.JxW(q);
                      cell_div(i, j) +=
                        div_phi_u[i] * div_phi_u[j] * fe_values.JxW(q);
                      cell_value(i, j) +=
                        phi_u[i] * phi_u[j] * fe_values.JxW(q);
                    }
                  const auto comp_i = fe->system_to_component_index(i).first;
                  cell_rhs(i) += fe_values.shape_value(i, q) *
                                 rhs_values[q](comp_i) * fe_values.JxW(q);
                }
            }

          // Boundary conditions
          for (const auto &f : cell->face_indices())
            if (cell->face(f)->at_boundary())
              {
                // Weak Dirichlet conditions
                if (par.weak_dirichlet_ids.find(cell->face(f)->boundary_id()) !=
                    par.weak_dirichlet_ids.end())
                  {
                    const auto &dirichlet_bc =
                      par.get_dirichlet_bc(cell->face(f)->boundary_id());
                    fe_face_values.reinit(cell, f);
                    const auto cell_diameter = cell->diameter();

                    for (unsigned int q = 0;
                         q < fe_face_values.n_quadrature_points;
                         ++q)
                      {
                        const auto n = fe_face_values.normal_vector(q);
                        for (unsigned int k = 0; k < dofs_per_cell; ++k)
                          {
                            phi_u[k] = fe_face_values[displacement].value(k, q);
                            grad_phi_u[k] =
                              fe_face_values[displacement].symmetric_gradient(
                                k, q);
                            div_phi_u[k] =
                              fe_face_values[displacement].divergence(k, q);
                          }
                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                          {
                            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                              {
                                cell_penalty_value(i, j) +=
                                  par.penalty_term * (1.0 / cell_diameter) *
                                  phi_u[i] * phi_u[j] * fe_face_values.JxW(q);
                                cell_penalty_grad(i, j) +=
                                  (-grad_phi_u[j] * n * phi_u[i] -
                                   grad_phi_u[i] * n * phi_u[j]) *
                                  fe_face_values.JxW(q);
                                cell_penalty_div(i, j) +=
                                  (-div_phi_u[j] * (n * phi_u[i]) -
                                   div_phi_u[i] * (n * phi_u[j])) *
                                  fe_face_values.JxW(q);
                              }
                            const auto comp_i =
                              fe->system_to_component_index(i).first;
                            Tensor<1, spacedim> g;
                            g[comp_i] =
                              phase_factor(par.dirichlet_phase_by_id,
                                           cell->face(f)->boundary_id(),
                                           par.dirichlet_phase,
                                           imaginary) *
                              dirichlet_bc.value(
                                fe_face_values.quadrature_point(q), comp_i);

                            cell_penalty_value_rhs(i) +=
                              par.penalty_term * (1.0 / cell_diameter) * g *
                              phi_u[i] * fe_face_values.JxW(q);

                            cell_penalty_grad_rhs(i) +=
                              -grad_phi_u[i] * n * g * fe_face_values.JxW(q);

                            cell_penalty_div_rhs(i) +=
                              -div_phi_u[i] * (n * g) * fe_face_values.JxW(q);
                          }
                      }
                  }
                // Neumann Boundary conditions
                else if (par.neumann_ids.find(cell->face(f)->boundary_id()) !=
                         par.neumann_ids.end())
                  {
                    const auto &neumann_bc =
                      par.get_neumann_bc(cell->face(f)->boundary_id());
                    fe_face_values.reinit(cell, f);
                    for (unsigned int q = 0;
                         q < fe_face_values.n_quadrature_points;
                         ++q)
                      {
                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                          {
                            const auto comp_i =
                              fe->system_to_component_index(i).first;
                            const auto un =
                              phase_factor(par.neumann_phase_by_id,
                                           cell->face(f)->boundary_id(),
                                           par.neumann_phase,
                                           imaginary) *
                              neumann_bc.value(
                                fe_face_values.quadrature_point(q), comp_i);
                            cell_rhs(i) += un *
                                           fe_face_values.shape_value(i, q) *
                                           fe_face_values.JxW(q);
                          }
                      }
                  }
              }

          cell->get_dof_indices(local_dof_indices);
          cell_stiffness.equ(2 * mp.Lame_mu,
                             cell_grad,
                             mp.Lame_lambda,
                             cell_div,
                             1.0,
                             cell_penalty_value);
          cell_mass.equ(mp.rho, cell_value);
          cell_damping.equ(mp.rayleigh_beta,
                           cell_stiffness,
                           mp.rayleigh_alpha,
                           cell_mass,
                           mp.neta,
                           cell_grad);
          cell_stiffness.add(2 * mp.Lame_mu,
                             cell_penalty_grad,
                             mp.Lame_lambda,
                             cell_penalty_div);

          cell_rhs.add(2 * mp.Lame_mu,
                       cell_penalty_grad_rhs,
                       mp.Lame_lambda,
                       cell_penalty_div_rhs);

          cell_rhs += cell_penalty_value_rhs;
          if (positive_surrogate)
            {
              FullMatrix<double> local_P(dofs_per_cell, dofs_per_cell);
              local_P.equ(1.0,
                          cell_stiffness,
                          omega * omega,
                          cell_mass,
                          omega,
                          cell_damping);
              if (build_matrix)
                fc.distribute_local_to_global(local_P,
                                              local_dof_indices,
                                              matrix);
              continue;
            }
          FullMatrix<double> H(2 * dofs_per_cell, 2 * dofs_per_cell);
          Vector<double>     b(2 * dofs_per_cell);
          std::vector<types::global_dof_index> ids(2 * dofs_per_cell);
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              ids[2 * i]                     = 2 * local_dof_indices[i];
              ids[2 * i + 1]                 = ids[2 * i] + 1;
              b[2 * i + (imaginary ? 1 : 0)] = cell_rhs[i];
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  const double R =
                    cell_stiffness(i, j) - omega * omega * cell_mass(i, j);
                  const double I          = omega * cell_damping(i, j);
                  H(2 * i, 2 * j)         = R;
                  H(2 * i + 1, 2 * j + 1) = R;
                  H(2 * i, 2 * j + 1)     = -I;
                  H(2 * i + 1, 2 * j)     = I;
                }
            }
          if (build_matrix)
            fc.distribute_local_to_global(H, ids, matrix);
          fc.distribute_local_to_global(b, ids, rhs, H);
        }
    if (build_matrix)
      matrix.compress(VectorOperation::add);
    rhs.compress(VectorOperation::add);
  }

  template <int dim, int spacedim>
  void
  ElasticityFreqProblem<dim, spacedim>::output_frequency(
    const LA::MPI::Vector &real,
    const LA::MPI::Vector &imag) const
  { // output disp real and aimagianry, material, subdomain, amplitude and phase
    LA::MPI::Vector gr, gi, amplitude, phase, ga, gp;
    gr.reinit(owned_dofs[0], relevant_dofs[0], mpi_communicator);
    gi.reinit(owned_dofs[0], relevant_dofs[0], mpi_communicator);
    ga.reinit(owned_dofs[0], relevant_dofs[0], mpi_communicator);
    gp.reinit(owned_dofs[0], relevant_dofs[0], mpi_communicator);
    amplitude.reinit(owned_dofs[0], mpi_communicator);
    phase.reinit(owned_dofs[0], mpi_communicator);
    for (const auto i : owned_dofs[0])
      {
        amplitude[i] = std::hypot(real[i], imag[i]);
        phase[i]     = amplitude[i] > 0.0 ?
                         std::atan2(imag[i], real[i]) * 180.0 / std::acos(-1.0) :
                         0.0;
      }
    amplitude.compress(VectorOperation::insert);
    phase.compress(VectorOperation::insert);
    gr = real;
    gi = imag;
    ga = amplitude;
    gp = phase;
    DataOut<spacedim> data;
    data.attach_dof_handler(dh);
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(spacedim,
                     DataComponentInterpretation::component_is_part_of_vector);
    data.add_data_vector(gr,
                         std::vector<std::string>(spacedim,
                                                  "displacement_real"),
                         DataOut<spacedim>::type_dof_data,
                         interpretation);
    data.add_data_vector(gi,
                         std::vector<std::string>(spacedim,
                                                  "displacement_imag"),
                         DataOut<spacedim>::type_dof_data,
                         interpretation);
    std::vector<std::string>         amplitudes, phases;
    const std::array<std::string, 3> axes{{"x", "y", "z"}};
    for (unsigned int d = 0; d < spacedim; ++d)
      {
        amplitudes.push_back("amplitude_" + axes[d]);
        phases.push_back("phase_degrees_" + axes[d]);
      }
    data.add_data_vector(ga, amplitudes, DataOut<spacedim>::type_dof_data);
    data.add_data_vector(gp, phases, DataOut<spacedim>::type_dof_data);
    Vector<float> subdomain(tria->n_active_cells()),
      materials(tria->n_active_cells());
    unsigned int index = 0;
    for (const auto &cell : tria->active_cell_iterators())
      {
        subdomain[index] = tria->locally_owned_subdomain();
        materials[index] = cell->material_id();
        ++index;
      }
    data.add_data_vector(subdomain, "subdomain");
    data.add_data_vector(materials, "material_id");
    data.build_patches(par.fe_degree);
    const std::string stem =
      par.output_name + "_cycle_" + std::to_string(cycle);
    data.write_vtu_with_pvtu_record(
      par.output_directory, stem, time_step, mpi_communicator, 3);
  }

  template <int dim, int spacedim>
  void
  ElasticityFreqProblem<dim, spacedim>::run()
  {
    print_parameters();
    make_grid();
    setup_fe();
    setup_dofs();
    AssertThrow(
      par.refinement_strategy != "inclusions",
      ExcMessage(
        "No inclusions: select global, fixed_fraction or fixed_number refinement."));
    if (pcout.is_active())
      {
        std::ofstream metadata(par.output_directory + "/" + par.output_name +
                               "_frequencies.csv");
        metadata << "index,frequency_hz\n";
        for (unsigned int i = 0; i < par.frequencies.size(); ++i)
          metadata << i << "," << std::setprecision(17) << par.frequencies[i]
                   << "\n";
      }
    for (cycle = 0; cycle < par.n_refinement_cycles; ++cycle)
      {
        const auto nd = dh.n_dofs();
        IndexSet   own(2 * nd), rel(2 * nd);
        for (const auto i : owned_dofs[0])
          own.add_range(2 * i, 2 * i + 2);
        for (const auto i : relevant_dofs[0])
          rel.add_range(2 * i, 2 * i + 2);
        own.compress();
        rel.compress();
        setup_constraints();
        AffineConstraints<double> final_c, graph;
        final_c.reinit(own, rel);
        for (unsigned int k = 0; k < 2; ++k)
          for (const auto &line :
               (k == 0 ? constraints : imaginary_constraints).get_lines())
            {
              final_c.add_line(2 * line.index + k);
              for (const auto &entry : line.entries)
                final_c.add_entry(2 * line.index + k,
                                  2 * entry.first + k,
                                  entry.second);
              final_c.set_inhomogeneity(2 * line.index + k, line.inhomogeneity);
            }
        final_c.close();
        graph.copy_from(final_c);
        for (const auto &line : graph.get_lines())
          graph.set_inhomogeneity(line.index, 0.0);
        DynamicSparsityPattern               dsp(rel);
        std::vector<types::global_dof_index> ids(2 * fe->n_dofs_per_cell());
        std::vector<types::global_dof_index> scalar_ids(fe->n_dofs_per_cell());
        for (const auto &cell : dh.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(scalar_ids);
              for (unsigned int i = 0; i < scalar_ids.size(); ++i)
                {
                  ids[2 * i]     = 2 * scalar_ids[i];
                  ids[2 * i + 1] = 2 * scalar_ids[i] + 1;
                }
              graph.add_entries_local_to_global(ids, dsp, true);
            }
        for (const auto i : own)
          dsp.add(i, i);
        SparsityTools::distribute_sparsity_pattern(dsp,
                                                   own,
                                                   mpi_communicator,
                                                   rel);
        for (time_step = 0; time_step < par.frequencies.size(); ++time_step)
          {
            current_frequency  = par.frequencies[time_step];
            const double omega = 2 * std::acos(-1.0) * current_frequency;
            LA::MPI::SparseMatrix H;
            H.reinit(own, own, dsp, mpi_communicator);
            LA::MPI::Vector b, x;
            b.reinit(own, mpi_communicator);
            x.reinit(own, mpi_communicator);
            b = 0;
            x = 0;
            for (unsigned int k = 0; k < 2; ++k)
              {
                AffineConstraints<double> part;
                part.copy_from(final_c);
                for (const auto &line : part.get_lines())
                  if (line.index % 2 != k)
                    part.set_inhomogeneity(line.index, 0.0);
                assemble_frequency_part(omega, k == 1, k == 0, part, H, b);
              }
            for (const auto &line : graph.get_lines())
              if (own.is_element(line.index))
                {
                  H.set(line.index, line.index, 1.0);
                  b[line.index] = 0.0;
                }
            H.compress(VectorOperation::insert);
            b.compress(VectorOperation::insert);

            const double rhs_norm = b.l2_norm();

            const double target =
              std::max(par.displacement_solver_control.tolerance(),
                       par.displacement_solver_control.reduction() * rhs_norm);

            SolverControl control(par.displacement_solver_control.max_steps(),
                                  target,
                                  false,
                                  false);

            // control.set_log_frequency(50);

            pcout << "RHS norm: " << rhs_norm << '\n'
                  << "Target absolute residual: " << target << std::endl;

            // control.set_log_frequency(10);

            AffineConstraints<double> scalar_graph;
            scalar_graph.copy_from(constraints);
            for (const auto &line : scalar_graph.get_lines())
              scalar_graph.set_inhomogeneity(line.index, 0.0);
            DynamicSparsityPattern scalar_dsp(relevant_dofs[0]);
            DoFTools::make_sparsity_pattern(dh, scalar_dsp, scalar_graph, true);
            for (const auto i : owned_dofs[0])
              scalar_dsp.add(i, i);
            SparsityTools::distribute_sparsity_pattern(scalar_dsp,
                                                       owned_dofs[0],
                                                       mpi_communicator,
                                                       relevant_dofs[0]);
            LA::MPI::SparseMatrix P0;
            P0.reinit(owned_dofs[0],
                      owned_dofs[0],
                      scalar_dsp,
                      mpi_communicator);
            LA::MPI::Vector unused_rhs;
            unused_rhs.reinit(owned_dofs[0], mpi_communicator);
            unused_rhs = 0.0;
            assemble_frequency_part(
              omega, false, true, scalar_graph, P0, unused_rhs);
            for (const auto &line : scalar_graph.get_lines())
              if (owned_dofs[0].is_element(line.index))
                P0.set(line.index, line.index, 1.0);
            P0.compress(VectorOperation::insert);

            const unsigned int n_modes = spacedim * (spacedim + 1) / 2;
            Epetra_MultiVector near(P0.trilinos_matrix().DomainMap(), n_modes);
            near.PutScalar(0.0);
            std::vector<std::unique_ptr<LA::MPI::Vector>> modes;
            for (unsigned int m = 0; m < n_modes; ++m)
              {
                auto v = std::make_unique<LA::MPI::Vector>();
                v->reinit(owned_dofs[0], mpi_communicator);
                VectorTools::interpolate(dh, RigidBodyMotion<spacedim>(m), *v);
                for (const auto &line : scalar_graph.get_lines())
                  if (owned_dofs[0].is_element(line.index))
                    (*v)[line.index] = 0.0;
                v->compress(VectorOperation::insert);
                for (unsigned int pass = 0; pass < 2; ++pass)
                  for (const auto &previous : modes)
                    v->add(-((*v) * (*previous)), *previous);
                const double norm = v->l2_norm();
                AssertThrow(norm > 0.0 && std::isfinite(norm),
                            ExcMessage(
                              "Degenerate elasticity near-nullspace mode."));
                (*v) *= 1.0 / norm;
                for (const auto i : owned_dofs[0])
                  {
                    const auto local = near.Map().LID(static_cast<int>(i));
                    AssertThrow(local >= 0, ExcInternalError());
                    near[m][local] = (*v)[i];
                  }
                modes.emplace_back(std::move(v));
              }
            Teuchos::ParameterList ml;
            ml.set("PDE equations", spacedim);
            ml.set("null space: type", "pre-computed");
            ml.set("null space: dimension", static_cast<int>(n_modes));
            ml.set("null space: vectors", near.Values());
            ml.set("aggregation: type", "Uncoupled");
            ml.set("aggregation: threshold", 0.02);
            ml.set("smoother: type", "Chebyshev");
            ml.set("smoother: sweeps", 2);
            ml.set("smoother: pre or post", "both");
            ml.set("coarse: type", "symmetric Gauss-Seidel");
            ml.set("coarse: sweeps", 10);
            ml.set("coarse: max size", 2000);
            TrilinosWrappers::PreconditionAMG amg;
            amg.initialize(P0, ml);
            SplitDisplacementAppl<LA::MPI::SparseMatrix> doubled_P(
              P0, owned_dofs[0], mpi_communicator);
            SplitDisplacementAppl<TrilinosWrappers::PreconditionAMG>
              doubled_amg(amg, owned_dofs[0], mpi_communicator);
            SurrogateInverse<decltype(doubled_P), decltype(doubled_amg)>
              preconditioner(doubled_P,
                             doubled_amg,
                             owned_dofs[0],
                             mpi_communicator);
            deallog.depth_console(
              Utilities::MPI::this_mpi_process(mpi_communicator) == 0 ? 10 : 0);

            pcout
              << "Starting complex-Arnoldi FGMRES with displacement-surrogate AMG"
              << std::endl;
            deallog.push("physical");
            try
              {
                complex_fgmres(H,
                               x,
                               b,
                               preconditioner,
                               owned_dofs[0],
                               mpi_communicator,
                               control,
                               false,
                               true);
              }
            catch (...)
              {
                deallog.pop();
                throw;
              }
            deallog.pop();
            LA::MPI::Vector residual;
            residual.reinit(own, mpi_communicator);
            H.vmult(residual, x);
            residual.sadd(-1.0, 1.0, b);
            const double physical_residual = residual.l2_norm();
            AssertThrow(std::isfinite(physical_residual) &&
                          physical_residual <= target,
                        ExcMessage("Physical residual target was not met."));
            pcout << "FGMRES finished; residual=" << physical_residual
                  << std::endl;
            final_c.distribute(x);
            LA::MPI::Vector gx, ur, ui;
            gx.reinit(own, rel, mpi_communicator);
            gx = x;
            ur.reinit(owned_dofs[0], mpi_communicator);
            ui.reinit(owned_dofs[0], mpi_communicator);
            for (const auto i : owned_dofs[0])
              {
                ur[i] = gx[2 * i];
                ui[i] = gx[2 * i + 1];
              }
            ur.compress(VectorOperation::insert);
            ui.compress(VectorOperation::insert);
            output_frequency(ur, ui);
            solution.block(0)         = ur;
            locally_relevant_solution = solution;
            stress_part               = "real";
            if (par.domain_type == "generate")
              compute_internal_and_boundary_stress(time_step == 0);
            solution.block(0)         = ui;
            locally_relevant_solution = solution;
            stress_part               = "imag";
            if (par.domain_type == "generate")
              compute_internal_and_boundary_stress(time_step == 0);
            // Refinement uses component amplitudes at the last solved
            // frequency.
            for (const auto i : owned_dofs[0])
              solution.block(0)[i] = std::hypot(ur[i], ui[i]);
            solution.block(0).compress(VectorOperation::insert);
            locally_relevant_solution = solution;
            pcout << "Frequency " << current_frequency
                  << " Hz; FGMRES iterations=" << control.last_step()
                  << std::endl;
          }
        if (cycle + 1 < par.n_refinement_cycles)
          refine_and_transfer();
      }
  }
  template class ElasticityFreqProblem<2>;
  template class ElasticityFreqProblem<2, 3>;
  template class ElasticityFreqProblem<3>;
} // namespace ImmersX
