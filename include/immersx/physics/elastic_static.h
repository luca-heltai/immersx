// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_elastic_static_h
#define immersx_elastic_static_h

#include <deal.II/base/data_out_base.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/patterns.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#if DEAL_II_VERSION_GTE(9, 8, 0)
#  include <deal.II/numerics/solution_transfer.h>
#else
#  include <deal.II/distributed/solution_transfer.h>
#endif
#include <deal.II/numerics/vector_tools.h>

#include <immersx/algebra/linear_algebra.h>
#include <immersx/algebra/local_preconditioner.h>
#include <immersx/core/contributor.h>
#include <immersx/core/domain.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/robin_boundary.h>
#include <immersx/io/utils.h>
#include <immersx/physics/detail/robin_parameter_overrides.h>
#include <immersx/physics/material_properties.h>
#include <immersx/physics/modulated_parsed_function.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace ImmersX
{
  using namespace dealii;

  namespace elastic_static_detail
  {
    template <int dim, typename VectorType, int spacedim>
#if DEAL_II_VERSION_GTE(9, 8, 0)
    using CheckpointTransfer =
      dealii::SolutionTransfer<dim, VectorType, spacedim>;
#else
    using CheckpointTransfer = dealii::parallel::distributed::
      SolutionTransfer<dim, VectorType, spacedim>;
#endif

    inline std::string
    normalize_subsection(std::string subsection)
    {
      if (subsection.empty())
        return "/Elastic static/";
      if (subsection.front() != '/')
        subsection.insert(subsection.begin(), '/');
      if (subsection.back() != '/')
        subsection.push_back('/');
      return subsection;
    }
  } // namespace elastic_static_detail

  /** Parameter configuration for a standalone static elasticity Problem. */
  template <int dim, int spacedim = dim>
  class ElasticStaticParameters : public ParameterAcceptor
  {
    std::string subsection_;

  public:
    /**
     * Set expose_refinement_cycles to false when the caller never runs the
     * standalone refinement loop, so the input does not offer an entry that
     * has no effect.
     */
    explicit ElasticStaticParameters(
      const std::string &subsection               = "/Elastic static/",
      const bool         expose_refinement_cycles = true)
      : ParameterAcceptor(
          elastic_static_detail::normalize_subsection(subsection))
      , subsection_(elastic_static_detail::normalize_subsection(subsection))
      , domain_parameters(domain_subsection(subsection_))
      , space_parameters(subsection_ + "Displacement space/", "auto")
      , default_material_properties("default",
                                    elastic_static_detail::normalize_subsection(
                                      subsection) +
                                      "Material properties/")
      , rhs(elastic_static_detail::normalize_subsection(subsection) +
              "Functions/Right hand side",
            spacedim)
      , exact_solution(elastic_static_detail::normalize_subsection(subsection) +
                         "Functions/Exact solution",
                       spacedim)
      , bc(elastic_static_detail::normalize_subsection(subsection) +
             "Functions/Dirichlet boundary conditions",
           spacedim)
      , neumann_bc(elastic_static_detail::normalize_subsection(subsection) +
                     "Functions/Neumann boundary conditions",
                   spacedim)
      , robin_coefficient(subsection_ + "Functions/Robin coefficient", 1)
      , robin_rhs(subsection_ + "Functions/Robin boundary conditions", spacedim)
      , solver_control(elastic_static_detail::normalize_subsection(subsection) +
                       "Solver/Control")
      , convergence_table(std::vector<std::string>(spacedim, "u"))
    {
      domain_parameters.initial_refinement = 2;
      domain_parameters.arguments_for_grid = "0: 1: false";
      add_parameter(
        "FE degree", fe_degree, "", this->prm, Patterns::Integer(1));
      add_parameter("Dirichlet boundary ids", dirichlet_ids);
      add_parameter("Neumann boundary ids", neumann_ids);
      add_parameter("Robin boundary ids", robin_ids);
      add_parameter("Rhs material ids", rhs_material_ids);
      add_parameter("Output directory", output_directory);
      add_parameter("Output name", output_name);


      if (expose_refinement_cycles)
        {
          enter_subsection("Refinement");
          add_parameter("Number of refinement cycles", n_refinement_cycles);
          leave_subsection();
        }


      enter_subsection("Material properties");
      add_parameter("Material tags by material id",
                    material_tags_by_material_id);
      leave_subsection();

      this->enter_my_subsection(this->prm);
      this->prm.enter_subsection("Error");
      convergence_table.add_parameters(this->prm);
      this->prm.leave_subsection();
      this->leave_my_subsection(this->prm);

      parse_parameters_call_back.connect([this]() {
        bool created_dynamic_acceptors =
          robin_detail::register_parameter_overrides(*this,
                                                     robin_ids,
                                                     subsection_ + "Functions/",
                                                     robin_rhs,
                                                     robin_coefficient_by_id,
                                                     robin_rhs_by_id,
                                                     robin_explicit_entries);

        const auto split_subsection_path = [](const std::string &path) {
          std::vector<std::string> subsections;
          std::string              subsection;
          std::istringstream       stream(path);
          while (std::getline(stream, subsection, '/'))
            if (!subsection.empty())
              subsections.emplace_back(subsection);
          return subsections;
        };

        struct SubsectionScope
        {
          SubsectionScope(ParameterHandler               &prm,
                          const std::vector<std::string> &subsections)
            : prm(prm)
            , n_subsections(subsections.size())
          {
            for (const auto &subsection : subsections)
              prm.enter_subsection(subsection);
          }

          ~SubsectionScope()
          {
            for (unsigned int i = 0; i < n_subsections; ++i)
              prm.leave_subsection();
          }

          ParameterHandler &prm;
          unsigned int      n_subsections;
        };

        const auto copy_function_entries = [this](const std::string &from,
                                                  const std::string &to,
                                                  const auto &split_path) {
          const auto get_entry = [this,
                                  &split_path](const std::string &subsection,
                                               const std::string &entry) {
            const SubsectionScope scope(this->prm, split_path(subsection));
            return this->prm.get(entry);
          };
          const auto set_entry = [this,
                                  &split_path](const std::string &subsection,
                                               const std::string &entry,
                                               const std::string &value) {
            const SubsectionScope scope(this->prm, split_path(subsection));
            this->prm.set(entry, value);
          };

          for (const auto &entry : {"Function constants",
                                    "Function expression",
                                    "Variable names",
                                    "Modulation frequency",
                                    "Phase shift"})
            set_entry(to, entry, get_entry(from, entry));
        };

        const auto ensure_function_overrides =
          [this,
           &created_dynamic_acceptors,
           &copy_function_entries,
           &split_subsection_path](const auto        &ids,
                                   auto              &overrides,
                                   const auto        &fallback,
                                   const std::string &prefix) {
            for (const auto id : ids)
              if (overrides.find(id) == overrides.end())
                {
                  const auto subsection =
                    prefix + " " +
                    std::to_string(static_cast<unsigned int>(id));
                  auto function =
                    std::make_shared<ModulatedParsedFunction<spacedim>>(
                      subsection, spacedim);
                  function->set_fallback_configuration_source(&fallback);
                  function->enter_my_subsection(this->prm);
                  function->declare_parameters(this->prm);
                  function->leave_my_subsection(this->prm);
                  overrides.emplace(id, function);
                  created_dynamic_acceptors = true;
                  copy_function_entries(prefix,
                                        subsection,
                                        split_subsection_path);
                }
          };

        this->leave_my_subsection(this->prm);
        ensure_function_overrides(rhs_material_ids,
                                  rhs_by_material_id,
                                  rhs,
                                  subsection_ + "Functions/Right hand side");
        ensure_function_overrides(dirichlet_ids,
                                  dirichlet_bc_by_id,
                                  bc,
                                  subsection_ +
                                    "Functions/Dirichlet boundary conditions");
        ensure_function_overrides(neumann_ids,
                                  neumann_bc_by_id,
                                  neumann_bc,
                                  subsection_ +
                                    "Functions/Neumann boundary conditions");

        if (material_properties_by_id.empty() &&
            !material_tags_by_material_id.empty())
          {
            for (const auto &[material_id, material_tag] :
                 material_tags_by_material_id)
              material_properties_by_id.emplace(
                material_id,
                std::make_unique<MaterialProperties>(material_tag,
                                                     subsection_ +
                                                       "Material properties/"));
            created_dynamic_acceptors = true;
          }

        this->enter_my_subsection(this->prm);

        if (created_dynamic_acceptors)
          return;
      });
    }

    const MaterialProperties &
    get_material_properties(const types::material_id material_id) const
    {
      const auto it = material_properties_by_id.find(material_id);
      return it == material_properties_by_id.end() ?
               default_material_properties :
               *it->second;
    }

    const ModulatedParsedFunction<spacedim> &
    get_dirichlet_bc(const types::boundary_id boundary_id) const
    {
      const auto it = dirichlet_bc_by_id.find(boundary_id);
      return it == dirichlet_bc_by_id.end() ? bc : *it->second;
    }

    const ModulatedParsedFunction<spacedim> &
    get_neumann_bc(const types::boundary_id boundary_id) const
    {
      const auto it = neumann_bc_by_id.find(boundary_id);
      return it == neumann_bc_by_id.end() ? neumann_bc : *it->second;
    }

    const Functions::ParsedFunction<spacedim> &
    get_robin_coefficient(const types::boundary_id id) const
    {
      const auto found = robin_coefficient_by_id.find(id);
      return found == robin_coefficient_by_id.end() ? robin_coefficient :
                                                      *found->second;
    }

    const ModulatedParsedFunction<spacedim> &
    get_robin_rhs(const types::boundary_id id) const
    {
      const auto found = robin_rhs_by_id.find(id);
      return found == robin_rhs_by_id.end() ? robin_rhs : *found->second;
    }

    const ModulatedParsedFunction<spacedim> &
    get_rhs(const types::material_id material_id) const
    {
      const auto it = rhs_by_material_id.find(material_id);
      return it == rhs_by_material_id.end() ? rhs : *it->second;
    }

    std::string output_directory = ".";
    std::string output_name      = "elastic_static";

    unsigned int                                fe_degree = 1;
    DomainParameters<dim, spacedim>             domain_parameters;
    FiniteElementSpaceParameters<dim, spacedim> space_parameters;
    unsigned int                                n_refinement_cycles = 1;

    std::set<types::boundary_id> dirichlet_ids{0};
    std::set<types::boundary_id> neumann_ids{};
    std::set<types::boundary_id> robin_ids{};
    std::set<types::material_id> rhs_material_ids{};


    std::map<types::material_id, std::string> material_tags_by_material_id;
    MaterialProperties                        default_material_properties;
    std::map<types::material_id, std::unique_ptr<MaterialProperties>>
      material_properties_by_id;

    mutable ModulatedParsedFunction<spacedim> rhs;
    std::map<types::material_id,
             std::shared_ptr<ModulatedParsedFunction<spacedim>>>
                                              rhs_by_material_id;
    mutable ModulatedParsedFunction<spacedim> exact_solution;
    mutable ModulatedParsedFunction<spacedim> bc;
    std::map<types::boundary_id,
             std::shared_ptr<ModulatedParsedFunction<spacedim>>>
                                              dirichlet_bc_by_id;
    mutable ModulatedParsedFunction<spacedim> neumann_bc;
    std::map<types::boundary_id,
             std::shared_ptr<ModulatedParsedFunction<spacedim>>>
      neumann_bc_by_id;
    mutable ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
                                              robin_coefficient;
    mutable ModulatedParsedFunction<spacedim> robin_rhs;
    std::map<types::boundary_id,
             std::shared_ptr<
               ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>>>
      robin_coefficient_by_id;
    std::map<types::boundary_id,
             std::shared_ptr<ModulatedParsedFunction<spacedim>>>
                                                      robin_rhs_by_id;
    std::map<types::boundary_id, std::array<bool, 3>> robin_explicit_entries;
    /** Canonical linear solver control shared by native and adapter paths. */
    mutable ParameterAcceptorProxy<ReductionControl> solver_control;
    mutable ParsedConvergenceTable                   convergence_table;
  };

  /**
   * Parameter-driven assembled linear elasticity Problem for a static solve.
   *
   * The Problem owns the mesh, displacement DoFs, constraints, stiffness
   * matrix, forcing, accepted solution, and native linear solver control. It
   * does not own any coupling state.
   */
  template <int dim, int spacedim = dim>
  class ElasticStaticProblem
  {
    static_assert(dim >= 1 && dim <= spacedim && spacedim <= 3,
                  "ElasticStaticProblem requires 1 <= dim <= spacedim <= 3.");

  public:
    using VectorType = ImmersXLA::MPI::Vector;
    using MatrixType = ImmersXLA::MPI::SparseMatrix;

    explicit ElasticStaticProblem(
      const ElasticStaticParameters<dim, spacedim> &parameters,
      const MPI_Comm communicator = MPI_COMM_WORLD)
      : parameters_(parameters)
      , communicator_(communicator)
      , domain_(parameters_.domain_parameters, communicator_)
    {}

    /** Reuse the first assembled stiffness on each mesh.
     * Call before setup, including checkpoint setup. The caller guarantees
     * fixed material and Robin coefficients and a fixed constraint pattern.
     * Loads and inhomogeneous boundary values may still depend on time.
     */
    void
    use_fixed_stiffness()
    {
      AssertThrow(space_ == nullptr,
                  ExcMessage("Select fixed stiffness before FE setup."));
      fixed_stiffness_ = true;
    }

    bool
    uses_fixed_stiffness() const
    {
      return fixed_stiffness_;
    }

    /** Number of stiffness assembly passes, including initial setup. */
    unsigned int
    stiffness_assemblies() const
    {
      return stiffness_assemblies_;
    }

    /** Create the mesh, DoFs, constraints, stiffness matrix, and vectors. */
    void
    setup()
    {
      make_grid();
      setup_fe();
    }

    /** Create the mesh before attaching finite element spaces.
     *
     * An optional initialization policy may prepare, co-refine and finalize
     * the Domain with another mesh. It must leave the Domain finalized.
     */
    void
    make_grid(
      const std::function<void(Domain<dim, spacedim> &)> &initialize = {})
    {
      AssertThrow(space_ == nullptr &&
                    domain_.triangulation().n_global_active_cells() == 0,
                  ExcMessage(
                    "Static elasticity mesh may only be created once."));
      if (initialize)
        initialize(domain_);
      else
        domain_.make_grid();
      AssertThrow(!domain_.grid_is_prepared() &&
                    domain_.triangulation().n_global_active_cells() > 0,
                  ExcMessage(
                    "The grid policy must finalize the solid Domain."));
    }

    /** Create DoFs, constraints, stiffness and vectors on the finalized mesh.
     */
    void
    setup_fe()
    {
      AssertThrow(space_ == nullptr,
                  ExcMessage(
                    "Static elasticity FE setup may only be called once."));
      AssertThrow(!domain_.grid_is_prepared() &&
                    domain_.triangulation().n_global_active_cells() > 0,
                  ExcMessage(
                    "Call make_grid() before static elasticity FE setup."));
      setup_system();
      assemble_system(nullptr);
    }

    /**
     * Save the fixed mesh and accepted displacement with native
     * SolutionTransfer.
     *
     * This collective operation requires fullydistributed storage and loading
     * with the same MPI size. Geometry, material ids, boundary ids, and the FE
     * displacement are self-contained. Parameters and physical time belong to
     * the application's checkpoint manifest.
     */
    void
    save_checkpoint(const std::string &filename) const
    {
      AssertThrow(space_ != nullptr,
                  ExcMessage(
                    "Call setup() before checkpointing static elasticity."));
      AssertThrow(domain_.uses_fully_distributed_triangulation(),
                  ExcMessage("Static checkpoints require "
                             "Triangulation type = fullydistributed."));
      update_locally_relevant_solution();
      elastic_static_detail::CheckpointTransfer<dim, VectorType, spacedim>
        transfer(dof_handler());
      transfer.prepare_for_serialization(locally_relevant_solution_);
      triangulation().save(filename);
    }

    /**
     * Restore a fixed linear-mapping mesh and its accepted displacement.
     *
     * Domain parameters must select fullydistributed storage and the same FE
     * as the writer. Manifold objects and refinement policies are not restored;
     * the stored vertices define the linear mapping without further refinement.
     * The optional restore_geometry callback attaches required manifolds to the
     * empty mesh on every rank before native reconstruction.
     * Call update_time() with the manifest time before evaluating equilibrium.
     */
    void
    setup_from_checkpoint(
      const std::string &filename,
      const std::function<void(Triangulation<dim, spacedim> &)>
        &restore_geometry = {})
    {
      AssertThrow(space_ == nullptr,
                  ExcMessage(
                    "Static elasticity setup may only be called once."));
      AssertThrow(domain_.uses_fully_distributed_triangulation(),
                  ExcMessage("Static checkpoint loading requires "
                             "Triangulation type = fullydistributed."));
      auto &mesh = dynamic_cast<
        typename Domain<dim, spacedim>::FullyDistributedTriangulation &>(
        domain_.triangulation());
      if (restore_geometry)
        restore_geometry(mesh);
      mesh.load(filename);
      setup_system();
      assemble_system(nullptr);
      elastic_static_detail::CheckpointTransfer<dim, VectorType, spacedim>
        transfer(dof_handler());
      transfer.deserialize(solution_);
      update_locally_relevant_solution();
    }

    /**
     * Globally refine a distributed mesh and rebuild the static system.
     *
     * The candidate solution is reset because each static refinement cycle
     * solves the refined problem anew.
     */
    void
    refine_global()
    {
      AssertThrow(
        space_ != nullptr,
        ExcMessage("Call setup() before refining the static elasticity mesh."));
      AssertThrow(
        !domain_.uses_fully_distributed_triangulation(),
        ExcMessage(
          "ElasticStaticProblem::refine_global() is unavailable for "
          "parallel::fullydistributed::Triangulation because its mesh is "
          "immutable after copy_triangulation()."));

      space_.reset();
      dynamic_cast<parallel::distributed::Triangulation<dim, spacedim> &>(
        domain_.triangulation())
        .refine_global(1);
      setup_system();
      assemble_system(nullptr);
    }

    const dealii::parallel::TriangulationBase<dim, spacedim> &
    triangulation() const
    {
      return domain_.triangulation();
    }

    const dealii::IndexSet &
    locally_owned_dofs() const
    {
      return space_->locally_owned_dofs();
    }

    const dealii::IndexSet &
    locally_relevant_dofs() const
    {
      return space_->locally_relevant_dofs();
    }

    const dealii::DoFHandler<dim, spacedim> &
    dof_handler() const
    {
      return space_->dof_handler();
    }

    /** Return the linear reference-cell mapping used for FE evaluations. */
    const dealii::Mapping<dim, spacedim> &
    mapping() const
    {
      AssertThrow(
        space_ != nullptr,
        ExcMessage(
          "Call setup() before accessing the static elasticity mapping."));
      return compatibility::linear_mapping<dim, spacedim>(
        space_->finite_element().reference_cell());
    }

    const dealii::AffineConstraints<double> &
    constraints() const
    {
      return space_->constraints();
    }

    const MatrixType &
    stiffness_operator() const
    {
      return stiffness_matrix_;
    }

    const VectorType &
    forcing() const
    {
      return forcing_;
    }

    const VectorType &
    solution() const
    {
      return solution_;
    }

    /**
     * Reassemble the body force from a programmatic Function.
     *
     * Parameter-driven applications should configure the parsed rhs instead.
     * This convenience is useful for focused tests and callers that already
     * own a deal.II Function.
     */
    void
    set_forcing(const dealii::Function<spacedim> &function)
    {
      AssertThrow(function.n_components == spacedim,
                  ExcDimensionMismatch(function.n_components, spacedim));
      AssertThrow(space_ != nullptr,
                  ExcMessage(
                    "Call setup() before setting the elasticity forcing."));
      assemble_system(&function);
      forcing_override_ = true;
    }

    /**
     * Reassemble equilibrium at a candidate time without accepting a solution.
     *
     * The mesh, material coefficients, and constrained DoF pattern stay fixed.
     * Parsed body loads, tractions, springs, and Dirichlet data use this time.
     * With use_fixed_stiffness(), only loads and Dirichlet lifting are updated;
     * the stiffness and Robin coefficient remain those of the first assembly.
     * This selects parameter-driven forcing instead of a set_forcing override.
     */
    void
    update_time(const double time)
    {
      AssertThrow(space_ != nullptr,
                  ExcMessage(
                    "Call setup() before updating static elasticity time."));
      if (time == assembly_time_ && !forcing_override_)
        return;
      assembly_time_    = time;
      const auto update = [time](auto &fallback, const auto &overrides) {
        fallback.set_time(time);
        for (const auto &[id, function] : overrides)
          function->set_time(time);
      };
      update(parameters_.rhs, parameters_.rhs_by_material_id);
      update(parameters_.bc, parameters_.dirichlet_bc_by_id);
      update(parameters_.neumann_bc, parameters_.neumann_bc_by_id);
      update(parameters_.robin_rhs, parameters_.robin_rhs_by_id);
      update(parameters_.robin_coefficient,
             parameters_.robin_coefficient_by_id);
      update_constraints();
      assemble_system(nullptr);
      forcing_override_ = false;
    }

    /** Solve the assembled static elasticity system natively. */
    void
    solve()
    {
      AssertThrow(space_ != nullptr,
                  ExcMessage("Call setup() before solving static elasticity."));

      ImmersXLA::MPI::PreconditionAMG                          preconditioner;
      typename ImmersXLA::MPI::PreconditionAMG::AdditionalData data;
#ifdef IMMERSX_USE_PETSC_LA
      data.symmetric_operator = true;
#endif
      preconditioner.initialize(stiffness_matrix_, data);

      constraints().distribute(solution_);
      SolverCG<VectorType> solver(parameters_.solver_control);
      solver.solve(stiffness_matrix_, solution_, forcing_, preconditioner);
      constraints().distribute(solution_);
      update_locally_relevant_solution();
    }

    /** Run the ordinary standalone static elasticity lifecycle. */
    void
    run()
    {
      AssertThrow(
        !domain_.uses_fully_distributed_triangulation() ||
          parameters_.n_refinement_cycles <= 1,
        ExcMessage(
          "parallel::fullydistributed::Triangulation supports only one static "
          "refinement cycle because its mesh is immutable after "
          "copy_triangulation()."));

      setup();
      for (unsigned int cycle = 0; cycle < parameters_.n_refinement_cycles;
           ++cycle)
        {
          solve();
          if (space_ != nullptr &&
              space_->finite_element().reference_cell().is_hyper_cube())
            compute_error(parameters_.convergence_table);
          else if (space_ != nullptr &&
                   space_->finite_element().reference_cell().is_simplex() &&
                   Utilities::MPI::this_mpi_process(communicator_) == 0)
            std::cout
              << "   Error table skipped: not available on simplex meshes."
              << std::endl;
          output_results(cycle);

          if (cycle + 1 < parameters_.n_refinement_cycles)
            refine_global();
        }

      if (Utilities::MPI::this_mpi_process(communicator_) == 0 &&
          space_ != nullptr &&
          space_->finite_element().reference_cell().is_hyper_cube())
        parameters_.convergence_table.output_table(std::cout);
    }

    void
    set_solution(const VectorType &new_solution)
    {
      AssertThrow(solution_.size() == new_solution.size(),
                  ExcDimensionMismatch(solution_.size(), new_solution.size()));
      solution_ = new_solution;
      constraints().distribute(solution_);
      update_locally_relevant_solution();
    }

    /** ParsedConvergenceTable currently supports only hypercube meshes. */
    void
    compute_error(dealii::ParsedConvergenceTable &table) const
    {
      AssertThrow(
        space_ != nullptr &&
          space_->finite_element().reference_cell().is_hyper_cube(),
        ExcMessage("Static elasticity error tables require a hypercube mesh."));
      table.error_from_exact(dof_handler(),
                             locally_relevant_solution_,
                             parameters_.exact_solution);
    }

    /** Write the accepted displacement and subdomain id for one observation
     * cycle to configured output.
     */
    void
    output_results(const unsigned int cycle) const
    {
      output_results(cycle, static_cast<double>(cycle));
    }

    /** Write one accepted quasi-static state at its physical time. */
    void
    output_results(const unsigned int cycle, const double time) const
    {
      std::filesystem::create_directories(parameters_.output_directory);

      DataOut<dim, spacedim> data_out;
      data_out.attach_dof_handler(dof_handler());
      const std::vector<std::string> names(spacedim, "displacement");
      const std::vector<
        DataComponentInterpretation::DataComponentInterpretation>
        interpretation(
          spacedim, DataComponentInterpretation::component_is_part_of_vector);
      data_out.add_data_vector(locally_relevant_solution_,
                               names,
                               DataOut<dim, spacedim>::type_dof_data,
                               interpretation);

      Vector<float> subdomain(triangulation().n_active_cells());
      Vector<float> material_id(triangulation().n_active_cells());
      for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation().locally_owned_subdomain();
      for (const auto &cell : triangulation().active_cell_iterators())
        material_id(cell->active_cell_index()) = cell->material_id();
      data_out.add_data_vector(subdomain, "subdomain");
      data_out.add_data_vector(material_id, "material_id");
      data_out.build_patches(mapping());

      const std::string filename =
        parameters_.output_name + "_" + std::to_string(cycle) + ".vtu";
      data_out.write_vtu_in_parallel(parameters_.output_directory + "/" +
                                       filename,
                                     communicator_);

      output_records_.emplace_back(time, filename);

      if (Utilities::MPI::this_mpi_process(communicator_) == 0)
        {
          std::ofstream pvd(parameters_.output_directory + "/" +
                            parameters_.output_name + ".pvd");
          DataOutBase::write_pvd_record(pvd, output_records_);
        }
    }

  private:
    void
    setup_system()
    {
      const auto reference_cell =
        detail::uniform_reference_cell(triangulation());
      AssertThrow(reference_cell.has_value(),
                  ExcMessage("The static elasticity mesh has no cells."));
      std::unique_ptr<FiniteElement<dim, spacedim>> element;
      if (parameters_.space_parameters.finite_element != "auto")
        element = FETools::get_fe_by_name<dim, spacedim>(
          parameters_.space_parameters.finite_element);
      else if (reference_cell->is_hyper_cube())
        element = std::make_unique<FESystem<dim, spacedim>>(
          FE_Q<dim, spacedim>(parameters_.fe_degree), spacedim);
      else if constexpr (dim > 1)
        element = std::make_unique<FESystem<dim, spacedim>>(
          FE_SimplexP<dim, spacedim>(parameters_.fe_degree), spacedim);
      AssertThrow(
        element != nullptr && element->reference_cell() == *reference_cell,
        ExcMessage(
          "The displacement finite element must match the mesh reference cell."));
      space_ = std::make_unique<FiniteElementSpace<dim, spacedim>>(
        domain_.triangulation(), std::move(element));
      AssertThrow(
        space_->finite_element().n_components() == spacedim &&
          space_->finite_element().is_primitive(),
        ExcMessage(
          "The displacement FE must be primitive with spacedim components."));
      const auto &dofs     = dof_handler();
      const auto &owned    = locally_owned_dofs();
      const auto &relevant = locally_relevant_dofs();
      update_constraints();
      const auto &constraints = space_->constraints();

      DynamicSparsityPattern sparsity(relevant);
      DoFTools::make_sparsity_pattern(dofs, sparsity, constraints, true);
      SparsityTools::distribute_sparsity_pattern(sparsity,
                                                 owned,
                                                 communicator_,
                                                 relevant);

      stiffness_matrix_.reinit(owned, owned, sparsity, communicator_);
      stiffness_initialized_ = false;
      if (fixed_stiffness_ && !parameters_.dirichlet_ids.empty())
        {
          uncondensed_stiffness_.reinit(owned, owned, sparsity, communicator_);
          lifting_.reinit(owned, communicator_);
          lifting_action_.reinit(owned, communicator_);
          owned_indices_.assign(owned.begin(), owned.end());
          local_lifting_action_.reinit(owned.n_elements());
        }
      forcing_.reinit(owned, communicator_);
      solution_.reinit(owned, communicator_);
      locally_relevant_solution_.reinit(owned, relevant, communicator_);
      forcing_                   = 0.;
      solution_                  = 0.;
      locally_relevant_solution_ = 0.;
    }

    void
    update_constraints()
    {
      auto &constraints = space_->constraints();
      constraints.clear();
      constraints.reinit(locally_owned_dofs(), locally_relevant_dofs());
      DoFTools::make_hanging_node_constraints(dof_handler(), constraints);
      // Keep modulation consistent with body and boundary traction data.
      class ScaledFunction : public Function<spacedim>
      {
      public:
        ScaledFunction(const ModulatedParsedFunction<spacedim> &function,
                       const double                             time)
          : Function<spacedim>(spacedim)
          , function(function)
          , factor(function.scale(time))
        {}
        double
        value(const Point<spacedim> &point,
              const unsigned int     component = 0) const override
        {
          return factor * function.value(point, component);
        }

      private:
        const ModulatedParsedFunction<spacedim> &function;
        const double                             factor;
      };
      for (const auto boundary_id : parameters_.dirichlet_ids)
        {
          const ScaledFunction function(
            parameters_.get_dirichlet_bc(boundary_id), assembly_time_);
          VectorTools::interpolate_boundary_values(
            mapping(), dof_handler(), boundary_id, function, constraints);
        }
      constraints.close();
    }

    void
    assemble_system(const dealii::Function<spacedim> *body_force_override)
    {
      const bool assemble_stiffness =
        !fixed_stiffness_ || !stiffness_initialized_;
      const bool retain_lifting =
        fixed_stiffness_ && !parameters_.dirichlet_ids.empty();
      const auto &fe_          = space_->finite_element();
      const auto &constraints_ = constraints();
      const auto  quadrature =
        compatibility::gauss_quadrature<dim>(fe_.reference_cell(),
                                             fe_.degree + 1);
      FEValues<dim, spacedim> fe_values(
        mapping(),
        fe_,
        quadrature,
        update_values | update_quadrature_points | update_JxW_values |
          (assemble_stiffness ? update_gradients : update_default));

      const unsigned int dofs_per_cell = fe_.n_dofs_per_cell();
      const unsigned int n_q_points    = quadrature.size();
      FullMatrix<double> cell_matrix(assemble_stiffness ? dofs_per_cell : 0,
                                     assemble_stiffness ? dofs_per_cell : 0);
      Vector<double>                   cell_rhs(dofs_per_cell);
      std::vector<Tensor<2, spacedim>> symmetric_gradients(dofs_per_cell);
      std::vector<double>              divergences(dofs_per_cell);
      std::vector<Vector<double>> values(n_q_points, Vector<double>(spacedim));
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      robin_detail::assert_disjoint_boundary_ids(parameters_.robin_ids,
                                                 parameters_.dirichlet_ids,
                                                 parameters_.neumann_ids,
                                                 communicator_);
      FEFaceValues<dim, spacedim> robin_values(
        mapping(),
        fe_,
        compatibility::gauss_quadrature<dim - 1>(
          fe_.reference_cell().face_reference_cell(0), fe_.degree + 1),
        update_values | update_quadrature_points | update_JxW_values);
      parameters_.robin_coefficient.set_time(assembly_time_);
      parameters_.robin_rhs.set_time(assembly_time_);
      unsigned int invalid_robin = 0;
      if (assemble_stiffness)
        {
          stiffness_matrix_ = 0.;
          if (retain_lifting)
            uncondensed_stiffness_ = 0.;
        }
      forcing_ = 0.;

      for (const auto &cell : dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            if (assemble_stiffness)
              cell_matrix = 0.;
            cell_rhs = 0.;
            fe_values.reinit(cell);

            const auto &material =
              parameters_.get_material_properties(cell->material_id());
            if (assemble_stiffness)
              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                      symmetric_gradients[k] =
                        fe_values[displacement_].symmetric_gradient(k, q);
                      divergences[k] =
                        fe_values[displacement_].divergence(k, q);
                    }

                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      cell_matrix(i, j) +=
                        (2. * material.Lame_mu *
                           scalar_product(symmetric_gradients[i],
                                          symmetric_gradients[j]) +
                         material.Lame_lambda * divergences[i] *
                           divergences[j]) *
                        fe_values.JxW(q);
                }

            if (body_force_override != nullptr)
              body_force_override->vector_value_list(
                fe_values.get_quadrature_points(), values);
            else
              {
                const auto &rhs = parameters_.get_rhs(cell->material_id());
                rhs.vector_value_list(fe_values.get_quadrature_points(),
                                      values);
                const double scale = rhs.scale(assembly_time_);
                for (auto &value : values)
                  value *= scale;
              }

            for (unsigned int q = 0; q < n_q_points; ++q)
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  const auto component = fe_.system_to_component_index(i).first;
                  cell_rhs(i) +=
                    fe_values[displacement_].value(i, q)[component] *
                    values[q][component] * fe_values.JxW(q);
                }

            for (const auto face : cell->face_indices())
              if (cell->face(face)->at_boundary() &&
                  parameters_.robin_ids.count(cell->face(face)->boundary_id()))
                {
                  robin_values.reinit(cell, face);
                  if (assemble_stiffness)
                    invalid_robin |= robin_detail::add_face_terms(
                      robin_values,
                      displacement_,
                      parameters_.get_robin_coefficient(
                        cell->face(face)->boundary_id()),
                      parameters_.get_robin_rhs(
                        cell->face(face)->boundary_id()),
                      cell_matrix,
                      cell_rhs,
                      1.,
                      parameters_.get_robin_rhs(cell->face(face)->boundary_id())
                        .scale(assembly_time_));
                  else
                    {
                      const auto &load = parameters_.get_robin_rhs(
                        cell->face(face)->boundary_id());
                      const double scale = load.scale(assembly_time_);
                      for (unsigned int q = 0;
                           q < robin_values.n_quadrature_points;
                           ++q)
                        {
                          Tensor<1, spacedim> datum;
                          bool                valid = std::isfinite(scale);
                          try
                            {
                              for (unsigned int d = 0; d < spacedim; ++d)
                                {
                                  datum[d] =
                                    load.value(robin_values.quadrature_point(q),
                                               d);
                                  valid = valid && std::isfinite(datum[d]);
                                }
                            }
                          catch (const std::exception &)
                            {
                              valid = false;
                            }
                          if (!valid)
                            {
                              invalid_robin = 1;
                              continue;
                            }
                          for (unsigned int i = 0; i < dofs_per_cell; ++i)
                            cell_rhs(i) +=
                              scale *
                              (robin_values[displacement_].value(i, q) *
                               datum) *
                              robin_values.JxW(q);
                        }
                    }
                }
            cell->get_dof_indices(local_dof_indices);
            if (assemble_stiffness)
              {
                if (retain_lifting)
                  uncondensed_stiffness_.add(local_dof_indices, cell_matrix);
                constraints_.distribute_local_to_global(cell_matrix,
                                                        cell_rhs,
                                                        local_dof_indices,
                                                        stiffness_matrix_,
                                                        forcing_);
              }
            else
              constraints_.distribute_local_to_global(cell_rhs,
                                                      local_dof_indices,
                                                      forcing_);

            if (!parameters_.neumann_ids.empty())
              {
                FEFaceValues<dim, spacedim> face_values(
                  mapping(),
                  fe_,
                  compatibility::gauss_quadrature<dim - 1>(
                    fe_.reference_cell().face_reference_cell(0),
                    fe_.degree + 1),
                  update_values | update_quadrature_points | update_JxW_values);
                std::vector<Vector<double>> face_values_list(
                  face_values.n_quadrature_points, Vector<double>(spacedim));
                Vector<double> face_rhs(dofs_per_cell);

                for (unsigned int face = 0; face < cell->n_faces(); ++face)
                  if (cell->face(face)->at_boundary() &&
                      parameters_.neumann_ids.find(
                        cell->face(face)->boundary_id()) !=
                        parameters_.neumann_ids.end())
                    {
                      face_rhs = 0.;
                      face_values.reinit(cell, face);
                      const auto &traction = parameters_.get_neumann_bc(
                        cell->face(face)->boundary_id());
                      traction.vector_value_list(
                        face_values.get_quadrature_points(), face_values_list);
                      const double scale = traction.scale(assembly_time_);
                      for (auto &value : face_values_list)
                        value *= scale;

                      for (unsigned int q = 0;
                           q < face_values.n_quadrature_points;
                           ++q)
                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                          {
                            const auto component =
                              fe_.system_to_component_index(i).first;
                            face_rhs(i) +=
                              face_values[displacement_].value(i,
                                                               q)[component] *
                              face_values_list[q][component] *
                              face_values.JxW(q);
                          }

                      constraints_.distribute_local_to_global(face_rhs,
                                                              local_dof_indices,
                                                              forcing_);
                    }
              }
          }

      robin_detail::assert_valid_boundary_data(invalid_robin, communicator_);
      if (assemble_stiffness)
        {
          stiffness_matrix_.compress(VectorOperation::add);
          if (retain_lifting)
            uncondensed_stiffness_.compress(VectorOperation::add);
          stiffness_initialized_ = true;
          ++stiffness_assemblies_;
        }
      forcing_.compress(VectorOperation::add);
      if (!assemble_stiffness && retain_lifting)
        {
          // C^T(f-Kg): g satisfies current inhomogeneous constraints, while
          // free coefficients vanish. Scatter each owned row exactly once;
          // this also applies the transpose of hanging-node interpolation.
          lifting_ = 0.;
          constraints_.distribute(lifting_);
          uncondensed_stiffness_.vmult(lifting_action_, lifting_);
          for (unsigned int i = 0; i < owned_indices_.size(); ++i)
            local_lifting_action_[i] = -lifting_action_[owned_indices_[i]];
          constraints_.distribute_local_to_global(local_lifting_action_,
                                                  owned_indices_,
                                                  forcing_);
          forcing_.compress(VectorOperation::add);
        }
    }

    void
    update_locally_relevant_solution() const
    {
      locally_relevant_solution_ = solution_;
      locally_relevant_solution_.update_ghost_values();
    }

    const ElasticStaticParameters<dim, spacedim>      &parameters_;
    MPI_Comm                                           communicator_;
    Domain<dim, spacedim>                              domain_;
    std::unique_ptr<FiniteElementSpace<dim, spacedim>> space_;

    double                               assembly_time_         = 0.;
    bool                                 forcing_override_      = false;
    bool                                 fixed_stiffness_       = false;
    bool                                 stiffness_initialized_ = false;
    unsigned int                         stiffness_assemblies_  = 0;
    MatrixType                           uncondensed_stiffness_;
    VectorType                           lifting_, lifting_action_;
    std::vector<types::global_dof_index> owned_indices_;
    dealii::Vector<double>               local_lifting_action_;
    MatrixType                           stiffness_matrix_;
    VectorType                           forcing_;
    VectorType                           solution_;
    mutable VectorType                   locally_relevant_solution_;
    mutable std::vector<std::pair<double, std::string>> output_records_;

    FEValuesExtractors::Vector displacement_{0};
  };

  template <int dim, int spacedim = dim>
  struct ElasticStaticFields
  {
    using Space = FiniteElementSpaceView<dim, spacedim>;
    using VectorField =
      Field<dim, spacedim, dealii::FEValuesExtractors::Vector>;

    VectorField                  displacement;
    std::shared_ptr<const Space> space;
  };

  /** Register the static elasticity residual with an execution adapter. */
  template <typename Builder, int dim, int spacedim>
  ElasticStaticFields<dim, spacedim>
  contribute(Builder                                   &builder,
             const ElasticStaticProblem<dim, spacedim> &problem)
  {
    const auto displacement_id =
      builder.algebraic_field("displacement",
                              problem.locally_owned_dofs(),
                              problem.locally_relevant_dofs());
    using Space = typename ElasticStaticFields<dim, spacedim>::Space;
    auto space  = std::make_shared<Space>(finite_element_space_view(problem));
    const auto displacement =
      space->field(displacement_id,
                   "displacement",
                   dealii::FEValuesExtractors::Vector(0));
    const auto stiffness =
      builder.matrix_operator(problem.stiffness_operator());
    builder.preconditioner(displacement_id,
                           [](const auto &linearized_matrix,
                              const auto &reinit_vector) {
                             return make_amg_preconditioner(linearized_matrix,
                                                            reinit_vector);
                           });

    builder.term(displacement_id, "elastic-static")
      .residual([displacement_id, &problem, stiffness](const auto &context) {
        return stiffness.view * context.state(displacement_id) -
               problem.forcing();
      })
      .state(displacement_id, stiffness);

    return {displacement, std::move(space)};
  }
} // namespace ImmersX

#endif // immersx_elastic_static_h
