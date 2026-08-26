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

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/contributor.h>
#include <immersx/io/utils.h>
#include <immersx/physics/material_properties.h>
#include <immersx/physics/modulated_parsed_function.h>

#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace ImmersX
{
  using namespace dealii;

  namespace elastic_static_detail
  {
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
    explicit ElasticStaticParameters(
      const std::string &subsection = "/Elastic static/")
      : ParameterAcceptor(
          elastic_static_detail::normalize_subsection(subsection))
      , subsection_(elastic_static_detail::normalize_subsection(subsection))
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
      , convergence_table(std::vector<std::string>(spacedim, "u"))
    {
      add_parameter(
        "FE degree", fe_degree, "", this->prm, Patterns::Integer(1));
      add_parameter("Initial refinement", initial_refinement);
      add_parameter("Dirichlet boundary ids", dirichlet_ids);
      add_parameter("Neumann boundary ids", neumann_ids);
      add_parameter("Rhs material ids", rhs_material_ids);
      add_parameter("Output directory", output_directory);
      add_parameter("Output name", output_name);

      enter_subsection("Refinement");
      add_parameter("Number of refinement cycles", n_refinement_cycles);
      leave_subsection();

      enter_subsection("Grid generation");
      add_parameter("Domain type",
                    domain_type,
                    "",
                    this->prm,
                    Patterns::Selection("generate|file"));
      add_parameter("Grid generator", name_of_grid);
      add_parameter("Grid generator arguments", arguments_for_grid);
      add_parameter("Grid scale", grid_scale);
      add_parameter("Triangulation type",
                    triangulation_type,
                    "",
                    this->prm,
                    Patterns::Selection("distributed|fullydistributed"));
      leave_subsection();

      enter_subsection("Material properties");
      add_parameter("Material tags by material id",
                    material_tags_by_material_id);
      leave_subsection();

      this->prm.enter_subsection("Error");
      convergence_table.add_parameters(this->prm);
      this->prm.leave_subsection();

      parse_parameters_call_back.connect([this]() {
        bool created_dynamic_acceptors = false;

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

    const ModulatedParsedFunction<spacedim> &
    get_rhs(const types::material_id material_id) const
    {
      const auto it = rhs_by_material_id.find(material_id);
      return it == rhs_by_material_id.end() ? rhs : *it->second;
    }

    std::string output_directory = ".";
    std::string output_name      = "elastic_static";

    unsigned int fe_degree           = 1;
    unsigned int initial_refinement  = 2;
    unsigned int n_refinement_cycles = 1;

    std::set<types::boundary_id> dirichlet_ids{0};
    std::set<types::boundary_id> neumann_ids{};
    std::set<types::material_id> rhs_material_ids{};

    std::string domain_type        = "generate";
    std::string name_of_grid       = "hyper_cube";
    std::string arguments_for_grid = "0: 1: false";
    double      grid_scale         = 1.0;
    std::string triangulation_type = "distributed";

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
    mutable ParsedConvergenceTable convergence_table;
  };

  /**
   * Parameter-driven assembled linear elasticity Problem for a static solve.
   *
   * The Problem owns the mesh, displacement DoFs, constraints, stiffness
   * matrix, forcing, and accepted solution. It deliberately does not own a
   * solver or any coupling state.
   */
  template <int dim, int spacedim = dim>
  class ElasticStaticProblem
  {
    static_assert(dim == spacedim,
                  "ElasticStaticProblem currently requires dim == spacedim.");

  public:
    using VectorType = ImmersXLA::MPI::Vector;
    using MatrixType = ImmersXLA::MPI::SparseMatrix;

    using DistributedTriangulation =
      parallel::distributed::Triangulation<dim, spacedim>;
    using FullyDistributedTriangulation =
      parallel::fullydistributed::Triangulation<dim, spacedim>;
    using TriangulationVariant =
      std::variant<DistributedTriangulation, FullyDistributedTriangulation>;

    explicit ElasticStaticProblem(
      const ElasticStaticParameters<dim, spacedim> &parameters,
      const MPI_Comm communicator = MPI_COMM_WORLD)
      : parameters_(parameters)
      , communicator_(communicator)
      , triangulation_storage_(make_triangulation_storage(communicator_))
      , tria_(nullptr)
      , fe_(FE_Q<dim, spacedim>(parameters_.fe_degree), spacedim)
    {
      reset_triangulation();
    }

    /** Create the mesh, DoFs, constraints, stiffness matrix, and vectors. */
    void
    setup()
    {
      AssertThrow(dof_handler_->n_dofs() == 0,
                  ExcMessage(
                    "ElasticStaticProblem::setup() may only be called once."));

      make_grid();
      dof_handler_->distribute_dofs(fe_);

      locally_owned_dofs_ = dof_handler_->locally_owned_dofs();
      locally_relevant_dofs_ =
        DoFTools::extract_locally_relevant_dofs(*dof_handler_);

      constraints_.reinit(locally_owned_dofs_, locally_relevant_dofs_);
      DoFTools::make_hanging_node_constraints(*dof_handler_, constraints_);
      for (const auto boundary_id : parameters_.dirichlet_ids)
        VectorTools::interpolate_boundary_values(*dof_handler_,
                                                 boundary_id,
                                                 parameters_.get_dirichlet_bc(
                                                   boundary_id),
                                                 constraints_);
      constraints_.close();

      DynamicSparsityPattern sparsity(locally_relevant_dofs_);
      DoFTools::make_sparsity_pattern(*dof_handler_,
                                      sparsity,
                                      constraints_,
                                      true);
      SparsityTools::distribute_sparsity_pattern(sparsity,
                                                 locally_owned_dofs_,
                                                 communicator_,
                                                 locally_relevant_dofs_);

      stiffness_matrix_.reinit(locally_owned_dofs_,
                               locally_owned_dofs_,
                               sparsity,
                               communicator_);
      forcing_.reinit(locally_owned_dofs_, communicator_);
      solution_.reinit(locally_owned_dofs_, communicator_);
      locally_relevant_solution_.reinit(locally_owned_dofs_,
                                        locally_relevant_dofs_,
                                        communicator_);
      forcing_  = 0.;
      solution_ = 0.;

      assemble_system(nullptr);
    }

    const dealii::parallel::TriangulationBase<dim, spacedim> &
    triangulation() const
    {
      return *tria_;
    }

    const dealii::IndexSet &
    locally_owned_dofs() const
    {
      return locally_owned_dofs_;
    }

    const dealii::IndexSet &
    locally_relevant_dofs() const
    {
      return locally_relevant_dofs_;
    }

    const dealii::DoFHandler<dim, spacedim> &
    dof_handler() const
    {
      return *dof_handler_;
    }

    const dealii::AffineConstraints<double> &
    constraints() const
    {
      return constraints_;
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
      AssertThrow(dof_handler_->n_dofs() != 0,
                  ExcMessage(
                    "Call setup() before setting the elasticity forcing."));
      assemble_system(&function);
    }

    void
    set_solution(const VectorType &new_solution)
    {
      solution_ = new_solution;
      constraints_.distribute(solution_);
      update_locally_relevant_solution();
    }

    /** Write the accepted displacement and subdomain id to configured output.
     */
    void
    output_results() const
    {
      std::filesystem::create_directories(parameters_.output_directory);

      DataOut<dim, spacedim> data_out;
      data_out.attach_dof_handler(*dof_handler_);
      const std::vector<std::string> names(spacedim, "displacement");
      const std::vector<
        DataComponentInterpretation::DataComponentInterpretation>
        interpretation(
          spacedim, DataComponentInterpretation::component_is_part_of_vector);
      data_out.add_data_vector(locally_relevant_solution_,
                               names,
                               DataOut<dim, spacedim>::type_dof_data,
                               interpretation);

      Vector<float> subdomain(tria_->n_active_cells());
      Vector<float> material_id(tria_->n_active_cells());
      for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = tria_->locally_owned_subdomain();
      for (const auto &cell : tria_->active_cell_iterators())
        material_id(cell->active_cell_index()) = cell->material_id();
      data_out.add_data_vector(subdomain, "subdomain");
      data_out.add_data_vector(material_id, "material_id");
      data_out.build_patches();

      const std::string filename = parameters_.output_name + ".vtu";
      data_out.write_vtu_in_parallel(parameters_.output_directory + "/" +
                                       filename,
                                     communicator_);

      if (Utilities::MPI::this_mpi_process(communicator_) == 0)
        {
          std::ofstream pvd(parameters_.output_directory + "/" +
                            parameters_.output_name + ".pvd");
          DataOutBase::write_pvd_record(pvd, {{0., filename}});
        }
    }

  private:
    static TriangulationVariant
    make_triangulation_storage(const MPI_Comm communicator)
    {
      return TriangulationVariant(
        std::in_place_type<DistributedTriangulation>,
        communicator,
        typename Triangulation<dim, spacedim>::MeshSmoothing(
          Triangulation<dim, spacedim>::smoothing_on_refinement |
          Triangulation<dim, spacedim>::smoothing_on_coarsening),
        DistributedTriangulation::construct_multigrid_hierarchy);
    }

    void
    reset_triangulation()
    {
      tria_ = &std::visit(
        [](
          auto &selected_tria) -> parallel::TriangulationBase<dim, spacedim> & {
          return selected_tria;
        },
        triangulation_storage_);
      dof_handler_ = std::make_unique<DoFHandler<dim, spacedim>>(*tria_);
    }

    template <typename TriangulationType>
    void
    make_grid_in(TriangulationType &tria)
    {
      if (parameters_.domain_type == "generate")
        {
          GridGenerator::generate_from_name_and_arguments(
            tria, parameters_.name_of_grid, parameters_.arguments_for_grid);
        }
      else
        {
          read_grid_and_cad_files(parameters_.name_of_grid,
                                  parameters_.arguments_for_grid,
                                  tria);
        }

      if (parameters_.grid_scale != 1.)
        GridTools::scale(parameters_.grid_scale, tria);
      tria.refine_global(parameters_.initial_refinement);
    }

    void
    make_grid()
    {
      const bool fully_distributed =
        parameters_.triangulation_type == "fullydistributed";

      if ((fully_distributed &&
           std::holds_alternative<DistributedTriangulation>(
             triangulation_storage_)) ||
          (!fully_distributed &&
           std::holds_alternative<FullyDistributedTriangulation>(
             triangulation_storage_)))
        dof_handler_.reset();

      if (fully_distributed &&
          !std::holds_alternative<FullyDistributedTriangulation>(
            triangulation_storage_))
        triangulation_storage_.template emplace<FullyDistributedTriangulation>(
          communicator_);
      else if (!fully_distributed &&
               !std::holds_alternative<DistributedTriangulation>(
                 triangulation_storage_))
        triangulation_storage_.template emplace<DistributedTriangulation>(
          communicator_,
          typename Triangulation<dim, spacedim>::MeshSmoothing(
            Triangulation<dim, spacedim>::smoothing_on_refinement |
            Triangulation<dim, spacedim>::smoothing_on_coarsening),
          DistributedTriangulation::construct_multigrid_hierarchy);

      reset_triangulation();

      if (!fully_distributed)
        {
          make_grid_in(
            std::get<DistributedTriangulation>(triangulation_storage_));
          return;
        }

      Triangulation<dim, spacedim> serial_tria(
        typename Triangulation<dim, spacedim>::MeshSmoothing(
          Triangulation<dim, spacedim>::smoothing_on_refinement |
          Triangulation<dim, spacedim>::smoothing_on_coarsening));
      make_grid_in(serial_tria);
      std::get<FullyDistributedTriangulation>(triangulation_storage_)
        .copy_triangulation(serial_tria);
      reset_triangulation();
    }

    void
    assemble_system(const dealii::Function<spacedim> *body_force_override)
    {
      const QGauss<dim>       quadrature(fe_.degree + 1);
      FEValues<dim, spacedim> fe_values(fe_,
                                        quadrature,
                                        update_values | update_gradients |
                                          update_quadrature_points |
                                          update_JxW_values);

      const unsigned int dofs_per_cell = fe_.n_dofs_per_cell();
      const unsigned int n_q_points    = quadrature.size();
      FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
      Vector<double>     cell_rhs(dofs_per_cell);
      std::vector<Tensor<2, spacedim>> symmetric_gradients(dofs_per_cell);
      std::vector<double>              divergences(dofs_per_cell);
      std::vector<Vector<double>> values(n_q_points, Vector<double>(spacedim));
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      stiffness_matrix_ = 0.;
      forcing_          = 0.;

      for (const auto &cell : dof_handler_->active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell_matrix = 0.;
            cell_rhs    = 0.;
            fe_values.reinit(cell);

            const auto &material =
              parameters_.get_material_properties(cell->material_id());
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                  {
                    symmetric_gradients[k] =
                      fe_values[displacement_].symmetric_gradient(k, q);
                    divergences[k] = fe_values[displacement_].divergence(k, q);
                  }

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    cell_matrix(i, j) +=
                      (2. * material.Lame_mu *
                         scalar_product(symmetric_gradients[i],
                                        symmetric_gradients[j]) +
                       material.Lame_lambda * divergences[i] * divergences[j]) *
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
                const double scale = rhs.scale(0.);
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

            cell->get_dof_indices(local_dof_indices);
            constraints_.distribute_local_to_global(cell_matrix,
                                                    cell_rhs,
                                                    local_dof_indices,
                                                    stiffness_matrix_,
                                                    forcing_);

            if (!parameters_.neumann_ids.empty())
              {
                FEFaceValues<dim, spacedim> face_values(
                  fe_,
                  QGauss<dim - 1>(fe_.degree + 1),
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
                      const double scale = traction.scale(0.);
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

      stiffness_matrix_.compress(VectorOperation::add);
      forcing_.compress(VectorOperation::add);
    }

    void
    update_locally_relevant_solution() const
    {
      locally_relevant_solution_ = solution_;
      locally_relevant_solution_.update_ghost_values();
    }

    const ElasticStaticParameters<dim, spacedim> &parameters_;
    MPI_Comm                                      communicator_;
    TriangulationVariant                          triangulation_storage_;
    parallel::TriangulationBase<dim, spacedim>   *tria_;
    std::unique_ptr<DoFHandler<dim, spacedim>>    dof_handler_;
    FESystem<dim, spacedim>                       fe_;
    AffineConstraints<double>                     constraints_;

    IndexSet locally_owned_dofs_;
    IndexSet locally_relevant_dofs_;

    MatrixType         stiffness_matrix_;
    VectorType         forcing_;
    VectorType         solution_;
    mutable VectorType locally_relevant_solution_;

    FEValuesExtractors::Vector displacement_{0};
  };

  struct ElasticStaticFields
  {
    FieldId displacement;
  };

  /** Register the static elasticity residual with an execution adapter. */
  template <typename Builder, int dim, int spacedim>
  ElasticStaticFields
  contribute(Builder                                   &builder,
             const ElasticStaticProblem<dim, spacedim> &problem)
  {
    using VectorType = typename ElasticStaticProblem<dim, spacedim>::VectorType;
    const auto displacement =
      builder.algebraic_field("displacement",
                              problem.locally_owned_dofs(),
                              problem.locally_relevant_dofs());
    const auto stiffness =
      ImmersX::payload_free(dealii::linear_operator<VectorType, VectorType>(
        problem.stiffness_operator()));

    builder.term(displacement, "elastic-static")
      .residual([displacement, &problem](const auto &context) {
        const auto &state = context.state(displacement);
        dealii::PackagedOperation<VectorType> result;
        result.reinit_vector = [state](VectorType &vector, const bool omit) {
          vector.reinit(state, omit);
        };
        result.apply = [&problem, &state](VectorType &vector) {
          problem.stiffness_operator().vmult(vector, state);
          vector -= problem.forcing();
        };
        result.apply_add = [&problem, &state](VectorType &vector) {
          VectorType contribution;
          contribution.reinit(state);
          problem.stiffness_operator().vmult(contribution, state);
          contribution -= problem.forcing();
          vector += contribution;
        };
        return result;
      })
      .state(displacement, stiffness);

    return {displacement};
  }
} // namespace ImmersX

#endif // immersx_elastic_static_h
