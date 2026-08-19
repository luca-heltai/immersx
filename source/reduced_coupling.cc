#include "reduced_coupling.h"

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>

#include <fstream>
#include <regex>
#include <stdexcept>

#include "immersed_repartitioner.h"
#include "tensor_product_space.h"
#include "utils.h"

namespace
{
  bool
  expression_contains_symbol(const std::string &expression,
                             const std::string &symbol)
  {
    const std::regex token("(^|[^A-Za-z0-9_])" + symbol + "([^A-Za-z0-9_]|$)");
    return std::regex_search(expression, token);
  }
} // namespace

template <int reduced_dim, int dim, int spacedim, int n_components>
ReducedCouplingParameters<reduced_dim, dim, spacedim, n_components>::
  ReducedCouplingParameters()
  : ParameterAcceptor("/Tensor product coupling/")
{
  this->enter_subsection("Representative domain");
  this->add_parameter("Reduced right hand side",
                      coupling_rhs_expressions,
                      "",
                      this->prm,
                      Patterns::List(Patterns::Anything(),
                                     1,
                                     Patterns::List::max_int_value,
                                     ";"));
  this->leave_subsection();
}

template <int reduced_dim, int dim, int spacedim, int n_components>
ReducedCoupling<reduced_dim, dim, spacedim, n_components>::ReducedCoupling(
  parallel::TriangulationBase<spacedim> &background_tria,
  const ReducedCouplingParameters<reduced_dim, dim, spacedim, n_components>
    &par)
  : TensorProductSpace<reduced_dim, dim, spacedim, n_components>(
      par.tensor_product_space_parameters,
#if DEAL_II_VERSION_GTE(9, 6, 0)
      background_tria.get_mpi_communicator())
#else
background_tria.get_mpi_communicator()
#endif
  , ParticleCoupling<spacedim>(par.particle_coupling_parameters)
  , mpi_communicator(
#if DEAL_II_VERSION_GTE(9, 6, 0)
      background_tria.get_mpi_communicator()
#else
  background_tria.get_mpi_communicator()
#endif
        )
  , par(par)
  , background_tria(&background_tria)
  , immersed_partitioner(background_tria)
{
  if constexpr (reduced_dim > 0)
    {
      this->preprocess_serial_triangulation =
        [&](Triangulation<reduced_dim, spacedim> &tria) {
          adjust_grids(*(this->background_tria),
                       tria,
                       par.refinement_parameters);
        };
      this->set_partitioner = [&](auto &tria) {
        tria.set_partitioner(immersed_partitioner.value,
                             TriangulationDescription::Settings());
      };
    }
}

template <int reduced_dim, int dim, int spacedim, int n_components>
void
ReducedCoupling<reduced_dim, dim, spacedim, n_components>::initialize(
  const Mapping<spacedim> &mapping)
{
  // Initialize the background particle coupling first: its global bounding
  // boxes define geometric ownership for both representative and lifted
  // particles.
  bool prepared_for_refinement = false;
  if constexpr (reduced_dim == 0)
    {
      if (!zero_dimensional_background_refined)
        {
          this->prepare();
          refine_space_around_points(*background_tria,
                                     this->point_cloud.points,
                                     this->entity_thickness,
                                     par.refinement_parameters,
                                     mpi_communicator);
          zero_dimensional_background_refined = true;
          prepared_for_refinement             = true;
        }
    }

  ParticleCoupling<spacedim>::initialize_particle_handler(
    *this->background_tria, mapping);

  if constexpr (reduced_dim == 0)
    {
      if (!prepared_for_refinement)
        this->prepare();
      this->initialize_representative_particle_handler(
        *this->background_tria, mapping, this->get_global_bounding_boxes());
      this->compute_points_and_weights();
    }
  else
    TensorProductSpace<reduced_dim, dim, spacedim, n_components>::initialize();

  // Initialize lifted particles only after representative ownership has been
  // established. Their quadrature weights remain particle properties; the 0D
  // representative handler stores no artificial point weight.
  const auto &qpoints = this->get_locally_owned_qpoints();
  const auto &weights = this->get_locally_owned_weights();
  auto        q_index = this->insert_points(qpoints, weights);
  // ParticleHandler assigns ids globally by source-rank prefix when ids
  // are omitted. Build the explicit id-to-entity map from that documented
  // assignment before any assembly uses particle ids.
  this->register_particle_id_mapping();
  this->update_local_dof_indices(q_index);

  auto locally_owned_dofs    = this->locally_owned_representative_dofs();
  auto locally_relevant_dofs = this->locally_relevant_representative_dofs();
  coupling_constraints.clear();
  coupling_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  if constexpr (reduced_dim > 0)
    DoFTools::make_hanging_node_constraints(this->get_dof_handler(),
                                            coupling_constraints);
  coupling_constraints.close();

  const unsigned int n_basis =
    this->get_reference_cross_section().n_selected_basis();
  AssertThrow(par.coupling_rhs_expressions.size() == n_basis,
              ExcMessage("Reduced right hand side has " +
                         std::to_string(par.coupling_rhs_expressions.size()) +
                         " expressions, but the selected basis requires " +
                         std::to_string(n_basis) + "."));

  typename FunctionParser<spacedim>::ConstMap constants;
  constants["pi"] = numbers::PI;
  constants["E"]  = numbers::E;
  rhs_time        = 0.;
  std::vector<std::string> field_symbols;
  field_symbols.reserve(this->get_properties_bindings().size());
  for (const auto &binding : this->get_properties_bindings())
    field_symbols.push_back(binding.symbol_name);

  if (SymbolicFieldEvaluator::available())
    {
      symbolic_coupling_rhs = std::make_unique<SymbolicFieldEvaluator>();
      symbolic_coupling_rhs->initialize(par.coupling_rhs_expressions,
                                        field_symbols,
                                        {{"pi", numbers::PI},
                                         {"E", numbers::E}});
    }
  else
    {
      // Keep the coordinate-only FunctionParser path available in builds
      // without SymEngine. If an expression names an imported field, the
      // evaluator below reports that the requested feature is unavailable.
      coupling_rhs = std::make_unique<FunctionParser<spacedim>>(n_basis);
      coupling_rhs->initialize(
        FunctionParser<spacedim>::default_variable_names() + ",t",
        par.coupling_rhs_expressions,
        constants,
        true);
      if constexpr (reduced_dim > 0)
        AssertDimension(coupling_rhs->n_components,
                        this->get_dof_handler().get_fe().n_components());
      else
        AssertDimension(coupling_rhs->n_components,
                        this->n_representative_dofs_per_entity());
      if (!field_symbols.empty())
        {
          for (const auto &expression : par.coupling_rhs_expressions)
            for (const auto &symbol : field_symbols)
              if (expression_contains_symbol(expression, symbol))
                throw std::runtime_error(
                  "Reduced right-hand-side expression '" + expression +
                  "' refers to imported field '" + symbol +
                  "', but this build has no SymEngine support.");
        }
    }

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::cout
        << "Selected degree: "
        << par.tensor_product_space_parameters.section.inclusion_degree
        << ", selected basis functions: "
        << Patterns::Tools::to_string(
             par.tensor_product_space_parameters.section.selected_coefficients)
        << std::endl;
      std::cout << "Tensor-product coupling initialized" << std::endl;
      std::cout << "Reduced grid name: "
                << par.tensor_product_space_parameters.reduced_grid_name
                << std::endl;
    }
}

template <int reduced_dim, int dim, int spacedim, int n_components>
void
ReducedCoupling<reduced_dim, dim, spacedim, n_components>::set_time(
  const double time)
{
  AssertThrow(
    coupling_rhs || symbolic_coupling_rhs,
    ExcMessage(
      "Tensor-product coupling must be initialized before setting time"));
  rhs_time = time;
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::set_time(time);
  if (coupling_rhs)
    coupling_rhs->set_time(time);
}

template <int reduced_dim, int dim, int spacedim, int n_components>
void
ReducedCoupling<reduced_dim, dim, spacedim, n_components>::
  assemble_coupling_sparsity(DynamicSparsityPattern          &dsp,
                             const DoFHandler<spacedim>      &dh,
                             const AffineConstraints<double> &constraints) const
{
  const auto                          &fe = dh.get_fe();
  std::vector<types::global_dof_index> background_dof_indices(
    fe.n_dofs_per_cell());

  auto particle = this->get_particles().begin();
  while (particle != this->get_particles().end())
    {
      const auto &cell = particle->get_surrounding_cell();
      const auto  dh_cell =
        typename DoFHandler<spacedim>::cell_iterator(*cell, &dh);
      dh_cell->get_dof_indices(background_dof_indices);

      const auto pic = this->get_particles().particles_in_cell(cell);
      Assert(pic.begin() == particle, ExcInternalError());

      types::global_cell_index previous_cell_id = numbers::invalid_unsigned_int;
      for (const auto &p : pic)
        {
          const auto [immersed_cell_id, immersed_q, section_q] =
            this->particle_id_to_representative_indices(p.get_id());
          // If cell id is the same, we can skip the rest of the loop. We
          // already added these entries
          if (immersed_cell_id != previous_cell_id)
            {
              const auto &immersed_dof_indices =
                this->get_dof_indices(immersed_cell_id);

              constraints.add_entries_local_to_global(background_dof_indices,
                                                      coupling_constraints,
                                                      immersed_dof_indices,
                                                      dsp);

              previous_cell_id = immersed_cell_id;
            }
        }
      particle = pic.end();
    }
}

template <int reduced_dim, int dim, int spacedim, int n_components>
const AffineConstraints<double> &
ReducedCoupling<reduced_dim, dim, spacedim, n_components>::
  get_coupling_constraints() const
{
  return coupling_constraints;
}

// Explicit instantiations for ReducedCouplingParameters
template struct ReducedCouplingParameters<0, 2, 2, 1>;
template struct ReducedCouplingParameters<0, 1, 3, 1>;
template struct ReducedCouplingParameters<0, 2, 3, 1>;
template struct ReducedCouplingParameters<0, 3, 3, 1>;
template struct ReducedCouplingParameters<0, 2, 2, 2>;
template struct ReducedCouplingParameters<0, 1, 2, 1>;
template struct ReducedCouplingParameters<0, 1, 2, 2>;
template struct ReducedCouplingParameters<1, 2, 2, 1>;
template struct ReducedCouplingParameters<1, 2, 3, 1>;
template struct ReducedCouplingParameters<1, 3, 3, 1>;
template struct ReducedCouplingParameters<2, 3, 3, 1>;

template struct ReducedCouplingParameters<1, 2, 2, 2>;
template struct ReducedCouplingParameters<1, 2, 3, 3>;
template struct ReducedCouplingParameters<1, 3, 3, 3>;
template struct ReducedCouplingParameters<2, 3, 3, 3>;


template struct ReducedCoupling<0, 2, 2, 1>;
template struct ReducedCoupling<0, 1, 3, 1>;
template struct ReducedCoupling<0, 2, 3, 1>;
template struct ReducedCoupling<0, 3, 3, 1>;
template struct ReducedCoupling<0, 2, 2, 2>;
template struct ReducedCoupling<0, 1, 2, 1>;
template struct ReducedCoupling<0, 1, 2, 2>;
template struct ReducedCoupling<1, 2, 2, 1>;
template struct ReducedCoupling<1, 2, 3, 1>;
template struct ReducedCoupling<1, 3, 3, 1>;
template struct ReducedCoupling<2, 3, 3, 1>;

template struct ReducedCoupling<1, 2, 2, 2>;
template struct ReducedCoupling<1, 2, 3, 3>;
template struct ReducedCoupling<1, 3, 3, 3>;
template struct ReducedCoupling<2, 3, 3, 3>;
