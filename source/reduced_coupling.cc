#include "reduced_coupling.h"

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>

#include <fstream>

#include "immersed_repartitioner.h"
#include "tensor_product_space.h"
#include "vtk_utils.h"

template <int reduced_dim, int dim, int spacedim, int n_components>
ReducedCouplingParameters<reduced_dim, dim, spacedim, n_components>::
  ReducedCouplingParameters()
  : ParameterAcceptor("Reduced coupling/")
{
  this->enter_subsection("Representative domain");
  this->add_parameter("Pre-refinement level", pre_refinement);
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
  const parallel::TriangulationBase<spacedim> &background_tria,
  const ReducedCouplingParameters<reduced_dim, dim, spacedim, n_components>
    &par)
  : TensorProductSpace<reduced_dim, dim, spacedim, n_components>(
      par.tensor_product_space_parameters,
#if DEAL_II_VERSION_GTE(9, 6, 0)
      background_tria.get_communicator())
#else 
background_tria.get_mpi_communicator()
#endif
  , ParticleCoupling<spacedim>(par.particle_coupling_parameters)
  , mpi_communicator(
#if DEAL_II_VERSION_GTE(9, 6, 0)
      background_tria.get_communicator()
#else 
  background_tria.get_mpi_communicator()
#endif
        )
  , par(par)
  , background_tria(&background_tria)
  , immersed_partitioner(background_tria)
{
  this->set_partitioner = [&](auto &tria) {
    tria.set_partitioner(immersed_partitioner,
                         TriangulationDescription::Settings());
  };
}

template <int reduced_dim, int dim, int spacedim, int n_components>
void
ReducedCoupling<reduced_dim, dim, spacedim, n_components>::initialize(
  const Mapping<spacedim> &mapping)
{
  // Initialize the tensor product space
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::initialize();

  auto locally_owned_dofs = this->get_dof_handler().locally_owned_dofs();
  auto locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(this->get_dof_handler());

  coupling_constraints.clear();
  coupling_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(this->get_dof_handler(),
                                          coupling_constraints);
  coupling_constraints.close();

  // Initialize the particle coupling
  ParticleCoupling<spacedim>::initialize_particle_handler(
    *this->background_tria, mapping);

  // Initialize the particles
  const auto &qpoints = this->get_locally_owned_qpoints();
  const auto &weights = this->get_locally_owned_weights();
  auto        q_index = this->insert_points(qpoints, weights);
  this->update_local_dof_indices(q_index);

  // Initialize the coupling rhs
  typename FunctionParser<spacedim>::ConstMap constants;
  constants["pi"] = numbers::PI;
  constants["E"]  = numbers::E;

  coupling_rhs = std::make_unique<FunctionParser<spacedim>>(
    this->get_reference_cross_section().n_selected_basis());

  coupling_rhs->initialize(FunctionParser<spacedim>::default_variable_names() +
                             ",t",
                           par.coupling_rhs_expressions,
                           constants,
                           true);

  // This should be true. Let's double check
  AssertDimension(coupling_rhs->n_components,
                  this->get_dof_handler().get_fe().n_components());

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::cout
        << "Selected degree: "
        << par.tensor_product_space_parameters.section.inclusion_degree
        << ", selected basis functions: "
        << Patterns::Tools::to_string(
             par.tensor_product_space_parameters.section.selected_coefficients)
        << std::endl;
      std::cout << "Reduced coupling initialized" << std::endl;
      std::cout << "Reduced grid name: "
                << par.tensor_product_space_parameters.reduced_grid_name
                << std::endl;
    }
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
            this->particle_id_to_cell_and_qpoint_indices(p.get_id());
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
template struct ReducedCouplingParameters<1, 2, 2, 1>;
template struct ReducedCouplingParameters<1, 2, 3, 1>;
template struct ReducedCouplingParameters<1, 3, 3, 1>;
template struct ReducedCouplingParameters<2, 3, 3, 1>;

template struct ReducedCouplingParameters<1, 2, 2, 2>;
template struct ReducedCouplingParameters<1, 2, 3, 3>;
template struct ReducedCouplingParameters<1, 3, 3, 3>;
template struct ReducedCouplingParameters<2, 3, 3, 3>;


template struct ReducedCoupling<1, 2, 2, 1>;
template struct ReducedCoupling<1, 2, 3, 1>;
template struct ReducedCoupling<1, 3, 3, 1>;
template struct ReducedCoupling<2, 3, 3, 1>;

template struct ReducedCoupling<1, 2, 2, 2>;
template struct ReducedCoupling<1, 2, 3, 3>;
template struct ReducedCoupling<1, 3, 3, 3>;
template struct ReducedCoupling<2, 3, 3, 3>;
