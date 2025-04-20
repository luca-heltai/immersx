#include "tensor_product_space.h"

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/particles/data_out.h>
#include <deal.II/particles/utilities.h>

template <int reduced_dim, int dim, int spacedim, int n_components>
TensorProductSpaceParameters<reduced_dim, dim, spacedim, n_components>::
  TensorProductSpaceParameters()
  : ParameterAcceptor("Tensor product space")
{
  enter_subsection("Representative domain");
  add_parameter("Refinement level", refinement_level);
  add_parameter("Finite element degree", fe_degree);
  add_parameter("RTree extraction level", rtree_extraction_level);
  leave_subsection();
}

// Constructor for TensorProductSpace
template <int reduced_dim, int dim, int spacedim, int n_components>
TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
  TensorProductSpace(
    const TensorProductSpaceParameters<reduced_dim, dim, spacedim, n_components>
      &par)
  : par(par)
  , reference_cross_section(par.section)
  , fe(FE_Q<reduced_dim, spacedim>(par.fe_degree),
       reference_cross_section.n_selected_basis())
  , quadrature_formula(2 * par.fe_degree + 1)
  , dof_handler(triangulation)
{
  make_reduced_grid = [](Triangulation<reduced_dim, spacedim> &tria) {
    GridGenerator::hyper_cube(tria, 0, 1);
  };
}

// Initialize the tensor product space
template <int reduced_dim, int dim, int spacedim, int n_components>
void
TensorProductSpace<reduced_dim, dim, spacedim, n_components>::initialize()
{
  // Create the reduced grid
  make_reduced_grid(triangulation);
  triangulation.refine_global(par.refinement_level);

  // Setup degrees of freedom
  setup_dofs();
}

template <int reduced_dim, int dim, int spacedim, int n_components>
const ReferenceCrossSection<dim - reduced_dim, spacedim, n_components> &
TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
  get_reference_cross_section() const
{
  return reference_cross_section;
}

template <int reduced_dim, int dim, int spacedim, int n_components>
const DoFHandler<reduced_dim, spacedim> &
TensorProductSpace<reduced_dim, dim, spacedim, n_components>::get_dof_handler()
  const
{
  return dof_handler;
}

template <int reduced_dim, int dim, int spacedim, int n_components>
void
TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
  setup_qpoints_particles(
    const parallel::distributed::Triangulation<spacedim> &tria,
    const Mapping<spacedim>                              &mapping)
{
  qpoints_as_particles.initialize(tria, mapping);
  mpi_communicator = tria.get_communicator();

  std::vector<BoundingBox<spacedim>> all_boxes;
  all_boxes.reserve(tria.n_locally_owned_active_cells());
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      all_boxes.emplace_back(mapping.get_bounding_box(cell));

  const auto tree = pack_rtree(all_boxes);
  const auto local_boxes =
    extract_rtree_level(tree, par.rtree_extraction_level);

  auto global_bounding_boxes =
    Utilities::MPI::all_gather(mpi_communicator, local_boxes);

  // now that we have the global bounding boxes, we can set up the particles
  std::vector<Point<spacedim>> particles_positions;
  particles_positions.reserve(triangulation.n_active_cells() *
                              quadrature_formula.size() *
                              reference_cross_section.n_quadrature_points());

  UpdateFlags flags = reduced_dim == 1 ?
                        update_quadrature_points :
                        update_quadrature_points | update_normal_vectors;

  FEValues<reduced_dim, spacedim> fev(fe, quadrature_formula, flags);

  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fev.reinit(cell);
        const auto         &qpoints = fev.get_quadrature_points();
        Tensor<1, spacedim> new_vertical;
        if constexpr (reduced_dim == 1)
          new_vertical = cell->vertex(1) - cell->vertex(0);

        for (const auto &q : fev.quadrature_point_indices())
          {
            const auto &qpoint = qpoints[q];
            if constexpr (dim == 2)
              new_vertical = fev.normal_vector(q);
            auto cross_section_qpoints =
              reference_cross_section.get_transformed_quadrature(qpoint,
                                                                 new_vertical,
                                                                 1.0);
            particles_positions.insert(
              particles_positions.end(),
              cross_section_qpoints.get_points().begin(),
              cross_section_qpoints.get_points().end());
          }
      }
  qpoints_as_particles.insert_global_particles(particles_positions,
                                               global_bounding_boxes);
}



template <int reduced_dim, int dim, int spacedim, int n_components>
void
TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
  output_qpoints_particles(const std::string &filename) const
{
  Particles::DataOut<spacedim> particles_out;
  particles_out.build_patches(qpoints_as_particles);
  particles_out.write_vtu_in_parallel(filename, mpi_communicator);
}



// Setup degrees of freedom for the tensor product space
template <int reduced_dim, int dim, int spacedim, int n_components>
void
TensorProductSpace<reduced_dim, dim, spacedim, n_components>::setup_dofs()
{
  dof_handler.distribute_dofs(fe);
  // Additional setup can be done here if needed
}

template struct TensorProductSpaceParameters<1, 2, 3, 1>;
template struct TensorProductSpaceParameters<1, 3, 3, 1>;
template struct TensorProductSpaceParameters<2, 3, 3, 1>;

template struct TensorProductSpaceParameters<1, 2, 3, 3>;
template struct TensorProductSpaceParameters<1, 3, 3, 3>;
template struct TensorProductSpaceParameters<2, 3, 3, 3>;

template class TensorProductSpace<1, 2, 3, 1>;
template class TensorProductSpace<1, 3, 3, 1>;
template class TensorProductSpace<2, 3, 3, 1>;

template class TensorProductSpace<1, 2, 3, 3>;
template class TensorProductSpace<1, 3, 3, 3>;
template class TensorProductSpace<2, 3, 3, 3>;