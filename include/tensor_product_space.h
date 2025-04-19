#ifndef tensor_product_space_h
#define tensor_product_space_h

#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/tria.h>

#include "reference_cross_section.h"

using namespace dealii;

template <int reduced_dim, int dim, int spacedim, int n_components>
struct TensorProductSpaceParameters : public ParameterAcceptor
{
  TensorProductSpaceParameters();
  static constexpr int cross_section_dim = dim - reduced_dim;

  ReferenceCrossSectionParameters<cross_section_dim, spacedim, n_components>
    section;

  /// Refinement level of the mesh.
  unsigned int refinement_level = 0;
  unsigned int fe_degree        = 1;
};


/**
 * @brief A class representing a tensor product space combining a lower-dimensional
 * triangulation and a reference cross-section.
 *
 * @tparam reduced_dim The dimension of the reduced triangulation.
 * @tparam dim The dimension of the full-order object.
 * @tparam spacedim The dimension of the ambient space
 * @tparam n_components The number of components of the problem.
 */
template <int reduced_dim, int dim, int spacedim, int n_components>
class TensorProductSpace
{
public:
  /**
   * Constructor.
   */
  TensorProductSpace(
    const TensorProductSpaceParameters<reduced_dim, dim, spacedim, n_components>
      &par);

  static constexpr int cross_section_dim = dim - reduced_dim;

  std::function<void(Triangulation<reduced_dim, spacedim> &)> make_reduced_grid;

  void
  initialize();

  const ReferenceCrossSection<dim - reduced_dim, spacedim, n_components> &
  get_reference_cross_section() const;

  const DoFHandler<reduced_dim, spacedim> &
  get_dof_handler() const;

private:
  void
  setup_dofs();

  const TensorProductSpaceParameters<reduced_dim, dim, spacedim, n_components>
    &par;


  /**
   * The reference cross-section.
   */
  ReferenceCrossSection<cross_section_dim, spacedim, n_components>
    reference_cross_section;

  /**
   * The triangulation representing the reduced domain.
   */
  Triangulation<reduced_dim, spacedim> triangulation;

  /**
   * The finite element used for the reduced domain.
   */
  FESystem<reduced_dim, spacedim> fe;

  /**
   * The DoFHandler for the reduced domain.
   */
  DoFHandler<reduced_dim, spacedim> dof_handler;
};


#endif // tensor_product_space_h
