/**
 * @brief Header file that includes all necessary headers and defines the Inclusions class.
 */
#ifndef rdlm_mf_utils
#define rdlm_mf_utils


#include <deal.II/base/mpi_remote_point_evaluation.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <inclusions.h>

#include <fstream>

using namespace dealii;



/**
 * Class responsible to provide the action of the coupling operator in a
 * matrix-free fashion.
 */
template <int dim, typename number, int n_components = 1>
class CouplingOperator
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<number>;

  /**
   * Constructor. Takes an @p Inclusions instance, a reference to the background
   * triangulation and an optional mapping to initialize the evaluator on remote
   * points.
   */
  CouplingOperator(
    const Inclusions<dim>           &inclusions,
    const DoFHandler<dim>           &dof_handler,
    const AffineConstraints<number> &constraints = AffineConstraints<number>(),
    const MappingQ<dim>             &mapping     = MappingQ1<dim>(),
    const FiniteElement<dim>        &fe          = FE_Q<dim>(1));

  void
  initialize_dof_vector(VectorType &vec) const;

  void
  vmult(VectorType &dst, const VectorType &src) const;

  void
  Tvmult(VectorType &dst, const VectorType &src) const;

  void
  vmult_add(VectorType &dst, const VectorType &src) const;

  void
  Tvmult_add(VectorType &dst, const VectorType &src) const;


private:
  Utilities::MPI::RemotePointEvaluation<dim> rpe;
  const Mapping<dim>                        *mapping;
  const FiniteElement<dim>                  *fe;
  const DoFHandler<dim>                     *dof_handler;
  const Inclusions<dim>                     *inclusions;
  const AffineConstraints<number>           *constraints;
  const unsigned int                         n_coefficients;
};



#endif