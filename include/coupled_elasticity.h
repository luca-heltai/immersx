/* ---------------------------------------------------------------------
 */
#ifdef ENABLE_COUPLED_PROBLEMS

#  ifndef dealii_distributed_lagrange_multiplier_coupled_elasticity_h
#    define dealii_distributed_lagrange_multiplier_coupled_elasticity_h

#    include "elasticity.h"



template <int dim, int spacedim = dim>
class CoupledElasticityProblem : public ElasticityProblem<dim, spacedim>
{
public:
  CoupledElasticityProblem(
    const ElasticityProblemParameters<dim, spacedim> &par);

  void
  run();
  void
  run_timestep0();
  void
  run_timestep();
  void
  compute_coupling_pressure();

  void
  update_inclusions_data(std::vector<double> new_data);
  void
  update_inclusions_data(std::vector<double> new_data,
                         std::vector<int>    cells_per_vessel);

  std::vector<std::vector<double>>
    split_pressure_over_inclusions(std::vector<int>, Vector<double>) const;

  unsigned int
  n_vessels() const
  {
    return this->inclusions.get_n_vessels();
  };

  TrilinosWrappers::MPI::Vector coupling_pressure;
  TrilinosWrappers::MPI::Vector coupling_pressure_at_inclusions;

private:
  void
  reassemble_coupling_rhs();

  void
  refine_and_transfer_around_inclusions();

  void
  output_coupling_pressure(bool openfilefirsttime) const;
};

#  endif

#endif