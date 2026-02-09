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
  TrilinosWrappers::MPI::Vector
    // Vector<double>
    coupling_pressure_at_inclusions;

private:
  // void
  // make_grid();
  // void
  // setup_fe();
  // void
  // setup_dofs();
  // void
  // assemble_elasticity_system();
  // void
  // assemble_coupling();
  void
  reassemble_coupling_rhs();

  // void
  // check_boundary_ids();

  // /**
  //  * Builds coupling sparsity, and returns locally relevant inclusion dofs.
  //  */
  // IndexSet
  // assemble_coupling_sparsity(DynamicSparsityPattern &dsp);

  // void
  // solve();

  // void
  // refine_and_transfer();

  void
  refine_and_transfer_around_inclusions();

  // void
  // execute_actual_refine_and_transfer();

  // std::string
  // output_solution() const;

  // void
  // output_results() const;

  // void
  // print_parameters() const;

  // void
  // compute_internal_and_boundary_stress(
  //   bool openfilefirsttime) const;

  // void
  // compute_face_stress(bool /* openfilefirsttime */){};

  // // TrilinosWrappers::MPI::Vector
  // // output_pressure(bool openfilefirsttime) /*const*/;
  // void
  // output_coupling_pressure(bool) const;
};

#  endif

#endif