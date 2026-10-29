#ifndef dealii_distributed_lagrange_multiplier_coupled_elasticity_h
#  define dealii_distributed_lagrange_multiplier_coupled_elasticity_h

#  include <immersx/physics/elasticity.h>



namespace ImmersX
{
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
    compute_pressure_at_immersed_surface();

    void
    update_vessel_radius_data(std::vector<double> new_data);
    void
    update_vessel_radius_data(std::vector<double> new_data,
                              std::vector<int>    cells_per_vessel);

    std::vector<std::vector<double>>
      split_pressure_over_inclusions(std::vector<int>, Vector<double>) const;

    /**
     * Fill vessel_index_of_a_cell from the "vessel_id" input file field and
     * set n_vessels(). Throws if the ids are not 0, 1, ..., n_vessels() - 1.
     */
    void
    make_vessel_index_of_a_cell();

    /** Vessel id of each reduced cell, indexed by active_cell_index(). Entries
     * of cells not owned by this rank are numbers::invalid_unsigned_int. */
    const std::vector<unsigned int> &
    get_vessel_index_of_a_cell() const
    {
      return vessel_index_of_a_cell;
    }

    unsigned int
    n_vessels() const
    {
      if (!this->uses_tensor_product_coupling())
        return this->inclusions.get_n_vessels();
      else
        return n_vessels_from_input;
    };

    TrilinosWrappers::MPI::Vector pressure_at_immersed_surface; ///<
    TrilinosWrappers::MPI::Vector
      pressure_at_immersed_surface_by_inclusion; ///<
    /// @}

  private:
    void
    reassemble_coupling_rhs();

    void
    refine_and_transfer_around_inclusions();

    void
    output_pressure_at_immersed_surface(bool) const;

    std::vector<unsigned int> vessel_index_of_a_cell;

    unsigned int n_vessels_from_input = 0;
    // unsigned int n_vessels;
  };

#endif

} // namespace ImmersX
