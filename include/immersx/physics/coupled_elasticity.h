/* ---------------------------------------------------------------------
 */
#ifndef immersx_coupled_elasticity_h
#define immersx_coupled_elasticity_h

#include <immersx/physics/elasticity.h>



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

    /**
     * Compute tissue pressure from physical wall samples with fixed overlap
     * flags. Missing tissue contributes zero; normalization always uses the
     * complete nominal section. Every selected multiplier mode enters the
     * physical traction before taking its normal component.
     *
     * For point coupling the coefficients are forces. Inclusion k uses its
     * own multiplier block b(k), or its segment's block when clustering:
     * @f$\mathbf t_q=|S_k|^{-1}\sum_c\lambda_{b(k),c}
     * \boldsymbol\phi_c(q)@f$. The inclusion and vessel values are
     * @f[
     * p_k=\frac{\sum_{q\in S_k}w_q\chi_q\sigma_q}{|S_k|},\qquad
     * p_v=\frac{\sum_{k\in v}\sum_{q\in S_k}w_q\chi_q\sigma_q}
     *                 {\sum_{k\in v}|S_k|},\qquad
     * \sigma_q=\mathbf t_q\cdot\mathbf n_q+\sigma^-_{nn,q}.
     * @f]
     * On a complete circle with resolved Fourier products, only the radial
     * moments cos(theta) e_x and sin(theta) e_y survive, each with mean 1/2.
     * These have coefficient indices 3 and 7 in 3D, or 2 and 5 in 2D.
     * Their cancellation with other modes does not hold on a partial section.
     *
     * For tensor-product coupling, coefficients are traction densities:
     * @f$\mathbf t_{j\ell}=\sum_c\Lambda_c(s_j)
     * \boldsymbol\phi_c(j,\ell)@f$. First compute the covered-section mean
     * using all nominal section weights in its denominator:
     * @f[
     * p_{\rm cov}(s_j)=
     * \frac{\sum_\ell w_{j\ell}\chi_{j\ell}\sigma_{j\ell}}
     *      {\sum_\ell w_{j\ell}}.
     * @f]
     * Preserve the centerline averaging convention, including for a variable
     * radius. If v_j are the representative quadrature weights, then
     * @f$p_v=\sum_{j\in\gamma_v}v_j p_{\rm cov}(s_j)/
     * \sum_{j\in\gamma_v}v_j@f$. The values stored per reduced support point
     * are coefficients of a scalar L2 projection with the nominal centerline
     * mass:
     * @f[
     * M_{ik}=\sum_jv_j\psi_i(s_j)\psi_k(s_j),\qquad
     * b_i=\sum_jv_j p_{\rm cov}(s_j)\psi_i(s_j),\qquad Mp=b.
     * @f]
     * Vessel values are stored in pressure_at_immersed_surface; inclusion
     * values or projected coefficients are stored in
     * pressure_at_immersed_surface_by_inclusion.
     *
     * If ElasticityProblemParameters::remove_interior_stress_from_pressure
     * is true, retain the axisymmetric plane-strain correction
     * @f$\sigma^-_{nn,q}=\overline K\bar a_n(s_q)/r(s_q)@f$.
     * Here @f$\overline K@f$ is the arithmetic mean of 2(lambda_L+mu) over the
     * located particles of the inclusion or reduced cell; it is zero when
     * there are no such particles. No default material is assigned outside.
     * The radius and mean radial datum use the nominal geometry. Point
     * coupling uses the full nominal rhs moments, averaged over a segment's
     * inclusions when clustering. Tensor coupling first projects the original
     * datum into the modal FE space using the full nominal Gram and rhs,
     * independently of overlap and without changing the accepted multiplier.
     * On a resolved circle the radial mean is
     * @f$\bar a_n=(a_d+a_{2d+1})/2@f$. Only these radial modes enter the
     * interior correction. A rigid translation contributes no interior
     * stress, including on a partial wall. This closure retains the previous
     * mean radial model and does not infer a general angular interior stress.
     *
     * See doc/how-to/support-aware-weak-terms.md for the fixed quadrature
     * support and geometry convention.
     */
    void
    compute_pressure_at_immersed_surface();

    void
    update_vessel_radius_data(std::vector<double> new_data);
    void
    update_vessel_radius_data(std::vector<double> new_data,
                              std::vector<int>    cells_per_vessel);

    /** Interpolate gathered inclusion pressures to vessel samples.
     * A single inclusion supplies a constant pressure; a single requested
     * sample receives the vessel mean. Each vessel needs at least one sample.
     * This callback performs no MPI operations and may run only on rank zero
     * after the caller gathers the pressure vector.
     */
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

    /**
     * Mean normal stress on the wall of each vessel, @f$p_v@f$, indexed by
     * vessel id. See compute_pressure_at_immersed_surface().
     */
    TrilinosWrappers::MPI::Vector pressure_at_immersed_surface;

    /**
     * With point coupling: mean normal stress on the wall of each inclusion,
     * @f$p_k@f$, indexed by inclusion id. With tensor-product coupling: the
     * pressure @f$p(s)@f$ at each support point of the reduced grid, in the
     * order of the degrees of freedom of the first component of the reduced
     * DoFHandler. See compute_pressure_at_immersed_surface().
     */
    TrilinosWrappers::MPI::Vector pressure_at_immersed_surface_by_inclusion;

  private:
    void
    refine_and_transfer_around_inclusions();

    void
    output_pressure_at_immersed_surface(bool) const;

    std::vector<unsigned int> vessel_index_of_a_cell;

    unsigned int n_vessels_from_input = 0;
    // unsigned int n_vessels;
  };

} // namespace ImmersX

#endif
