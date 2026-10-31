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
     * Compute the pressure that the tissue exerts on each vessel wall from
     * the reduced Lagrange multiplier, and store it in
     * pressure_at_immersed_surface (one value per vessel) and
     * pressure_at_immersed_surface_by_inclusion (one value per inclusion, or
     * per support point of the reduced grid with tensor-product coupling).
     *
     * The pressure is the mean normal stress of the tissue on the wall,
     * @f[
     *   p_v = \frac{1}{|\Gamma_v|} \int_{\Gamma_v} \mathbf{t}\cdot\mathbf{n}
     *   \, d\Gamma ,
     * @f]
     * where @f$\mathbf{t}@f$ is the force per unit area that the multiplier
     * applies to the tissue and @f$\mathbf{n}@f$ is the outward unit normal
     * of the vessel wall @f$\Gamma_v@f$. Only the multiplier coefficients of
     * @f$\cos\theta\,\mathbf{e}_x@f$ and @f$\sin\theta\,\mathbf{e}_y@f$ (in
     * the frame of the vessel) have a nonzero mean normal component, so only
     * they enter the pressure: the coefficient indices 3 and 7 in 3D, 2 and 5
     * in 2D. The result has the sign of the multiplier.
     *
     * <b>Point coupling.</b> Each multiplier coefficient is a force. Inclusion
     * @f$k@f$ uses the multiplier block @f$b(k)@f$: its own block, or the
     * block of its segment with "Cluster inclusions with segments". With
     * @f$|S_k| = 2\pi r_k |\mathbf{d}_k|@f$ the area of its wall (@f$2\pi
     * r_k@f$ in 2D),
     * @f[
     *   p_k = \frac{\lambda_{b(k),3} + \lambda_{b(k),7}}{2\,|S_k|},
     *   \qquad
     *   p_v = \frac{\sum_{k \in v} \frac12 (\lambda_{b(k),3} +
     *   \lambda_{b(k),7})}{\sum_{k \in v} |S_k|},
     * @f]
     * so @f$p_v@f$ is the mean of @f$p_k@f$ weighted by the wall areas, and
     * for a constant radius it is the sum of the normal forces divided by
     * @f$2\pi r L_v@f$, with @f$L_v@f$ the vessel length. Both values do not
     * change if the vessel is split into more, shorter inclusions.
     *
     * <b>Tensor-product coupling.</b> The multiplier is a finite element field
     * on the reduced grid whose component @f$c@f$ is the coefficient
     * @f$\Lambda_c(s)@f$ of the cross-section mode @f$\phi_c@f$, scaled to
     * maximum one as in point coupling, so the radial modes are
     * @f$\cos\theta\,\mathbf{e}_x@f$ and @f$\sin\theta\,\mathbf{e}_y@f$.
     * The mean normal stress on the cross section at @f$s@f$ and on vessel
     * @f$v@f$ are
     * @f[
     *   p(s) = \frac{\Lambda_3(s) + \Lambda_7(s)}{2},
     *   \qquad
     *   p_v = \frac{1}{|\gamma_v|} \int_{\gamma_v} p(s)\, ds .
     * @f]
     * The factor @f$1/2@f$ is computed with the quadrature of the
     * reference cross section. The value stored per support point is the
     * nodal value of @f$p@f$, and @f$p_v@f$ is a mean of these values
     * weighted by @f$\int_{\gamma_v} \psi_n\, ds@f$.
     *
     * <b>Material inside the vessels.</b> The elastic material also fills
     * the vessels, so the multiplier gives the jump
     * @f$\sigma^+_{nn} - \sigma^-_{nn}@f$ between the normal stress of the
     * tissue and that of the interior material. If
     * ElasticityProblemParameters::remove_interior_stress_from_pressure is
     * true (the default), the analytic plane-strain interior stress
     * @f[
     *   \bar\sigma^-_{nn} = \frac{2(\lambda+\mu)}{r}\, \bar u_n
     * @f]
     * is added to @f$p_k@f$ and @f$p(s)@f$ (and therefore to @f$p_v@f$), so
     * that the pressure is the normal stress of the tissue alone. Here
     * @f$\bar u_n@f$ is the mean normal wall displacement of the imposed data,
     * @f$r@f$ the radius, and @f$\lambda, \mu@f$ the Lame parameters of the
     * cells that contain the coupling points.
     *
     * The derivation is in doc/concepts/vessel-pressure.md.
     */
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
    /// @}

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
