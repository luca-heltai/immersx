#include <immersx/physics/elasticity.h>

// Implement coupling-specific accessors and helpers.

namespace ImmersX
{
  template <int dim, int spacedim>
  bool
  ElasticityProblem<dim, spacedim>::uses_tensor_product_coupling() const
  {
    return par.coupling_type == CouplingType::TensorProduct;
  }

  template <int dim, int spacedim>
  types::global_dof_index
  ElasticityProblem<dim, spacedim>::n_multiplier_dofs() const
  {
    if (uses_tensor_product_coupling())
      {
        if (tensor_product_coupling)
          return tensor_product_coupling->get_dof_handler().n_dofs();
        else
          return static_cast<types::global_dof_index>(0);
      }
    else
      return inclusions.n_dofs();
  }

  template <int dim, int spacedim>
  unsigned int
  ElasticityProblem<dim, spacedim>::n_multiplier_components_per_coupling_dof()
    const
  {
    if (uses_tensor_product_coupling())
      return spacedim;
    else
      return inclusions.n_dofs_per_inclusion();
  }

  // Setup helper implementations

  template <int dim, int spacedim>
  void
  ElasticityProblem<dim, spacedim>::setup_point_coupling_dofs()
  {
    if (inclusions.n_dofs() > 0)
      {
        if (inclusions.cluster_with_segments)
          {
            auto inclusions_segment_set_vector =
              Utilities::MPI::create_ascending_partitioning(
                mpi_communicator, inclusions.n_local_segments());

            auto inclusions_segment_set =
              inclusions_segment_set_vector[Utilities::MPI::this_mpi_process(
                mpi_communicator)];

            owned_dofs[1] = inclusions_segment_set.tensor_product(
              complete_index_set(inclusions.n_dofs_per_inclusion()));
          }
        else
          {
            auto inclusions_set =
              Utilities::MPI::create_evenly_distributed_partitioning(
                mpi_communicator, inclusions.n_inclusions());
            owned_dofs[1] = inclusions_set.tensor_product(
              complete_index_set(inclusions.n_dofs_per_inclusion()));
          }

        DynamicSparsityPattern dsp(dh.n_dofs(),
                                   inclusions.n_dofs(),
                                   relevant_dofs[0]);

        relevant_dofs[1] = assemble_coupling_sparsity(dsp);
        relevant_dofs[1].add_indices(owned_dofs[1]);
        SparsityTools::distribute_sparsity_pattern(dsp,
                                                   owned_dofs[0],
                                                   mpi_communicator,
                                                   relevant_dofs[0]);
        coupling_matrix.clear();
        coupling_matrix.reinit(owned_dofs[0],
                               owned_dofs[1],
                               dsp,
                               mpi_communicator);

        DynamicSparsityPattern idsp(relevant_dofs[1]);
        for (const auto i : relevant_dofs[1])
          idsp.add(i, i);

        SparsityTools::distribute_sparsity_pattern(idsp,
                                                   owned_dofs[1],
                                                   mpi_communicator,
                                                   relevant_dofs[1]);
        inclusion_matrix.clear();
        inclusion_matrix.reinit(owned_dofs[1],
                                owned_dofs[1],
                                idsp,
                                mpi_communicator);
      }
  }


  template <int dim, int spacedim>
  void
  ElasticityProblem<dim, spacedim>::setup_tensor_product_coupling_dofs()
  {
#if defined(DEAL_II_WITH_VTK)
    AssertThrow(tensor_product_coupling,
                ExcMessage("Tensor-product coupling not initialized"));

    const auto &tensor_product_dh = tensor_product_coupling->get_dof_handler();

    owned_dofs[1] = tensor_product_dh.locally_owned_dofs();
    relevant_dofs[1] =
      DoFTools::extract_locally_relevant_dofs(tensor_product_dh);

    DynamicSparsityPattern dsp(dh.n_dofs(),
                               tensor_product_dh.n_dofs(),
                               relevant_dofs[0]);
    tensor_product_coupling->assemble_coupling_sparsity(dsp, dh, constraints);

    SparsityTools::distribute_sparsity_pattern(dsp,
                                               owned_dofs[0],
                                               mpi_communicator,
                                               relevant_dofs[0]);

    coupling_matrix.clear();
    coupling_matrix.reinit(owned_dofs[0], owned_dofs[1], dsp, mpi_communicator);

    DynamicSparsityPattern               idsp(relevant_dofs[1]);
    std::vector<types::global_dof_index> tensor_product_cell_dofs(
      tensor_product_dh.get_fe().n_dofs_per_cell());
    for (const auto &cell : tensor_product_dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(tensor_product_cell_dofs);
          for (const auto row : tensor_product_cell_dofs)
            for (const auto col : tensor_product_cell_dofs)
              idsp.add(row, col);
        }
    SparsityTools::distribute_sparsity_pattern(idsp,
                                               owned_dofs[1],
                                               mpi_communicator,
                                               relevant_dofs[1]);

    inclusion_matrix.clear();
    inclusion_matrix.reinit(owned_dofs[1],
                            owned_dofs[1],
                            idsp,
                            mpi_communicator);
#else
    (void)dim;
    (void)spacedim; // silence warnings if compiled without VTK
#endif
  }


  template <int dim, int spacedim>
  void
  ElasticityProblem<dim, spacedim>::assemble_tensor_product_coupling()
  {
#if defined(DEAL_II_WITH_VTK)
    AssertThrow(tensor_product_coupling,
                ExcMessage("Tensor-product coupling not initialized"));

    coupling_matrix     = 0;
    inclusion_matrix    = 0;
    system_rhs.block(1) = 0;
    force_rhs.block(1)  = 0;

    // Build the coupling matrix between background and reduced space
    tensor_product_coupling->assemble_coupling_matrix(coupling_matrix,
                                                      dh,
                                                      constraints);

    // Assemble the tensor-product rhs and copy it into the system rhs.
    tensor_product_coupling->assemble_reduced_rhs(force_rhs.block(1));
    system_rhs.block(1) = force_rhs.block(1);

    // Assemble the tensor-product mass matrix.
    tensor_product_coupling->assemble_coupling_mass_matrix(inclusion_matrix);

    if (n_multiplier_dofs() > 0)
      {
        Teuchos::ParameterList amg_parameter_list;
        amg_parameter_list.set("smoother: type", "Chebyshev");
        amg_parameter_list.set("smoother: sweeps", 2);
        amg_parameter_list.set("smoother: pre or post", "both");
        amg_parameter_list.set("coarse: type", "Amesos-KLU");
        amg_parameter_list.set("coarse: max size", 2000);
        amg_parameter_list.set("aggregation: threshold", 0.02);
        prec_M.initialize(inclusion_matrix, amg_parameter_list);
      }
#else
    (void)dim;
    (void)spacedim;
#endif
  }


  // Solver helpers

  template <int dim, int spacedim>
  bool
  ElasticityProblem<dim, spacedim>::has_immersed_coupling() const
  {
    return n_multiplier_dofs() > 0;
  }


  template <int dim, int spacedim>
  void
  ElasticityProblem<dim, spacedim>::distribute_multiplier_solution(
    LA::MPI::Vector &lambda) const
  {
#if defined(DEAL_II_WITH_VTK)
    if (uses_tensor_product_coupling())
      {
        tensor_product_coupling->get_coupling_constraints().distribute(lambda);
        return;
      }
#endif
    inclusion_constraints.distribute(lambda);
  }


  template <int dim, int spacedim>
  void
  ElasticityProblem<dim, spacedim>::output_immersed_particles(
    const std::string &filename) const
  {
#if defined(DEAL_II_WITH_VTK)
    if (uses_tensor_product_coupling())
      {
        AssertThrow(tensor_product_coupling,
                    ExcMessage("Tensor-product coupling not initialized"));
        tensor_product_coupling->output_particles(filename);
        return;
      }
#endif
    inclusions.output_particles(filename);
  }

  // Explicit instantiations for supported templates
  template class ElasticityProblem<2>;
  template class ElasticityProblem<2, 3>;
  template class ElasticityProblem<3>;
} // namespace ImmersX
