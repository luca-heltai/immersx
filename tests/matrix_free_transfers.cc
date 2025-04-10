// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the reduced_lagrange_multipliers application, based on
// the deal.II library.
//
// The reduced_lagrange_multipliers application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the file LICENSE.md at the top level
// of the reduced_lagrange_multipliers distribution.
//
// ---------------------------------------------------------------------

#include <deal.II/base/function_lib.h>

#include <matrix_free_utils.h>

#include "../tests/tests.h"
#include "laplacian.h"

// Test the matrix-free transfer.

template <int dim>
void
test(const std::vector<std::vector<double>> &inclusions,
     const MPI_Comm                          mpi_comm)
{
  parallel::distributed::Triangulation<dim> tria{mpi_comm};
  GridGenerator::hyper_cube(tria, -1, 1);
  tria.refine_global(6);

  const unsigned int Nq = 100;
  const unsigned int Nc = 5;
  Inclusions<dim>    inclusion;
  inclusion.set_n_q_points(Nq);
  inclusion.set_n_coefficients(Nc);
  inclusion.inclusions = inclusions;
  inclusion.initialize();

  AffineConstraints<double> constraints;
  inclusion.setup_inclusions_particles(tria);
  DoFHandler<dim> dof_handler(tria);
  MappingQ1<dim>  mapping;
  FE_Q<dim>       fe{1};
  dof_handler.distribute_dofs(fe);


  // TODO: remove usage of particles here
  IndexSet                             relevant(inclusion.n_dofs());
  std::vector<types::global_dof_index> inclusion_dof_indices;
  auto particle = inclusion.inclusions_as_particles.begin();
  while (particle != inclusion.inclusions_as_particles.end())
    {
      const auto &cell = particle->get_surrounding_cell();

      const auto pic =
        inclusion.inclusions_as_particles.particles_in_cell(cell);
      Assert(pic.begin() == particle, ExcInternalError());
      std::set<types::global_dof_index> inclusion_dof_indices_set;
      for (const auto &p : pic)
        {
          const auto ids = inclusion.get_dof_indices(p.get_id());
          inclusion_dof_indices_set.insert(ids.begin(), ids.end());
        }
      inclusion_dof_indices.resize(0);
      inclusion_dof_indices.insert(inclusion_dof_indices.begin(),
                                   inclusion_dof_indices_set.begin(),
                                   inclusion_dof_indices_set.end());


      relevant.add_indices(inclusion_dof_indices.begin(),
                           inclusion_dof_indices.end());
      particle = pic.end();
    }



  CouplingOperator<dim, double, 1> coupling_mf{
    inclusion, dof_handler, constraints, mapping, fe};

  LinearAlgebra::distributed::Vector<double> src, dst;

  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
  src.reinit(dof_handler.locally_owned_dofs(), locally_relevant_dofs, mpi_comm);
  Tensor<1, dim> exponents;
  exponents[0] = 1;
  exponents[1] = 2;
  if constexpr (dim == 3)
    exponents[2] = 1;
  VectorTools::interpolate(dof_handler, Functions::Monomial{exponents}, src);


  auto inclusions_set = Utilities::MPI::create_evenly_distributed_partitioning(
    mpi_comm, inclusion.n_inclusions());
  IndexSet owned_dofs = inclusions_set.tensor_product(
    complete_index_set(inclusion.get_n_coefficients()));

  dst.reinit(owned_dofs, relevant, mpi_comm);

  coupling_mf.Tvmult(dst, src);
  const double dst_norm = dst.l2_norm();
  deallog << "dst norm (Tvmult) = " << dst_norm << std::endl;

  LinearAlgebra::distributed::Vector<double> dstvmult;
  dstvmult.reinit(src);
  coupling_mf.vmult(dstvmult, dst);
  const double dstvmult_norm = dstvmult.l2_norm();
  deallog << "dst norm (vmult) = " << dstvmult_norm << std::endl;
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;
  deallog.precision(4);

  test<2>({{0, 0, .2}}, MPI_COMM_WORLD);
  // Half a radius circle, Dirichlet data
  test<2>({{0, 0, .5}}, MPI_COMM_WORLD);
}
