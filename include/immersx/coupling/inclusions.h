// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// The ImmersX application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the file LICENSE.md at the top level
// of the ImmersX distribution.
//
// ---------------------------------------------------------------------

/**
 * @brief Header file that includes all necessary headers and defines the Inclusions class.
 *
 * This class represents a set of inclusions in a domain. It stores the
 * positions of the inclusions, their radii, and their Fourier coefficients. It
 * also provides methods to initialize the inclusions, compute their normals,
 * and extract their degrees of freedom.
 *
 * @tparam spacedim The dimension of the space in which the inclusions are embedded.
 */
#ifndef rdlm_inclusions
#define rdlm_inclusions

#include <deal.II/base/hdf5.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_function.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/tria_base.h>

#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/particles/data_out.h>
#include <deal.II/particles/generators.h>
#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/utilities.h>

#include <deal.II/physics/transformations.h>

#include <boost/algorithm/string.hpp>

#include <fstream>
#include <map>

using namespace dealii;

namespace ImmersX
{

  /**
   * @brief Class for handling inclusions in an immersed boundary method.
   *
   * This class provides functionality for handling inclusions in an immersed
   * boundary method. It stores the positions, radii, and Fourier coefficients
   * of the inclusions, and provides methods for computing their normals,
   * center, and direction. It also provides methods for initializing the
   * inclusions from a file, setting up particles for the inclusions, and
   * getting degrees of freedom associated with each inclusion.
   *
   * <h3>Fourier coefficients</h3>
   *
   * Data on the boundary of an inclusion (a circle in 2D, the cross-section
   * of a cylinder in 3D) is represented in the Fourier basis
   * @f[
   *   1,\ \cos\theta,\ \sin\theta,\ \cos 2\theta,\ \sin 2\theta,\ \dots
   * @f]
   * Fourier mode @f$m@f$ is the constant for @f$m=0@f$, @f$\cos(k\theta)@f$
   * for @f$m=2k-1@f$, and @f$\sin(k\theta)@f$ for @f$m=2k@f$. The basis
   * functions are not normalized: each of them has maximum value one.
   *
   * For a problem with @p n_vector_components components, every mode is
   * repeated once per component. A coefficient index @f$c@f$ refers to
   * Fourier mode @f$c / n_{\text{vector components}}@f$ and vector component
   * @f$c \% n_{\text{vector components}}@f$. For example, in 3D the indices
   * 0, 1, 2 are the constant mode in x, y, z, the indices 3, 4, 5 are
   * @f$\cos\theta@f$ in x, y, z, and the indices 6, 7, 8 are
   * @f$\sin\theta@f$ in x, y, z. The constant mode always has index 0, and
   * every data line (from the "Data file" or the "Reference inclusion data")
   * uses this same indexing, starting from the constant mode.
   *
   * The coefficients in use are chosen with two parameters:
   *
   * - "Selection of Fourier coefficients": an explicit list of coefficient
   *   indices. Each selected index becomes one degree of freedom of the
   *   Lagrange multiplier of each inclusion, in the order of the list.
   * - "Number of fourier coefficients": the number of coefficients in use.
   *
   * initialize() combines them as follows:
   *
   * - Selection given, number 0: the selection is used, and the number is
   *   set to the size of the selection.
   * - Selection given, number equal to its size: the selection is used.
   * - Selection given, number different from its size: an exception is
   *   thrown.
   * - No selection, number 0: no coefficients are used. The Lagrange
   *   multiplier has no degrees of freedom, and the inclusions do not couple
   *   to the bulk problem, even if inclusions and data are given. A message
   *   on rank zero reports this case when inclusions exist.
   * - No selection, number @f$N>0@f$: the indices
   *   @f$0,\dots,N\,n_{\text{vector components}}-1@f$ are used, i.e., all
   *   vector components of the first @f$N@f$ Fourier modes.
   *
   * n_dofs_per_inclusion() always returns the number of coefficients in use.
   * Code that groups the Lagrange multiplier by inclusion must use it rather
   * than get_n_coefficients(): in the last case the two differ by a factor
   * @p n_vector_components.
   */
  template <int spacedim>
  class Inclusions : public ParameterAcceptor
  {
  public:
    template <int dim, typename number, int n_components>
    friend class MatrixFreeCouplingOperator;

    /**
     * @brief Class for computing the inclusions of a given mesh.
     *
     * @param n_vector_components Number of vector components.
     */
    Inclusions(const unsigned int n_vector_components = 1)
      : ParameterAcceptor("/Immersed Problem/Immersed inclusions")
      , inclusions_rhs("/Immersed Problem/Immersed inclusions/Boundary data",
                       n_vector_components)
      , n_vector_components(n_vector_components)
    {
      static_assert(spacedim > 1, "Not implemented in dim = 1");
      add_parameter("Inclusions refinement", n_q_points);
      add_parameter("Inclusions", inclusions);
      add_parameter("Reference inclusion data", reference_inclusion_data);

      add_parameter(
        "Number of fourier coefficients",
        n_coefficients,
        "Number of Fourier coefficients used on each inclusion. If "
        "\"Selection of Fourier coefficients\" is given, this must be 0 "
        "(it is then set to the size of the selection) or equal to the size "
        "of the selection; any other value is an error. If no selection is "
        "given and this is N > 0, all vector components of the first N "
        "Fourier modes are used, that is, the coefficient indices "
        "0, ..., N * n_vector_components - 1. If no selection is given and "
        "this is 0, no coefficients are used and the Lagrange multiplier has "
        "no degrees of freedom, even if inclusions and data are given.");
      add_parameter(
        "Selection of Fourier coefficients",
        selected_coefficients,
        "Explicit list of the Fourier coefficient indices used on each "
        "inclusion. Each index becomes one degree of freedom of the Lagrange "
        "multiplier of each inclusion, in the order of the list. Index c "
        "refers to Fourier mode c / n_vector_components and vector component "
        "c % n_vector_components. Mode 0 is the constant, mode 2k-1 is "
        "cos(k theta) and mode 2k is sin(k theta). For example, in 3D, 3 is "
        "cos(theta) in x and 7 is sin(theta) in y; in 2D, 2 is cos(theta) in "
        "x and 5 is sin(theta) in y. Data lines use the same indexing, "
        "starting from the constant mode, so the data for index c is entry c "
        "of the line. Leave empty to select coefficients with \"Number of "
        "fourier coefficients\" instead.");
      add_parameter("Bounding boxes extraction level", rtree_extraction_level);
      add_parameter("Inclusions file", inclusions_file);
      add_parameter("Data file", data_file);
      add_parameter("Cluster inclusions with segments", cluster_with_segments);

      enter_subsection("Boundary data");
      add_parameter("Modulation frequency", modulation_frequency);
      add_parameter("Phase shift", phase_shift);
      leave_subsection();

      auto reset_function = [this]() {
        this->prm.set("Function expression",
                      (spacedim == 2 ? "0; 0" : "0; 0; 0"));
      };
      inclusions_rhs.declare_parameters_call_back.connect(reset_function);
    }


    types::global_dof_index
    /**
     * Returns the number of degrees of freedom in the system.
     *
     * @return The number of degrees of freedom.
     */
    n_dofs() const
    {
      if (cluster_with_segments)
        return n_global_segments() * n_dofs_per_inclusion();
      else
        return inclusions.size() * n_dofs_per_inclusion();
    }


    types::global_dof_index
    /**
     * Returns the number of particles in the system.
     *
     * @return The number of particles in the system.
     */
    n_particles() const
    {
      return inclusions.size() * n_q_points;
    }


    types::global_dof_index
    /**
     * @brief Returns the number of inclusions in the mesh.
     *
     * @return The number of inclusions in the mesh.
     */
    n_inclusions() const
    {
      return inclusions.size();
    }

    unsigned int
    n_global_segments() const
    {
      return global_max_segment_index;
    }

    unsigned int
    n_local_segments() const
    {
      return local_max_segment_index;
    }

    /**
     * @brief Number of degrees of freedom associated to each inclusion.
     *
     * This is the number of Fourier coefficients in use, i.e., the size of
     * the selection after initialize(). It can be zero. See the class
     * documentation for how the selection is built.
     */
    unsigned int
    n_dofs_per_inclusion() const
    {
      return selected_coefficients.size();
    }

    /**
     * Reinit all reference points and normals, read the inclusions file and
     * the data file, and build the selection of Fourier coefficients.
     *
     * The selection is built from "Number of fourier coefficients" and
     * "Selection of Fourier coefficients" as described in the class
     * documentation. An explicit selection whose size differs from a nonzero
     * number of coefficients throws an exception. An empty selection is
     * replaced by all vector components of the first get_n_coefficients()
     * modes, and an explicit selection with zero coefficients sets the
     * number of coefficients to the size of the selection.
     *
     * This function appends the inclusions read from file and fills the
     * selection, so it must be called only once per set of parameters.
     */
    void
    initialize()
    {
      AssertThrow(n_q_points > 0,
                  ExcMessage(
                    "Refinement of inclusions must be greater than zero."));
      support_points.resize(n_q_points);
      normals.resize(n_q_points);
      theta.resize(n_q_points);
      current_support_points.resize(n_q_points);

      for (unsigned int i = 0; i < n_q_points; ++i)
        {
          theta[i]             = i * 2 * numbers::PI / n_q_points;
          support_points[i][0] = std::cos(theta[i]);
          support_points[i][1] = std::sin(theta[i]);
          normals[i]           = support_points[i];
        }

      if (!selected_coefficients.empty())
        {
          AssertThrow(
            n_coefficients == 0 ||
              n_coefficients == selected_coefficients.size(),
            ExcMessage(
              "\"Number of fourier coefficients\" is " +
              std::to_string(n_coefficients) + ", but \"Selection of Fourier "
              "coefficients\" has " +
              std::to_string(selected_coefficients.size()) +
              " entries. Set the number to 0 or to the size of the "
              "selection."));
          n_coefficients = selected_coefficients.size();
        }
      else
        {
          // Use all vector components of the first n_coefficients modes.
          selected_coefficients.resize(n_coefficients * n_vector_components);
          for (unsigned int i = 0; i < n_coefficients * n_vector_components;
               ++i)
            selected_coefficients[i] = i;
        }

      // This MUST be here, otherwise n_dofs_per_inclusions() would be wrong.
      current_fe_values.resize(n_dofs_per_inclusion());

      if (inclusions_file != "")
        {
          std::ifstream infile(inclusions_file);
          Assert(infile, ExcIO());

          double buffer_double;
          // cx, cy, R or cx,cy,cz,dx,dy,dz,R,vesselID
          const unsigned int  size = (spacedim == 2 ? 3 : 8);
          std::vector<double> inclusion(size);

          while (infile >> buffer_double)
            {
              inclusion[0] = buffer_double;
              for (unsigned int i = 1; i < size; ++i)
                {
                  Assert(infile, ExcIO());
                  infile >> inclusion[i];
                }
              inclusions.push_back(inclusion);
            }
        }

      if (data_file != "")
        {
          std::ifstream infile(data_file);
          Assert(infile, ExcIO());

          std::string line;
          while (std::getline(infile, line))
            {
              std::vector<double> double_line;
              std::istringstream  iss(line);
              double              buffer_double;
              while (iss >> buffer_double)
                {
                  double_line.push_back(buffer_double);
                }
              inclusions_data.push_back(double_line);
            }
          AssertThrow(inclusions_data.size() == n_inclusions(),
                      ExcDimensionMismatch(inclusions_data.size(),
                                           n_inclusions()));
          if (inclusions_data.size() > 0)
            {
              const auto N = inclusions_data[0].size();
              for (const auto &l : inclusions_data)
                {
                  AssertThrow(l.size() == N, ExcDimensionMismatch(l.size(), N));
                }
              if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
                {
                  std::cout
                    << "rank "
                    << Utilities::MPI::this_mpi_process(mpi_communicator)
                    << ": Read " << N << " coefficients per "
                    << inclusions.size() << " inclusion" << std::endl;
                }
            }
        }
      else
        {
          while (inclusions_data.size() < inclusions.size())
            {
              inclusions_data.push_back(reference_inclusion_data);
            }
        }

      if (n_dofs_per_inclusion() == 0 && n_inclusions() > 0 &&
          Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        std::cout << "Inclusions: " << n_inclusions()
                  << " inclusions are defined, but no Fourier coefficients "
                     "are selected (\"Number of fourier coefficients\" = 0 "
                     "and empty \"Selection of Fourier coefficients\"). The "
                     "Lagrange multiplier has no degrees of freedom and the "
                     "inclusions do not couple to the bulk problem."
                  << std::endl;
      check_vessels();
    }


    /**
     * @brief Returns the degrees of freedom indices associated with a given quadrature point.
     *
     * @param quadrature_id The global index of the quadrature point.
     * @return A vector containing the degrees of freedom indices.
     */
    std::vector<types::global_dof_index>
    get_dof_indices(const types::global_dof_index &quadrature_id) const
    {
      AssertIndexRange(quadrature_id, n_particles());
      std::vector<types::global_dof_index> dofs(n_dofs_per_inclusion());
      unsigned int                         start_index;
      if (cluster_with_segments)
        start_index = get_segment_index(quadrature_id) * n_dofs_per_inclusion();
      else
        start_index = get_inclusion_id(quadrature_id) * n_dofs_per_inclusion();
      for (auto &d : dofs)
        d = start_index++;
      return dofs;
    }

    /**
     * @brief Sets up the inclusions particles for the given triangulation.
     *
     * @param tria The triangulation to set up the inclusions particles for.
     */
    void
    setup_inclusions_particles(
      const parallel::TriangulationBase<spacedim> &tria)
    {
      mpi_communicator = tria.get_mpi_communicator();
      initialize();

      inclusions_as_particles.initialize(tria,
                                         StaticMappingQ1<spacedim>::mapping);
      particles_on_centerline.initialize(tria,
                                         StaticMappingQ1<spacedim>::mapping);
      if (n_inclusions() == 0)
        return;

      // Only add particles once.
      auto inclusions_set =
        Utilities::MPI::create_evenly_distributed_partitioning(mpi_communicator,
                                                               n_inclusions());

      std::vector<Point<spacedim>> particles_positions;
      std::vector<Point<spacedim>> central_particle_positions;
      particles_positions.reserve(n_particles());
      central_particle_positions.reserve(n_inclusions());
      for (const auto i : inclusions_set)
        {
          const auto &p = get_current_support_points(i);
          particles_positions.insert(particles_positions.end(),
                                     p.begin(),
                                     p.end());
          central_particle_positions.push_back(get_center(i));
        }

      std::vector<BoundingBox<spacedim>> all_boxes;
      all_boxes.reserve(tria.n_locally_owned_active_cells());
      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
          all_boxes.emplace_back(cell->bounding_box());
      const auto tree = pack_rtree(all_boxes);
      const auto local_boxes =
        extract_rtree_level(tree, rtree_extraction_level);

      auto global_bounding_boxes =
        Utilities::MPI::all_gather(mpi_communicator, local_boxes);

      Assert(!global_bounding_boxes.empty(),
             ExcInternalError(
               "I was expecting the "
               "global_bounding_boxes to be filled at this stage. "
               "Make sure you fill this vector before trying to use it "
               "here. Bailing out."));
      inclusions_as_particles.insert_global_particles(particles_positions,
                                                      global_bounding_boxes);
      particles_on_centerline.insert_global_particles(
        central_particle_positions, global_bounding_boxes);

      // Sanity check.
      AssertDimension(inclusions_as_particles.n_global_particles(),
                      n_particles());
      AssertDimension(particles_on_centerline.n_global_particles(),
                      n_inclusions());

      if (cluster_with_segments)
        build_segment_index_vector();
    }

    /**
     * Quadrature id to inclusion id.
     *
     * @param quadrature_id
     * @return const types::global_dof_index
     */
    inline types::global_dof_index
    get_inclusion_id(const types::global_dof_index &quadrature_id) const
    {
      AssertIndexRange(quadrature_id, n_particles());
      return (quadrature_id / n_q_points);
    }

    /**
     * @brief Get the ith component for the given dof index.
     *
     * @param dof_index A number in [0,n_dofs())
     * @return unsigned int The index of the current component
     */
    inline unsigned int
    get_component(const types::global_dof_index &dof_index) const
    {
      AssertIndexRange(dof_index, n_dofs());
      // return dof_index % n_vector_components;
      return selected_coefficients[dof_index % n_dofs_per_inclusion()] %
             n_vector_components;
    }


    /**
     * @brief Get the ith Fourier component for the given dof index.
     *
     * @param dof_index A number in [0,n_dofs())
     * @return unsigned int The index of the current component
     */
    inline unsigned int
    get_fourier_component(const types::global_dof_index &dof_index) const
    {
      AssertIndexRange(dof_index, n_dofs());
      // return dof_index % n_vector_components;
      return selected_coefficients[dof_index % n_dofs_per_inclusion()];
    }


    /**
     * @brief Get the Fourier data for the given local dof index.
     *
     * @param inclusion_id A number in [0,n_inclusions())
     * @param dof_index A number in [0,n_dofs_per_inclusion())
     * @return unsigned int The index of the current component
     */
    inline double
    get_inclusion_data(const types::global_dof_index &inclusion_id,
                       const types::global_dof_index &dof_index) const
    {
      AssertIndexRange(inclusion_id, n_inclusions());
      AssertIndexRange(dof_index, n_dofs());
      // return dof_index % n_vector_components;
      return inclusions_data[inclusion_id][get_fourier_component(dof_index)];
    }

    /**
     * @brief Get the Fourier data for the given local dof index, computed at the
     * given quadrature point.
     *
     * @param inclusion_id A number in [0,n_inclusions())
     * @param dof_index A number in [0,n_dofs_per_inclusion())
     * @param point The inclusion quadrature point location
     * @return unsigned int The index of the current component
     */
    inline double
    get_inclusion_data(const types::global_dof_index &inclusion_id,
                       const types::global_dof_index &dof_index,
                       const Point<spacedim>         &point) const
    {
      AssertIndexRange(inclusion_id, n_inclusions());
      AssertIndexRange(dof_index, n_dofs());
      // return dof_index % n_vector_components;
      if (inclusions_data.size() > 0) // If we have data
        {
          if (n_vector_components == 1)
            return inclusions_data[inclusion_id]
                                  [get_fourier_component(dof_index)];
          else if (n_vector_components == spacedim)
            {
              return inclusions_data[inclusion_id]
                                    [get_fourier_component(dof_index)];
            }
          else
            {
              AssertThrow(false, ExcNotImplemented());
              return 0.0;
            }
        }
      else
        {
          return inclusions_rhs.value(point, get_component(dof_index));
        }
    }



    /**
     * @brief Get the normal
     *
     * @param quadrature_id
     * @return const Tensor<1, spacedim>&
     */
    inline const Tensor<1, spacedim> &
    get_normal(const types::global_dof_index &quadrature_id) const
    {
      AssertIndexRange(quadrature_id, n_particles());
      if constexpr (spacedim == 2)
        return (normals[quadrature_id % n_q_points]);
      else
        {
          const auto inclusion_id = get_inclusion_id(quadrature_id);
          const auto rotation     = get_rotation(inclusion_id);
          return (rotation * normals[quadrature_id % n_q_points]);
        }
    }

    /**
     * @brief return weight for integration normal direction of an inclusion at a quadrature point
     *
     * @param particle_id
     * @return const Tensor<1, spacedim>&
     */
    inline double
    get_JxW(const types::global_dof_index &particle_id) const
    {
      AssertIndexRange(particle_id, n_particles());
      const auto id = particle_id / n_q_points;
      AssertIndexRange(id, inclusions.size());
      const auto r = get_radius(id);
      return 2 * numbers::PI * r / n_q_points * get_direction(id).norm();
    }


    /**
     * Get a list of fe values
     *
     * @param particle_id
     * @return const std::vector<double>&
     */
    const std::vector<double> &
    get_fe_values(const types::global_dof_index particle_id) const
    {
      AssertIndexRange(particle_id, n_particles());
      // No coefficients in use: the multiplier has no dofs on this inclusion.
      if (n_dofs_per_inclusion() == 0)
        return current_fe_values;
      const auto q  = particle_id % n_q_points;
      const auto id = particle_id / n_q_points;
      AssertIndexRange(id, inclusions.size());
      (void)id;
      const auto scaling_factor = 1.0;

      unsigned int basis_local_id = 0;
      for (unsigned int basis : selected_coefficients)
        {
          const unsigned int fourier_index =
            basis / n_vector_components + 0; // coefficient_offset;
          unsigned int omega = (fourier_index + 1) / 2;

          if (fourier_index == 0)
            current_fe_values[basis_local_id] = scaling_factor;
          else if ((fourier_index - 1) % 2 == 0)
            current_fe_values[basis_local_id] =
              scaling_factor * std::cos(theta[q] * omega);
          else
            current_fe_values[basis_local_id] =
              scaling_factor * std::sin(theta[q] * omega);
          ++basis_local_id;
        }
      return current_fe_values;
    }


    /**
     * @brief Get the center of the inclusion
     *
     * @param inclusion_id
     * @return Point<spacedim>
     */
    Point<spacedim>
    get_center(const types::global_dof_index &inclusion_id) const
    {
      AssertIndexRange(inclusion_id, inclusions.size());
      const auto &inclusion = inclusions[inclusion_id];
      Assert(inclusion.size() > spacedim, ExcInternalError());
      Point<spacedim> center;
      for (unsigned int d = 0; d < spacedim; ++d)
        center[d] = inclusion[d];
      return center;
    }


    /**
     * @brief Get the radius of the inclusion
     *
     * @param inclusion_id
     * @return double
     */
    double
    get_radius(const types::global_dof_index &inclusion_id) const
    {
      AssertIndexRange(inclusion_id, inclusions.size());
      const auto &inclusion = inclusions[inclusion_id];
      if constexpr (spacedim == 2)
        {
          AssertDimension(inclusion.size(), spacedim + 1);
          return inclusion[spacedim];
        }
      else
        {
          AssertDimension(inclusion.size(), 2 * spacedim + 2);
          return inclusion[2 * spacedim];
        }
    }

    /**
     * @brief Get the measure of the section of the inclusion
     *
     * @param inclusion_id
     * @return double
     */
    double
    get_section_measure(const types::global_dof_index &inclusion_id) const
    {
      auto r = get_radius(inclusion_id);
      if constexpr (spacedim == 2)
        return 2 * numbers::PI * r;
      else
        {
          auto ds = get_direction(inclusion_id).norm();
          return 2 * numbers::PI * r * ds;
        }
    }


    /**
     * @brief Get the direction of the inclusion
     *
     * @param inclusion_id
     * @return Tensor<1, spacedim>
     */
    Tensor<1, spacedim>
    get_direction(const types::global_dof_index &inclusion_id) const
    {
      AssertIndexRange(inclusion_id, inclusions.size());
      if constexpr (spacedim == 2)
        {
          // No direction in 2d. But the norm is used.
          Tensor<1, spacedim> ret;
          ret[0] = 1.0;
          return ret;
        }
      else
        {
          const auto &inclusion = inclusions[inclusion_id];
          AssertDimension(inclusion.size(), 2 * spacedim + 2);
          Tensor<1, spacedim> direction;
          for (unsigned int d = 0; d < spacedim; ++d)
            direction[d] = inclusion[spacedim + d];
          AssertThrow(direction.norm() > 1e-10,
                      ExcMessage("Expecting a direction with non-zero norm"));
          return direction;
        }
    }


    /**
     * @brief Get the rotation of the inclusion
     *
     * @param inclusion_id
     * @return Tensor<2, spacedim>
     */
    Tensor<2, spacedim>
    get_rotation(const types::global_dof_index &inclusion_id) const
    {
      Tensor<2, spacedim> rotation = unit_symmetric_tensor<spacedim>();
      if constexpr (spacedim == 2)
        {
          return rotation;
        }
      else if constexpr (spacedim == 3)
        {
          auto direction = get_direction(inclusion_id);
          direction /= direction.norm();

          // Build rotation w.r.t. z axis
          static const auto z_axis = Tensor<1, spacedim>({0, 0, 1});
          auto              v      = cross_product_3d(z_axis, direction);
          const auto        cos_t  = direction * z_axis;

          if (std::abs(cos_t + 1) < 1e-10)
            {
              rotation[1][1] = -1;
              rotation[2][2] = -1;
            }
          else
            {
              Tensor<2, spacedim> vx;
              vx[0]    = Tensor<1, spacedim>({0, -v[2], v[1]});
              vx[1]    = Tensor<1, spacedim>({v[2], 0, -v[0]});
              vx[2]    = Tensor<1, spacedim>({-v[1], v[0], 0});
              auto vx2 = vx * vx;
              rotation += vx + vx2 * (1 / (1 + cos_t));
            }

          return rotation;
        }
    }

    /**
     * Return support points of one inclusion in current configuration.
     */
    const std::vector<Point<spacedim>> &
    get_current_support_points(
      const types::global_dof_index &inclusion_id) const
    {
      const auto center   = get_center(inclusion_id);
      const auto radius   = get_radius(inclusion_id);
      const auto rotation = get_rotation(inclusion_id);

      for (unsigned int q = 0; q < n_q_points; ++q)
        current_support_points[q] =
          center + rotation * (support_points[q] * radius);
      return current_support_points;
    }


    /**
     * @brief print the inclusions in parallel on a .vtu file
     *
     * @param filename
     */
    void
    output_particles(const std::string &filename) const
    {
      Particles::DataOut<spacedim> particles_out;
      particles_out.build_patches(inclusions_as_particles);
      particles_out.write_vtu_in_parallel(filename, mpi_communicator);
    }

    /**
     * @brief Update the displacement data after the initialization reading from a hdf5 file
     */
    void
    update_displacement_hdf5()
    {
#ifdef DEAL_II_WITH_HDF5
      //
      inclusions_data.clear();
      inclusions_data.resize(n_inclusions());
      data_file_h =
        std::make_unique<HDF5::File>(data_file,
                                     HDF5::File::FileAccessMode::open,
                                     mpi_communicator);
      auto group = data_file_h->open_group("data");
      // Read new displacement
      {
        auto h5data         = group.open_dataset("displacement_data");
        auto vector_of_data = h5data.template read<Vector<double>>();

        auto inclusions_set =
          Utilities::MPI::create_evenly_distributed_partitioning(
            mpi_communicator, n_inclusions());
        for (const auto i : inclusions_set)
          {
            AssertIndexRange(i, vector_of_data.size());
            inclusions_data[i] = vector_of_data[i];
          }
      }
      // check that the new data is size consistente
      const auto N = inclusions_data[0].size();
      for (const auto &l : inclusions_data)
        {
          AssertThrow(l.size() == N, ExcDimensionMismatch(l.size(), N));
        }

#else
      AssertThrow(false, ExcNeedsHDF5());
#endif
    }

    /**
     * @brief Update the displacement data after the initialization, 3D case only
     *
     * @param new_data  vector of lenght equal to the number of inclusions or to the number of vessels, elements of the vector are the new values to be assigned for normal expansion
     */
    void
    update_inclusions_data(std::vector<double> new_data)
    {
      AssertThrow(
        spacedim == 3,
        ExcMessage(
          "update of inclusions data is only available for spacedim = 3"));

      // if (new_data.size() != 0 && inclusions.size() == 0)

      if (new_data.size() == n_vessels)
        {
          std::map<unsigned int, std::vector<types::global_dof_index>>::iterator
            it = map_vessel_inclusions.begin();
          while (it != map_vessel_inclusions.end())
            {
              // inclusions_data[it->second] = {new_data[it->first],
              // 0,0,0,new_data[it->first]};
              for (auto inclusion_id : it->second)
                update_single_inclusion_data_along_normal(inclusion_id,
                                                          new_data[it->first]);
              ++it;
            }
        }
      else if (new_data.size() == inclusions.size())
        {
          for (long unsigned int id = 0; id < new_data.size(); ++id)
            update_single_inclusion_data_along_normal(id, new_data[id]);
        }
      else
        AssertThrow(
          new_data.size() == 0,
          ExcMessage(
            "dimensions of new data for the update does not match the inclusions"));
    }

    /**
     * Update inclusion data by vessel-wise sampled profiles.
     */
    void
    update_inclusions_data(std::vector<std::vector<double>> new_data)
    {
      AssertThrow(
        spacedim == 3,
        ExcMessage(
          "update of inclusions data is only available for spacedim = 3"));

      AssertThrow(
        new_data.size() == n_vessels,
        ExcMessage(
          "dimensions of new data for the update does not match the inclusions"));

      for (unsigned int current_vessel = 0; current_vessel < n_vessels;
           ++current_vessel)
        {
          AssertIndexRange(current_vessel, new_data.size());
          auto &current_new_data   = new_data[current_vessel];
          auto &current_inclusions = map_vessel_inclusions[current_vessel];

          auto N1 = current_inclusions.size(); // inclusions in vessel
          auto N2 = current_new_data.size();   // points in new_data

          AssertThrow(
            N2 > 0,
            ExcMessage(
              "dimensions of new data for the update does not match the inclusions"));
          AssertThrow(
            N1 > 1,
            ExcMessage(
              "insufficient number of inclusion int the vessel for the update"));
          if (N2 == 1)
            {
              for (const auto inclusion_id : current_inclusions)
                update_single_inclusion_data_along_normal(inclusion_id,
                                                          current_new_data[0]);
            }
          else
            {
              // The samples are equally spaced from the first to the last
              // inclusion of the vessel, in the order of the inclusions file.
              for (unsigned int i = 0; i < N1; ++i)
                {
                  const double X =
                    static_cast<double>(i) * (N2 - 1) / (N1 - 1);
                  const unsigned int j =
                    std::min<unsigned int>(std::floor(X), N2 - 2);
                  const double w = X - j;
                  update_single_inclusion_data_along_normal(
                    current_inclusions[i],
                    (1 - w) * current_new_data[j] +
                      w * current_new_data[j + 1]);
                }
            }
        }
    }

    /**
     * @brief 3D return the vessel that a given inclusion belongs to,
     * 2D return 0
     *
     * @param inclusion_id
     * @return unsigned int of vessel id
     */
    int
    get_vesselID(const types::global_dof_index &inclusion_id) const
    {
      AssertIndexRange(inclusion_id, inclusions.size());
      const auto &inclusion = inclusions[inclusion_id];
      if constexpr (spacedim == 2)
        {
          return inclusion_id;
        }
      else
        {
          AssertDimension(inclusion.size(), 2 * spacedim + 2);
          return int(inclusion[2 * spacedim + 1]);
        }
    }

    /**
     * Update one inclusion using a scalar normal-displacement value.
     */
    void
    update_single_inclusion_data_along_normal(
      const types::global_dof_index &inclusion_id,
      const double                   nd)
    {
      AssertThrow(
        spacedim == 3,
        ExcMessage(
          "update of inclusions data is only available for spacedim = 3"));

      AssertIndexRange(inclusion_id, inclusions_data.size());
      // AssertIndexRange(inclusion_id, inclusions.size());
      // // update old radius with new value
      // if constexpr (spacedim == 2)
      //   {
      //     AssertDimension(inclusions[inclusion_id].size(), spacedim + 1);
      //     inclusions[inclusion_id][spacedim] +=
      //     inclusions_data[inclusion_id][0];
      //   }
      // else
      //   {
      //     AssertDimension(inclusions[inclusion_id].size(), 2 * spacedim + 2);
      //     inclusions[inclusion_id][2 * spacedim] +=
      //     inclusions_data[inclusion_id][0];
      //   }
      inclusions_data[inclusion_id] = {0, 0, 0, nd, 0, 0, 0, nd, 0, 0, 0, 0};
    }

    /**
     * Placeholder for vessel-wise updates with vector-valued payloads.
     */
    void
    update_single_vessel_data(const types::global_dof_index &,
                              const std::vector<double>)
    {}

    /**
     * Return number of vessel currently detected.
     */
    unsigned int
    get_n_vessels() const
    {
      return n_vessels;
    }

    /**
     * Return the value of "Number of fourier coefficients".
     *
     * After initialize() this is the size of the selection when a selection
     * was given, and the number of Fourier modes otherwise. In the second
     * case each mode contributes @p n_vector_components coefficients, so this
     * is not the number of degrees of freedom per inclusion for vector
     * problems. Use n_dofs_per_inclusion() for that.
     */
    unsigned int
    get_n_coefficients() const
    {
      return n_coefficients;
    }

    // /**
    //  * Return coefficient offset used for harmonic indexing.
    //  */
    // unsigned int
    // get_offset_coefficients() const
    // {
    //   return offset_coefficients;
    // }

    /**
     * Return number of inclusions mapped to a given vessel id.
     */
    unsigned int
    get_inclusions_in_vessel(unsigned int vessel_id) const
    {
      AssertIndexRange(vessel_id, n_vessels);
      unsigned int s = map_vessel_inclusions.at(vessel_id).size();
      return s;
    }

    /**
     * Return the Fourier coefficient indices in use. After initialize() this
     * is the explicit selection, or the indices generated from
     * get_n_coefficients() when no selection was given. Entry j is the
     * coefficient of local degree of freedom j of each inclusion.
     */
    std::vector<unsigned int>
    get_selected_coefficients() const
    {
      return selected_coefficients;
    }

    void
    set_n_q_points(unsigned int n_q)
    {
      n_q_points = n_q;
    }


    /**
     * Set "Number of fourier coefficients". Call before initialize(). See the
     * class documentation for how it combines with the selection.
     */
    void
    set_n_coefficients(unsigned int n_coefficients)
    {
      this->n_coefficients = n_coefficients;
    }

    /**
     * Set "Selection of Fourier coefficients". Call before initialize(). An
     * empty vector means no explicit selection.
     */
    void
    set_fourier_coefficients(std::vector<unsigned int> temp)
    {
      this->selected_coefficients = temp;
    }


    void
    build_segment_index_vector()
    {
      std::vector<unsigned int> owned_segment_indices;
      owned_segment_indices.reserve(
        particles_on_centerline.n_locally_owned_particles());
      auto         particle        = particles_on_centerline.begin();
      unsigned int current_segment = 0;
      while (particle != particles_on_centerline.end())
        {
          const auto        &cell0 = particle->get_surrounding_cell();
          const unsigned int n_pic =
            particles_on_centerline.n_particles_in_cell(cell0);
          const auto pic = particles_on_centerline.particles_in_cell(cell0);

          for (unsigned int i = 0; i < n_pic; i++)
            {
              owned_segment_indices.push_back(current_segment);
            }
          particle = pic.end();
          current_segment++;
        }
      MPI_Barrier(mpi_communicator);


      local_max_segment_index = current_segment;
      auto [local_shift, global_size] =
        Utilities::MPI::partial_and_total_sum(local_max_segment_index,
                                              mpi_communicator);

      global_max_segment_index = global_size;

      for (auto &segment_index : owned_segment_indices)
        segment_index += local_shift;

      // now we build segment_indices out of the local vectors each processor
      // has
      auto owned_particle_indices =
        particles_on_centerline.locally_owned_particle_ids().get_index_vector();
      AssertDimension(owned_particle_indices.size(),
                      owned_segment_indices.size());

      std::vector<int> send_buffer;
      for (unsigned int i = 0; i < owned_segment_indices.size(); i++)
        {
          send_buffer.push_back(owned_particle_indices[i]);
          send_buffer.push_back(owned_segment_indices[i]);
        }

      int send_count = send_buffer.size();

      std::vector<int> recv_counts(
        Utilities::MPI::n_mpi_processes(mpi_communicator));
      MPI_Allgather(&send_count,
                    1,
                    MPI_INT,
                    recv_counts.data(),
                    1,
                    MPI_INT,
                    mpi_communicator);

      // displacements
      std::vector<int> displs(Utilities::MPI::n_mpi_processes(mpi_communicator),
                              0);
      int              total_size = 0;
      for (unsigned int i = 0;
           i < Utilities::MPI::n_mpi_processes(mpi_communicator);
           i++)
        {
          displs[i] = total_size;
          total_size += recv_counts[i];
        }

      // receive buffer
      std::vector<int> recv_buffer(total_size);

      MPI_Allgatherv(send_buffer.data(),
                     send_count,
                     MPI_INT,
                     recv_buffer.data(),
                     recv_counts.data(),
                     displs.data(),
                     MPI_INT,
                     mpi_communicator);

      segment_indices.resize(n_inclusions());
      for (int i = 0; i < total_size; i += 2)
        {
          int idx              = recv_buffer[i];
          int val              = recv_buffer[i + 1];
          segment_indices[idx] = val;
        }

      return;
    }


    unsigned int
    get_segment_index(const types::global_dof_index &quadrature_id) const
    {
      AssertIndexRange(get_inclusion_id(quadrature_id), n_inclusions());
      return segment_indices[get_inclusion_id(quadrature_id)];
      // return
      // segment_indices.nth_index_in_set(get_inclusion_id(quadrature_id));
    }

    std::vector<unsigned int>
    get_inclusions_in_a_segment(unsigned int segment_index) const
    {
      std::vector<unsigned int> inclusions_in_segment;
      for (unsigned int i = 0; i < segment_indices.size(); ++i)
        if (segment_indices[i] == segment_index)
          inclusions_in_segment.push_back(i);
      return inclusions_in_segment;
    }


    double
    get_average_vessel_radius(unsigned int vessel_id)
    {
      AssertIndexRange(vessel_id, n_vessels);
      auto   inc_vec = map_vessel_inclusions.at(vessel_id);
      double radius  = 0;
      for (const auto i : inc_vec)
        {
          radius += get_radius(i);
        }
      Assert(inc_vec.size() > 0, ExcInternalError());
      radius /= inc_vec.size();
      Assert(radius > 0, ExcInternalError());
      return radius;
    }

    double
    get_vessel_length(unsigned int vessel_id)
    {
      AssertIndexRange(vessel_id, n_vessels);
      auto   inc_vec = map_vessel_inclusions.at(vessel_id);
      double length  = 0;
      for (const auto i : inc_vec)
        length += get_direction(i).norm();
      Assert(length > 0, ExcInternalError());
      return length;
    }

    ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>> inclusions_rhs;
    /**
     * Frequency used to modulate inclusion boundary data in time.
     */
    double modulation_frequency = 0.0;
    double phase_shift          = 0.0;

    /**
     * Particle representation of inclusion quadrature points.
     */
    Particles::ParticleHandler<spacedim> inclusions_as_particles;
    Particles::ParticleHandler<spacedim> particles_on_centerline;
    std::vector<std::vector<double>>     inclusions;

    /**
     * Optional ASCII file containing coefficient data.
     */
    std::string data_file = "";
#ifdef DEAL_II_WITH_HDF5
    /**
     * Optional HDF5 handle used for time-updated inclusion data.
     */
    mutable std::unique_ptr<HDF5::File> data_file_h;
#endif
    /**
     * Per-inclusion coefficient vectors used by coupling assembly.
     */
    std::vector<std::vector<double>> inclusions_data;
    /**
     * Default coefficients replicated when no data file is provided.
     */
    std::vector<double> reference_inclusion_data;

    /**
     * Mapping from vessel id to indices of associated inclusions.
     */
    std::map<unsigned int, std::vector<types::global_dof_index>>
      map_vessel_inclusions;

    /**
     * Switch for the clustering of inclusions dofs into segments
     */
    bool cluster_with_segments = false; // make const

  private:
    unsigned int              n_q_points          = 100;
    unsigned int              n_coefficients      = 0;
    // unsigned int              offset_coefficients = 0;
    std::vector<unsigned int> selected_coefficients;
    unsigned int              global_max_segment_index;
    unsigned int              local_max_segment_index;

    /**
     * Fixed number of vector components in the coupled bulk field.
     */
    const unsigned int n_vector_components;
    /**
     * MPI communicator associated with the background triangulation.
     */
    MPI_Comm mpi_communicator = MPI_COMM_WORLD;
    /**
     * Reference support points on the unit cross-section.
     */
    std::vector<Point<spacedim>> support_points;
    /**
     * Angular coordinates used for Fourier basis evaluation.
     */
    std::vector<double> theta;

    /**
     * Cached geometric data in current inclusion configuration.
     */
    /// @{
    mutable std::vector<Tensor<1, spacedim>> normals;
    mutable std::vector<Point<spacedim>>
      current_support_points; ///< Cached moved support points.
    mutable std::vector<double>
      current_fe_values; ///< Cached basis values at one point.
    /// @}

    /**
     * Number of vessel groups inferred from inclusion metadata.
     */
    unsigned int n_vessels = 1;

    /**
     * Input file for inclusion geometry and vessel metadata.
     */
    std::string inclusions_file = "";
    /**
     * Depth used when extracting R-tree bounding-box coverings.
     */
    // Use leaf-level cell boxes by default. Coarser boxes can include cells
    // that only overlap a particle's bounding box; recent deal.II versions
    // then attempt an inverse mapping on those false-positive candidates.
    unsigned int rtree_extraction_level = 0;

    std::vector<unsigned int> segment_indices;

    /**
     * @brief Check that all inclusions with the same vessel ID have parallel
     * directions.
     *
     * Directions only need to point the same way: their lengths may differ.
     * Opposite directions are rejected.
     */
    void
    check_vessel_directions() const
    {
      if constexpr (spacedim == 3)
        {
          // Tolerance on the sine of the angle between two directions.
          const double tolerance = 1e-6;
          for (const auto &[vessel_id, inclusion_ids] : map_vessel_inclusions)
            {
              AssertThrow(!inclusion_ids.empty(),
                          ExcMessage("Vessel " + std::to_string(vessel_id) +
                                     " has no inclusions."));
              const auto first_id = inclusion_ids.front();
              const auto first    = get_direction(first_id);
              for (const auto inclusion_id : inclusion_ids)
                {
                  const auto direction = get_direction(inclusion_id);
                  AssertThrow(
                    cross_product_3d(first, direction).norm() <=
                        tolerance * first.norm() * direction.norm() &&
                      first * direction > 0,
                    ExcMessage("Inclusions " + std::to_string(first_id) +
                               " and " + std::to_string(inclusion_id) +
                               " belong to vessel " +
                               std::to_string(vessel_id) +
                               " but their directions are not parallel and equally oriented."));
                }
            }
        }
    }

    /**
     * @brief Check that all vesselsID are present
     and create the map vessel_inclusions
     */
    void
    check_vessels()
    {
      if (inclusions.size() == 0)
        return;

      // if constexpr (spacedim == 2)
      //   n_vessels = inclusions.size();
      {
        for (types::global_dof_index inc_number = 0;
             inc_number < inclusions.size();
             ++inc_number)
          map_vessel_inclusions[get_vesselID(inc_number)].push_back(inc_number);

        types::global_dof_index id_check = 0;

        std::map<unsigned int, std::vector<types::global_dof_index>>::iterator
          it = map_vessel_inclusions.begin();

        while (it != map_vessel_inclusions.end() && id_check == it->first)
          {
            ++id_check;
            ++it;
          }
        AssertThrow(
          it == map_vessel_inclusions.end(),
          ExcMessage(
            "Vessel Ids from data file should be sequential, missing vessels ID(s)"));

        n_vessels = map_vessel_inclusions.size();
      }
      check_vessel_directions();
      /*
      {
      std::set<double> vessel_id_is_present;
      for (types::global_dof_index inc_number = 0; inc_number <
      inclusions.size();
           ++inc_number)
          vessel_id_is_present.insert(get_vesselID(inc_number));

      types::global_dof_index id_check = 0;
      while (id_check < vessel_id_is_present.size() &&
             vessel_id_is_present.find(id_check) != vessel_id_is_present.end())
        ++id_check;

      AssertThrow(
        id_check + 1 != vessel_id_is_present.size(),
        ExcMessage(
          "Vessel Ids from data file should be sequential, missing vessels
          ID(s)"));
      n_vessels = vessel_id_is_present.size();
      }
      */
    }
  };

} // namespace ImmersX

#endif
