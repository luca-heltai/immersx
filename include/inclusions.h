#ifndef rdlm_inclusions
#define rdlm_inclusions

#include <deal.II/base/hdf5.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_function.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/particles/data_out.h>
#include <deal.II/particles/generators.h>
#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/utilities.h>

#include <deal.II/physics/transformations.h>

#include <fstream>

using namespace dealii;

template <int spacedim>
class Inclusions : public ParameterAcceptor
{
public:
  Inclusions(const unsigned int n_vector_components = 1)
    : ParameterAcceptor("/Immersed Problem/Immersed inclusions")
    , inclusions_rhs("/Immersed Problem/Immersed inclusions/Boundary data",
                     n_vector_components)
    , n_vector_components(n_vector_components)
  {
    static_assert(spacedim > 1, "Not implemented in dim = 1");
    add_parameter("Inclusions refinement", n_q_points);
    add_parameter("Inclusions", inclusions);
    add_parameter("Number of fourier coefficients", n_coefficients);
    add_parameter("Start index of Fourier coefficients", offset_coefficients);
    add_parameter("Bounding boxes extraction level", rtree_extraction_level);
    add_parameter("Inclusions file", inclusions_file);
    add_parameter("Data file", data_file);
    add_parameter("3D 1D discretization", h3D1D);
  }


  types::global_dof_index
  n_dofs() const
  {
    return inclusions.size() * n_dofs_per_inclusion();
  }


  types::global_dof_index
  n_particles() const
  {
    return inclusions.size() * n_q_points;
  }


  types::global_dof_index
  n_inclusions() const
  {
    return inclusions.size();
  }

  /**
   * @brief Number of degrees of freedom associated to each inclusion.
   *
   * @return unsigned int
   */
  unsigned int
  n_dofs_per_inclusion() const
  {
    return n_coefficients * n_vector_components;
  }

  /**
   * Reinit all reference points and normals, and read inclusions file.
   */
  void
  initialize()
  {
    AssertThrow(n_q_points > 0,
                ExcMessage(
                  "Refinement of inclusions must be greater than zero."));
    AssertThrow(n_coefficients > 0,
                ExcMessage(
                  "Number of coefficients must be greater than zero."));
    support_points.resize(n_q_points);
    normals.resize(n_q_points);
    theta.resize(n_q_points);
    current_support_points.resize(n_q_points);
    current_fe_values.resize(n_dofs_per_inclusion());

    for (unsigned int i = 0; i < n_q_points; ++i)
      {
        theta[i]             = i * 2 * numbers::PI / n_q_points;
        support_points[i][0] = std::cos(theta[i]);
        support_points[i][1] = std::sin(theta[i]);
        normals[i]           = support_points[i];
      }

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
            std::cout << "rank " << Utilities::MPI::this_mpi_process(mpi_communicator)
                      << ": Read " << N << " coefficients per " << inclusions.size() 
                      << " inclusion" << std::endl;
          }
      }
    n_vessels = check_vessels();
  }


  std::vector<types::global_dof_index>
  get_dof_indices(const types::global_dof_index &quadrature_id) const
  {
    AssertIndexRange(quadrature_id, n_particles());
    std::vector<types::global_dof_index> dofs(n_dofs_per_inclusion());
    auto start_index = (quadrature_id / n_q_points) * n_dofs_per_inclusion();
    for (auto &d : dofs)
      d = start_index++;
    return dofs;
  }

  void
  setup_inclusions_particles(
    const parallel::distributed::Triangulation<spacedim> &tria)
  {
    initialize();
    mpi_communicator = tria.get_communicator();
    inclusions_as_particles.initialize(tria,
                                       StaticMappingQ1<spacedim>::mapping);

    if (n_dofs() == 0)
      return;

    // Only add particles once.
    auto inclusions_set =
      Utilities::MPI::create_evenly_distributed_partitioning(mpi_communicator,
                                                             n_inclusions());

    std::vector<Point<spacedim>> particles_positions;
    particles_positions.reserve(n_particles());
    for (const auto i : inclusions_set)
      {
        const auto &p = get_current_support_points(i);
        particles_positions.insert(particles_positions.end(),
                                   p.begin(),
                                   p.end());
      }

    std::vector<BoundingBox<spacedim>> all_boxes;
    all_boxes.reserve(tria.n_locally_owned_active_cells());
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned())
        all_boxes.emplace_back(cell->bounding_box());
    const auto tree        = pack_rtree(all_boxes);
    const auto local_boxes = extract_rtree_level(tree, rtree_extraction_level);

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

    // Sanity check.
    AssertDimension(inclusions_as_particles.n_global_particles(),
                    n_particles());
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
    return dof_index % n_vector_components;
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
    if (n_coefficients == 0)
      return current_fe_values;
    const auto q  = particle_id % n_q_points;
    const auto id = particle_id / n_q_points;
    AssertIndexRange(id, inclusions.size());
    const auto r = get_radius(id);
    for (unsigned int c = 0; c < n_coefficients; ++c)
      {
        unsigned int omega = (c + offset_coefficients + 1) / 2;
        const double rho   = std::pow(r, omega);
        for (unsigned int i = 0; i < n_vector_components; ++i)
          if ((std::max(c + offset_coefficients, 1u) + 1) % 2 == 0)
            current_fe_values[c * n_vector_components + i] =
              rho * std::cos(theta[q] * omega);
          else
            current_fe_values[c * n_vector_components + i] =
              rho * std::sin(theta[q] * omega);
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



  const std::vector<Point<spacedim>> &
  get_current_support_points(const types::global_dof_index &inclusion_id) const
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
   *
   */
  void
  read_displacement_hdf5()
  {
    data_file_h = std::make_unique<HDF5::File>(data_file,
                                               HDF5::File::FileAccessMode::open,
                                               mpi_communicator);
    auto group  = data_file_h->open_group("data");
    // Read new displacement
    {
      auto h5data         = group.open_dataset("displacement_data");
      auto vector_of_data = h5data.template read<Vector<double>>();

      for (unsigned int i = 0; i < vector_of_data.size(); ++i)
        {
          // inclusions_data.push_back(vector_of_data[i]);
        }
    }
    AssertThrow(inclusions_data.size() == n_inclusions(),
                ExcDimensionMismatch(inclusions_data.size(), n_inclusions()));
    if (inclusions_data.size() > 0)
      {
        const auto N = inclusions_data[0].size();
        for (const auto &l : inclusions_data)
          {
            AssertThrow(l.size() == N, ExcDimensionMismatch(l.size(), N));
          }

        std::cout << "Read " << N << " coefficients per inclusion" << std::endl;
      }
  }

  int
  get_vesselID(const types::global_dof_index &inclusion_id) const
  {
    AssertIndexRange(inclusion_id, inclusions.size());
    const auto &inclusion = inclusions[inclusion_id];
    if constexpr (spacedim == 2)
      {
        return 0.0;
      }
    else
      {
        AssertDimension(inclusion.size(), 2 * spacedim + 2);
        return int(inclusion[2 * spacedim + 1]);
      }
  }

  double
  get_h3D1D()
  {
    return h3D1D;
  }

  ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>> inclusions_rhs;

  std::vector<std::vector<double>> inclusions;
  unsigned int                     n_q_points          = 100;
  unsigned int                     n_coefficients      = 1;
  unsigned int                     offset_coefficients = 0;
  unsigned int                     n_vessels      = 0;
  double                           h3D1D               = 0.01;

  Particles::ParticleHandler<spacedim> inclusions_as_particles;

  std::string                         data_file = "";
  mutable std::unique_ptr<HDF5::File> data_file_h;
  std::vector<std::vector<double>>    inclusions_data;

private:
  const unsigned int           n_vector_components;
  MPI_Comm                     mpi_communicator;
  std::vector<Point<spacedim>> support_points;
  std::vector<double>          theta;

  // Current configuration
  mutable std::vector<Tensor<1, spacedim>> normals;
  mutable std::vector<Point<spacedim>>     current_support_points;
  mutable std::vector<double>              current_fe_values;

  std::string  inclusions_file        = "";
  unsigned int rtree_extraction_level = 1;

  /**
   * @brief Check that all vesselsID are present
   */
  unsigned int
  check_vessels() const
  {
    if (inclusions.size() == 0)
      return 0;

    if constexpr (spacedim == 2)
        return n_inclusions();

    std::set<double> vessel_id_is_present;
    for (types::global_dof_index inc_number = 0; inc_number < inclusions.size();
         ++inc_number)
      vessel_id_is_present.insert(get_vesselID(inc_number));

    types::global_dof_index id_check = 0;
    while (id_check < vessel_id_is_present.size() &&
           vessel_id_is_present.find(id_check) != vessel_id_is_present.end())
      ++id_check;

    AssertThrow(
      id_check+1 != vessel_id_is_present.size(),
      ExcMessage(
        "Vessel Ids from data file should be sequential, missing vessels ID(s)"));
    return vessel_id_is_present.size();
  }
};

#endif