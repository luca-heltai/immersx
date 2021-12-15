#ifndef reference_inclusion
#define reference_inclusion

#include <deal.II/particles/data_out.h>
#include <deal.II/particles/generators.h>
#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/utilities.h>


using namespace dealii;

template <int spacedim>
struct ReferenceInclusion
{
  ReferenceInclusion(unsigned int n_q_points, unsigned int n_coefficients)
    : n_q_points(n_q_points)
    , n_coefficients(n_coefficients)
    , support_points(n_q_points)
    , normals(n_q_points)
    , theta(n_q_points)
    , current_support_points(n_q_points)
    , current_fe_values(n_coefficients)
  {
    static_assert(spacedim > 1, "Not implemented in dim = 1");
    for (unsigned int i = 0; i < n_q_points; ++i)
      {
        theta[i]             = i * 2 * numbers::PI / n_q_points;
        support_points[i][0] = std::cos(theta[i]);
        support_points[i][1] = std::sin(theta[i]);
        normals[i]           = support_points[i];
      }
  }


  std::vector<types::global_dof_index>
  get_dof_indices(const types::global_dof_index &quadrature_id) const
  {
    std::vector<types::global_dof_index> dofs(n_coefficients);
    auto start_index = (quadrature_id / n_q_points) * n_coefficients;
    for (auto &d : dofs)
      d = start_index++;
    return dofs;
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
    return (quadrature_id / n_q_points);
  }

  inline const Tensor<1, spacedim> &
  get_normal(const types::global_dof_index &quadrature_id) const
  {
    return (normals[quadrature_id % n_q_points]);
  }


  const unsigned int               n_q_points;
  const unsigned int               n_coefficients;
  std::vector<Point<spacedim>>     support_points;
  std::vector<Tensor<1, spacedim>> normals;
  std::vector<double>              theta;

  // Current configuration
  unsigned int    current_inclusion_id = numbers::invalid_unsigned_int;
  double          current_radius;
  Point<spacedim> current_center;
  mutable std::vector<Point<spacedim>> current_support_points;
  std::vector<double>                  current_fe_values;

  const std::vector<double> &
  reinit(const types::global_dof_index           particle_id,
         const std::vector<std::vector<double>> &inclusions)
  {
    if (n_coefficients == 0)
      return current_fe_values;
    const auto q  = particle_id % n_q_points;
    const auto id = particle_id / n_q_points;
    AssertIndexRange(id, inclusions.size());
    AssertDimension(inclusions[id].size(), spacedim + 1);
    const auto r         = inclusions[id][spacedim];
    const auto ds        = 2 * numbers::PI * r / n_q_points;
    current_fe_values[0] = ds;
    for (unsigned int c = 1; c < n_coefficients; ++c)
      {
        unsigned int omega = (c + 1) / 2;
        const double rho   = std::pow(r, omega);
        if ((c + 1) % 2 == 0)
          current_fe_values[c] = ds * rho * std::cos(theta[q] * omega);
        else
          current_fe_values[c] = ds * rho * std::sin(theta[q] * omega);
      }
    return current_fe_values;
  }

  const std::vector<Point<spacedim>> &
  get_current_support_points(const std::vector<double> &inclusion) const
  {
    AssertDimension(inclusion.size(), spacedim + 1);
    Point<spacedim> center;
    for (unsigned int d = 0; d < spacedim; ++d)
      center[d] = inclusion[d];

    const auto &r = inclusion[spacedim];
    for (unsigned int q = 0; q < n_q_points; ++q)
      current_support_points[q] = center + support_points[q] * r;
    return current_support_points;
  }
};

#endif