// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/exceptions.h>
#include <deal.II/base/patterns.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <immersx/core/domain.h>

#if DEAL_II_VERSION_GTE(9, 8, 0) && defined(DEAL_II_WITH_VTK)
#  include <deal.II/vtk/utilities.h>
#elif !DEAL_II_VERSION_GTE(9, 8, 0)
#  include <immersx/compatibility/dealii_9_8/grid_in.h>
#  ifdef DEAL_II_WITH_VTK
#    include <immersx/compatibility/dealii_9_8/vtk/utilities.h>
#  endif
#endif

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>

namespace ImmersX
{
  namespace
  {
    template <int dim, int spacedim, typename TriangulationType>
    void
    read_vtk_file(const std::string &filename,
                  TriangulationType &tria,
                  const std::string &material_id_field,
                  const std::string &boundary_id_field)
    {
      dealii::Triangulation<dim, spacedim> serial_tria;

#ifdef DEAL_II_WITH_VTK
#  if DEAL_II_VERSION_GTE(9, 8, 0)
      dealii::VTKWrappers::read_tria(
        filename, serial_tria, true, 0., material_id_field, boundary_id_field);
#  else
      ImmersX::VTKWrappers::read_tria(
        filename, serial_tria, true, 0., material_id_field, boundary_id_field);
#  endif
#else
#  if DEAL_II_VERSION_GTE(9, 8, 0)
      dealii::GridIn<dim, spacedim> grid_in;
      grid_in.attach_triangulation(serial_tria);
      grid_in.read(filename);
#  else
      std::ifstream input(filename);
      AssertThrow(input.good(), dealii::ExcIO());
      ImmersX::dealii_9_8_compat::GridIn<dim, spacedim> grid_in;
      grid_in.attach_triangulation(serial_tria);
      grid_in.read_vtk(input);
#  endif
#endif

      tria.copy_triangulation(serial_tria);
    }


    template <int dim, int spacedim, typename TriangulationType>
    void
    read_mesh_file(const std::string &filename,
                   TriangulationType &tria,
                   const std::string &material_id_field,
                   const std::string &boundary_id_field)
    {
      std::string extension;
      const auto  dot = filename.find_last_of('.');
      if (dot != std::string::npos)
        extension = filename.substr(dot + 1);
      std::transform(extension.begin(),
                     extension.end(),
                     extension.begin(),
                     [](const unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                     });

      if (extension == "vtk" || extension == "vtu" || extension == "pvtu")
        {
          read_vtk_file<dim, spacedim>(filename,
                                       tria,
                                       material_id_field,
                                       boundary_id_field);
          return;
        }

      dealii::GridIn<dim, spacedim> grid_in;
      grid_in.attach_triangulation(tria);
#if defined(DEAL_II_WITH_GMSH_API) || defined(DEAL_II_GMSH_WITH_API)
      if (extension == "msh")
        {
          grid_in.read_msh(filename);
          return;
        }
#endif
      grid_in.read(filename);
    }


    template <int dim, int spacedim, typename TriangulationType>
    void
    make_grid_in(const DomainParameters<dim, spacedim> &parameters,
                 TriangulationType                     &tria)
    {
      if (std::filesystem::is_regular_file(parameters.name_of_grid))
        {
          read_mesh_file<dim, spacedim>(parameters.name_of_grid,
                                        tria,
                                        parameters.vtk_material_id_field,
                                        parameters.vtk_boundary_id_field);
          return;
        }

      dealii::GridGenerator::generate_from_name_and_arguments(
        tria, parameters.name_of_grid, parameters.arguments_for_grid);
    }
  } // namespace


  template <int dim, int spacedim>
  DomainParameters<dim, spacedim>::DomainParameters(
    const std::string &subsection)
    : dealii::ParameterAcceptor(subsection)
  {
    add_parameter("Grid generator", name_of_grid);
    add_parameter("Grid generator arguments", arguments_for_grid);
    add_parameter("Grid scale", grid_scale);
    add_parameter("Initial refinement", initial_refinement);
    add_parameter("Triangulation type",
                  triangulation_type,
                  "",
                  this->prm,
                  dealii::Patterns::Selection("distributed|fullydistributed"));
    enter_subsection("VTK support");
    add_parameter("Read fields", read_vtk_fields);
    add_parameter("Boundary id field", vtk_boundary_id_field);
    add_parameter("Material id field", vtk_material_id_field);
    leave_subsection();
  }


  template <int dim, int spacedim>
  typename Domain<dim, spacedim>::TriangulationVariant
  Domain<dim, spacedim>::make_triangulation_storage(
    const DomainParameters<dim, spacedim> &parameters,
    MPI_Comm                               communicator)
  {
    const bool fully_distributed =
      parameters.triangulation_type == "fullydistributed" || dim == 1;

    if (fully_distributed)
      return TriangulationVariant(
        std::in_place_type<FullyDistributedTriangulation>, communicator);

    return TriangulationVariant(
      std::in_place_type<DistributedTriangulation>,
      communicator,
      typename dealii::Triangulation<dim, spacedim>::MeshSmoothing(
        dealii::Triangulation<dim, spacedim>::smoothing_on_refinement |
        dealii::Triangulation<dim, spacedim>::smoothing_on_coarsening),
      DistributedTriangulation::construct_multigrid_hierarchy);
  }


  template <int dim, int spacedim>
  Domain<dim, spacedim>::Domain(
    const DomainParameters<dim, spacedim> &parameters,
    const MPI_Comm                         communicator)
    : parameters_(parameters)
    , communicator_(communicator)
    , triangulation_storage_(
        make_triangulation_storage(parameters_, communicator_))
  {
    triangulation_ =
      &std::visit([](auto &tria) -> TriangulationType & { return tria; },
                  triangulation_storage_);
  }


  template <int dim, int spacedim>
  Domain<dim, spacedim>::~Domain() = default;


  template <int dim, int spacedim>
  void
  Domain<dim, spacedim>::import_vtk_fields_if_requested()
  {
    vtk_fields_.reset();
    if (!parameters_.read_vtk_fields ||
        !std::filesystem::is_regular_file(parameters_.name_of_grid))
      return;

    std::string extension;
    const auto  dot = parameters_.name_of_grid.find_last_of('.');
    if (dot != std::string::npos)
      extension = parameters_.name_of_grid.substr(dot + 1);
    std::transform(extension.begin(),
                   extension.end(),
                   extension.begin(),
                   [](const unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    if (extension != "vtk" && extension != "vtu" && extension != "pvtu")
      return;

#ifdef DEAL_II_WITH_VTK
    vtk_fields_ = std::make_unique<ImportedFields>(parameters_.name_of_grid,
                                                   triangulation(),
                                                   communicator_);
#else
    AssertThrow(false,
                dealii::ExcMessage(
                  "Domain parameter 'Read fields' requires deal.II VTK "
                  "support."));
#endif
  }


  template <int dim, int spacedim>
  void
  Domain<dim, spacedim>::make_grid()
  {
    if (uses_fully_distributed_triangulation())
      {
        dealii::Triangulation<dim, spacedim> serial_tria(
          typename dealii::Triangulation<dim, spacedim>::MeshSmoothing(
            dealii::Triangulation<dim, spacedim>::smoothing_on_refinement |
            dealii::Triangulation<dim, spacedim>::smoothing_on_coarsening));

        make_grid_in(parameters_, serial_tria);
        if (parameters_.grid_scale != 1.)
          dealii::GridTools::scale(parameters_.grid_scale, serial_tria);
        AssertThrow(
          !parameters_.read_vtk_fields || parameters_.initial_refinement == 0,
          dealii::ExcMessage(
            "Domain cannot transfer imported VTK fields through initial "
            "refinement on a fullydistributed triangulation. Use the "
            "distributed backend or set Initial refinement to zero."));
        serial_tria.refine_global(parameters_.initial_refinement);

        auto &fully_distributed =
          std::get<FullyDistributedTriangulation>(triangulation_storage_);
        if constexpr (dim == 1)
          fully_distributed.set_partitioner(
            [](dealii::Triangulation<dim, spacedim> &tria,
               const unsigned int                    n_partitions) {
              dealii::GridTools::partition_triangulation_zorder(n_partitions,
                                                                tria,
                                                                false);
            },
            dealii::TriangulationDescription::Settings::default_setting);

        for (const auto manifold_id : serial_tria.get_manifold_ids())
          if (manifold_id != dealii::numbers::flat_manifold_id)
            fully_distributed.set_manifold(
              manifold_id, serial_tria.get_manifold(manifold_id));
        fully_distributed.copy_triangulation(serial_tria);
        import_vtk_fields_if_requested();
        return;
      }

    auto &distributed =
      std::get<DistributedTriangulation>(triangulation_storage_);
    make_grid_in(parameters_, distributed);
    if (parameters_.grid_scale != 1.)
      dealii::GridTools::scale(parameters_.grid_scale, distributed);
    import_vtk_fields_if_requested();
    // Imported fields are connected to the distributed triangulation before
    // refinement, so their SolutionTransfer hooks interpolate them onto the
    // refined mesh just as in the existing reduced-grid code.
    distributed.refine_global(parameters_.initial_refinement);
  }


  template <int dim, int spacedim>
  typename Domain<dim, spacedim>::TriangulationType &
  Domain<dim, spacedim>::triangulation()
  {
    return *triangulation_;
  }


  template <int dim, int spacedim>
  const typename Domain<dim, spacedim>::TriangulationType &
  Domain<dim, spacedim>::triangulation() const
  {
    return *triangulation_;
  }


  template <int dim, int spacedim>
  bool
  Domain<dim, spacedim>::uses_fully_distributed_triangulation() const
  {
    return std::holds_alternative<FullyDistributedTriangulation>(
      triangulation_storage_);
  }


  template <int dim, int spacedim>
  MPI_Comm
  Domain<dim, spacedim>::communicator() const
  {
    return communicator_;
  }


  template <int dim, int spacedim>
  bool
  Domain<dim, spacedim>::has_vtk_fields() const
  {
    return vtk_fields_ != nullptr;
  }


  template <int dim, int spacedim>
  const typename Domain<dim, spacedim>::ImportedFields &
  Domain<dim, spacedim>::vtk_fields() const
  {
    AssertThrow(vtk_fields_ != nullptr,
                dealii::ExcMessage("No VTK fields were imported for Domain."));
    return *vtk_fields_;
  }


  template <int dim, int spacedim>
  typename Domain<dim, spacedim>::ImportedFields::FieldView
  Domain<dim, spacedim>::field(const std::string &name,
                               const unsigned int component) const
  {
    return vtk_fields().field(name, component);
  }


#define INSTANTIATE_DOMAIN(dim, spacedim)         \
  template class DomainParameters<dim, spacedim>; \
  template class Domain<dim, spacedim>;

  INSTANTIATE_DOMAIN(1, 1)
  INSTANTIATE_DOMAIN(1, 2)
  INSTANTIATE_DOMAIN(1, 3)
  INSTANTIATE_DOMAIN(2, 2)
  INSTANTIATE_DOMAIN(2, 3)
  INSTANTIATE_DOMAIN(3, 3)

#undef INSTANTIATE_DOMAIN
} // namespace ImmersX
