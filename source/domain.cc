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
#include <deal.II/base/scope_exit.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/solution_transfer.h>

#include <immersx/core/domain.h>
#include <immersx/io/vtk_utils.h>

#include "surface_volume_mesh.h"

#if DEAL_II_VERSION_GTE(9, 8, 0) && defined(DEAL_II_WITH_VTK)
#  include <deal.II/vtk/utilities.h>
#elif !DEAL_II_VERSION_GTE(9, 8, 0)
#  include <immersx/compatibility/dealii_9_8/grid_in.h>
#  ifdef DEAL_II_WITH_VTK
#    include <immersx/compatibility/dealii_9_8/vtk/utilities.h>
#  endif
#endif

#ifdef DEAL_II_WITH_VTK
#  include <vtkGenericDataObjectReader.h>
#  include <vtkUnstructuredGrid.h>
#  include <vtkVersionMacros.h>
#  if VTK_MAJOR_VERSION > 9 || \
    (VTK_MAJOR_VERSION == 9 && VTK_MINOR_VERSION >= 3)
#    include <vtkCleanUnstructuredGrid.h>
#  endif
#endif

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#if defined(DEAL_II_WITH_GMSH_API) || defined(DEAL_II_GMSH_WITH_API)
#  include <gmsh.h>
#  include <unistd.h>
#endif

namespace ImmersX
{
  namespace
  {
#if defined(DEAL_II_WITH_GMSH_API) || defined(DEAL_II_GMSH_WITH_API)
    template <int dim, int spacedim>
    void
    read_gmsh_file(const std::string             &filename,
                   dealii::GridIn<dim, spacedim> &grid_in)
    {
      // A failed open leaves an existing Gmsh session unable to read later
      // meshes.
      std::ifstream input(filename);
      AssertThrow(input.good(), dealii::ExcFileNotOpen(filename));

      // Some Gmsh API readers ignore unnamed physical groups. Make only their
      // default ids explicit, retaining the native reader's named selectors.
      // A temporary MSH4 file also retains entities with no physical group.
      const bool owns_gmsh = gmsh::isInitialized() == 0;
      if (owns_gmsh)
        gmsh::initialize();
      const dealii::ScopeExit finalize_gmsh([owns_gmsh]() {
        if (owns_gmsh && gmsh::isInitialized())
          gmsh::finalize();
      });
      double version = 0., save_all = 0., binary = 0., scaling = 0.,
             orphans = 0.;
      gmsh::option::getNumber("Mesh.MshFileVersion", version);
      gmsh::option::getNumber("Mesh.SaveAll", save_all);
      gmsh::option::getNumber("Mesh.Binary", binary);
      gmsh::option::getNumber("Mesh.ScalingFactor", scaling);
      gmsh::option::getNumber("Mesh.SaveWithoutOrphans", orphans);
      const dealii::ScopeExit restore_options(
        [version, save_all, binary, scaling, orphans]() {
          // Older native readers finalize the Gmsh session they use.
          if (!gmsh::isInitialized())
            gmsh::initialize();
          gmsh::option::setNumber("Mesh.MshFileVersion", version);
          gmsh::option::setNumber("Mesh.SaveAll", save_all);
          gmsh::option::setNumber("Mesh.Binary", binary);
          gmsh::option::setNumber("Mesh.ScalingFactor", scaling);
          gmsh::option::setNumber("Mesh.SaveWithoutOrphans", orphans);
        });
      gmsh::option::setNumber("General.Verbosity", 0);
      gmsh::clear();
      gmsh::open(filename);
      gmsh::vectorpair groups;
      gmsh::model::getPhysicalGroups(groups);
      bool changed = false;
      for (const auto &[dimension, tag] : groups)
        {
          std::string name;
          gmsh::model::getPhysicalName(dimension, tag, name);
          if (name.empty())
            {
              gmsh::model::setPhysicalName(dimension,
                                           tag,
                                           std::string(dimension == dim ?
                                                         "MaterialID:" :
                                                         "BoundaryID:") +
                                             std::to_string(tag));
              changed = true;
            }
        }
      if (!changed)
        {
          grid_in.read_msh(filename);
          return;
        }

      auto pattern =
        (std::filesystem::temp_directory_path() / "immersx-gmsh-XXXXXX.msh")
          .string();
      std::vector<char> temporary_name(pattern.begin(), pattern.end());
      temporary_name.push_back('\0');
      const int descriptor = mkstemps(temporary_name.data(), 4);
      AssertThrow(descriptor != -1,
                  dealii::ExcMessage("Cannot create a temporary Gmsh mesh."));
      close(descriptor);
      const std::string       temporary_file(temporary_name.data());
      const dealii::ScopeExit remove_file([&temporary_file]() {
        std::error_code ignored;
        std::filesystem::remove(temporary_file, ignored);
      });
      gmsh::option::setNumber("Mesh.MshFileVersion", 4.1);
      gmsh::option::setNumber("Mesh.SaveAll", 1);
      gmsh::option::setNumber("Mesh.Binary", 1);
      gmsh::option::setNumber("Mesh.ScalingFactor", 1);
      gmsh::option::setNumber("Mesh.SaveWithoutOrphans", 0);
      gmsh::write(temporary_file);
      grid_in.read_msh(temporary_file);
    }
#endif

    template <int dim, int spacedim, typename TriangulationType>
    void
    read_vtk_file(const std::string &filename,
                  TriangulationType &tria,
                  const std::string &material_id_field,
                  const std::string &boundary_id_field)
    {
      dealii::Triangulation<dim, spacedim> serial_tria;

#ifdef DEAL_II_WITH_VTK
      // Legacy readers otherwise retain only the first SCALARS array. Read all
      // arrays so selecting an id field does not depend on its file ordering.
      std::string extension =
        std::filesystem::path(filename).extension().string();
      std::transform(extension.begin(),
                     extension.end(),
                     extension.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      if (extension == ".vtk")
        {
          std::ifstream input(filename);
          AssertThrow(input.good(), dealii::ExcIO());
          auto reader = vtkSmartPointer<vtkGenericDataObjectReader>::New();
          reader->SetFileName(filename.c_str());
          reader->ReadAllScalarsOn();
          reader->ReadAllVectorsOn();
          reader->ReadAllTensorsOn();
          reader->ReadAllNormalsOn();
          reader->ReadAllTCoordsOn();
          reader->ReadAllColorScalarsOn();
          reader->ReadAllFieldsOn();
          reader->Update();
#  if DEAL_II_VERSION_GTE(9, 8, 0)
          namespace vtk_api = dealii::VTKWrappers;
#  else
          namespace vtk_api = ImmersX::VTKWrappers;
#  endif
          auto grid = vtk_api::internal::convert_to_unstructured_grid(
            reader->GetOutput());
#  if VTK_MAJOR_VERSION > 9 || \
    (VTK_MAJOR_VERSION == 9 && VTK_MINOR_VERSION >= 3)
          auto cleaner = vtkSmartPointer<vtkCleanUnstructuredGrid>::New();
          cleaner->SetToleranceIsAbsolute(false);
          cleaner->SetTolerance(0.);
          cleaner->SetInputData(grid);
          cleaner->Update();
          grid->ShallowCopy(cleaner->GetOutput());
#  endif
          vtk_api::unstructured_grid_to_dealii_triangulation(*grid,
                                                             serial_tria,
                                                             material_id_field,
                                                             boundary_id_field);
          tria.copy_triangulation(serial_tria);
          return;
        }
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
      if (extension == "msh" || extension == "stl")
        {
          if (extension == "msh")
            read_gmsh_file(filename, grid_in);
          else if (extension == "stl")
            {
              // Reject failed opens before entering Gmsh's project lock.
              {
                std::ifstream input(filename, std::ios::binary);
                AssertThrow(input.good(), dealii::ExcFileNotOpen(filename));
              }
              grid_in.read_msh(filename);
            }
          return;
        }
#endif
#ifdef DEAL_II_WITH_ASSIMP
      if constexpr (dim == 2 && spacedim == 3)
        {
          // GridIn's ordinary extension parser does not dispatch Assimp
          // formats. Let Assimp detect additional surface formats itself.
          try
            {
              dealii::GridIn<dim, spacedim>::parse_format(extension);
            }
          catch (const dealii::ExceptionBase &)
            {
              grid_in.read_assimp(filename);
              return;
            }
        }
#endif
      grid_in.read(filename);
    }


    template <int dim, int spacedim, typename TriangulationType>
    void
    make_grid_in(const DomainParameters<dim, spacedim> &parameters,
                 TriangulationType                     &tria)
    {
      if (std::filesystem::path(parameters.name_of_grid).has_extension())
        {
          if constexpr (dim == 3 && spacedim == 3)
            {
              // Probe into serial storage: a failed read must not populate the
              // destination, which may already be a distributed triangulation.
              dealii::Triangulation<3> volume;
              std::string              volume_error;
              if (parameters.mesh_input != "surface")
                {
                  try
                    {
                      read_mesh_file<3, 3>(parameters.name_of_grid,
                                           volume,
                                           parameters.vtk_material_id_field,
                                           parameters.vtk_boundary_id_field);
                      AssertThrow(volume.n_active_cells() > 0,
                                  dealii::ExcMessage(
                                    "The input contains no volume cells."));
                    }
                  catch (const dealii::ExceptionBase &error)
                    {
                      if (parameters.mesh_input == "volume")
                        throw;
                      volume_error = error.what();
                    }
                  if (volume.n_active_cells() > 0 && volume_error.empty())
                    {
                      tria.copy_triangulation(volume);
                      return;
                    }
                }
              AssertThrow(
                !parameters.read_vtk_fields,
                dealii::ExcMessage(
                  "Surface remeshing cannot transfer input VTK fields."));
              dealii::Triangulation<2, 3> surface;
              try
                {
                  read_mesh_file<2, 3>(parameters.name_of_grid,
                                       surface,
                                       "",
                                       "");
                }
              catch (const dealii::ExceptionBase &error)
                {
                  AssertThrow(
                    false,
                    dealii::ExcMessage(
                      "Cannot read input as a volume or surface mesh. Volume: " +
                      volume_error + " Surface: " + error.what()));
                }
              volume.clear();
              detail::generate_volume_from_surface(surface, volume, parameters);
              AssertThrow(
                parameters.surface_volume_cells != "tetrahedra" ||
                  parameters.triangulation_type == "fullydistributed",
                dealii::ExcMessage(
                  "Tetrahedra require Triangulation type = fullydistributed."));
              tria.copy_triangulation(volume);
            }
          else
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


  // Serial input fields live only during preparation. Keep the serial vector
  // attached to the same mesh through every refinement, then redistribute by
  // CellId when the final triangulation has been constructed.
  template <int dim, int spacedim>
  struct Domain<dim, spacedim>::SerialFields
  {
    dealii::DoFHandler<dim, spacedim>                     dof_handler;
    std::unique_ptr<dealii::FiniteElement<dim, spacedim>> fe;
    dealii::Vector<double>                                coefficients;
    FieldCatalog                                          catalog;
    std::unique_ptr<
      dealii::SolutionTransfer<dim, dealii::Vector<double>, spacedim>>
                                       transfer;
    boost::signals2::scoped_connection before;
    boost::signals2::scoped_connection after;

    SerialFields(const std::string                    &filename,
                 dealii::Triangulation<dim, spacedim> &tria)
      : dof_handler(tria)
    {
#ifdef DEAL_II_WITH_VTK
      dealii::Triangulation<dim, spacedim> input_tria;
      dealii::DoFHandler<dim, spacedim>    input_dh(input_tria);
      dealii::Vector<double>               input_values;
      VTKUtils::read_vtk(filename, input_dh, input_values, catalog);
      fe = input_dh.get_fe().clone();
      dof_handler.distribute_dofs(*fe);
      coefficients.reinit(dof_handler.n_dofs());
      AssertDimension(input_tria.n_active_cells(), tria.n_active_cells());
      std::vector<dealii::types::global_dof_index> src(fe->n_dofs_per_cell());
      std::vector<dealii::types::global_dof_index> dst(src.size());
      auto input_cell = input_dh.begin_active();
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          AssertThrow(cell->id() == input_cell->id(),
                      dealii::ExcMessage(
                        "Input fields require matching CellIds."));
          input_cell->get_dof_indices(src);
          cell->get_dof_indices(dst);
          for (unsigned int i = 0; i < src.size(); ++i)
            coefficients[dst[i]] = input_values[src[i]];
          ++input_cell;
        }
      before = tria.signals.pre_refinement.connect([this]() {
        transfer = std::make_unique<
          dealii::SolutionTransfer<dim, dealii::Vector<double>, spacedim>>(
          dof_handler);
        transfer->prepare_for_coarsening_and_refinement(coefficients);
      });
      after  = tria.signals.post_refinement.connect([this]() {
        dof_handler.distribute_dofs(*fe);
        dealii::Vector<double> values(dof_handler.n_dofs());
        transfer->interpolate(values);
        coefficients.swap(values);
        transfer.reset();
      });
#else
      (void)filename;
      AssertThrow(false,
                  dealii::ExcMessage("Reading fields requires VTK support."));
#endif
    }
  };

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
    if constexpr (dim == 3 && spacedim == 3)
      {
        enter_subsection("Surface mesh");
        add_parameter(
          "Input",
          mesh_input,
          "Read a volume, mesh a closed surface, or detect automatically.",
          this->prm,
          dealii::Patterns::Selection("auto|volume|surface"));
        add_parameter("Target edge length",
                      surface_mesh_size,
                      "Length in input coordinates, before Grid scale.",
                      this->prm,
                      dealii::Patterns::Double(0.));
        add_parameter("Feature angle",
                      surface_feature_angle,
                      "Gmsh surface classification angle in degrees.",
                      this->prm,
                      dealii::Patterns::Double(0., 180.));
        add_parameter(
          "Volume cells",
          surface_volume_cells,
          "Hexahedra are obtained by conforming tetrahedron subdivision.",
          this->prm,
          dealii::Patterns::Selection("tetrahedra|hexahedra"));
        leave_subsection();
      }
    enter_subsection("VTK support");
    add_parameter("Read fields", read_vtk_fields);
    add_parameter("Boundary id field", vtk_boundary_id_field);
    add_parameter("Material id field", vtk_material_id_field);
    leave_subsection();
  }


  template <int dim, int spacedim>
  std::unique_ptr<typename Domain<dim, spacedim>::TriangulationType>
  Domain<dim, spacedim>::make_triangulation_storage(
    const DomainParameters<dim, spacedim> &parameters,
    MPI_Comm                               communicator)
  {
    const bool fully_distributed =
      parameters.triangulation_type == "fullydistributed" || dim == 1;

    if (fully_distributed)
      return std::make_unique<FullyDistributedTriangulation>(communicator);

    return std::make_unique<DistributedTriangulation>(
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
    , triangulation_(triangulation_storage_.get())
  {}


  template <int dim, int spacedim>
  Domain<dim, spacedim>::Domain(
    const DomainParameters<dim, spacedim> &parameters,
    TriangulationType                     &triangulation)
    : parameters_(parameters)
    , communicator_(triangulation.get_mpi_communicator())
    , triangulation_(&triangulation)
  {
    if (triangulation.n_global_active_cells() != 0)
      {
        detail::uniform_reference_cell(triangulation);
        import_vtk_fields_if_requested();
      }
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
    prepare_grid();
    finalize_grid();
  }

  template <int dim, int spacedim>
  void
  Domain<dim, spacedim>::prepare_grid(const bool serial)
  {
    prepare_grid_impl(serial, nullptr);
  }

  template <int dim, int spacedim>
  void
  Domain<dim, spacedim>::prepare_grid(
    const dealii::Triangulation<dim, spacedim> &coarse_mesh)
  {
    prepare_grid_impl(true, &coarse_mesh);
  }

  template <int dim, int spacedim>
  void
  Domain<dim, spacedim>::prepare_grid_impl(
    const bool                                  serial,
    const dealii::Triangulation<dim, spacedim> *coarse_mesh)
  {
    AssertThrow(!grid_prepared_ && triangulation_->n_global_active_cells() == 0,
                dealii::ExcMessage(
                  "Cannot prepare an already populated Domain."));
    if (serial || uses_fully_distributed_triangulation())
      {
        serial_triangulation_ =
          std::make_unique<dealii::Triangulation<dim, spacedim>>(
            typename dealii::Triangulation<dim, spacedim>::MeshSmoothing(
              dealii::Triangulation<dim, spacedim>::smoothing_on_refinement |
              dealii::Triangulation<dim, spacedim>::smoothing_on_coarsening));
        if (coarse_mesh != nullptr)
          serial_triangulation_->copy_triangulation(*coarse_mesh);
        else
          make_grid_in(parameters_, *serial_triangulation_);
        detail::uniform_reference_cell(*serial_triangulation_);
        // deal.II's optional smoothing rules use hypercube face numbering.
        if constexpr (dim > 1)
          if (serial_triangulation_->all_reference_cells_are_simplex())
            serial_triangulation_->set_mesh_smoothing(
              dealii::Triangulation<dim, spacedim>::none);
        AssertThrow(
          uses_fully_distributed_triangulation() ||
            serial_triangulation_->all_reference_cells_are_hyper_cube(),
          dealii::ExcMessage(
            "Simplex meshes require Triangulation type = fullydistributed."));
        if (parameters_.read_vtk_fields &&
            std::filesystem::is_regular_file(parameters_.name_of_grid))
          {
            std::string extension =
              std::filesystem::path(parameters_.name_of_grid)
                .extension()
                .string();
            std::transform(extension.begin(),
                           extension.end(),
                           extension.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            if (extension == ".vtk" || extension == ".vtu" ||
                extension == ".pvtu")
              serial_fields_ =
                std::make_unique<SerialFields>(parameters_.name_of_grid,
                                               *serial_triangulation_);
          }
      }
    else if (!uses_fully_distributed_triangulation())
      {
        make_grid_in(parameters_, *triangulation_);
        detail::uniform_reference_cell(*triangulation_);
        import_vtk_fields_if_requested();
      }
    grid_prepared_ = true;
    auto &tria     = refinement_triangulation();
    if (parameters_.grid_scale != 1.)
      dealii::GridTools::scale(parameters_.grid_scale, tria);
    tria.refine_global(parameters_.initial_refinement);
  }

  template <int dim, int spacedim>
  double
  Domain<dim, spacedim>::prepared_field_value(
    const std::string        &name,
    const dealii::CellId     &cell_id,
    const dealii::Point<dim> &unit_point,
    const unsigned int        component) const
  {
    AssertThrow(
      grid_prepared_ && serial_fields_ != nullptr,
      dealii::ExcMessage(
        "Prepared field evaluation requires retained serial input fields."));
    const auto &fields = *serial_fields_;
    const auto  entry =
      std::find_if(fields.catalog.begin(),
                   fields.catalog.end(),
                   [&](const auto &field) { return field.name == name; });
    AssertThrow(entry != fields.catalog.end(),
                dealii::ExcMessage("Unknown prepared input field '" + name +
                                   "'."));
    AssertThrow(component < entry->n_components,
                dealii::ExcMessage(
                  "Prepared input component is out of range."));
    const auto raw_cell =
      fields.dof_handler.get_triangulation().create_cell_iterator(cell_id);
    AssertThrow(raw_cell->is_active(),
                dealii::ExcMessage("Prepared field requires an active cell."));
    const auto cell = raw_cell->as_dof_handler_iterator(fields.dof_handler);
    std::vector<dealii::types::global_dof_index> indices(
      fields.fe->n_dofs_per_cell());
    cell->get_dof_indices(indices);
    double     value    = 0.;
    const auto selected = entry->first_fe_component + component;
    for (unsigned int i = 0; i < indices.size(); ++i)
      value += fields.coefficients[indices[i]] *
               fields.fe->shape_value_component(i, unit_point, selected);
    return value;
  }

  template <int dim, int spacedim>
  void
  Domain<dim, spacedim>::finalize_grid()
  {
    finalize_grid_impl(nullptr);
  }

  template <int dim, int spacedim>
  void
  Domain<dim, spacedim>::finalize_grid(
    const dealii::RepartitioningPolicyTools::Base<dim, spacedim> &policy)
  {
    AssertThrow(
      uses_fully_distributed_triangulation(),
      dealii::ExcMessage(
        "A final ownership policy requires a fullydistributed Domain."));
    finalize_grid_impl(&policy);
  }

  template <int dim, int spacedim>
  void
  Domain<dim, spacedim>::finalize_grid_impl(
    const dealii::RepartitioningPolicyTools::Base<dim, spacedim> *policy)
  {
    AssertThrow(grid_prepared_,
                dealii::ExcMessage("Prepare the Domain before finalizing it."));
    if (serial_triangulation_)
      {
        detail::uniform_reference_cell(*serial_triangulation_);
        if (uses_fully_distributed_triangulation())
          {
            auto &fully_distributed =
              dynamic_cast<FullyDistributedTriangulation &>(*triangulation_);
            if constexpr (dim == 1)
              fully_distributed.set_partitioner(
                [](dealii::Triangulation<dim, spacedim> &tria,
                   const unsigned int                    n_partitions) {
                  dealii::GridTools::partition_triangulation_zorder(
                    n_partitions, tria, false);
                },
                dealii::TriangulationDescription::Settings::default_setting);
          }
        for (const auto manifold_id : serial_triangulation_->get_manifold_ids())
          if (manifold_id != dealii::numbers::flat_manifold_id)
            triangulation_->set_manifold(
              manifold_id, serial_triangulation_->get_manifold(manifold_id));
        if (uses_fully_distributed_triangulation())
          triangulation_->copy_triangulation(*serial_triangulation_);
        else if (!uses_fully_distributed_triangulation())
          {
            // p4est cannot copy a refined serial mesh. Reconstruct its coarse
            // mesh and replay the retained hierarchy before attaching fields.
            const auto &[vertices, cells, subcells] =
              dealii::GridTools::get_coarse_mesh_description(
                *serial_triangulation_);
            triangulation_->create_triangulation(vertices, cells, subcells);
            bool changed = true;
            while (changed)
              {
                unsigned int marked = 0;
                for (const auto &cell : triangulation_->active_cell_iterators())
                  if (cell->is_locally_owned())
                    {
                      const auto source =
                        serial_triangulation_->create_cell_iterator(cell->id());
                      if (source->has_children())
                        {
                          cell->set_refine_flag();
                          ++marked;
                        }
                    }
                changed =
                  dealii::Utilities::MPI::sum(marked, communicator_) > 0;
                if (changed)
                  triangulation_->execute_coarsening_and_refinement();
              }
            for (const auto &cell : triangulation_->active_cell_iterators())
              if (!cell->is_artificial())
                {
                  const auto source =
                    serial_triangulation_->create_cell_iterator(cell->id());
                  AssertThrow(
                    source->is_active(),
                    dealii::ExcMessage(
                      "Serial refinement is incompatible with p4est balancing."));
                  cell->set_material_id(source->material_id());
                  cell->set_manifold_id(source->manifold_id());
                  for (const auto f : cell->face_indices())
                    {
                      cell->face(f)->set_manifold_id(
                        source->face(f)->manifold_id());
                      if (cell->face(f)->at_boundary())
                        cell->face(f)->set_boundary_id(
                          source->face(f)->boundary_id());
                    }
                }
          }
        if (policy != nullptr)
          {
            auto &fully_distributed =
              dynamic_cast<FullyDistributedTriangulation &>(*triangulation_);
            const auto description = dealii::TriangulationDescription::
              Utilities::create_description_from_triangulation(
                fully_distributed,
                policy->partition(fully_distributed),
                dealii::TriangulationDescription::Settings::default_setting);
            // Use a one-time description instead of retaining a pointer to the
            // caller's policy in the triangulation.
            fully_distributed.clear();
            for (const auto manifold_id :
                 serial_triangulation_->get_manifold_ids())
              if (manifold_id != dealii::numbers::flat_manifold_id)
                fully_distributed.set_manifold(
                  manifold_id,
                  serial_triangulation_->get_manifold(manifold_id));
            fully_distributed.create_triangulation(description);
          }
        AssertThrow(
          triangulation_->n_global_active_cells() ==
            serial_triangulation_->n_global_active_cells(),
          dealii::ExcMessage(
            "Finalization must preserve the prepared active-cell hierarchy."));
        if (serial_fields_)
          vtk_fields_ =
            std::make_unique<ImportedFields>(serial_fields_->dof_handler,
                                             serial_fields_->coefficients,
                                             serial_fields_->catalog,
                                             *triangulation_,
                                             communicator_);
        serial_fields_.reset();
        serial_triangulation_.reset();
      }
    grid_prepared_ = false;
  }

  template <int dim, int spacedim>
  bool
  Domain<dim, spacedim>::grid_is_prepared() const
  {
    return grid_prepared_;
  }

  template <int dim, int spacedim>
  dealii::Triangulation<dim, spacedim> &
  Domain<dim, spacedim>::refinement_triangulation()
  {
    AssertThrow(grid_prepared_,
                dealii::ExcMessage(
                  "The Domain is not prepared for refinement."));
    if (serial_triangulation_)
      return *serial_triangulation_;
    return *triangulation_;
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
    return dynamic_cast<const FullyDistributedTriangulation *>(
             triangulation_.get()) != nullptr;
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
