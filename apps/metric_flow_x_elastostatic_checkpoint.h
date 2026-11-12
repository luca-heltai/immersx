// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#ifndef immersx_app_metric_flow_x_elastostatic_checkpoint_h
#define immersx_app_metric_flow_x_elastostatic_checkpoint_h

#include <deal.II/base/config.h>

#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/manifold_lib.h>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

#include <immersx/physics/elastic_static.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

namespace ImmersX::elastostatic_checkpoint_detail
{
  struct SolidManifold
  {
    enum class Kind
    {
      flat,
      cylindrical
    };

    dealii::types::manifold_id id   = dealii::numbers::flat_manifold_id;
    Kind                       kind = Kind::flat;
    std::array<double, 3>      periodicity{}, direction{}, point{};
    double                     tolerance = 0.;

    bool
    operator==(const SolidManifold &other) const
    {
      return std::tie(id, kind, periodicity, direction, point, tolerance) ==
             std::tie(other.id,
                      other.kind,
                      other.periodicity,
                      other.direction,
                      other.point,
                      other.tolerance);
    }

    template <class Archive>
    void
    serialize(Archive &archive, const unsigned int)
    {
      archive &id &kind &periodicity &direction &point &tolerance;
    }
  };

  struct Metadata
  {
    std::string  magic          = "ImmersX MetricFlowX elastostatic checkpoint";
    unsigned int version        = 2;
    std::string  dealii_version = DEAL_II_PACKAGE_VERSION;
    unsigned int ranks          = 0;
    double       time           = 0.;
    std::string  parameters;
    std::string  reference_fe;
    unsigned int reference_component = 0;
    std::string  lambda_fe;
    std::vector<SolidManifold> solid_manifolds;

    template <class Archive>
    void
    serialize(Archive &archive, const unsigned int)
    {
      archive &magic &version &dealii_version &ranks &time &parameters
        &reference_fe &reference_component &lambda_fe;
      if (version >= 2)
        archive &solid_manifolds;
    }
  };

  template <class Operation>
  void
  collective_file_operation(const MPI_Comm comm, const Operation &operation)
  {
    std::string error;
    try
      {
        operation();
      }
    catch (const std::exception &exception)
      {
        error = exception.what();
      }
    const auto errors = dealii::Utilities::MPI::all_gather(comm, error);
    for (const auto &message : errors)
      AssertThrow(message.empty(),
                  dealii::ExcMessage("Elastostatic checkpoint: " + message));
  }

  inline std::vector<SolidManifold>
  describe_solid_manifolds(const dealii::Triangulation<3> &mesh,
                           const MPI_Comm                  comm)
  {
    std::vector<SolidManifold> local;
    collective_file_operation(comm, [&] {
      std::set<dealii::types::manifold_id> ids;
      // Native load reconstructs inactive levels before their children.
      for (const auto &cell : mesh.cell_iterators())
        {
          ids.insert(cell->manifold_id());
          for (const auto face : cell->face_indices())
            ids.insert(cell->face(face)->manifold_id());
          for (unsigned int line = 0; line < cell->n_lines(); ++line)
            ids.insert(cell->line(line)->manifold_id());
        }
      for (const auto id : ids)
        {
          SolidManifold descriptor;
          descriptor.id        = id;
          const auto &manifold = mesh.get_manifold(id);
          if (typeid(manifold) == typeid(dealii::FlatManifold<3>))
            {
              const auto &flat =
                static_cast<const dealii::FlatManifold<3> &>(manifold);
              for (unsigned int d = 0; d < 3; ++d)
                descriptor.periodicity[d] = flat.get_periodicity()[d];
              descriptor.tolerance = flat.get_tolerance();
            }
          else if (typeid(manifold) == typeid(dealii::CylindricalManifold<3>))
            {
              descriptor.kind = SolidManifold::Kind::cylindrical;
              const auto &cylinder =
                static_cast<const dealii::CylindricalManifold<3> &>(manifold);
              for (unsigned int d = 0; d < 3; ++d)
                {
                  descriptor.direction[d] = cylinder.get_direction()[d];
                  descriptor.point[d]     = cylinder.get_point_on_axis()[d];
                }
              descriptor.tolerance = cylinder.get_tolerance();
            }
          else
            AssertThrow(
              false,
              dealii::ExcMessage(
                "Solid checkpoints support exact FlatManifold and "
                "CylindricalManifold objects; unsupported manifold ID " +
                std::to_string(id) + "."));
          local.push_back(descriptor);
        }
    });
    std::map<dealii::types::manifold_id, SolidManifold> descriptions;
    for (const auto &rank : dealii::Utilities::MPI::all_gather(comm, local))
      for (const auto &descriptor : rank)
        {
          const auto inserted = descriptions.emplace(descriptor.id, descriptor);
          AssertThrow(
            inserted.second || inserted.first->second == descriptor,
            dealii::ExcMessage(
              "Solid manifold descriptions differ between MPI ranks."));
        }
    std::vector<SolidManifold> result;
    for (const auto &entry : descriptions)
      result.push_back(entry.second);
    return result;
  }

  inline void
  restore_solid_manifolds(dealii::Triangulation<3>         &mesh,
                          const std::vector<SolidManifold> &descriptions)
  {
    AssertThrow(
      mesh.n_cells() == 0 && !descriptions.empty(),
      dealii::ExcMessage(
        "Restore solid manifolds before loading the checkpoint mesh."));
    std::set<dealii::types::manifold_id> ids;
    for (const auto &descriptor : descriptions)
      {
        AssertThrow(ids.insert(descriptor.id).second &&
                      std::isfinite(descriptor.tolerance) &&
                      descriptor.tolerance > 0.,
                    dealii::ExcMessage("Invalid solid manifold description."));
        if (descriptor.kind == SolidManifold::Kind::flat)
          {
            dealii::Tensor<1, 3> periodicity;
            for (unsigned int d = 0; d < 3; ++d)
              {
                AssertThrow(std::isfinite(descriptor.periodicity[d]) &&
                              descriptor.periodicity[d] >= 0.,
                            dealii::ExcMessage(
                              "Invalid flat manifold periodicity."));
                periodicity[d] = descriptor.periodicity[d];
              }
            if (descriptor.id != dealii::numbers::flat_manifold_id)
              mesh.set_manifold(descriptor.id,
                                dealii::FlatManifold<3>(periodicity,
                                                        descriptor.tolerance));
          }
        else if (descriptor.kind == SolidManifold::Kind::cylindrical)
          {
            dealii::Tensor<1, 3> direction;
            dealii::Point<3>     point;
            for (unsigned int d = 0; d < 3; ++d)
              {
                AssertThrow(std::isfinite(descriptor.direction[d]) &&
                              std::isfinite(descriptor.point[d]),
                            dealii::ExcMessage(
                              "Invalid cylindrical manifold axis."));
                direction[d] = descriptor.direction[d];
                point[d]     = descriptor.point[d];
              }
            AssertThrow(direction.norm() > 0.,
                        dealii::ExcMessage("Empty cylindrical manifold axis."));
            mesh.set_manifold(descriptor.id,
                              dealii::CylindricalManifold<3>(
                                direction, point, descriptor.tolerance));
          }
        else
          AssertThrow(false,
                      dealii::ExcMessage("Unsupported solid manifold kind."));
      }
  }

  using Manifest = std::vector<std::pair<std::string, std::uintmax_t>>;

  inline Metadata
  read_metadata(const std::string &directory, const MPI_Comm comm)
  {
    Metadata metadata;
    collective_file_operation(comm, [&] {
      if (dealii::Utilities::MPI::this_mpi_process(comm) != 0)
        return;
      const std::filesystem::path root(directory);
      AssertThrow(root.extension() != ".incomplete",
                  dealii::ExcMessage("Uncommitted coupled checkpoint."));
      std::ifstream complete(root / "COMPLETE", std::ios::binary);
      AssertThrow(complete.good(),
                  dealii::ExcMessage("Missing complete coupled checkpoint."));
      Manifest files;
      {
        boost::archive::binary_iarchive archive(complete);
        archive >> files;
      }
      AssertThrow(!files.empty(),
                  dealii::ExcMessage("Empty checkpoint manifest."));
      for (const auto &[name, size] : files)
        {
          const std::filesystem::path relative(name);
          bool valid_path = !relative.empty() && !relative.is_absolute();
          for (const auto &component : relative)
            valid_path = valid_path && component != "..";
          AssertThrow(valid_path,
                      dealii::ExcMessage("Invalid checkpoint manifest path."));
          AssertThrow(std::filesystem::is_regular_file(root / relative) &&
                        std::filesystem::file_size(root / relative) == size,
                      dealii::ExcMessage(
                        "Missing or truncated checkpoint file: " + name));
        }
      std::ifstream file(root / "coupled.bin", std::ios::binary);
      AssertThrow(file.good(), dealii::ExcMessage("Missing coupled metadata."));
      boost::archive::binary_iarchive archive(file);
      archive >> metadata;
    });
    metadata = dealii::Utilities::MPI::broadcast(comm, metadata, 0);
    AssertThrow(metadata.magic == Metadata().magic &&
                  metadata.version == Metadata().version,
                dealii::ExcMessage("Unsupported coupled checkpoint version."));
    AssertThrow(metadata.dealii_version == DEAL_II_PACKAGE_VERSION,
                dealii::ExcMessage("Checkpoint deal.II version differs."));
    AssertThrow(metadata.ranks == dealii::Utilities::MPI::n_mpi_processes(comm),
                dealii::ExcMessage(
                  "Checkpoint requires the original MPI size."));
    AssertThrow(std::isfinite(metadata.time) && !metadata.parameters.empty() &&
                  !metadata.reference_fe.empty() && !metadata.lambda_fe.empty(),
                dealii::ExcMessage("Incomplete coupled checkpoint metadata."));
    return metadata;
  }

  template <int dim, int spacedim, class Vector>
  Vector
  ghosted_copy(const dealii::DoFHandler<dim, spacedim> &dofs,
               const Vector                            &source,
               const MPI_Comm                           comm)
  {
    Vector result;
    result.reinit(dofs.locally_owned_dofs(),
                  dealii::DoFTools::extract_locally_relevant_dofs(dofs),
                  comm);
    result = source;
    result.update_ghost_values();
    return result;
  }

  /**
   * Save one accepted state with native FE transfers in deterministic order.
   * The source meshes, fields, and transfer objects stay alive until mesh.save.
   */
  template <class Flow,
            class Solid,
            class FlowVector,
            int dim,
            int spacedim,
            class LambdaVector,
            class ReferenceVector>
  void
  save(const std::string                       &directory,
       Metadata                                 metadata,
       Flow                                    &flow,
       Solid                                   &solid,
       const FlowVector                        &flow_state,
       const FlowVector                        &flow_derivative,
       const dealii::DoFHandler<dim, spacedim> &lambda_dofs,
       const LambdaVector                      &lambda_state,
       const LambdaVector                      &lambda_dot,
       const dealii::DoFHandler<dim, spacedim> &reference_dofs,
       const ReferenceVector                   &reference_coefficients,
       const MPI_Comm                           comm)
  {
    AssertThrow(
      dealii::Utilities::MPI::min(
        !directory.empty() && std::isfinite(metadata.time) &&
            !metadata.parameters.empty() &&
            &lambda_dofs.get_triangulation() == &flow.triangulation() &&
            &reference_dofs.get_triangulation() == &flow.triangulation() &&
            lambda_state.size() == lambda_dofs.n_dofs() &&
            lambda_dot.size() == lambda_dofs.n_dofs() &&
            reference_coefficients.size() == reference_dofs.n_dofs() &&
            metadata.lambda_fe == lambda_dofs.get_fe().get_name() &&
            metadata.reference_fe == reference_dofs.get_fe().get_name() &&
            metadata.reference_component <
              reference_dofs.get_fe().n_components() ?
          1u :
          0u,
        comm) == 1,
      dealii::ExcMessage(
        "Checkpoint fields do not match the native flow mesh."));
    AssertThrow(dealii::Utilities::MPI::min(metadata.time, comm) ==
                  dealii::Utilities::MPI::max(metadata.time, comm),
                dealii::ExcMessage(
                  "Checkpoint times differ between MPI ranks."));
    metadata.ranks          = dealii::Utilities::MPI::n_mpi_processes(comm);
    metadata.magic          = Metadata().magic;
    metadata.version        = Metadata().version;
    metadata.dealii_version = DEAL_II_PACKAGE_VERSION;
    metadata.solid_manifolds =
      describe_solid_manifolds(solid.triangulation(), comm);
    const std::filesystem::path destination(directory);
    const std::filesystem::path staging(directory + ".incomplete");
    collective_file_operation(comm, [&] {
      if (dealii::Utilities::MPI::this_mpi_process(comm) != 0)
        return;
      AssertThrow(!std::filesystem::exists(destination) &&
                    !std::filesystem::exists(staging),
                  dealii::ExcMessage(
                    "Checkpoint destination or staging already exists."));
      if (!destination.parent_path().empty())
        std::filesystem::create_directories(destination.parent_path());
      AssertThrow(std::filesystem::create_directory(staging),
                  dealii::ExcMessage(
                    "Cannot create checkpoint staging directory."));
      std::ofstream file(staging / "coupled.bin", std::ios::binary);
      {
        boost::archive::binary_oarchive archive(file);
        archive << metadata;
      }
      file.close();
      AssertThrow(file.good(),
                  dealii::ExcMessage("Cannot write coupled metadata."));
    });
    MPI_Barrier(comm);

    auto lambda_ghost     = ghosted_copy(lambda_dofs, lambda_state, comm);
    auto lambda_dot_ghost = ghosted_copy(lambda_dofs, lambda_dot, comm);
    auto reference_ghost =
      ghosted_copy(reference_dofs, reference_coefficients, comm);
    elastic_static_detail::CheckpointTransfer<dim, LambdaVector, spacedim>
      lambda_transfer(lambda_dofs);
    elastic_static_detail::CheckpointTransfer<dim, ReferenceVector, spacedim>
      reference_transfer(reference_dofs);
    collective_file_operation(comm, [&] {
      solid.save_checkpoint((staging / "solid_mesh").string());
    });
    // Native flow registers its state pair first. These two transfers are the
    // second and third attachments on that same native mesh.
    flow.save_checkpoint(
      (staging / "flow").string(),
      metadata.time,
      flow_state,
      flow_derivative,
      [&](const std::string &) {
        lambda_transfer.prepare_for_serialization(
          std::vector<const LambdaVector *>{&lambda_ghost, &lambda_dot_ghost});
        reference_transfer.prepare_for_serialization(reference_ghost);
      });
    MPI_Barrier(comm);
    collective_file_operation(comm, [&] {
      if (dealii::Utilities::MPI::this_mpi_process(comm) != 0)
        return;
      Manifest files;
      for (const auto &entry :
           std::filesystem::recursive_directory_iterator(staging))
        if (entry.is_regular_file())
          files.emplace_back(
            std::filesystem::relative(entry.path(), staging).string(),
            entry.file_size());
      std::ofstream file(staging / "COMPLETE", std::ios::binary);
      {
        boost::archive::binary_oarchive archive(file);
        archive << files;
      }
      file.close();
      AssertThrow(file.good(),
                  dealii::ExcMessage("Cannot commit checkpoint manifest."));
      std::filesystem::rename(staging, destination);
    });
    MPI_Barrier(comm);
  }

  /** Restore additional FE attachments after native flow deserialization. */
  template <int dim, int spacedim, class LambdaVector, class ReferenceVector>
  void
  restore_fields(const dealii::DoFHandler<dim, spacedim> &lambda_dofs,
                 LambdaVector                            &lambda_state,
                 LambdaVector                            &lambda_dot,
                 const dealii::DoFHandler<dim, spacedim> &reference_dofs,
                 ReferenceVector &reference_coefficients)
  {
    const auto      comm = lambda_dofs.get_mpi_communicator();
    LambdaVector    owned_lambda, owned_lambda_dot;
    ReferenceVector owned_reference;
    owned_lambda.reinit(lambda_dofs.locally_owned_dofs(), comm);
    owned_lambda_dot.reinit(lambda_dofs.locally_owned_dofs(), comm);
    owned_reference.reinit(reference_dofs.locally_owned_dofs(), comm);
    elastic_static_detail::CheckpointTransfer<dim, LambdaVector, spacedim>
      lambda_transfer(lambda_dofs);
    elastic_static_detail::CheckpointTransfer<dim, ReferenceVector, spacedim>
                                reference_transfer(reference_dofs);
    std::vector<LambdaVector *> lambda_pair{&owned_lambda, &owned_lambda_dot};
    lambda_transfer.deserialize(lambda_pair);
    reference_transfer.deserialize(owned_reference);
    lambda_state           = owned_lambda;
    lambda_dot             = owned_lambda_dot;
    reference_coefficients = owned_reference;
    lambda_state.update_ghost_values();
    lambda_dot.update_ghost_values();
    reference_coefficients.update_ghost_values();
  }
} // namespace ImmersX::elastostatic_checkpoint_detail
#endif
