// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_output_handler_h
#define immersx_output_handler_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/numerics/data_out.h>

#include <immersx/core/fe_space.h>
#include <immersx/core/state.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace ImmersX
{
  /** Parameters shared by one family of semantic output handlers. */
  struct OutputHandlerParameters : public dealii::ParameterAcceptor
  {
    explicit OutputHandlerParameters(const std::string &subsection = "/Output/")
      : dealii::ParameterAcceptor(subsection)
    {
      add_parameter("Output directory", output_directory);
    }

    std::string output_directory = ".";
  };

  /** Write semantic execution fields on one native finite-element space. */
  template <int dim, int spacedim, typename VectorType>
  class OutputHandler
  {
  public:
    using global_index = dealii::types::global_dof_index;
    using Space        = FiniteElementSpaceView<dim, spacedim>;
    using ScalarField =
      Field<dim, spacedim, dealii::FEValuesExtractors::Scalar>;
    using VectorField =
      Field<dim, spacedim, dealii::FEValuesExtractors::Vector>;

    OutputHandler(const Space                   &space,
                  const OutputHandlerParameters &parameters,
                  std::string                    basename)
      : space_(space)
      , parameters_(parameters)
      , basename_(std::move(basename))
    {
      AssertThrow(!basename_.empty(),
                  dealii::ExcMessage("An output handler needs a basename."));
    }

    /** Register a scalar semantic field on this handler's FE space. */
    void
    add_field(const ScalarField &field)
    {
      add_field_impl(field, field.extractor().component, 1, false);
    }

    /** Register a vector semantic field on this handler's FE space. */
    void
    add_field(const VectorField &field)
    {
      add_field_impl(field,
                     field.extractor().first_vector_component,
                     spacedim,
                     true);
    }

    /** Write one output step from solver-neutral semantic state. */
    void
    write(const StateAccessor<VectorType> &state, const double time)
    {
      AssertThrow(!fields_.empty(),
                  dealii::ExcMessage(
                    "An OutputHandler needs at least one registered field."));

      const auto n_components = space_.finite_element().n_components();
      std::vector<const RegisteredField *> component_fields(n_components,
                                                            nullptr);
      for (const auto &field : fields_)
        {
          AssertThrow(
            field.first_component + field.n_components <= n_components,
            dealii::ExcMessage(
              "A registered output field exceeds the FE components."));
          for (unsigned int component = 0; component < field.n_components;
               ++component)
            {
              const auto component_index = field.first_component + component;
              AssertThrow(
                component_fields[component_index] == nullptr,
                dealii::ExcMessage(
                  "Registered output fields overlap in FE components."));
              component_fields[component_index] = &field;
            }
        }

      for (unsigned int component = 0; component < n_components; ++component)
        AssertThrow(component_fields[component] != nullptr,
                    dealii::ExcMessage(
                      "Registered output fields do not cover the FE space."));

      VectorType native;
      native.reinit(space_.locally_owned_dofs(),
                    space_.locally_relevant_dofs(),
                    space_.mpi_communicator());
      native = 0.;
      std::vector<dealii::IndexSet> component_dofs(n_components);
      for (unsigned int component = 0; component < n_components; ++component)
        component_dofs[component] = dealii::DoFTools::extract_dofs(
          space_.dof_handler(),
          space_.finite_element().component_mask(
            dealii::FEValuesExtractors::Scalar(component)));

      for (const auto native_index : space_.locally_owned_dofs())
        {
          unsigned int component = 0;
          while (component < n_components &&
                 !component_dofs[component].is_element(native_index))
            ++component;
          AssertThrow(component < n_components,
                      dealii::ExcMessage(
                        "A native DoF has no finite-element component."));
          const auto &field = *component_fields[component];
          AssertThrow(field.has_execution_index(native_index),
                      dealii::ExcMessage(
                        "A registered output field has no execution index for "
                        "a native DoF."));
          const auto  execution_index = field.execution_index(native_index);
          const auto &values          = state.field(field.id, time);
          AssertThrow(values.locally_owned_elements().is_element(
                        execution_index),
                      dealii::ExcMessage(
                        "A registered output field does not own the required "
                        "execution DoF."));
          native[native_index] = values[execution_index];
        }
      native.update_ghost_values();

      std::vector<std::string> names;
      std::vector<
        dealii::DataComponentInterpretation::DataComponentInterpretation>
        interpretation;
      names.reserve(n_components);
      interpretation.reserve(n_components);
      for (unsigned int component = 0; component < n_components; ++component)
        {
          const auto &field = *component_fields[component];
          names.push_back(field.name);
          interpretation.push_back(
            field.is_vector ?
              dealii::DataComponentInterpretation::component_is_part_of_vector :
              dealii::DataComponentInterpretation::component_is_scalar);
        }

      dealii::DataOut<dim, spacedim> data_out;
      data_out.attach_dof_handler(space_.dof_handler());
      data_out.add_data_vector(native,
                               names,
                               dealii::DataOut<dim, spacedim>::type_dof_data,
                               interpretation);
      data_out.build_patches(space_.mapping());

      const auto communicator = space_.mpi_communicator();
      const auto rank = dealii::Utilities::MPI::this_mpi_process(communicator);
      const std::filesystem::path output_directory(
        parameters_.output_directory);
      if (rank == 0)
        std::filesystem::create_directories(output_directory);
      MPI_Barrier(communicator);

      const auto filename =
        basename_ + "_" + std::to_string(output_cycle_) + ".vtu";
      data_out.write_vtu_in_parallel((output_directory / filename).string(),
                                     communicator);
      output_records_.emplace_back(time, filename);

      if (rank == 0)
        {
          std::ofstream pvd(output_directory / (basename_ + ".pvd"));
          dealii::DataOutBase::write_pvd_record(pvd, output_records_);
        }
      MPI_Barrier(communicator);
      ++output_cycle_;
    }

  private:
    struct RegisteredField
    {
      FieldId                                   id;
      std::string                               name;
      unsigned int                              first_component;
      unsigned int                              n_components;
      bool                                      is_vector;
      std::function<global_index(global_index)> execution_index;
      std::function<bool(global_index)>         has_execution_index;
    };

    template <typename FieldType>
    void
    add_field_impl(const FieldType   &field,
                   const unsigned int first_component,
                   const unsigned int n_components,
                   const bool         is_vector)
    {
      AssertThrow(
        field.is_registered(),
        dealii::ExcMessage(
          "An OutputHandler field must be registered in a StateLayout."));
      AssertThrow(&field.dof_handler() == &space_.dof_handler(),
                  dealii::ExcMessage(
                    "An OutputHandler field belongs to another FE space."));
      AssertThrow(first_component + n_components <=
                    space_.finite_element().n_components(),
                  dealii::ExcMessage(
                    "An OutputHandler field exceeds the FE components."));

      RegisteredField registered;
      registered.id              = field.id();
      registered.name            = field.name();
      registered.first_component = first_component;
      registered.n_components    = n_components;
      registered.is_vector       = is_vector;
      registered.execution_index = [field](const global_index native_index) {
        return field.execution_index(native_index);
      };
      registered.has_execution_index =
        [field](const global_index native_index) {
          return field.has_execution_index(native_index);
        };
      fields_.push_back(std::move(registered));
    }

    const Space                                &space_;
    const OutputHandlerParameters              &parameters_;
    std::string                                 basename_;
    std::vector<RegisteredField>                fields_;
    std::vector<std::pair<double, std::string>> output_records_;
    unsigned int                                output_cycle_ = 0;
  };
} // namespace ImmersX

#endif // immersx_output_handler_h
