// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II library.
//
// ---------------------------------------------------------------------

#ifndef immersx_coral_register_h
#define immersx_coral_register_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/numerics/data_out.h>

#include <coral.h>
#include <coral_log.h>
#include <coral_network.h>
#include <coral_plugin.h>
#include <immersx/coral/coupled_poisson_elasticity.h>
#include <immersx/coral/fiber_reinforced_elastodynamics.h>
#include <immersx/coral/ida_elastodynamics.h>
#include <immersx/coral/reduced_poisson.h>
#include <immersx/core/constraint.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/core/observable.h>
#include <immersx/physics/elastic_static.h>
#include <immersx/physics/elastodynamics.h>
#include <immersx/physics/fiber_reinforced_elastodynamics.h>
#include <immersx/physics/poisson.h>
#include <immersx/physics/poisson_residual.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ImmersX::Coral
{
  using json = nlohmann::json;

  inline std::string
  dimensions(const int dim, const int spacedim)
  {
    return std::to_string(dim) + "," + std::to_string(spacedim);
  }

  inline coral::RegistryMetadata
  finite_element_space_metadata(const std::string &problem_name,
                                const int          dim,
                                const int          spacedim)
  {
    coral::RegistryMetadata metadata;
    metadata.operation    = "Finite element space";
    metadata.display_name = "Finite element space";
    metadata.variant_name = problem_name + ". " + std::to_string(dim) + "D";
    if (dim != spacedim)
      metadata.variant_name += " in " + std::to_string(spacedim) + "D";
    metadata.description =
      "Extract the finite element space from the " + problem_name + " problem.";
    return metadata;
  }

  template <int dim, int spacedim>
  inline void
  register_finite_element_space_type()
  {
    using Space = ImmersX::FiniteElementSpaceView<dim, spacedim>;
    coral::detail::set_type_alias<Space>("ImmersX::FiniteElementSpaceView<" +
                                         dimensions(dim, spacedim) + ">");
    coral::NodeObject::register_output_type<Space>();
  }

  inline coral::RegistryMetadata
  field_metadata(const std::string &operation,
                 const std::string &field_kind,
                 const int          dim,
                 const int          spacedim)
  {
    coral::RegistryMetadata metadata;
    metadata.operation    = operation;
    metadata.display_name = field_kind;
    metadata.variant_name = field_kind + ". " + std::to_string(dim) + "D";
    if (dim != spacedim)
      metadata.variant_name += " in " + std::to_string(spacedim) + "D";
    metadata.description =
      "Describe a generic " + field_kind + " over the finite element space.";
    return metadata;
  }

  template <int dim, int spacedim>
  inline void
  register_scalar_field_types()
  {
    using Space = ImmersX::FiniteElementSpaceView<dim, spacedim>;
    using Field =
      ImmersX::Field<dim, spacedim, dealii::FEValuesExtractors::Scalar>;

    register_finite_element_space_type<dim, spacedim>();
    coral::detail::set_type_alias<Field>(
      "ImmersX::Field<" + dimensions(dim, spacedim) + ",Scalar>");
    coral::NodeObject::register_output_type<Field>();

    const auto metadata =
      field_metadata("Scalar field", "Scalar field", dim, spacedim);
    coral::NodeObject::register_function(
      std::function<Field(const Space &, const std::string &)>(
        [](const Space &space, const std::string &name) {
          return ImmersX::scalar_field(space, name);
        }),
      {"space", "name"},
      metadata);

    coral::RegistryMetadata registered_metadata;
    registered_metadata.operation    = "Registered scalar field";
    registered_metadata.display_name = "Registered scalar field";
    registered_metadata.variant_name =
      "Scalar field with semantic id. " + std::to_string(dim) + "D";
    if (dim != spacedim)
      registered_metadata.variant_name +=
        " in " + std::to_string(spacedim) + "D";
    registered_metadata.description =
      "Bind a semantic FieldId to a scalar field description.";
    coral::NodeObject::register_function(
      std::function<
        Field(const Space &, const ImmersX::FieldId &, const std::string &)>(
        [](const Space            &space,
           const ImmersX::FieldId &id,
           const std::string      &name) {
          return space.field(id, name, dealii::FEValuesExtractors::Scalar(0));
        }),
      {"space", "field", "name"},
      registered_metadata);
  }

  template <int dim, int spacedim>
  inline void
  register_vector_field_types()
  {
    using Space = ImmersX::FiniteElementSpaceView<dim, spacedim>;
    using Field =
      ImmersX::Field<dim, spacedim, dealii::FEValuesExtractors::Vector>;

    register_finite_element_space_type<dim, spacedim>();
    coral::detail::set_type_alias<Field>(
      "ImmersX::Field<" + dimensions(dim, spacedim) + ",Vector>");
    coral::NodeObject::register_output_type<Field>();

    const auto metadata =
      field_metadata("Vector field", "Vector field", dim, spacedim);
    coral::NodeObject::register_function(
      std::function<Field(const Space &, const std::string &)>(
        [](const Space &space, const std::string &name) {
          return ImmersX::vector_field(space, name);
        }),
      {"space", "name"},
      metadata);
  }

  template <int dim, int spacedim>
  inline void
  register_field_observable_operations();

  template <int dim, int spacedim>
  inline void
  register_expression_algebra();

  template <int dim, int spacedim>
  inline void
  register_field_types()
  {
    register_scalar_field_types<dim, spacedim>();
    register_vector_field_types<dim, spacedim>();
    register_field_observable_operations<dim, spacedim>();
    register_expression_algebra<dim, spacedim>();
  }

  inline coral::RegistryMetadata
  field_observable_metadata(const std::string &operation,
                            const std::string &field_kind,
                            const std::string &observable_kind,
                            const int          dim,
                            const int          spacedim)
  {
    coral::RegistryMetadata metadata;
    metadata.operation    = operation;
    metadata.display_name = operation;
    metadata.variant_name =
      field_kind + " field · " + std::to_string(dim) + "D";
    if (dim != spacedim)
      metadata.variant_name += " in " + std::to_string(spacedim) + "D";
    metadata.description =
      "Apply " + observable_kind + " to a generic " + field_kind + " field.";
    return metadata;
  }

  inline coral::RegistryMetadata
  expression_metadata(const std::string &operation,
                      const std::string &variant_name,
                      const std::string &description,
                      const int          dim,
                      const int          spacedim)
  {
    coral::RegistryMetadata metadata;
    metadata.operation    = operation;
    metadata.display_name = operation;
    metadata.variant_name = variant_name + ". " + std::to_string(dim) + "D";
    if (dim != spacedim)
      metadata.variant_name += " in " + std::to_string(spacedim) + "D";
    metadata.description = description;
    return metadata;
  }

  template <int dim, int spacedim, typename Extractor, typename Function>
  inline void
  register_field_observable_operation(const std::string &field_kind,
                                      const std::string &operation,
                                      const std::string &observable_kind,
                                      Function           function)
  {
    using Field = ImmersX::Field<dim, spacedim, Extractor>;

    using Observable =
      std::decay_t<decltype(function(std::declval<const Field &>()))>;
    coral::detail::set_type_alias<Observable>(
      "ImmersX::Observable<" + dimensions(dim, spacedim) + "," + field_kind +
      "," + observable_kind + ">");
    coral::NodeObject::register_output_type<Observable>();
    coral::NodeObject::register_function(
      std::function<Observable(const Field &)>(function),
      {"field"},
      field_observable_metadata(
        operation, field_kind, observable_kind, dim, spacedim));
  }

  template <int dim, int spacedim, typename Extractor>
  inline void
  register_field_observable_operations(const std::string &field_kind)
  {
    register_field_observable_operation<dim, spacedim, Extractor>(
      field_kind, "Field value", "value", [](const auto &field) {
        return ImmersX::value(field);
      });
    register_field_observable_operation<dim, spacedim, Extractor>(
      field_kind, "Field gradient", "gradient", [](const auto &field) {
        return ImmersX::gradient(field);
      });
  }

  template <int dim, int spacedim>
  inline void
  register_vector_field_observable_operations()
  {
    using Extractor = dealii::FEValuesExtractors::Vector;

    register_field_observable_operation<dim, spacedim, Extractor>(
      "Vector", "Field divergence", "divergence", [](const auto &field) {
        return ImmersX::divergence(field);
      });
    register_field_observable_operation<dim, spacedim, Extractor>(
      "Vector",
      "Field symmetric gradient",
      "symmetric gradient",
      [](const auto &field) { return ImmersX::symmetric_gradient(field); });
    if constexpr (spacedim > 1)
      register_field_observable_operation<dim, spacedim, Extractor>(
        "Vector", "Field curl", "curl", [](const auto &field) {
          return ImmersX::curl(field);
        });
  }

  template <int dim, int spacedim>
  inline void
  register_field_observable_operations()
  {
    register_field_observable_operations<dim,
                                         spacedim,
                                         dealii::FEValuesExtractors::Scalar>(
      "Scalar");
    register_field_observable_operations<dim,
                                         spacedim,
                                         dealii::FEValuesExtractors::Vector>(
      "Vector");
    register_vector_field_observable_operations<dim, spacedim>();
  }

  template <typename Expression>
  inline void
  register_scaled_expression(const std::string &variant_name,
                             const std::string &description,
                             const int          dim,
                             const int          spacedim)
  {
    using Scaled = std::decay_t<decltype(std::declval<double>() *
                                         std::declval<const Expression &>())>;
    coral::NodeObject::register_function(
      std::function<Scaled(double, const Expression &)>(
        [](const double coefficient, const Expression &expression) {
          return coefficient * expression;
        }),
      {"coefficient", "term"},
      expression_metadata(
        "Scale term", variant_name, description, dim, spacedim));
  }

  template <int dim, int spacedim, typename Extractor>
  inline void
  register_nonlinear_product_operation(const std::string &field_kind)
  {
    using Field = ImmersX::Field<dim, spacedim, Extractor>;
    using Gradient =
      std::decay_t<decltype(ImmersX::gradient(std::declval<const Field &>()))>;
    using Product = std::decay_t<decltype(std::declval<const Gradient &>() *
                                          std::declval<const Field &>())>;

    const auto variant_name = field_kind + " gradient times field";
    coral::detail::set_type_alias<Product>("ImmersX::NonlinearProduct<" +
                                           dimensions(dim, spacedim) + "," +
                                           field_kind + ">");
    coral::NodeObject::register_output_type<Product>();
    coral::NodeObject::register_function(
      std::function<Product(const Gradient &, const Field &)>(
        [](const Gradient &gradient, const Field &field) {
          return gradient * field;
        }),
      {"gradient", "field"},
      expression_metadata(
        "Nonlinear product",
        variant_name,
        "Build the nonlinear gradient-times-field term used by convection.",
        dim,
        spacedim));
    register_scaled_expression<Product>(
      variant_name,
      "Scale the nonlinear gradient-times-field term.",
      dim,
      spacedim);
  }

  template <int dim, int spacedim, typename Extractor>
  inline void
  register_expression_algebra(const std::string &field_kind)
  {
    using Field = ImmersX::Field<dim, spacedim, Extractor>;
    using Value =
      std::decay_t<decltype(ImmersX::value(std::declval<const Field &>()))>;
    using Gradient =
      std::decay_t<decltype(ImmersX::gradient(std::declval<const Field &>()))>;

    register_scaled_expression<Value>(field_kind + " value",
                                      "Scale a field value expression.",
                                      dim,
                                      spacedim);
    register_scaled_expression<Gradient>(field_kind + " gradient",
                                         "Scale a field gradient expression.",
                                         dim,
                                         spacedim);
    register_nonlinear_product_operation<dim, spacedim, Extractor>(field_kind);
  }

  template <int dim, int spacedim>
  inline void
  register_expression_algebra()
  {
    register_expression_algebra<dim,
                                spacedim,
                                dealii::FEValuesExtractors::Scalar>("Scalar");
    register_expression_algebra<dim,
                                spacedim,
                                dealii::FEValuesExtractors::Vector>("Vector");
  }

  inline auto
  parameter_acceptor_derived_types() -> std::vector<std::string> &;

  inline void
  register_common_types()
  {
    coral::detail::set_type_alias<unsigned int>("unsigned int");
    coral::detail::set_type_alias<std::string>("std::string");
    coral::detail::set_type_alias<ImmersX::FieldId>("ImmersX::FieldId");
    coral::detail::set_type_alias<dealii::ParameterAcceptor>(
      "dealii::ParameterAcceptor");

    parameter_acceptor_derived_types().clear();
    coral::NodeObject::register_abstract_type<dealii::ParameterAcceptor>();

    coral::NodeObject::register_elementary_type<std::string>();
    coral::NodeObject::register_elementary_type<bool>();
    coral::NodeObject::register_elementary_type<int>();
    coral::NodeObject::register_elementary_type<unsigned int>();
    coral::NodeObject::register_elementary_type<double>();
    coral::NodeObject::register_output_type<ImmersX::FieldId>();
    coral::Network::register_node();
  }

  template <int dim, int spacedim>
  inline std::string
  poisson_parameters_name()
  {
    return "ImmersX::PoissonParameters<" + dimensions(dim, spacedim) + ">";
  }

  template <int dim, int spacedim>
  inline std::string
  poisson_name()
  {
    return "ImmersX::Poisson<" + dimensions(dim, spacedim) + ">";
  }

  template <int dim, int spacedim>
  double
  poisson_solution_l2_norm(const ImmersX::PoissonSolver<dim, spacedim> &problem)
  {
    return problem.solution_l2_norm();
  }

  template <int dim, int spacedim>
  bool
  poisson_solution_is_finite(
    const ImmersX::PoissonSolver<dim, spacedim> &problem)
  {
    return problem.solution_is_finite();
  }

  inline auto
  parameter_acceptor_derived_types() -> std::vector<std::string> &
  {
    static std::vector<std::string> types;
    return types;
  }

  inline void
  initialize_parameter_file(const std::string &file_name)
  {
    const std::ifstream input(file_name);
    AssertThrow(input.good(),
                dealii::ExcMessage("Could not open Coral parameter file '" +
                                   file_name + "'."));
    dealii::ParameterAcceptor::initialize(file_name);
  }

  template <std::size_t>
  using parameter_acceptor_reference = dealii::ParameterAcceptor &;

  template <std::size_t... I>
  void
  register_initialize_parameters(std::index_sequence<I...>)
  {
    using Function = std::function<void(parameter_acceptor_reference<I>...,
                                        const std::string &)>;

    Function function = [](parameter_acceptor_reference<I>... parameters,
                           const std::string &file_name) {
      (static_cast<void>(parameters), ...);
      initialize_parameter_file(file_name);
    };

    std::vector<std::string> argument_names;
    argument_names.reserve(sizeof...(I) + 1);
    (argument_names.push_back("parameter_" + std::to_string(I + 1)), ...);
    argument_names.emplace_back("parameter_file");

    coral::RegistryMetadata metadata;
    metadata.operation    = "Initialize parameters";
    metadata.display_name = "Initialize parameters";
    metadata.variant_name =
      std::to_string(sizeof...(I)) +
      (sizeof...(I) == 1 ? " parameter object" : " parameter objects");
    metadata.description =
      "Initialize all live ParameterAcceptor objects from a parameter file.";

    coral::NodeObject::register_function(function, argument_names, metadata);
  }

  template <std::size_t... Arity>
  void
  register_initialize_parameter_families(std::index_sequence<Arity...>)
  {
    (register_initialize_parameters(std::make_index_sequence<Arity + 1>{}),
     ...);
  }

  inline void
  register_initialize_parameter_types()
  {
    register_initialize_parameter_families(std::make_index_sequence<8>{});
  }

  template <typename Parameters>
  void
  register_parameter_type(const std::string &type_name)
  {
    static_assert(std::is_base_of_v<dealii::ParameterAcceptor, Parameters>,
                  "Coral parameter types must derive from ParameterAcceptor.");

    coral::detail::set_type_alias<Parameters>(type_name);
    auto &initializer =
      coral::NodeObject::register_derived_type<dealii::ParameterAcceptor,
                                               Parameters,
                                               const std::string &>(
        "subsection");

    const auto derived_type =
      initializer.json_serializer.at("type").template get<std::string>();
    auto &known_derived_types = parameter_acceptor_derived_types();
    // Coral reinitializes the abstract base when registering a derived type.
    // Restore the complete derived-type list after each registration.
    if (std::find(known_derived_types.begin(),
                  known_derived_types.end(),
                  derived_type) == known_derived_types.end())
      known_derived_types.push_back(derived_type);

    auto &base_initializer =
      coral::NodeObject::register_abstract_type<dealii::ParameterAcceptor>();
    base_initializer.json_serializer["derived"] = known_derived_types;
  }

  template <int dim, int spacedim>
  void
  write_scalar_field(
    const ImmersX::FiniteElementSpaceView<dim, spacedim> &space,
    const ImmersXLA::MPI::Vector                         &field,
    const std::string                                    &output_file,
    const std::string                                    &field_name)
  {
    const std::filesystem::path pvd_path(output_file);
    if (!pvd_path.parent_path().empty())
      std::filesystem::create_directories(pvd_path.parent_path());

    dealii::DataOut<dim, spacedim> data_out;
    data_out.attach_dof_handler(space.dof_handler());
    data_out.add_data_vector(field,
                             field_name,
                             dealii::DataOut<dim, spacedim>::type_dof_data);
    data_out.build_patches();

    const auto rank =
      dealii::Utilities::MPI::this_mpi_process(space.mpi_communicator());
    const auto vtu_name =
      pvd_path.stem().string() + "-0." + std::to_string(rank) + ".vtu";
    std::ofstream vtu_file(pvd_path.parent_path() / vtu_name);
    data_out.write_vtu(vtu_file);

    MPI_Barrier(space.mpi_communicator());
    if (rank == 0)
      {
        std::ofstream pvd_file(pvd_path);
        dealii::DataOutBase::write_pvd_record(pvd_file, {{0., vtu_name}});
      }
  }

  inline void
  assert_finite_below(const double value, const double limit)
  {
    AssertThrow(std::isfinite(value) && value < limit,
                dealii::ExcMessage("The supplied residual is not below the "
                                   "requested finite limit."));
  }

  template <int dim, int spacedim>
  void
  register_owned_finite_element_space_types()
  {
    using Parameters = ImmersX::FiniteElementSpaceParameters<dim, spacedim>;
    using Space      = ImmersX::FiniteElementSpace<dim, spacedim>;
    using View       = ImmersX::FiniteElementSpaceView<dim, spacedim>;
    using OwnedSpace = std::shared_ptr<Space>;

    const auto parameter_name = "ImmersX::FiniteElementSpaceParameters<" +
                                dimensions(dim, spacedim) + ">";
    register_parameter_type<Parameters>(parameter_name);

    const auto space_name =
      "ImmersX::FiniteElementSpace<" + dimensions(dim, spacedim) + ">";
    coral::detail::set_type_alias<Space>(space_name);
    coral::detail::set_type_alias<OwnedSpace>(
      "ImmersX::OwnedFiniteElementSpace<" + dimensions(dim, spacedim) + ">");
    coral::NodeObject::register_output_type<OwnedSpace>();

    coral::RegistryMetadata create_metadata;
    create_metadata.operation    = "Create finite element space";
    create_metadata.display_name = "Create finite element space";
    create_metadata.variant_name =
      "From finite element space view. " + dimensions(dim, spacedim);
    create_metadata.description =
      "Create an owning finite element space on an existing geometry.";
    coral::NodeObject::register_function(
      std::function<OwnedSpace(const View &, const Parameters &)>(
        [](const View &view, const Parameters &parameters) {
          return std::make_shared<Space>(view.distributed_triangulation(),
                                         parameters);
        }),
      {"source", "parameters"},
      create_metadata);

    coral::RegistryMetadata view_metadata;
    view_metadata.operation    = "Finite element space";
    view_metadata.display_name = "Finite element space";
    view_metadata.variant_name =
      "Owning space view. " + dimensions(dim, spacedim);
    view_metadata.description =
      "Expose the non-owning view of an owning finite element space.";
    coral::NodeObject::register_function(
      std::function<View(const OwnedSpace &)>(
        [](const OwnedSpace &space) { return space->view(); }),
      {"space"},
      view_metadata);
  }

  template <int dim, int spacedim>
  using LinearAdapterFor =
    ImmersX::LinearAdapter<ImmersXLA::MPI::Vector, ImmersXLA::MPI::BlockVector>;

  template <int dim, int spacedim>
  void
  register_linear_problem_operations()
  {
    using Adapter           = LinearAdapterFor<dim, spacedim>;
    using Problem           = ImmersX::PoissonSolver<dim, spacedim>;
    using Fields            = ImmersX::PoissonFields<dim, spacedim>;
    using ProblemHandleType = ImmersX::ProblemHandle<Adapter, Fields>;
    using AdapterHandle     = std::shared_ptr<Adapter>;
    using Vector            = ImmersXLA::MPI::Vector;

    coral::detail::set_type_alias<Adapter>("ImmersX::LinearAdapter<2>");
    coral::detail::set_type_alias<AdapterHandle>("ImmersX::LinearExecution");
    coral::detail::set_type_alias<Vector>("ImmersX::LinearFieldVector");
    coral::NodeObject::register_output_type<AdapterHandle>();
    coral::NodeObject::register_output_type<Vector>();
    const auto handle_name =
      "ImmersX::LinearProblemHandle<" + dimensions(dim, spacedim) + ">";
    coral::detail::set_type_alias<ProblemHandleType>(handle_name);
    coral::NodeObject::register_output_type<ProblemHandleType>();

    coral::RegistryMetadata add_metadata;
    add_metadata.operation    = "Add problem to linear execution";
    add_metadata.display_name = "Add problem";
    add_metadata.variant_name = "Poisson problem. " + dimensions(dim, spacedim);
    add_metadata.description =
      "Add a generic assembled Problem to a LinearAdapter.";
    coral::NodeObject::register_function(
      std::function<ProblemHandleType(
        AdapterHandle &, const Problem &, const std::string &)>(
        [](AdapterHandle     &adapter,
           const Problem     &problem,
           const std::string &prefix) {
          return (*adapter).add(problem, prefix);
        }),
      {"adapter", "problem", "prefix"},
      add_metadata);

    coral::RegistryMetadata field_metadata;
    field_metadata.operation    = "Problem solution field";
    field_metadata.display_name = "Problem solution field";
    field_metadata.variant_name =
      "Poisson solution. " + dimensions(dim, spacedim);
    field_metadata.description =
      "Return the semantic solution field registered by a Problem handle.";
    coral::NodeObject::register_function(
      std::function<ImmersX::FieldId(const ProblemHandleType &)>(
        [](const ProblemHandleType &handle) {
          return handle.fields().solution;
        }),
      {"handle"},
      field_metadata);

    coral::NodeObject::register_method<Problem, void, const Vector &>(
      &Problem::set_solution,
      {poisson_name<dim, spacedim>() + "::set_solution",
       "problem",
       "solution"});
  }

  inline void
  register_linear_execution_types()
  {
    constexpr int spacedim  = 2;
    using Adapter           = LinearAdapterFor<2, spacedim>;
    using AdapterParameters = ImmersX::LinearAdapterParameters;
    using FieldVector       = ImmersXLA::MPI::Vector;
    using GlobalVector      = ImmersXLA::MPI::BlockVector;
    using AdapterHandle     = std::shared_ptr<Adapter>;
    using BulkField =
      ImmersX::Field<2, spacedim, dealii::FEValuesExtractors::Scalar>;
    using LineField =
      ImmersX::Field<1, spacedim, dealii::FEValuesExtractors::Scalar>;
    using Constraint =
      std::decay_t<decltype(ImmersX::make_continuity_constraint(
        std::declval<const BulkField &>(),
        std::declval<const LineField &>(),
        std::declval<const LineField &>()))>;
    using ConstraintFields = ImmersX::ConstraintFields;
    using ConstraintHandle = ImmersX::ProblemHandle<Adapter, ConstraintFields>;
    using View             = ImmersX::FiniteElementSpaceView<1, spacedim>;

    coral::detail::set_type_alias<Adapter>("ImmersX::LinearAdapter<2>");
    coral::detail::set_type_alias<AdapterHandle>("ImmersX::LinearExecution");
    coral::detail::set_type_alias<FieldVector>("ImmersX::LinearFieldVector");
    coral::detail::set_type_alias<GlobalVector>("ImmersX::LinearState");
    coral::detail::set_type_alias<Constraint>(
      "ImmersX::ContinuityConstraint<2>");
    coral::detail::set_type_alias<ConstraintHandle>(
      "ImmersX::LinearConstraintHandle");
    coral::NodeObject::register_output_type<AdapterHandle>();
    coral::NodeObject::register_output_type<FieldVector>();
    coral::NodeObject::register_output_type<GlobalVector>();
    coral::NodeObject::register_output_type<Constraint>();
    coral::NodeObject::register_output_type<ConstraintHandle>();

    register_parameter_type<AdapterParameters>(
      "ImmersX::LinearAdapterParameters");

    coral::RegistryMetadata adapter_metadata;
    adapter_metadata.operation    = "Create linear execution";
    adapter_metadata.display_name = "Linear execution";
    adapter_metadata.variant_name = "LinearAdapter";
    adapter_metadata.description =
      "Create the generic execution adapter for a composed linear system.";
    coral::NodeObject::register_function(
      std::function<AdapterHandle(const AdapterParameters &)>(
        [](const AdapterParameters &parameters) {
          return std::make_shared<Adapter>(parameters, MPI_COMM_WORLD);
        }),
      {"parameters"},
      adapter_metadata);

    coral::RegistryMetadata constraint_metadata;
    constraint_metadata.operation    = "Continuity constraint";
    constraint_metadata.display_name = "Continuity constraint";
    constraint_metadata.variant_name = "Scalar fields";
    constraint_metadata.description =
      "Build a generic Lagrange-multiplier continuity Constraint.";
    coral::NodeObject::register_function(
      std::function<
        Constraint(const BulkField &, const LineField &, const LineField &)>(
        [](const BulkField &bulk,
           const LineField &line,
           const LineField &multiplier) {
          return ImmersX::make_continuity_constraint(bulk, line, multiplier);
        }),
      {"left_field", "right_field", "multiplier_field"},
      constraint_metadata);

    coral::RegistryMetadata constraint_add_metadata;
    constraint_add_metadata.operation    = "Add constraint to linear execution";
    constraint_add_metadata.display_name = "Add constraint";
    constraint_add_metadata.variant_name = "Lagrange-multiplier Constraint";
    constraint_add_metadata.description =
      "Add an Interaction or Constraint to a LinearAdapter.";
    coral::NodeObject::register_function(
      std::function<ConstraintHandle(
        AdapterHandle &, const Constraint &, const std::string &)>(
        [](AdapterHandle     &adapter,
           const Constraint  &constraint,
           const std::string &prefix) {
          return (*adapter).add(constraint, prefix);
        }),
      {"adapter", "constraint", "prefix"},
      constraint_add_metadata);

    coral::RegistryMetadata multiplier_metadata;
    multiplier_metadata.operation    = "Constraint multiplier field";
    multiplier_metadata.display_name = "Constraint multiplier field";
    multiplier_metadata.variant_name = "Lagrange multiplier";
    multiplier_metadata.description =
      "Return the semantic multiplier field registered by a Constraint.";
    coral::NodeObject::register_function(
      std::function<ImmersX::FieldId(const ConstraintHandle &)>(
        [](const ConstraintHandle &handle) {
          return handle.fields().multiplier;
        }),
      {"handle"},
      multiplier_metadata);

    coral::RegistryMetadata state_metadata;
    state_metadata.operation    = "Create linear state";
    state_metadata.display_name = "Linear state";
    state_metadata.variant_name = "Global block vector";
    state_metadata.description =
      "Allocate the execution state owned by a LinearAdapter.";
    coral::NodeObject::register_function(
      std::function<GlobalVector(const AdapterHandle &)>(
        [](const AdapterHandle &adapter) { return adapter->make_state(); }),
      {"adapter"},
      state_metadata);

    coral::RegistryMetadata field_value_metadata;
    field_value_metadata.operation    = "Linear state field";
    field_value_metadata.display_name = "Linear state field";
    field_value_metadata.variant_name = "Distributed field vector";
    field_value_metadata.description =
      "Extract one semantic field vector from an execution state.";
    coral::NodeObject::register_function(
      std::function<FieldVector(
        const AdapterHandle &, const GlobalVector &, const ImmersX::FieldId &)>(
        [](const AdapterHandle    &adapter,
           const GlobalVector     &state,
           const ImmersX::FieldId &field) {
          return adapter->field(state, field);
        }),
      {"adapter", "state", "field"},
      field_value_metadata);

    coral::NodeObject::register_function(
      std::function<void(const AdapterHandle &, GlobalVector &)>(
        [](const AdapterHandle &adapter, GlobalVector &state) {
          adapter->solve(state);
        }),
      {"adapter", "state"},
      coral::RegistryMetadata{"Solve linear state",
                              "Solve",
                              "LinearAdapter",
                              "Solve the composed linear system."});

    coral::NodeObject::register_function(
      std::function<
        void(const AdapterHandle &, const GlobalVector &, GlobalVector &)>(
        [](const AdapterHandle &adapter,
           const GlobalVector  &state,
           GlobalVector        &residual) {
          adapter->evaluate_residual(state, residual);
        }),
      {"adapter", "state", "residual"},
      coral::RegistryMetadata{"Evaluate linear residual",
                              "Evaluate residual",
                              "LinearAdapter",
                              "Evaluate the residual of a composed state."});

    coral::NodeObject::register_function(
      std::function<double(const GlobalVector &)>(
        [](const GlobalVector &vector) { return vector.l2_norm(); }),
      {"vector"},
      coral::RegistryMetadata{"Linear state norm",
                              "State norm",
                              "Global block vector",
                              "Compute a distributed execution-state norm."});

    coral::NodeObject::register_function(
      std::function<void(const double &, const double &)>(&assert_finite_below),
      {"value", "limit"},
      coral::RegistryMetadata{
        "Assert finite below",
        "Assert finite below",
        "Scalar diagnostic",
        "Require a finite scalar below a prescribed limit."});

    coral::NodeObject::register_function(
      std::function<void(const View &,
                         const FieldVector &,
                         const std::string &,
                         const std::string &)>(
        &write_scalar_field<1, spacedim>),
      {"space", "field", "output_file", "field_name"},
      coral::RegistryMetadata{"Write scalar field",
                              "Write scalar field",
                              "Finite element space",
                              "Write a scalar field and its PVD record."});
  }

  template <int dim, int spacedim>
  void
  write_elastic_static_output(
    ImmersX::ElasticStaticProblem<dim, spacedim> &problem)
  {
    problem.output_results(0);
  }

  template <int dim, int spacedim>
  bool
  elastodynamics_state_is_finite(
    const ImmersX::ElastodynamicsSolver<dim, spacedim> &problem)
  {
    return problem.state_is_finite();
  }

  template <int dim, int spacedim>
  double
  elastodynamics_current_time(
    const ImmersX::ElastodynamicsSolver<dim, spacedim> &problem)
  {
    return problem.current_time();
  }

  template <int dim, int spacedim>
  void
  write_elastodynamics_output(
    ImmersX::ElastodynamicsSolver<dim, spacedim> &problem)
  {
    problem.output_results();
  }

  inline double
  coupled_poisson_elasticity_residual_norm(
    const CoupledPoissonElasticity3D &workflow)
  {
    return workflow.residual_norm();
  }

  inline double
  coupled_poisson_elasticity_pressure_scale_error(
    const CoupledPoissonElasticity3D &workflow)
  {
    return workflow.pressure_scale_error();
  }

  inline double
  coupled_poisson_elasticity_traction_balance_error(
    const CoupledPoissonElasticity3D &workflow)
  {
    return workflow.traction_balance_error();
  }

  inline void
  register_coupled_poisson_elasticity_types()
  {
    using Workflow         = CoupledPoissonElasticity3D;
    const std::string name = "ImmersX::CoupledPoissonElasticity<3>";
    coral::detail::set_type_alias<Workflow>(name);
    coral::NodeObject::register_type<Workflow, const std::string &>(
      "parameter_file");
    coral::NodeObject::register_method<Workflow, void>(&Workflow::run,
                                                       {name + "::run",
                                                        "workflow"});
    coral::NodeObject::register_function(
      std::function<double(const Workflow &)>(
        &coupled_poisson_elasticity_residual_norm),
      {name + "::residual_norm", "workflow", "residual"});
    coral::NodeObject::register_function(
      std::function<double(const Workflow &)>(
        &coupled_poisson_elasticity_pressure_scale_error),
      {name + "::pressure_scale_error", "workflow", "diagnostic"});
    coral::NodeObject::register_function(
      std::function<double(const Workflow &)>(
        &coupled_poisson_elasticity_traction_balance_error),
      {name + "::traction_balance_error", "workflow", "diagnostic"});
  }

  inline bool
  reduced_poisson_state_is_finite(const ReducedPoisson3D &workflow)
  {
    return workflow.state_is_finite();
  }

  inline void
  register_reduced_poisson_types()
  {
    using Workflow         = ReducedPoisson3D;
    const std::string name = "ImmersX::ReducedPoissonWorkflow<3>";
    coral::detail::set_type_alias<Workflow>(name);
    coral::NodeObject::register_type<Workflow, const std::string &>(
      "parameter_file");
    coral::NodeObject::register_method<Workflow, void>(&Workflow::run,
                                                       {name + "::run",
                                                        "workflow"});
    coral::NodeObject::register_function(
      std::function<bool(const Workflow &)>(&reduced_poisson_state_is_finite),
      {name + "::state_is_finite", "workflow", "is_finite"});
    coral::NodeObject::register_function(
      std::function<unsigned int(const Workflow &)>(&Workflow::n_reduced_dofs),
      {name + "::n_reduced_dofs", "workflow", "diagnostic"});
    coral::NodeObject::register_function(
      std::function<double(const Workflow &)>(
        &Workflow::coupling_matrix_frobenius_norm),
      {name + "::coupling_matrix_frobenius_norm", "workflow", "diagnostic"});
    coral::NodeObject::register_function(
      std::function<double(const Workflow &)>(&Workflow::bulk_solution_l2_norm),
      {name + "::bulk_solution_l2_norm", "workflow", "norm"});
    coral::NodeObject::register_function(
      std::function<double(const Workflow &)>(
        &Workflow::multiplier_solution_l2_norm),
      {name + "::multiplier_solution_l2_norm", "workflow", "norm"});
  }

#ifdef DEAL_II_WITH_SUNDIALS
  inline bool
  ida_elastodynamics_state_is_finite(const IDAElastodynamics2D &workflow)
  {
    return workflow.state_is_finite();
  }

  inline double
  ida_elastodynamics_current_time(const IDAElastodynamics2D &workflow)
  {
    return workflow.current_time();
  }

  inline void
  register_ida_elastodynamics_types()
  {
    using Workflow         = IDAElastodynamics2D;
    const std::string name = "ImmersX::IDAElastodynamics<2,2>";
    coral::detail::set_type_alias<Workflow>(name);
    coral::NodeObject::register_type<Workflow, const std::string &>(
      "parameter_file");
    coral::NodeObject::register_method<Workflow, void>(&Workflow::run,
                                                       {name + "::run",
                                                        "workflow"});
    coral::NodeObject::register_function(std::function<bool(const Workflow &)>(
                                           &ida_elastodynamics_state_is_finite),
                                         {name + "::state_is_finite",
                                          "workflow",
                                          "is_finite"});
    coral::NodeObject::register_function(
      std::function<double(const Workflow &)>(&ida_elastodynamics_current_time),
      {name + "::current_time", "workflow", "time"});
  }
#endif

  template <int dim>
  std::string
  fiber_reinforced_name()
  {
    return "ImmersX::FiberReinforcedElastodynamics<" + std::to_string(dim) +
           ">";
  }

  template <int dim>
  bool
  fiber_reinforced_state_is_finite(
    const ImmersX::FiberReinforcedElastodynamics<dim> &workflow)
  {
    return workflow.matrix_problem().state_is_finite() &&
           workflow.fiber_problem().state_is_finite();
  }

  template <int dim>
  double
  fiber_reinforced_matrix_velocity_residual(
    const ImmersX::FiberReinforcedElastodynamics<dim> &workflow)
  {
    return workflow.residuals().matrix_velocity;
  }

  template <int dim>
  double
  fiber_reinforced_fiber_velocity_residual(
    const ImmersX::FiberReinforcedElastodynamics<dim> &workflow)
  {
    return workflow.residuals().fiber_velocity;
  }

  template <int dim>
  double
  fiber_reinforced_constraint_residual(
    const ImmersX::FiberReinforcedElastodynamics<dim> &workflow)
  {
    return workflow.residuals().velocity_constraint;
  }

  template <int dim>
  double
  fiber_reinforced_displacement_residual(
    const ImmersX::FiberReinforcedElastodynamics<dim> &workflow)
  {
    return workflow.residuals().displacement_compatibility;
  }

  template <int dim>
  void
  register_fiber_reinforced_types()
  {
    using Parameters = ImmersX::FiberReinforcedElastodynamicsParameters<dim>;
    using Workflow   = ImmersX::FiberReinforcedElastodynamics<dim>;
    const auto name  = fiber_reinforced_name<dim>();

    coral::detail::set_type_alias<Workflow>(name);
    register_parameter_type<Parameters>(
      "ImmersX::FiberReinforcedElastodynamicsParameters<" +
      std::to_string(dim) + ">");
    coral::NodeObject::register_type<Workflow, const Parameters &>(
      "parameters");
    coral::NodeObject::register_method<Workflow, void>(&Workflow::run,
                                                       {name + "::run",
                                                        "workflow"});
    coral::NodeObject::register_function(
      std::function<bool(const Workflow &)>(
        &fiber_reinforced_state_is_finite<dim>),
      {name + "::state_is_finite", "workflow", "is_finite"});
    coral::NodeObject::register_function(
      std::function<double(const Workflow &)>(
        &fiber_reinforced_matrix_velocity_residual<dim>),
      {name + "::matrix_velocity_residual", "workflow", "diagnostic"});
    coral::NodeObject::register_function(
      std::function<double(const Workflow &)>(
        &fiber_reinforced_fiber_velocity_residual<dim>),
      {name + "::fiber_velocity_residual", "workflow", "diagnostic"});
    coral::NodeObject::register_function(
      std::function<double(const Workflow &)>(
        &fiber_reinforced_constraint_residual<dim>),
      {name + "::velocity_constraint_residual", "workflow", "diagnostic"});
    coral::NodeObject::register_function(
      std::function<double(const Workflow &)>(
        &fiber_reinforced_displacement_residual<dim>),
      {name + "::displacement_compatibility", "workflow", "diagnostic"});
  }

  template <int dim>
  void
  register_fiber_reinforced_composition_types()
  {
    using Graph       = ImmersX::Coral::FiberReinforcedElastodynamicsGraph<dim>;
    using Matrix      = ImmersX::Coral::FiberMatrixProblem<dim>;
    using Fiber       = ImmersX::Coral::FiberEmbeddedProblem<dim>;
    using Interaction = ImmersX::Coral::FiberVelocityContinuity<dim>;
    using Adapter     = ImmersX::Coral::FiberExecutionAdapter<dim>;

    const auto graph_name = "ImmersX::FiberReinforcedElastodynamicsGraph<" +
                            std::to_string(dim) + ">";
    const auto matrix_name =
      "ImmersX::FiberMatrixProblem<" + std::to_string(dim) + ">";
    const auto fiber_name =
      "ImmersX::FiberEmbeddedProblem<" + std::to_string(dim) + ">";
    const auto interaction_name =
      "ImmersX::FiberVelocityContinuity<" + std::to_string(dim) + ">";
    const auto adapter_name =
      "ImmersX::FiberExecutionAdapter<" + std::to_string(dim) + ">";

    coral::detail::set_type_alias<Graph>(graph_name);
    coral::detail::set_type_alias<Matrix>(matrix_name);
    coral::detail::set_type_alias<Fiber>(fiber_name);
    coral::detail::set_type_alias<Interaction>(interaction_name);
    coral::detail::set_type_alias<Adapter>(adapter_name);

    coral::NodeObject::register_type<Graph, const std::string &>(
      "parameter_file");
    coral::NodeObject::register_type<Matrix>();
    coral::NodeObject::register_type<Fiber>();
    coral::NodeObject::register_type<Interaction>();
    coral::NodeObject::register_type<Adapter>();

    coral::NodeObject::register_function(
      std::function<Matrix(const Graph &)>(
        [](const Graph &graph) { return graph.matrix_problem(); }),
      {graph_name + "::matrix_problem", "problem", "graph"});
    coral::NodeObject::register_function(
      std::function<Fiber(const Graph &)>(
        [](const Graph &graph) { return graph.embedded_problem(); }),
      {graph_name + "::embedded_problem", "problem", "graph"});
    coral::NodeObject::register_function(
      std::function<Interaction(const Graph &)>(
        [](const Graph &graph) { return graph.velocity_continuity(); }),
      {graph_name + "::velocity_continuity", "interaction", "graph"});
    coral::NodeObject::register_function(
      std::function<Adapter(const Graph &)>(
        [](const Graph &graph) { return graph.execution_adapter(); }),
      {graph_name + "::execution_adapter", "adapter", "graph"});

    coral::NodeObject::register_method<Matrix, void>(&Matrix::prepare,
                                                     {matrix_name + "::prepare",
                                                      "problem"});
    coral::NodeObject::register_method<Fiber, void>(&Fiber::prepare,
                                                    {fiber_name + "::prepare",
                                                     "problem"});
    coral::NodeObject::register_method<Interaction, void>(
      &Interaction::prepare,
      {interaction_name + "::prepare", "interaction", "matrix", "fiber"});
    coral::NodeObject::register_method<Adapter, void>(
      &Adapter::run,
      {adapter_name + "::run", "adapter", "matrix", "fiber", "interaction"});

    coral::NodeObject::register_function(
      std::function<bool(const Adapter &)>(
        [](const Adapter &adapter) { return adapter.state_is_finite(); }),
      {adapter_name + "::state_is_finite", "adapter", "is_finite"});
    coral::NodeObject::register_function(
      std::function<double(const Adapter &)>([](const Adapter &adapter) {
        return adapter.matrix_velocity_residual();
      }),
      {adapter_name + "::matrix_velocity_residual", "adapter", "diagnostic"});
    coral::NodeObject::register_function(
      std::function<double(const Adapter &)>([](const Adapter &adapter) {
        return adapter.fiber_velocity_residual();
      }),
      {adapter_name + "::fiber_velocity_residual", "adapter", "diagnostic"});
    coral::NodeObject::register_function(
      std::function<double(const Adapter &)>([](const Adapter &adapter) {
        return adapter.velocity_constraint_residual();
      }),
      {adapter_name + "::velocity_constraint_residual",
       "adapter",
       "diagnostic"});
    coral::NodeObject::register_function(
      std::function<double(const Adapter &)>([](const Adapter &adapter) {
        return adapter.displacement_compatibility();
      }),
      {adapter_name + "::displacement_compatibility", "adapter", "diagnostic"});
  }

  template <int dim, int spacedim>
  void
  register_poisson_types()
  {
    using Parameters = ImmersX::PoissonParameters<dim, spacedim>;
    using Problem    = ImmersX::PoissonSolver<dim, spacedim>;

    coral::detail::set_type_alias<Problem>(poisson_name<dim, spacedim>());
    register_parameter_type<Parameters>(
      poisson_parameters_name<dim, spacedim>());

    coral::NodeObject::register_type<Problem, const Parameters &>("parameters");

    const auto name = poisson_name<dim, spacedim>();
    using Space     = ImmersX::FiniteElementSpaceView<dim, spacedim>;
    register_finite_element_space_type<dim, spacedim>();
    coral::NodeObject::register_function(
      std::function<Space(const Problem &)>([](const Problem &problem) {
        return ImmersX::finite_element_space_view(problem);
      }),
      {"problem"},
      finite_element_space_metadata("Poisson", dim, spacedim));
    coral::NodeObject::register_method<Problem, void>(&Problem::make_grid,
                                                      {name + "::make_grid",
                                                       "problem"});
    coral::NodeObject::register_method<Problem, void>(&Problem::setup_fe,
                                                      {name + "::setup_fe",
                                                       "problem"});
    coral::NodeObject::register_method<Problem, void>(&Problem::setup_system,
                                                      {name + "::setup_system",
                                                       "problem"});
    coral::NodeObject::register_method<Problem, void>(
      &Problem::assemble_system, {name + "::assemble_system", "problem"});
    coral::NodeObject::register_method<Problem, void>(&Problem::solve,
                                                      {name + "::solve",
                                                       "problem"});
    coral::NodeObject::register_method<Problem, void>(
      &Problem::output_results, {name + "::output_results", "problem"});
    coral::NodeObject::register_function(
      std::function<double(const Problem &)>(
        &poisson_solution_l2_norm<dim, spacedim>),
      {name + "::solution_l2_norm", "problem", "norm"});
    coral::NodeObject::register_function(
      std::function<bool(const Problem &)>(
        &poisson_solution_is_finite<dim, spacedim>),
      {name + "::solution_is_finite", "problem", "is_finite"});
  }

  template <int dim, int spacedim>
  void
  register_elastic_static_types()
  {
    using Parameters = ImmersX::ElasticStaticParameters<dim, spacedim>;
    using Problem    = ImmersX::ElasticStaticProblem<dim, spacedim>;
    const auto name =
      "ImmersX::ElasticStatic<" + dimensions(dim, spacedim) + ">";

    coral::detail::set_type_alias<Problem>(name);
    register_parameter_type<Parameters>("ImmersX::ElasticStaticParameters<" +
                                        dimensions(dim, spacedim) + ">");
    coral::NodeObject::register_type<Problem, const Parameters &>("parameters");
    using Space = ImmersX::FiniteElementSpaceView<dim, spacedim>;
    register_finite_element_space_type<dim, spacedim>();
    coral::NodeObject::register_function(
      std::function<Space(const Problem &)>([](const Problem &problem) {
        return ImmersX::finite_element_space_view(problem);
      }),
      {"problem"},
      finite_element_space_metadata("Elastic static", dim, spacedim));
    coral::NodeObject::register_method<Problem, void>(&Problem::setup,
                                                      {name + "::setup",
                                                       "problem"});
    coral::NodeObject::register_method<Problem, void>(&Problem::solve,
                                                      {name + "::solve",
                                                       "problem"});
    coral::NodeObject::register_method<Problem, void>(&Problem::run,
                                                      {name + "::run",
                                                       "problem"});
    coral::NodeObject::register_function(
      std::function<void(Problem &)>(
        &write_elastic_static_output<dim, spacedim>),
      {name + "::output_results", "problem"});
  }

  template <int dim, int spacedim>
  void
  register_elastodynamics_types()
  {
    using Parameters = ImmersX::ElastodynamicsParameters<dim, spacedim>;
    using Problem    = ImmersX::ElastodynamicsSolver<dim, spacedim>;
    const auto name =
      "ImmersX::Elastodynamics<" + dimensions(dim, spacedim) + ">";

    coral::detail::set_type_alias<Problem>(name);
    register_parameter_type<Parameters>("ImmersX::ElastodynamicsParameters<" +
                                        dimensions(dim, spacedim) + ">");
    coral::NodeObject::register_type<Problem, const Parameters &>("parameters");
    using Space = ImmersX::FiniteElementSpaceView<dim, spacedim>;
    register_finite_element_space_type<dim, spacedim>();
    coral::NodeObject::register_function(
      std::function<Space(const Problem &)>([](const Problem &problem) {
        return ImmersX::finite_element_space_view(problem);
      }),
      {"problem"},
      finite_element_space_metadata("Elastodynamics", dim, spacedim));
    coral::NodeObject::register_method<Problem, void>(&Problem::make_grid,
                                                      {name + "::make_grid",
                                                       "problem"});
    coral::NodeObject::register_method<Problem, void>(&Problem::setup_fe,
                                                      {name + "::setup_fe",
                                                       "problem"});
    coral::NodeObject::register_method<Problem, void>(&Problem::setup_system,
                                                      {name + "::setup_system",
                                                       "problem"});
    coral::NodeObject::register_method<Problem, void>(
      &Problem::assemble_operators, {name + "::assemble_operators", "problem"});
    coral::NodeObject::register_method<Problem, void>(
      &Problem::set_initial_conditions,
      {name + "::set_initial_conditions", "problem"});
    coral::NodeObject::register_method<Problem, void>(
      &Problem::advance_one_timestep,
      {name + "::advance_one_timestep", "problem"});
    coral::NodeObject::register_method<Problem, void>(&Problem::solve,
                                                      {name + "::solve",
                                                       "problem"});
    coral::NodeObject::register_method<Problem, void>(&Problem::run,
                                                      {name + "::run",
                                                       "problem"});
    coral::NodeObject::register_function(
      std::function<void(Problem &)>(
        &write_elastodynamics_output<dim, spacedim>),
      {name + "::output_results", "problem"});
    coral::NodeObject::register_function(
      std::function<bool(const Problem &)>(
        &elastodynamics_state_is_finite<dim, spacedim>),
      {name + "::state_is_finite", "problem", "is_finite"});
    coral::NodeObject::register_function(
      std::function<double(const Problem &)>(
        &elastodynamics_current_time<dim, spacedim>),
      {name + "::current_time", "problem", "time"});
  }

  template <int spacedim>
  void
  register_immersx_types()
  {
    register_common_types();
    register_initialize_parameter_types();
    register_field_types<1, spacedim>();
    register_poisson_types<1, spacedim>();
    register_elastic_static_types<1, spacedim>();
    register_elastodynamics_types<1, spacedim>();
    if constexpr (spacedim >= 2)
      {
        register_field_types<2, spacedim>();
        register_poisson_types<2, spacedim>();
        register_elastic_static_types<2, spacedim>();
        register_elastodynamics_types<2, spacedim>();
        if constexpr (spacedim == 2)
          {
            register_owned_finite_element_space_types<1, spacedim>();
            register_linear_problem_operations<1, spacedim>();
            register_linear_problem_operations<2, spacedim>();
            register_linear_execution_types();
            register_fiber_reinforced_types<2>();
            register_fiber_reinforced_composition_types<2>();
#ifdef DEAL_II_WITH_SUNDIALS
            register_ida_elastodynamics_types();
#endif
          }
      }
    if constexpr (spacedim >= 3)
      {
        register_field_types<3, spacedim>();
        register_poisson_types<3, spacedim>();
        register_elastic_static_types<3, spacedim>();
        register_elastodynamics_types<3, spacedim>();
        if constexpr (spacedim == 3)
          {
            register_coupled_poisson_elasticity_types();
            register_fiber_reinforced_types<3>();
            register_fiber_reinforced_composition_types<3>();
            register_reduced_poisson_types();
          }
      }
  }

  template <int spacedim>
  struct PluginState
  {
    static inline std::unique_ptr<dealii::Utilities::MPI::MPI_InitFinalize>
      mpi_session;
  };

  template <int spacedim>
  inline std::string
  plugin_name()
  {
    return "immersx-coral-" + std::to_string(spacedim) + "d";
  }

  template <int spacedim>
  int
  load_plugin(const char *subjson, const CoralLogger *logger)
  {
    coral_active_logger      = logger;
    coral_active_plugin_name = coral_plugin_name();

    unsigned int max_num_threads = dealii::numbers::invalid_unsigned_int;
    std::vector<std::string> args;

    if (subjson != nullptr)
      {
        try
          {
            const auto init = json::parse(subjson);
            if (init.contains("MPI"))
              {
                const auto &mpi = init.at("MPI");
                max_num_threads = mpi.value("max_num_threads", max_num_threads);
                args            = mpi.value("args", args);
              }
          }
        catch (const std::exception &exception)
          {
            coral_log_error("Invalid ImmersX plugin initialization JSON: %s",
                            exception.what());
            return 1;
          }
      }

    // ImmersX uses MPI-backed deal.II data structures even for a one-rank
    // local run.  The platform-level MPI setting controls whether the host
    // is launched with multiple ranks, not whether the MPI runtime exists.
    // Initialize it here when the host has not already done so, before Coral
    // creates its worker threads or constructs any workflow node.
    if (!dealii::Utilities::MPI::job_supports_mpi())
      {
        std::vector<char *> argv;
        argv.reserve(args.size());
        for (auto &argument : args)
          argv.push_back(argument.data());
        int    argc     = static_cast<int>(argv.size());
        char **argv_ptr = argv.data();
        PluginState<spacedim>::mpi_session =
          std::make_unique<dealii::Utilities::MPI::MPI_InitFinalize>(
            argc, argv_ptr, max_num_threads);
      }

    register_immersx_types<spacedim>();
    return 0;
  }

  template <int spacedim>
  void
  unload_plugin()
  {
    PluginState<spacedim>::mpi_session.reset();
  }
} // namespace ImmersX::Coral

#endif // immersx_coral_register_h
