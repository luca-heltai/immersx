// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// The ImmersX application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception (the "License"); either version 3.0 of the
// License, or (at your option) any later version. The full text of the
// license can be found in the LICENSE.md file at the top level of the
// ImmersX distribution.
//
// ---------------------------------------------------------------------

#ifndef immersx_field_h
#define immersx_field_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>

#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ImmersX
{
  /** Whether a field has a differential or algebraic time role. */
  enum class TimeRole
  {
    differential,
    algebraic
  };

  /** A stable semantic identifier assigned by StateLayout. */
  class FieldId
  {
  public:
    inline static constexpr std::size_t invalid_value =
      std::numeric_limits<std::size_t>::max();

    constexpr FieldId() = default;

    explicit constexpr FieldId(const std::size_t value)
      : value_(value)
    {}

    constexpr std::size_t
    value() const
    {
      return value_;
    }

    constexpr bool
    is_valid() const
    {
      return value_ != invalid_value;
    }

    friend constexpr bool
    operator==(const FieldId left, const FieldId right)
    {
      return left.value_ == right.value_;
    }

    friend constexpr bool
    operator!=(const FieldId left, const FieldId right)
    {
      return !(left == right);
    }

    friend constexpr bool
    operator<(const FieldId left, const FieldId right)
    {
      return left.value_ < right.value_;
    }

  private:
    std::size_t value_ = invalid_value;
  };

  /** Stable identifier for a group of fields sharing one history timeline. */
  class HistoryGroupId
  {
  public:
    inline static constexpr std::size_t invalid_value =
      std::numeric_limits<std::size_t>::max();

    constexpr HistoryGroupId() = default;

    explicit constexpr HistoryGroupId(const std::size_t value)
      : value_(value)
    {}

    constexpr std::size_t
    value() const
    {
      return value_;
    }

    constexpr bool
    is_valid() const
    {
      return value_ != invalid_value;
    }

    friend constexpr bool
    operator==(const HistoryGroupId left, const HistoryGroupId right)
    {
      return left.value_ == right.value_;
    }

    friend constexpr bool
    operator!=(const HistoryGroupId left, const HistoryGroupId right)
    {
      return !(left == right);
    }

    friend constexpr bool
    operator<(const HistoryGroupId left, const HistoryGroupId right)
    {
      return left.value_ < right.value_;
    }

  private:
    std::size_t value_ = invalid_value;
  };

  /** Semantic metadata and algebraic ownership information for a field. */
  struct FieldDescriptor
  {
    std::string      name;
    TimeRole         time_role = TimeRole::differential;
    HistoryGroupId   history_group;
    dealii::IndexSet locally_owned;
    dealii::IndexSet locally_relevant;
  };

  /** Registry mapping semantic field names to stable field identifiers. */
  class StateLayout
  {
  public:
    /** Add a field and return its stable semantic identifier. */
    FieldId
    add_field(FieldDescriptor descriptor)
    {
      AssertThrow(!descriptor.name.empty(),
                  dealii::ExcMessage("A field must have a name."));
      AssertThrow(!has_field(descriptor.name),
                  dealii::ExcMessage("Duplicate field name '" +
                                     descriptor.name + "'."));

      const FieldId id(fields_.size());
      field_names_.emplace(descriptor.name, id);
      fields_.push_back(std::move(descriptor));
      return id;
    }

    /** Return whether a field with the given semantic name is registered. */
    bool
    has_field(const std::string &name) const
    {
      return field_names_.find(name) != field_names_.end();
    }

    /** Look up a field identifier by semantic name. */
    FieldId
    field_id(const std::string &name) const
    {
      const auto it = field_names_.find(name);
      AssertThrow(it != field_names_.end(),
                  dealii::ExcMessage("Unknown field name '" + name + "'."));
      return it->second;
    }

    /** Return whether an identifier belongs to this layout. */
    bool
    contains(const FieldId id) const
    {
      return id.is_valid() && id.value() < fields_.size();
    }

    /** Return the descriptor for a registered field. */
    const FieldDescriptor &
    field(const FieldId id) const
    {
      AssertThrow(contains(id),
                  dealii::ExcMessage("Field identifier is not in the state "
                                     "layout."));
      return fields_[id.value()];
    }

    /** Return the number of semantic fields in this layout. */
    std::size_t
    n_fields() const
    {
      return fields_.size();
    }

  private:
    std::vector<FieldDescriptor>             fields_;
    std::unordered_map<std::string, FieldId> field_names_;
  };
} // namespace ImmersX

#endif // immersx_field_h
