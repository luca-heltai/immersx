// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_native_field_layout_h
#define immersx_native_field_layout_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>

#include <cstddef>
#include <map>
#include <optional>
#include <vector>

#include "residual.h"
#include "state.h"

namespace ImmersX
{
  /**
   * Local mapping between semantic fields and native block-vector blocks.
   *
   * Native block numbers are deliberately local to this object. A Problem
   * can therefore keep its native BlockMatrix while the rest of the model
   * addresses rows and columns only through FieldId.
   */
  class NativeFieldLayout
  {
  public:
    explicit NativeFieldLayout(const StateLayout &layout)
      : layout_(layout)
    {}

    /** Append a native block and return its local block number. */
    unsigned int
    add_block(const FieldId field)
    {
      AssertThrow(layout_.contains(field),
                  dealii::ExcMessage(
                    "Native block field is not in the state layout."));
      const unsigned int block = static_cast<unsigned int>(fields_.size());
      fields_.push_back(field);
      return block;
    }

    unsigned int
    n_blocks() const
    {
      return static_cast<unsigned int>(fields_.size());
    }

    FieldId
    field(const unsigned int native_block) const
    {
      AssertThrow(native_block < fields_.size(),
                  dealii::ExcMessage("Unknown native field block."));
      return fields_[native_block];
    }

    /** Gather semantic fields into a native deal.II-style BlockVector. */
    template <typename VectorType, typename NativeBlockVectorType>
    void
    gather(const StateAccessor<VectorType> &state,
           const double                     time,
           NativeBlockVectorType           &native) const
    {
      native.reinit(fields_.size());
      for (unsigned int block = 0; block < fields_.size(); ++block)
        {
          const auto &values = state.field(fields_[block], time);
          native.block(block).reinit(values);
          native.block(block) = values;
        }
      native.collect_sizes();
    }

    /** Scatter native block residual rows additively into semantic rows. */
    template <typename VectorType, typename NativeBlockVectorType>
    void
    scatter_add(const NativeBlockVectorType     &native,
                ResidualAccumulator<VectorType> &residual) const
    {
      AssertThrow(native.n_blocks() == fields_.size(),
                  dealii::ExcMessage(
                    "Native vector does not match its field layout."));
      for (unsigned int block = 0; block < fields_.size(); ++block)
        residual.field(fields_[block]).add(native.block(block));
    }

  private:
    const StateLayout   &layout_;
    std::vector<FieldId> fields_;
  };

  /**
   * Local adapter for a monolithic vector such as an IDA vector.
   *
   * The index sets are adapter data, not semantic identities. The time role
   * and field ownership remain in StateLayout and FieldDescriptor.
   */
  template <typename VectorType>
  class MonolithicFieldLayout
  {
  public:
    MonolithicFieldLayout(const StateLayout &layout,
                          const std::size_t  global_size)
      : layout_(layout)
      , global_size_(global_size)
      , fields_(layout.n_fields())
    {}

    void
    add_field(const FieldId field, const dealii::IndexSet &indices)
    {
      AssertThrow(layout_.contains(field),
                  dealii::ExcMessage(
                    "Monolithic field is not in the state layout."));
      AssertThrow(indices.size() == global_size_,
                  dealii::ExcMessage(
                    "Monolithic field has the wrong global size."));
      AssertThrow(!fields_[field.value()].has_value(),
                  dealii::ExcMessage("Monolithic field was registered twice."));
      fields_[field.value()] = indices;
    }

    void
    add_field(const FieldId     field,
              const std::size_t begin,
              const std::size_t end)
    {
      dealii::IndexSet indices(global_size_);
      indices.add_range(begin, end);
      add_field(field, indices);
    }

    bool
    has_field(const FieldId field) const
    {
      return layout_.contains(field) && fields_[field.value()].has_value();
    }

    const StateLayout &
    state_layout() const
    {
      return layout_;
    }

    /** Bind extracted field vectors to a semantic StateView. */
    void
    bind(StateView<VectorType>         &view,
         const std::vector<VectorType> &fields) const
    {
      AssertThrow(fields.size() == fields_.size(),
                  dealii::ExcMessage("Field vector count does not match the "
                                     "monolithic field layout."));
      for (std::size_t field = 0; field < fields_.size(); ++field)
        if (fields_[field].has_value())
          view.bind(FieldId(field), fields[field]);
    }

    /** Extract registered monolithic fields into independently owned vectors.
     */
    void
    extract(const VectorType &monolithic, std::vector<VectorType> &fields) const
    {
      AssertThrow(monolithic.size() == global_size_,
                  dealii::ExcMessage(
                    "Monolithic vector has the wrong global size."));
      fields.assign(layout_.n_fields(), VectorType());
      for (std::size_t field = 0; field < fields_.size(); ++field)
        if (fields_[field].has_value())
          {
            const auto &indices = *fields_[field];
            fields[field].reinit(indices.n_elements());
            for (std::size_t local = 0; local < indices.n_elements(); ++local)
              fields[field][local] =
                monolithic[indices.nth_index_in_set(local)];
          }
    }

    /** Add independently stored field rows into a monolithic vector. */
    void
    scatter_add(const std::vector<VectorType> &fields,
                VectorType                    &monolithic) const
    {
      AssertThrow(monolithic.size() == global_size_,
                  dealii::ExcMessage(
                    "Monolithic vector has the wrong global size."));
      AssertThrow(fields.size() == fields_.size(),
                  dealii::ExcMessage("Field vector count does not match the "
                                     "monolithic field layout."));
      for (std::size_t field = 0; field < fields_.size(); ++field)
        if (fields_[field].has_value())
          {
            const auto &indices = *fields_[field];
            AssertThrow(fields[field].size() == indices.n_elements(),
                        dealii::ExcMessage(
                          "Field vector does not match its IndexSet."));
            for (std::size_t local = 0; local < indices.n_elements(); ++local)
              monolithic[indices.nth_index_in_set(local)] +=
                fields[field][local];
          }
    }

  private:
    const StateLayout                           &layout_;
    std::size_t                                  global_size_;
    std::vector<std::optional<dealii::IndexSet>> fields_;
  };
} // namespace ImmersX

#endif // immersx_native_field_layout_h
