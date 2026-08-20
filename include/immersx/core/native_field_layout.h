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

#include <immersx/core/residual.h>
#include <immersx/core/state.h>

#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <vector>

namespace ImmersX
{
  /**
   * Local mapping between semantic fields and native block-vector blocks.
   *
   * Native block numbers are deliberately local to this object. A Problem
   * can therefore keep its native BlockMatrix while the rest of the model
   * addresses rows and columns only through FieldId. The StateLayout is a
   * non-owning reference and must outlive this layout.
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
        residual.field(fields_[block]).add(1., native.block(block));
    }

  private:
    const StateLayout   &layout_;
    std::vector<FieldId> fields_;
  };

  /**
   * Adapter-local mapping between semantic fields and distributed blocks.
   *
   * A block is the storage selected by one execution adapter.  It is not a
   * semantic field number and it is not required to agree with a Problem's
   * native block ordering.  In particular, the field vectors are the actual
   * vectors stored inside @p GlobalBlockVectorType: binding does not gather
   * entries into temporary vectors.
   */
  template <typename FieldVectorType, typename GlobalBlockVectorType>
  class BlockFieldLayout
  {
  public:
    explicit BlockFieldLayout(const StateLayout &layout)
      : layout_(layout)
      , blocks_by_field_(layout.n_fields())
    {}

    /** Associate one semantic field with one execution block. */
    void
    add_field(const FieldId field, const unsigned int block)
    {
      AssertThrow(layout_.contains(field),
                  dealii::ExcMessage(
                    "Block field is not in the state layout."));
      AssertThrow(!blocks_by_field_[field.value()].has_value(),
                  dealii::ExcMessage("Block field was registered twice."));

      if (block >= fields_by_block_.size())
        fields_by_block_.resize(block + 1);
      AssertThrow(!fields_by_block_[block].has_value(),
                  dealii::ExcMessage("Execution block was registered twice."));

      blocks_by_field_[field.value()] = block;
      fields_by_block_[block]         = field;
      block_sizes_.resize(fields_by_block_.size());
      block_owned_.resize(fields_by_block_.size());
    }

    /** Return the execution block assigned to a semantic field. */
    unsigned int
    block(const FieldId field) const
    {
      validate_field(field);
      AssertThrow(blocks_by_field_[field.value()].has_value(),
                  dealii::ExcMessage("Field has no execution block."));
      return *blocks_by_field_[field.value()];
    }

    /** Return the semantic field assigned to an execution block. */
    FieldId
    field(const unsigned int block) const
    {
      AssertThrow(block < fields_by_block_.size() &&
                    fields_by_block_[block].has_value(),
                  dealii::ExcMessage("Execution block has no field."));
      return *fields_by_block_[block];
    }

    unsigned int
    n_blocks() const
    {
      return static_cast<unsigned int>(fields_by_block_.size());
    }

    const StateLayout &
    state_layout() const
    {
      return layout_;
    }

    /**
     * Register the global size and locally owned indices of one block.
     *
     * This information is only needed for the flattened IDA differential
     * mask.  The binding operations below use the block vectors themselves.
     */
    void
    set_block_distribution(const unsigned int      block,
                           const std::size_t       global_size,
                           const dealii::IndexSet &locally_owned)
    {
      AssertThrow(
        block < fields_by_block_.size() && fields_by_block_[block].has_value(),
        dealii::ExcMessage("Cannot set distribution for an unknown block."));
      AssertThrow(locally_owned.size() == global_size,
                  dealii::ExcMessage("Block IndexSet has the wrong size."));
      block_sizes_[block] = global_size;
      block_owned_[block] = locally_owned;
    }

    /** Bind the actual global blocks as semantic state fields. */
    void
    bind_state(StateView<FieldVectorType>  &view,
               const GlobalBlockVectorType &global) const
    {
      validate_complete();
      AssertThrow(global.n_blocks() == fields_by_block_.size(),
                  dealii::ExcMessage(
                    "Global block vector does not match its field layout."));
      for (unsigned int block = 0; block < fields_by_block_.size(); ++block)
        view.bind(*fields_by_block_[block], global.block(block));
    }

    /** Bind the actual global blocks as additive semantic residual rows. */
    void
    bind_residual(ResidualAccumulator<FieldVectorType> &accumulator,
                  GlobalBlockVectorType                &global) const
    {
      validate_complete();
      AssertThrow(global.n_blocks() == fields_by_block_.size(),
                  dealii::ExcMessage(
                    "Global block vector does not match its field layout."));
      for (unsigned int block = 0; block < fields_by_block_.size(); ++block)
        accumulator.bind(*fields_by_block_[block], global.block(block));
    }

    /** Return the flattened locally owned differential IDA components. */
    dealii::IndexSet
    differential_components() const
    {
      validate_complete();
      std::size_t total_size = 0;
      for (const auto size : block_sizes_)
        {
          AssertThrow(size.has_value(),
                      dealii::ExcMessage(
                        "All block distributions are required for an IDA "
                        "differential mask."));
          total_size += *size;
        }

      dealii::IndexSet result(total_size);
      std::size_t      offset = 0;
      for (unsigned int block = 0; block < fields_by_block_.size(); ++block)
        {
          AssertThrow(block_owned_[block].has_value(),
                      dealii::ExcMessage(
                        "Locally owned indices are missing for an IDA "
                        "block."));
          if (layout_.field(*fields_by_block_[block]).time_role ==
              TimeRole::differential)
            for (const auto index : *block_owned_[block])
              result.add_index(offset + index);
          offset += *block_sizes_[block];
        }
      result.compress();
      return result;
    }

  private:
    void
    validate_field(const FieldId field) const
    {
      AssertThrow(layout_.contains(field),
                  dealii::ExcMessage(
                    "Field identifier is not in the state layout."));
    }

    void
    validate_complete() const
    {
      for (const auto &block : blocks_by_field_)
        AssertThrow(block.has_value(),
                    dealii::ExcMessage(
                      "Every semantic field must have an execution block."));
      for (const auto &field : fields_by_block_)
        AssertThrow(field.has_value(),
                    dealii::ExcMessage("Every execution block must have a "
                                       "semantic field."));
    }

    const StateLayout                           &layout_;
    std::vector<std::optional<unsigned int>>     blocks_by_field_;
    std::vector<std::optional<FieldId>>          fields_by_block_;
    std::vector<std::optional<std::size_t>>      block_sizes_;
    std::vector<std::optional<dealii::IndexSet>> block_owned_;
  };

  namespace detail
  {
    /**
     * Local adapter for a monolithic vector such as an IDA vector.
     *
     * The index sets are adapter data, not semantic identities. The time role
     * and field ownership remain in StateLayout and FieldDescriptor. This is
     * an implementation detail of solver adapters, not an application-level
     * state container.
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
                    dealii::ExcMessage(
                      "Monolithic field was registered twice."));
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
      extract(const VectorType        &monolithic,
              std::vector<VectorType> &fields) const
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
  } // namespace detail
} // namespace ImmersX

#endif // immersx_native_field_layout_h
