// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_detail_second_order_condensation_h
#define immersx_detail_second_order_condensation_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector_operation.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

namespace ImmersX
{
  namespace detail
  {
    /**
     * Eliminate the kinematic rows of second-order pairs from a materialized
     * Jacobian before a sparse direct factorization.
     *
     * A pair with kinematic residual M u' - M v linearizes to
     * [rho M_f, -M_f] on free displacement rows, where rho is the derivative
     * weight, and to a nonzero diagonal D on constrained rows c, so that
     * u_c = r_c / D. M is the mass or a diagonal approximation of it. If M_f
     * reaches a constrained column, the velocity row c must be a unit row,
     * which fixes v_c from the right-hand side. With Mhat = [M_ff, M_fc; 0, I],
     * the kinematic rows give z = Mhat^{-1} w, where w_f = r_f / rho and w_c =
     * u_c - v_c / rho (w_f = -r_f and w_c = v_c when rho = 0). After z_c is
     * replaced by u_c:
     *
     * - rho > 0: u = z + P_f v / rho, and each velocity column i of the
     *   reduced matrix is J_Rv(:, i) + J_Ru(:, i) / rho for free i;
     * - rho = 0: v_f = z_f and u_c = z_c, and the velocity slot i carries u_i
     *   for free i, v_i otherwise.
     *
     * The reduced matrix S keeps all other blocks and has one spatial block
     * per pair instead of two coupled ones. Mhat does not depend on rho, so
     * callers can keep its factorization across Jacobians.
     *
     * build() verifies this structure on the matrix values and returns
     * nullptr when any pair does not have it.
     */
    template <typename MatrixType, typename FieldVectorType>
    class SecondOrderCondensation
    {
    public:
      using size_type = dealii::types::global_dof_index;

      struct Pair
      {
        unsigned int displacement;
        unsigned int velocity;
      };

      /** Locally owned rows of a distributed matrix in compressed form. */
      struct CompressedRows
      {
        std::vector<size_type>   rows;
        std::vector<std::size_t> offsets{0};
        std::vector<size_type>   columns;
        std::vector<double>      values;

        /** Append one row; duplicate columns are summed. */
        void
        add_row(const size_type                            row,
                std::vector<std::pair<size_type, double>> &entries)
        {
          std::sort(entries.begin(), entries.end());
          rows.push_back(row);
          for (std::size_t k = 0; k < entries.size(); ++k)
            if (k > 0 && entries[k].first == entries[k - 1].first)
              values.back() += entries[k].second;
            else
              {
                columns.push_back(entries[k].first);
                values.push_back(entries[k].second);
              }
          offsets.push_back(columns.size());
        }

        bool
        operator==(const CompressedRows &other) const
        {
          return rows == other.rows && offsets == other.offsets &&
                 columns == other.columns && values == other.values;
        }

        std::shared_ptr<MatrixType>
        make_matrix(const dealii::IndexSet &partition,
                    const MPI_Comm          communicator) const
        {
          dealii::DynamicSparsityPattern sparsity(partition.size(),
                                                  partition.size(),
                                                  partition);
          for (std::size_t row = 0; row < rows.size(); ++row)
            sparsity.add_entries(rows[row],
                                 columns.begin() + offsets[row],
                                 columns.begin() + offsets[row + 1],
                                 true);
          auto result = std::make_shared<MatrixType>();
          result->reinit(partition, partition, sparsity, communicator, false);
          for (std::size_t row = 0; row < rows.size(); ++row)
            if (offsets[row + 1] > offsets[row])
              result->set(rows[row],
                          offsets[row + 1] - offsets[row],
                          columns.data() + offsets[row],
                          values.data() + offsets[row],
                          false);
          result->compress(dealii::VectorOperation::insert);
          return result;
        }
      };

      /**
       * Return the condensation of @p jacobian, or nullptr when a pair does
       * not have the kinematic structure. All ranks return the same result.
       */
      template <typename BlockMatrixType>
      static std::shared_ptr<const SecondOrderCondensation>
      build(const BlockMatrixType               &jacobian,
            const std::vector<Pair>             &pairs,
            const std::vector<dealii::IndexSet> &partitions,
            const MPI_Comm                       communicator)
      {
        if (pairs.empty())
          return nullptr;
        const auto all_ranks = [communicator](const bool value) {
          return dealii::Utilities::MPI::min(value ? 1 : 0, communicator) == 1;
        };
        auto result           = std::make_shared<SecondOrderCondensation>();
        result->communicator_ = communicator;
        result->partitions_   = partitions;
        const unsigned int n_blocks = partitions.size();
        result->slot_.assign(n_blocks, none);
        for (std::size_t p = 0; p < pairs.size(); ++p)
          {
            AssertThrow(pairs[p].displacement < n_blocks &&
                          pairs[p].velocity < n_blocks &&
                          result->slot_[pairs[p].displacement] == none &&
                          result->slot_[pairs[p].velocity] == none,
                        dealii::ExcInternalError());
            result->slot_[pairs[p].displacement] = p;
            result->slot_[pairs[p].velocity]     = p;
          }

        bool valid = true;
        for (const auto &pair : pairs)
          valid =
            valid && partitions[pair.displacement] == partitions[pair.velocity];
        if (!all_ranks(valid))
          return nullptr;
        // Kinematic rows may only couple the pair itself.
        for (const auto &pair : pairs)
          for (unsigned int block = 0; block < n_blocks; ++block)
            if (block != pair.displacement && block != pair.velocity &&
                jacobian.block(pair.displacement, block).linfty_norm() != 0.)
              return nullptr;

        std::vector<std::pair<size_type, double>> uu, uv;
        for (const auto &pair : pairs)
          {
            PairData data;
            data.pair         = pair;
            const auto &owned = partitions[pair.displacement];
            const auto &mass =
              jacobian.block(pair.displacement, pair.displacement);
            const auto &kinematic =
              jacobian.block(pair.displacement, pair.velocity);
            data.constrained.set_size(owned.size());
            double     rho_min  = std::numeric_limits<double>::max();
            double     rho_max  = std::numeric_limits<double>::lowest();
            const auto diagonal = [](const auto &entries, const size_type row) {
              double value = 0.;
              for (const auto &entry : entries)
                if (entry.first == row)
                  value += entry.second;
              return value;
            };
            const auto check_rows = [&]() {
              for (const auto row : owned)
                {
                  row_entries(mass, row, uu);
                  row_entries(kinematic, row, uv);
                  double uv_max = 0.;
                  for (const auto &entry : uv)
                    uv_max = std::max(uv_max, std::abs(entry.second));
                  if (uv_max == 0.)
                    {
                      // A constrained row: a nonzero diagonal and nothing else.
                      const double d = diagonal(uu, row);
                      if (d == 0.)
                        return false;
                      for (const auto &entry : uu)
                        if (entry.first != row &&
                            std::abs(entry.second) > tolerance * std::abs(d))
                          return false;
                      data.constrained.add_index(row);
                      data.constrained_diagonal.emplace_back(row, d);
                      std::vector<std::pair<size_type, double>> unit{{row, 1.}};
                      data.mass.add_row(row, unit);
                      continue;
                    }
                  // A free row: J_uu = -rho J_uv entry by entry.
                  const double uv_diagonal = diagonal(uv, row);
                  if (uv_diagonal == 0.)
                    return false;
                  const double rho = -diagonal(uu, row) / uv_diagonal;
                  if (!std::isfinite(rho) || rho < 0.)
                    return false;
                  rho_min       = std::min(rho_min, rho);
                  rho_max       = std::max(rho_max, rho);
                  std::size_t i = 0, j = 0;
                  while (i < uu.size() || j < uv.size())
                    {
                      double a = 0., b = 0.;
                      if (j == uv.size() ||
                          (i < uu.size() && uu[i].first < uv[j].first))
                        a = uu[i++].second;
                      else if (i == uu.size() || uv[j].first < uu[i].first)
                        b = uv[j++].second;
                      else
                        {
                          a = uu[i++].second;
                          b = uv[j++].second;
                        }
                      if (std::abs(a + rho * b) >
                          tolerance * std::max(rho, 1.) * uv_max)
                        return false;
                    }
                  for (auto &entry : uv)
                    entry.second = -entry.second;
                  data.mass.add_row(row, uv);
                }
              return true;
            };
            if (!all_ranks(check_rows()))
              return nullptr;
            rho_min = dealii::Utilities::MPI::min(rho_min, communicator);
            rho_max = dealii::Utilities::MPI::max(rho_max, communicator);
            if (!(rho_max >= rho_min) ||
                rho_max - rho_min > tolerance * rho_max)
              return nullptr;
            data.rho = rho_max;
            data.constrained.compress();
            // Constrained velocity rows that are unit rows fix v_c directly.
            for (const auto &[row, d] : data.constrained_diagonal)
              {
                double diagonal = 0.;
                bool   unit     = true;
                for (unsigned int column = 0; column < n_blocks && unit;
                     ++column)
                  {
                    row_entries(jacobian.block(pair.velocity, column), row, uu);
                    for (const auto &entry : uu)
                      if (column == pair.velocity && entry.first == row)
                        diagonal = entry.second;
                      else if (entry.second != 0.)
                        unit = false;
                  }
                (void)d;
                if (unit && diagonal != 0.)
                  data.velocity_diagonal.emplace(row, diagonal);
              }
            result->pairs_.push_back(std::move(data));
          }

        // Offsets of the remaining blocks in the reduced monolithic vector.
        // A pair occupies the slot of its velocity block.
        result->reduced_offsets_.assign(n_blocks,
                                        dealii::numbers::invalid_dof_index);
        size_type reduced_size = 0;
        for (unsigned int block = 0; block < n_blocks; ++block)
          if (!result->is_displacement(block))
            {
              result->reduced_offsets_[block] = reduced_size;
              reduced_size += partitions[block].size();
            }
        result->reduced_partition_.set_size(reduced_size);
        for (unsigned int block = 0; block < n_blocks; ++block)
          if (!result->is_displacement(block))
            result->reduced_partition_.add_indices(
              partitions[block], result->reduced_offsets_[block]);
        result->reduced_partition_.compress();

        // Columns of a pair are folded into its slot once the constrained
        // entries are known on every rank.
        struct PendingRow
        {
          std::vector<std::pair<size_type, double>>                     direct;
          std::vector<std::tuple<std::size_t, bool, size_type, double>> paired;
        };
        std::vector<size_type>        pending_rows;
        std::vector<PendingRow>       pending;
        std::vector<dealii::IndexSet> ghosts;
        for (const auto &data : result->pairs_)
          ghosts.emplace_back(partitions[data.pair.displacement].size());
        for (unsigned int block = 0; block < n_blocks; ++block)
          if (!result->is_displacement(block))
            for (const auto row : partitions[block])
              {
                PendingRow entries;
                for (unsigned int column = 0; column < n_blocks; ++column)
                  {
                    row_entries(jacobian.block(block, column), row, uu);
                    const auto p = result->slot_[column];
                    if (p == none)
                      for (const auto &entry : uu)
                        entries.direct.emplace_back(
                          result->reduced_offsets_[column] + entry.first,
                          entry.second);
                    else
                      for (const auto &entry : uu)
                        {
                          entries.paired.emplace_back(p,
                                                      column ==
                                                        pairs[p].velocity,
                                                      entry.first,
                                                      entry.second);
                          ghosts[p].add_index(entry.first);
                        }
                  }
                pending_rows.push_back(result->reduced_offsets_[block] + row);
                pending.push_back(std::move(entries));
              }
        for (std::size_t p = 0; p < result->pairs_.size(); ++p)
          for (const auto column : result->pairs_[p].mass.columns)
            ghosts[p].add_index(column);

        // Constrained flags, including ghost columns.
        std::vector<dealii::LinearAlgebra::distributed::Vector<double>> flags(
          result->pairs_.size());
        for (std::size_t p = 0; p < result->pairs_.size(); ++p)
          {
            const auto &owned = partitions[result->pairs_[p].pair.displacement];
            ghosts[p].compress();
            auto relevant = ghosts[p];
            relevant.add_indices(owned);
            relevant.compress();
            flags[p].reinit(owned, relevant, communicator);
            // 1: constrained with a unit velocity row, 2: other constrained.
            for (const auto index : result->pairs_[p].constrained)
              flags[p](index) =
                result->pairs_[p].velocity_diagonal.count(index) ? 1. : 2.;
            flags[p].update_ghost_values();
          }

        // Free mass rows may reach a constrained column only if v_c is known
        // from its unit velocity row.
        valid = true;
        for (std::size_t p = 0; p < result->pairs_.size() && valid; ++p)
          {
            const auto &mass = result->pairs_[p].mass;
            for (std::size_t row = 0; row < mass.rows.size() && valid; ++row)
              if (flags[p](mass.rows[row]) == 0.)
                {
                  double row_max = 0.;
                  for (auto k = mass.offsets[row]; k < mass.offsets[row + 1];
                       ++k)
                    row_max = std::max(row_max, std::abs(mass.values[k]));
                  for (auto k = mass.offsets[row]; k < mass.offsets[row + 1];
                       ++k)
                    if (flags[p](mass.columns[k]) == 2. &&
                        std::abs(mass.values[k]) > tolerance * row_max)
                      valid = false;
                }
          }
        if (!all_ranks(valid))
          return nullptr;

        CompressedRows reduced;
        for (std::size_t row = 0; row < pending.size(); ++row)
          {
            auto &entries = pending[row];
            for (const auto &[p, from_velocity, column, value] : entries.paired)
              {
                const auto  &data = result->pairs_[p];
                const bool   free = flags[p](column) == 0.;
                const double coefficient =
                  data.rho > 0. ?
                    (from_velocity ? 1. : (free ? 1. / data.rho : 0.)) :
                    (from_velocity ? (free ? 0. : 1.) : (free ? 1. : 0.));
                if (coefficient != 0.)
                  entries.direct.emplace_back(
                    result->reduced_offsets_[data.pair.velocity] + column,
                    coefficient * value);
              }
            reduced.add_row(pending_rows[row], entries.direct);
          }
        result->reduced_matrix_ =
          reduced.make_matrix(result->reduced_partition_, communicator);

        // Keep the couplings that move the kinematic solution to the
        // right-hand side: J_Ru, and J_Rv when rho = 0.
        for (unsigned int block = 0; block < n_blocks; ++block)
          if (!result->is_displacement(block))
            for (std::size_t p = 0; p < result->pairs_.size(); ++p)
              for (const bool from_velocity : {false, true})
                {
                  if (from_velocity && result->pairs_[p].rho > 0.)
                    continue;
                  const auto &coupling =
                    jacobian.block(block,
                                   from_velocity ? pairs[p].velocity :
                                                   pairs[p].displacement);
                  if (coupling.n_nonzero_elements() == 0 ||
                      coupling.linfty_norm() == 0.)
                    continue;
                  auto copy = std::make_shared<MatrixType>();
                  copy->reinit(coupling);
                  copy->copy_from(coupling);
                  result->couplings_.push_back(
                    {block, p, from_velocity, std::move(copy)});
                }
        return result;
      }

      /** The reduced matrix S. */
      const MatrixType &
      reduced_matrix() const
      {
        return *reduced_matrix_;
      }

      std::size_t
      n_pairs() const
      {
        return pairs_.size();
      }

      /** Mhat for pair @p p; it does not depend on the derivative weight. */
      const CompressedRows &
      kinematic_mass(const std::size_t p) const
      {
        return pairs_[p].mass;
      }

      const dealii::IndexSet &
      kinematic_partition(const std::size_t p) const
      {
        return partitions_[pairs_[p].pair.displacement];
      }

      /** The right-hand side w of Mhat z = w for pair @p p; see the class
       * documentation. Call finish_kinematic_solution() on z afterwards. */
      template <typename GlobalVectorType>
      void
      kinematic_rhs(const GlobalVectorType &rhs,
                    const std::size_t       p,
                    FieldVectorType        &destination) const
      {
        const auto  &data  = pairs_[p];
        const auto  &u     = rhs.block(data.pair.displacement);
        const double scale = data.rho > 0. ? 1. / data.rho : -1.;
        destination.reinit(kinematic_partition(p), communicator_);
        for (const auto index : kinematic_partition(p))
          if (!data.constrained.is_element(index))
            destination(index) = scale * u(index);
        const auto &v = rhs.block(data.pair.velocity);
        for (const auto &[index, diagonal] : data.constrained_diagonal)
          {
            const auto unit = data.velocity_diagonal.find(index);
            if (unit == data.velocity_diagonal.end())
              destination(index) = u(index) / diagonal;
            else if (data.rho > 0.)
              destination(index) =
                u(index) / diagonal - v(index) / unit->second / data.rho;
            else
              destination(index) = v(index) / unit->second;
          }
        destination.compress(dealii::VectorOperation::insert);
      }

      /** Replace the constrained entries of z = Mhat^{-1} w by u_c. */
      template <typename GlobalVectorType>
      void
      finish_kinematic_solution(const GlobalVectorType &rhs,
                                const std::size_t       p,
                                FieldVectorType        &solution) const
      {
        const auto &data = pairs_[p];
        const auto &u    = rhs.block(data.pair.displacement);
        for (const auto &[index, diagonal] : data.constrained_diagonal)
          solution(index) = u(index) / diagonal;
        solution.compress(dealii::VectorOperation::insert);
      }

      /** The reduced right-hand side for the kinematic solutions z. */
      template <typename GlobalVectorType>
      void
      reduced_rhs(const GlobalVectorType             &rhs,
                  const std::vector<FieldVectorType> &kinematic_solutions,
                  FieldVectorType                    &destination) const
      {
        destination.reinit(reduced_partition_, communicator_);
        for (unsigned int block = 0; block < partitions_.size(); ++block)
          if (!is_displacement(block))
            for (const auto index : partitions_[block])
              destination(reduced_offsets_[block] + index) =
                rhs.block(block)(index);
        FieldVectorType known, product;
        for (const auto &coupling : couplings_)
          {
            const auto &data = pairs_[coupling.pair];
            known            = kinematic_solutions[coupling.pair];
            if (data.rho == 0.)
              {
                // u_c and v_f are known; u_f and v_c stay in the slot.
                for (const auto index : kinematic_partition(coupling.pair))
                  if (data.constrained.is_element(index) ==
                      coupling.from_velocity)
                    known(index) = 0.;
                known.compress(dealii::VectorOperation::insert);
              }
            product.reinit(partitions_[coupling.block], communicator_);
            coupling.matrix->vmult(product, known);
            // Keep every write an insertion; Trilinos rejects mixing them.
            for (const auto index : partitions_[coupling.block])
              {
                const auto row = reduced_offsets_[coupling.block] + index;
                destination(row) =
                  static_cast<double>(destination(row)) - product(index);
              }
          }
        destination.compress(dealii::VectorOperation::insert);
      }

      /** Recover all blocks from the reduced solution and z. */
      template <typename GlobalVectorType>
      void
      expand(const FieldVectorType              &reduced_solution,
             const std::vector<FieldVectorType> &kinematic_solutions,
             GlobalVectorType                   &destination) const
      {
        for (unsigned int block = 0; block < partitions_.size(); ++block)
          if (!is_displacement(block))
            for (const auto index : partitions_[block])
              destination.block(block)(index) =
                reduced_solution(reduced_offsets_[block] + index);
        for (std::size_t p = 0; p < pairs_.size(); ++p)
          {
            const auto &data = pairs_[p];
            const auto &z    = kinematic_solutions[p];
            const auto  slot = reduced_offsets_[data.pair.velocity];
            auto       &u    = destination.block(data.pair.displacement);
            auto       &v    = destination.block(data.pair.velocity);
            for (const auto index : kinematic_partition(p))
              {
                const bool   free = !data.constrained.is_element(index);
                const double x    = reduced_solution(slot + index);
                if (data.rho > 0.)
                  u(index) = z(index) + (free ? x / data.rho : 0.);
                else
                  {
                    u(index) = free ? x : z(index);
                    v(index) = free ? z(index) : x;
                  }
              }
          }
        destination.compress(dealii::VectorOperation::insert);
      }

    private:
      static constexpr double      tolerance = 1.e-12;
      static constexpr std::size_t none =
        std::numeric_limits<std::size_t>::max();

      struct PairData
      {
        Pair                                      pair;
        double                                    rho = 0.;
        dealii::IndexSet                          constrained;
        std::vector<std::pair<size_type, double>> constrained_diagonal;
        std::map<size_type, double>               velocity_diagonal;
        CompressedRows                            mass;
      };

      struct Coupling
      {
        unsigned int                block;
        std::size_t                 pair;
        bool                        from_velocity;
        std::shared_ptr<MatrixType> matrix;
      };

      template <typename BlockType>
      static void
      row_entries(const BlockType                           &matrix,
                  const size_type                            row,
                  std::vector<std::pair<size_type, double>> &entries)
      {
        entries.clear();
        if (matrix.row_length(row) != 0)
          for (auto entry = matrix.begin(row), end = matrix.end(row);
               entry != end;
               ++entry)
            entries.emplace_back(entry->column(), entry->value());
        std::sort(entries.begin(), entries.end());
      }

      bool
      is_displacement(const unsigned int block) const
      {
        return slot_[block] != none && pairs_.size() > slot_[block] &&
               pairs_[slot_[block]].pair.displacement == block;
      }

      MPI_Comm                      communicator_ = MPI_COMM_SELF;
      std::vector<dealii::IndexSet> partitions_;
      std::vector<std::size_t>      slot_;
      std::vector<size_type>        reduced_offsets_;
      dealii::IndexSet              reduced_partition_;
      std::vector<PairData>         pairs_;
      std::vector<Coupling>         couplings_;
      std::shared_ptr<MatrixType>   reduced_matrix_;
    };
  } // namespace detail
} // namespace ImmersX

#endif
