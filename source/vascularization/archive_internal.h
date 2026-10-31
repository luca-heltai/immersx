// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_archive_h
#define immersx_internal_archive_h
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

#include <immersx/vascularization/generation.h>

namespace ImmersX::Vascularization::internal
{
  void
  validate_archive_data(const NetworkData                  &network,
                        const std::vector<Territory>       &territories,
                        const std::vector<TerminalPair>    &pairs,
                        MPI_Comm                            communicator,
                        const std::vector<TerminalTriplet> &triplets = {});
  struct TerritoryMetadata
  {
    std::vector<Territory>       territories;
    std::vector<TerminalPair>    pairs;
    std::vector<TerminalTriplet> triplets;
    template <class Archive>
    void
    serialize(Archive &ar, unsigned int)
    {
      ar &territories &pairs &triplets;
    }
  };
  struct RetainedMetadata
  {
    std::vector<Territory>       territories;
    std::vector<TerminalPair>    pairs;
    std::vector<TerminalTriplet> triplets;
    ValidationReport             validation;
  };
  RetainedMetadata
  retain_metadata(const NetworkData                  &before,
                  const std::vector<Territory>       &territories,
                  const std::vector<TerminalPair>    &pairs,
                  const NetworkData                  &after,
                  MPI_Comm                            communicator,
                  const std::vector<TerminalTriplet> &triplets = {});
} // namespace ImmersX::Vascularization::internal
namespace boost::serialization
{
  template <class Archive>
  void
  serialize(Archive &ar, ImmersX::Vascularization::Territory &t, unsigned int)
  {
    ar &t.id &t.position &t.owner &t.demand;
  }
  template <class Archive>
  void
  serialize(Archive                                &ar,
            ImmersX::Vascularization::TerminalPair &p,
            unsigned int)
  {
    ar &p.territory &p.arterial &p.venous &p.resistance;
  }
  template <class Archive>
  void
  serialize(Archive                                   &ar,
            ImmersX::Vascularization::TerminalTriplet &t,
            unsigned int)
  {
    ar &t.territory &t.arterial &t.portal &t.venous &t.resistances;
  }
} // namespace boost::serialization
#endif
