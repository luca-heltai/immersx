// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_network_serialization_h
#define immersx_internal_network_serialization_h
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>

#include <immersx/vascularization/network.h>

// Wire serialization keeps physical ids as integers. Particle ids used during
// point location are temporary and independent of these namespaces.
namespace boost::serialization
{
  template <class Archive>
  void
  serialize(Archive &ar, ImmersX::Vascularization::Node &n, unsigned int)
  {
    ar &n.position &n.owner &n.port &n.provenance &n.inward_port_normal;
  }
  template <class Archive>
  void
  serialize(Archive &ar, ImmersX::Vascularization::Vessel &v, unsigned int)
  {
    ar &v.nodes &v.radii &v.viscosity &v.system &v.provenance &v.owner &v
      .physical_properties;
  }
  template <class Archive>
  void
  serialize(Archive &ar, ImmersX::Vascularization::Terminal &t, unsigned int)
  {
    ar &t.node &t.territory &t.system &t.owner &t.demand;
  }
  template <class Archive>
  void
  serialize(Archive &ar, ImmersX::Vascularization::NetworkData &n, unsigned int)
  {
    ar &n.nodes &n.vessels &n.terminals;
  }
} // namespace boost::serialization

#endif
