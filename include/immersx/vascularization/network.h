// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_vascularization_network_h
#define immersx_vascularization_network_h

#include <deal.II/base/point.h>
#include <deal.II/base/types.h>

#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace ImmersX::Vascularization
{
  /** Persistent physical identifiers, independent of cell and DoF numbering. */
  using Id                       = std::uint64_t;
  inline constexpr Id invalid_id = std::numeric_limits<Id>::max();

  enum class System : unsigned int
  {
    arterial,
    venous
  };

  enum class Provenance : unsigned int
  {
    imported,
    generated
  };

  enum class Port : unsigned int
  {
    none,
    arterial_inlet,
    venous_outlet,
    arterial_outlet
  };

  struct Node
  {
    dealii::Point<3>            position;
    dealii::types::subdomain_id owner      = 0;
    Port                        port       = Port::none;
    Provenance                  provenance = Provenance::imported;
  };

  /**
   * A physical straight vessel, with linearly interpolated radius in metres.
   * Endpoints follow growth from the root towards tissue. Reference flow is
   * along this orientation for arteries and against it for veins. Numerical
   * subdivision never creates another Vessel or changes these endpoints.
   */
  struct Vessel
  {
    std::array<Id, 2>     nodes{{invalid_id, invalid_id}};
    std::array<double, 2> radii{{0., 0.}};
    double                viscosity  = std::numeric_limits<double>::quiet_NaN();
    System                system     = System::arterial;
    Provenance            provenance = Provenance::imported;
    dealii::types::subdomain_id owner = 0;
    /** Additional input physical properties, with caller-defined SI names. */
    std::map<std::string, double> physical_properties;
  };

  struct Terminal
  {
    Id                          node      = invalid_id;
    Id                          territory = invalid_id;
    System                      system    = System::arterial;
    dealii::types::subdomain_id owner     = 0;
    /** Relative tissue demand; normalization belongs to the reference solve. */
    double demand = 1.;
  };

  /**
   * Locally relevant physical data. Map keys are persistent IDs in separate
   * node, vessel and terminal namespaces. An entry is owned iff its owner is
   * the local rank; other entries are ghosts. A locally relevant vessel must
   * carry both endpoint nodes. Remote topology need not be stored here.
   */
  struct NetworkData
  {
    std::map<Id, Node>     nodes;
    std::map<Id, Vessel>   vessels;
    std::map<Id, Terminal> terminals;
  };

  struct ValidationIssue
  {
    std::string entity;
    Id          id = invalid_id;
    std::string message;
  };

  /** Local data checks only; geometry and collective checks are separate. */
  struct ValidationReport
  {
    std::vector<ValidationIssue> issues;

    bool
    valid() const;
  };

  ValidationReport
  validate_data(const NetworkData &network, unsigned int communicator_size);

  /** Physical centreline length in metres. Missing endpoints are an error. */
  double
  length(const NetworkData &network, Id vessel);

  /** Additive lumen volume in m^3 on the reference interval [begin,end]. */
  double
  volume(const NetworkData &network,
         Id                 vessel,
         double             begin = 0.,
         double             end   = 1.);

  /** Poiseuille resistance in Pa s/m^3, integrated over a tapered vessel. */
  double
  resistance(const NetworkData &network,
             Id                 vessel,
             double             begin = 0.,
             double             end   = 1.);

  /** Generalized Murray parent radius; all children and gamma must be positive.
   */
  double
  murray_radius(const std::vector<double> &children, double gamma);

  /** A numerical cell refers to its physical vessel and reference interval. */
  struct VesselCell
  {
    Id     vessel = invalid_id;
    double begin  = 0.;
    double end    = 1.;
  };

  /**
   * Partition one physical vessel at strictly increasing reference coordinates
   * including exactly 0 and 1. Does not change the physical graph or radii.
   */
  std::vector<VesselCell>
  subdivide(const NetworkData         &network,
            Id                         vessel,
            const std::vector<double> &coordinates);
} // namespace ImmersX::Vascularization

#endif
