// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/utility.hpp>

#include <immersx/vascularization/distribution.h>
#include <immersx/vascularization/io.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>

#include "archive_internal.h"
#include "network_serialization.h"

namespace boost::serialization
{
  template <class Archive>
  void
  serialize(Archive                                  &ar,
            ImmersX::Vascularization::NetworkArchive &a,
            unsigned int)
  {
    ar &a.network &a.territories &a.terminal_pairs;
  }
} // namespace boost::serialization
namespace ImmersX::Vascularization
{
  namespace
  {
    using namespace dealii;
    using Rank   = types::subdomain_id;
    using Counts = std::array<Id, 5>;
    using Metadata =
      std::pair<std::vector<Territory>, std::vector<TerminalPair>>;
    constexpr unsigned int format_version = 2;

    void
    require_all(bool valid, MPI_Comm comm, const std::string &message)
    {
      AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(valid), comm) ==
                    1,
                  ExcMessage(message));
    }
    std::filesystem::path
    common_directory(const std::filesystem::path &directory, MPI_Comm comm)
    {
      bool                  valid = !directory.empty();
      std::filesystem::path path;
      try
        {
          path = std::filesystem::absolute(directory).lexically_normal();
        }
      catch (const std::exception &)
        {
          valid = false;
        }
      for (const auto &other : Utilities::MPI::all_gather(comm, path.string()))
        valid = valid && other == path.string();
      require_all(
        valid,
        comm,
        "All archive participants must use the same nonempty directory.");
      return path;
    }
    bool
    same_node(const Node &a, const Node &b)
    {
      return a.position == b.position && a.owner == b.owner &&
             a.port == b.port && a.provenance == b.provenance &&
             a.inward_port_normal == b.inward_port_normal;
    }
    // Authority is explicit here because old owner numbers are checked before
    // remapping them into a differently sized reader communicator.
    std::map<Id, Rank>
    resolve_nodes(const std::map<Id, Node> &nodes,
                  const std::set<Id>       &owned,
                  MPI_Comm                  comm)
    {
      using Record    = std::pair<Id, std::pair<Node, bool>>;
      const Rank size = Utilities::MPI::n_mpi_processes(comm);
      std::map<Rank, std::vector<Record>> send;
      for (const auto &[id, node] : nodes)
        send[id % size].push_back({id, {node, owned.count(id) != 0}});
      bool                         valid = true;
      std::map<Id, Node>           registered;
      std::map<Id, unsigned int>   authorities;
      std::map<Id, Rank>           owners;
      std::map<Id, std::set<Rank>> readers;
      for (const auto &[source, records] :
           Utilities::MPI::some_to_some(comm, send))
        for (const auto &[id, record] : records)
          {
            const auto [it, inserted] = registered.emplace(id, record.first);
            valid = (inserted || same_node(it->second, record.first)) && valid;
            authorities[id] += record.second;
            readers[id].insert(source);
            if (record.second)
              owners[id] = source;
          }
      for (const auto &[id, node] : registered)
        valid = authorities[id] == 1 && valid;
      require_all(
        valid,
        comm,
        "Archived nodes require one authority and matching endpoint ghosts.");
      std::map<Rank, std::map<Id, Rank>> replies;
      for (const auto &[id, targets] : readers)
        for (Rank target : targets)
          replies[target].emplace(id, owners.at(id));
      std::map<Id, Rank> result;
      for (const auto &[source, entries] :
           Utilities::MPI::some_to_some(comm, replies))
        result.insert(entries.begin(), entries.end());
      return result;
    }
    struct Tissue
    {
      Id       id = invalid_id;
      Terminal terminal;
      Point<3> position;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &id &terminal &position;
      }
    };
  } // namespace

  void
  internal::validate_archive_data(const NetworkData               &n,
                                  const std::vector<Territory>    &territories,
                                  const std::vector<TerminalPair> &pairs,
                                  MPI_Comm                         comm)
  {
    const Rank rank = Utilities::MPI::this_mpi_process(comm),
               size = Utilities::MPI::n_mpi_processes(comm);
    require_all(validate_data(n, size).valid(),
                comm,
                "Cannot archive invalid physical vascular data.");
    std::set<Id> owned_nodes;
    for (const auto &[id, node] : n.nodes)
      if (node.owner == rank)
        owned_nodes.insert(id);
    resolve_nodes(n.nodes, owned_nodes, comm);
    std::map<Rank, std::vector<std::pair<unsigned int, Id>>> identities;
    for (const auto &[id, v] : n.vessels)
      if (v.owner == rank)
        identities[id % size].emplace_back(0, id);
    std::set<Id>                        tissue_nodes;
    bool                                valid = true;
    std::map<Rank, std::vector<Tissue>> tissue_send;
    for (const auto &[id, t] : n.terminals)
      if (t.owner == rank)
        {
          valid = n.nodes.at(t.node).owner == rank &&
                  tissue_nodes.insert(t.node).second && valid;
          identities[id % size].emplace_back(1, id);
          if (t.territory != invalid_id)
            tissue_send[t.territory % size].push_back(
              {id, t, n.nodes.at(t.node).position});
        }
    std::set<std::pair<unsigned int, Id>> registered_ids;
    for (const auto &[source, ids] :
         Utilities::MPI::some_to_some(comm, identities))
      for (const auto &id : ids)
        valid = registered_ids.insert(id).second && valid;
    require_all(
      valid,
      comm,
      "Archive vessel/terminal ids must be unique, with one native terminal per tissue node.");
    using End = std::pair<System, bool>;
    std::map<Rank, std::vector<std::pair<Id, End>>>    incidence_send;
    std::map<Rank, std::vector<std::pair<Id, System>>> leaf_send;
    for (const auto &[id, v] : n.vessels)
      if (v.owner == rank)
        for (unsigned int end = 0; end < 2; ++end)
          incidence_send[v.nodes[end] % size].push_back(
            {v.nodes[end], {v.system, end == 1}});
    for (const auto &[id, t] : n.terminals)
      if (t.owner == rank)
        leaf_send[t.node % size].emplace_back(t.node, t.system);
    std::map<Id, std::vector<End>> incidence;
    for (const auto &[source, entries] :
         Utilities::MPI::some_to_some(comm, incidence_send))
      for (const auto &[node, end] : entries)
        incidence[node].push_back(end);
    for (const auto &[source, entries] :
         Utilities::MPI::some_to_some(comm, leaf_send))
      for (const auto &[node, system] : entries)
        {
          const auto &ends = incidence[node];
          valid            = ends.size() == 1 && ends[0].first == system &&
                  ends[0].second && valid;
        }
    require_all(
      valid,
      comm,
      "Archived terminals must end exactly one outward vessel of their declared system.");
    std::map<Rank, Metadata> metadata_send;
    for (const auto &t : territories)
      {
        valid = t.id != invalid_id && t.owner == rank &&
                std::isfinite(t.demand) && t.demand > 0. && valid;
        for (unsigned int d = 0; d < 3; ++d)
          valid = std::isfinite(t.position[d]) && valid;
        metadata_send[t.id % size].first.push_back(t);
      }
    for (const auto &p : pairs)
      {
        valid = p.territory != invalid_id && p.arterial != p.venous &&
                std::isfinite(p.resistance) && p.resistance > 0. && valid;
        metadata_send[p.territory % size].second.push_back(p);
      }
    std::map<Id, Territory>    registered;
    std::map<Id, TerminalPair> beds;
    for (const auto &[source, metadata] :
         Utilities::MPI::some_to_some(comm, metadata_send))
      {
        for (const auto &t : metadata.first)
          valid = registered.emplace(t.id, t).second && valid;
        for (const auto &p : metadata.second)
          valid = beds.emplace(p.territory, p).second && valid;
      }
    std::map<Id, std::vector<Tissue>> tissue;
    for (const auto &[source, records] :
         Utilities::MPI::some_to_some(comm, tissue_send))
      for (const auto &t : records)
        if (registered.count(t.terminal.territory))
          tissue[t.terminal.territory].push_back(t);
    for (const auto &[id, territory] : registered)
      {
        const Tissue *a = nullptr, *v = nullptr;
        for (const auto &t : tissue[id])
          {
            if (t.terminal.system == System::arterial)
              {
                valid = a == nullptr && valid;
                a     = &t;
              }
            else if (t.terminal.system == System::venous)
              {
                valid = v == nullptr && valid;
                v     = &t;
              }
          }
        if (!a)
          {
            valid = false;
            continue;
          }
        valid = a->terminal.owner == territory.owner &&
                a->terminal.demand == territory.demand &&
                a->position == territory.position && valid;
        if (v || beds.count(id))
          {
            if (!v || !beds.count(id))
              {
                valid = false;
                continue;
              }
            const auto &bed = beds.at(id);
            valid           = tissue[id].size() == 2 && a->id == bed.arterial &&
                    v->id == bed.venous &&
                    a->terminal.demand == v->terminal.demand &&
                    a->terminal.node != v->terminal.node &&
                    a->position != v->position && valid;
          }
      }
    for (const auto &[id, pair] : beds)
      valid = registered.count(id) && valid;
    require_all(
      valid,
      comm,
      "Archive territories and beds must be unique, positive and match their arterial/venous terminal identities, demand and writer.");
  }
  namespace
  {
    constexpr std::size_t records_per_part = 1024;
    Counts
    counts(const NetworkArchive &a, const std::set<Id> &owned_nodes)
    {
      return {{owned_nodes.size(),
               a.network.vessels.size(),
               a.network.terminals.size(),
               a.territories.size(),
               a.terminal_pairs.size()}};
    }
    std::filesystem::path
    part_path(const std::filesystem::path &path, Rank writer, unsigned int part)
    {
      return path / ("part-" + std::to_string(writer) + "-" +
                     std::to_string(part) + ".txt");
    }
  } // namespace

  void
  save_network(const std::filesystem::path     &directory,
               const NetworkData               &network,
               const std::vector<Territory>    &territories,
               const std::vector<TerminalPair> &pairs,
               MPI_Comm                         comm)
  {
    const auto path = common_directory(directory, comm);
    const Rank rank = dealii::Utilities::MPI::this_mpi_process(comm),
               size = dealii::Utilities::MPI::n_mpi_processes(comm);
    internal::validate_archive_data(network, territories, pairs, comm);
    bool valid = true;
    try
      {
        valid = !std::filesystem::exists(path);
      }
    catch (const std::exception &)
      {
        valid = false;
      }
    require_all(valid,
                comm,
                "Archive output directory must not already exist.");
    if (rank == 0)
      try
        {
          std::filesystem::create_directories(path.parent_path());
          valid = std::filesystem::create_directory(path);
        }
      catch (const std::exception &)
        {
          valid = false;
        }
    require_all(valid,
                comm,
                "Archive output directory must be new and writable.");
    std::vector<Id> node_ids, vessel_ids, terminal_ids;
    for (const auto &[id, node] : network.nodes)
      if (node.owner == rank)
        node_ids.push_back(id);
    for (const auto &[id, v] : network.vessels)
      if (v.owner == rank)
        vessel_ids.push_back(id);
    for (const auto &[id, t] : network.terminals)
      if (t.owner == rank)
        terminal_ids.push_back(id);
    const auto maximum = std::max({node_ids.size(),
                                   vessel_ids.size(),
                                   terminal_ids.size(),
                                   territories.size(),
                                   pairs.size()});
    const auto n_parts =
      std::max(std::size_t{1},
               maximum / records_per_part +
                 static_cast<std::size_t>(maximum % records_per_part != 0));
    require_all(n_parts <= std::numeric_limits<unsigned int>::max(),
                comm,
                "Too many vascular archive parts.");
    std::vector<Counts> local_table;
    for (unsigned int part = 0; part < n_parts; ++part)
      try
        {
          NetworkArchive chunk;
          std::set<Id>   owned_nodes;
          const auto     begin = part * records_per_part;
          for (std::size_t i = begin;
               i < std::min(node_ids.size(), begin + records_per_part);
               ++i)
            {
              const Id id = node_ids[i];
              chunk.network.nodes.emplace(id, network.nodes.at(id));
              owned_nodes.insert(id);
            }
          for (std::size_t i = begin;
               i < std::min(vessel_ids.size(), begin + records_per_part);
               ++i)
            {
              const auto &v = network.vessels.at(vessel_ids[i]);
              chunk.network.vessels.emplace(vessel_ids[i], v);
              for (Id id : v.nodes)
                chunk.network.nodes.emplace(id, network.nodes.at(id));
            }
          for (std::size_t i = begin;
               i < std::min(terminal_ids.size(), begin + records_per_part);
               ++i)
            {
              const auto &t = network.terminals.at(terminal_ids[i]);
              chunk.network.terminals.emplace(terminal_ids[i], t);
              chunk.network.nodes.emplace(t.node, network.nodes.at(t.node));
            }
          for (std::size_t i = begin;
               i < std::min(territories.size(), begin + records_per_part);
               ++i)
            chunk.territories.push_back(territories[i]);
          for (std::size_t i = begin;
               i < std::min(pairs.size(), begin + records_per_part);
               ++i)
            chunk.terminal_pairs.push_back(pairs[i]);
          const auto header = counts(chunk, owned_nodes);
          local_table.push_back(header);
          std::ofstream out;
          out.exceptions(std::ios::failbit | std::ios::badbit);
          out.open(part_path(path, rank, part));
          {
            boost::archive::text_oarchive ar(out);
            ar << std::string("ImmersX vascular part") << format_version << rank
               << size << part << header << owned_nodes << chunk;
          }
          out.close();
        }
      catch (const std::exception &)
        {
          valid = false;
        }
    require_all(
      valid,
      comm,
      "A vascular archive part could not be written; no manifest was published.");
    const auto table = dealii::Utilities::MPI::all_gather(comm, local_table);
    if (rank == 0)
      try
        {
          std::ofstream out;
          out.exceptions(std::ios::failbit | std::ios::badbit);
          out.open(path / "manifest.tmp");
          {
            boost::archive::text_oarchive ar(out);
            ar << std::string("ImmersX vascular network") << format_version
               << table;
          }
          out.close();
          std::filesystem::rename(path / "manifest.tmp", path / "manifest");
        }
      catch (const std::exception &)
        {
          valid = false;
        }
    require_all(
      valid,
      comm,
      "Vascular archive parts were written, but the manifest could not be published.");
  }

  NetworkArchive
  load_network(const std::filesystem::path    &directory,
               const dealii::Triangulation<3> &background,
               const dealii::Mapping<3>       &mapping,
               double                          halo,
               MPI_Comm                        comm)
  {
    using namespace dealii;
    const auto path = common_directory(directory, comm);
    const Rank rank = Utilities::MPI::this_mpi_process(comm),
               size = Utilities::MPI::n_mpi_processes(comm);
    std::vector<std::vector<Counts>> table;
    bool                             valid = true;
    if (rank == 0)
      try
        {
          std::ifstream                 in(path / "manifest");
          boost::archive::text_iarchive ar(in);
          std::string                   magic;
          unsigned int                  version;
          ar >> magic >> version;
          AssertThrow(
            magic == "ImmersX vascular network" && version == format_version,
            ExcMessage("Unsupported vascular archive manifest version."));
          ar >> table;
          valid = magic == "ImmersX vascular network" &&
                  version == format_version && !table.empty() &&
                  table.size() <= std::numeric_limits<Rank>::max();
          for (const auto &parts : table)
            valid = !parts.empty() &&
                    parts.size() <= std::numeric_limits<unsigned int>::max() &&
                    valid;
        }
      catch (const std::exception &)
        {
          valid = false;
        }
    require_all(
      valid,
      comm,
      "Missing, incomplete or unsupported vascular archive manifest.");
    table = Utilities::MPI::broadcast(comm, table, 0);
    const Rank                                 old_size = table.size();
    std::vector<std::pair<Rank, unsigned int>> files;
    for (Rank writer = 0; writer < old_size; ++writer)
      for (unsigned int part = 0; part < table[writer].size(); ++part)
        files.emplace_back(writer, part);
    NetworkArchive result;
    std::set<Id>   authoritative_nodes;
    for (std::size_t file = rank; file < files.size(); file += size)
      try
        {
          const auto [expected_writer, expected_part] = files[file];
          std::ifstream  in(part_path(path, expected_writer, expected_part));
          NetworkArchive chunk;
          std::set<Id>   owned_nodes;
          std::string    magic;
          unsigned int   version, part;
          Rank           writer, writer_size;
          Counts         header;
          {
            boost::archive::text_iarchive ar(in);
            ar >> magic >> version;
            AssertThrow(
              magic == "ImmersX vascular part" && version == format_version,
              ExcMessage("Unsupported vascular archive part version."));
            ar >> writer >> writer_size >> part >> header >> owned_nodes >>
              chunk;
          }
          valid = magic == "ImmersX vascular part" &&
                  version == format_version && writer == expected_writer &&
                  part == expected_part && writer_size == old_size &&
                  header == table[expected_writer][expected_part] &&
                  counts(chunk, owned_nodes) == header &&
                  validate_data(chunk.network, old_size).valid() && valid;
          for (Id count : header)
            valid = count <= records_per_part && valid;
          in >> std::ws;
          valid = in.eof() && valid;
          for (Id id : owned_nodes)
            valid = chunk.network.nodes.count(id) &&
                    chunk.network.nodes.at(id).owner == writer &&
                    authoritative_nodes.insert(id).second && valid;
          for (const auto &[id, node] : chunk.network.nodes)
            {
              const auto [it, inserted] =
                result.network.nodes.emplace(id, node);
              valid = (inserted || same_node(it->second, node)) && valid;
            }
          for (auto &[id, v] : chunk.network.vessels)
            {
              valid   = v.owner == writer && valid;
              v.owner = rank;
              valid = result.network.vessels.emplace(id, std::move(v)).second &&
                      valid;
            }
          for (auto &[id, t] : chunk.network.terminals)
            {
              valid   = t.owner == writer && valid;
              t.owner = rank;
              valid =
                result.network.terminals.emplace(id, std::move(t)).second &&
                valid;
            }
          for (auto &t : chunk.territories)
            {
              valid   = t.owner == writer && valid;
              t.owner = rank;
              result.territories.push_back(std::move(t));
            }
          result.terminal_pairs.insert(result.terminal_pairs.end(),
                                       chunk.terminal_pairs.begin(),
                                       chunk.terminal_pairs.end());
        }
      catch (const std::exception &)
        {
          valid = false;
        }
    require_all(
      valid,
      comm,
      "A vascular archive part is missing, truncated, inconsistent or invalid.");
    const auto node_owners =
      resolve_nodes(result.network.nodes, authoritative_nodes, comm);
    for (auto &[id, node] : result.network.nodes)
      node.owner = node_owners.at(id);
    result.network =
      redistribute(background, mapping, result.network, halo, comm);

    // Route physical metadata to the restored arterial point owner. Reference
    // beds may connect remote endpoints; they do not require graph replication.
    std::map<Rank, std::vector<std::pair<Id, Rank>>> arterial_send;
    for (const auto &[id, t] : result.network.terminals)
      if (t.owner == rank && t.system == System::arterial &&
          t.territory != invalid_id)
        arterial_send[t.territory % size].emplace_back(t.territory, rank);
    std::map<Id, std::vector<Rank>> writers;
    for (const auto &[source, entries] :
         Utilities::MPI::some_to_some(comm, arterial_send))
      for (const auto &[id, owner] : entries)
        writers[id].push_back(owner);
    std::map<Rank, Metadata> metadata_send;
    for (const auto &t : result.territories)
      metadata_send[t.id % size].first.push_back(t);
    for (const auto &p : result.terminal_pairs)
      metadata_send[p.territory % size].second.push_back(p);
    std::map<Rank, Metadata> routes;
    for (const auto &[source, metadata] :
         Utilities::MPI::some_to_some(comm, metadata_send))
      {
        for (auto t : metadata.first)
          if (writers[t.id].size() != 1)
            valid = false;
          else
            {
              t.owner = writers.at(t.id)[0];
              routes[t.owner].first.push_back(t);
            }
        for (const auto &p : metadata.second)
          if (writers[p.territory].size() != 1)
            valid = false;
          else
            routes[writers.at(p.territory)[0]].second.push_back(p);
      }
    require_all(
      valid,
      comm,
      "Restored territory metadata has no unique arterial point owner.");
    result.territories.clear();
    result.terminal_pairs.clear();
    for (const auto &[source, metadata] :
         Utilities::MPI::some_to_some(comm, routes))
      {
        result.territories.insert(result.territories.end(),
                                  metadata.first.begin(),
                                  metadata.first.end());
        result.terminal_pairs.insert(result.terminal_pairs.end(),
                                     metadata.second.begin(),
                                     metadata.second.end());
      }
    internal::validate_archive_data(result.network,
                                    result.territories,
                                    result.terminal_pairs,
                                    comm);
    std::sort(result.territories.begin(),
              result.territories.end(),
              [](const auto &a, const auto &b) { return a.id < b.id; });
    std::sort(result.terminal_pairs.begin(),
              result.terminal_pairs.end(),
              [](const auto &a, const auto &b) {
                return a.territory < b.territory;
              });
    return result;
  }
} // namespace ImmersX::Vascularization
