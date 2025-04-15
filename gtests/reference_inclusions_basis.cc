// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the reduced_lagrange_multipliers application, based on
// the deal.II library.
//
// The reduced_lagrange_multipliers application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the file LICENSE.md at the top level
// of the reduced_lagrange_multipliers distribution.
//
// ---------------------------------------------------------------------

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <vector>

#include "inclusions.h"

using namespace dealii;

TEST(TestInclusionsBasis2, CheckIntegrals) // NOLINT
{
  // cx, cy, r
  Inclusions<2> ref;
  ref.set_n_q_points(100);
  ref.set_n_coefficients(3);
  ref.inclusions.push_back({{0, 0, 1.0}});
  ref.initialize();
  double integral[4] = {0, 0, 0, 0};
  for (unsigned int q = 0; q < ref.n_particles(); ++q)
    {
      const auto &fe_values = ref.get_fe_values(q);
      const auto &ds        = ref.get_JxW(q);
      integral[0] += ds;
      integral[1] += fe_values[0] * ds;
      integral[2] += fe_values[1] * ds;
      integral[3] += fe_values[2] * ds;
    }

  ASSERT_NEAR(integral[0], 2 * numbers::PI, 1e-10);
  ASSERT_NEAR(integral[1], 2 * numbers::PI, 1e-10);
  ASSERT_NEAR(integral[2], 0, 1e-10);
  ASSERT_NEAR(integral[3], 0, 1e-10);
}

TEST(TestInclusionsBasis2, CheckScaling) // NOLINT
{
  // cx, cy, r
  Inclusions<2> ref;
  double        radius = 9;
  ref.set_n_q_points(100);
  ref.set_n_coefficients(13);
  ref.inclusions.push_back({{0, 0, radius}});
  ref.initialize();
  std::vector<double> integral(ref.get_n_coefficients() + 1, 0.0);
  for (unsigned int q = 0; q < ref.n_particles(); ++q)
    {
      const auto &fe_values = ref.get_fe_values(q);
      const auto &ds        = ref.get_JxW(q);
      integral[0] += ds;
      for (unsigned int i = 0; i < ref.get_n_coefficients(); ++i)
        integral[i + 1] += fe_values[i] * fe_values[i] * ds;
    }
  // Take the square root of the integral of the basis functions
  // and check that they are equal to sqrt(|D|) where |D| is the length of the
  // inclusion
  auto D = 2 * numbers::PI * radius;
  for (unsigned int i = 0; i < ref.get_n_coefficients() + 1; ++i)
    {
      auto expected = D;
      // Take into account
      if (i == 2 || i == 3)
        expected /= 2;
      EXPECT_NEAR(integral[i], expected, 1e-10);
    }
}


std::string
indent(int level)
{
  std::string s;
  for (int i = 0; i < level; i++)
    s += "  ";
  return s;
}

void
print_tree(boost::property_tree::ptree &pt, int level)
{
  if (pt.empty())
    {
      std::cout << "\"" << pt.data() << "\"";
    }

  else
    {
      if (level)
        std::cout << std::endl;

      std::cout << indent(level) << "{" << std::endl;

      for (boost::property_tree::ptree::iterator pos = pt.begin();
           pos != pt.end();)
        {
          std::cout << indent(level + 1) << "\"" << pos->first << "\": ";

          print_tree(pos->second, level + 1);
          ++pos;
          if (pos != pt.end())
            {
              std::cout << ",";
            }
          std::cout << std::endl;
        }

      std::cout << indent(level) << " }";
    }
  std::cout << std::endl;
  return;
}

TEST(CCO, XmlConverter)
{
  using namespace boost;
  using namespace property_tree;

  ptree root;
  read_xml(SOURCE_DIR "/data/tree_3D.xml", root);

  // print_tree(root, 0);

  // std::vector<Point<3>>              nodes;
  // std::vector<std::array<size_t, 2>> edges;
  // std::vector<double>                radii;
  std::map<std::string, Point<3>>                   nodes;
  std::map<std::string, std::array<std::string, 2>> edges;
  std::map<std::string, double>                     radii;
  std::map<std::string, size_t>                     ids;

  for (const auto &node : root.get_child("gxl").get_child("graph"))
    {
      if (node.first == "node")
        {
          auto id = node.second.get<std::string>("<xmlattr>.id");
          std::cout << "Node: " << id;

          Point<3> n;
          for (const auto &attr : node.second)
            if (attr.first == "attr")
              for (const auto &attr : attr.second)
                if (attr.first == "tup")
                  {
                    for (unsigned int i = 0; i < 3; ++i)
                      n[i] = attr.second.get<double>("float");
                    nodes[id] = n;
                    std::cout << ": " << n << std::endl;
                  }
        }
      else if (node.first == "edge")
        {
          auto from = node.second.get<std::string>("<xmlattr>.from");
          auto to   = node.second.get<std::string>("<xmlattr>.to");
          auto id   = node.second.get<std::string>("<xmlattr>.id");
          edges[id] = {{from, to}};
          for (const auto &attr : node.second)
            {
              if (attr.first == "attr" &&
                  attr.second.get_optional<std::string>("name").has_value() &&
                  attr.second.get_optional<std::string>("name").value() ==
                    "radius")
                {
                  radii[id] = attr.second.get<double>("float");
                }
            }
        }
    }
  std::cout << "N nodes: " << nodes.size() << std::endl;
  std::cout << "N edges: " << edges.size() << std::endl;
  for (const auto &n : nodes)
    ids[n.first] = ids.size();

  std::vector<Point<3>>              v_nodes;
  std::vector<std::array<size_t, 2>> v_edges;
  std::vector<double>                v_radii;
  for (const auto &n : nodes)
    v_nodes.push_back(n.second);
  for (const auto &e : edges)
    v_edges.push_back({{ids[e.second[0]], ids[e.second[1]]}});
  for (const auto &r : radii)
    v_radii.push_back(r.second);
}
