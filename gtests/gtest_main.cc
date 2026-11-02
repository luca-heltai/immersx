// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// The ImmersX application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the file LICENSE.md at the top level
// of the ImmersX distribution.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>

#include <gtest/gtest.h>

#include <string>

#include "test_watchdog.h"

namespace
{
  class TestTimeLimit : public ::testing::EmptyTestEventListener
  {
  public:
    void
    OnTestStart(const ::testing::TestInfo &info) override
    {
      ImmersX::TestWatchdog::instance().arm(
        std::string(info.test_suite_name()) + "." + info.name());
    }

    void
    OnTestEnd(const ::testing::TestInfo &) override
    {
      ImmersX::TestWatchdog::instance().disarm();
    }
  };
} // namespace

// The category guard needs a typed registry entry to check glob matching.
// Keep this fixture in the common driver so it needs no separate translation
// unit.
template <typename T>
class TestCategoryTyped : public ::testing::Test
{};

using RegistryTypes = ::testing::Types<int, double>;
TYPED_TEST_SUITE(TestCategoryTyped, RegistryTypes, );

TYPED_TEST(TestCategoryTyped, RegistryEntry)
{
  SUCCEED();
}

int
main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);

  // List the complete registry without initializing MPI or solver backends.
  // This also lets every process in a discovery job exit without collectives.
  if (::testing::GTEST_FLAG(list_tests))
    return RUN_ALL_TESTS();

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const auto n_ranks = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  if (n_ranks == 1)
    {
      constexpr const char *mpi_test_pattern = "*.MPI_*";
      std::string           filter           = ::testing::GTEST_FLAG(filter);

      if (filter.empty() || filter == "*")
        filter = "*-" + std::string(mpi_test_pattern);
      else if (filter.find("-*.MPI_*") == std::string::npos &&
               filter.find("MPI_") == std::string::npos &&
               filter.find("BOTH_") == std::string::npos)
        filter += "-" + std::string(mpi_test_pattern);

      ::testing::GTEST_FLAG(filter) = filter;
    }
  else if (n_ranks > 1)
    {
      constexpr const char *mpi_test_pattern  = "*.MPI_*";
      constexpr const char *both_test_pattern = "*.BOTH_*";
      std::string           filter            = ::testing::GTEST_FLAG(filter);

      if (filter.empty() || filter == "*")
        filter = std::string(mpi_test_pattern) + ":" + both_test_pattern;
      else if (filter.find("MPI_") == std::string::npos &&
               filter.find("BOTH_") == std::string::npos)
        filter += ":" + std::string(mpi_test_pattern);

      ::testing::GTEST_FLAG(filter) = filter;
    }

  ::testing::TestEventListeners &listeners =
    ::testing::UnitTest::GetInstance()->listeners();
  listeners.Append(new TestTimeLimit());
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) != 0)
    {
      delete listeners.Release(listeners.default_result_printer());
      // XML is written by rank zero too. Concurrent listeners would overwrite
      // the same report; MPI/CTest exit status still includes every rank.
      if (listeners.default_xml_generator() != nullptr)
        delete listeners.Release(listeners.default_xml_generator());
    }
  return RUN_ALL_TESTS();
}
