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

#include <deal.II/base/mpi.h>

#include <gtest/gtest.h>

#include <string>

int
main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  testing::InitGoogleTest(&argc, argv);

  const auto n_ranks = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  if (n_ranks == 1)
    {
      constexpr const char *mpi_test_pattern = "*.MPI_*";
      std::string           filter           = ::testing::GTEST_FLAG(filter);

      if (filter.empty() || filter == "*")
        filter = "*-" + std::string(mpi_test_pattern);
      else if (const auto dash_pos = filter.find('-');
               dash_pos == std::string::npos)
        filter += "-" + std::string(mpi_test_pattern);
      else
        filter += ":" + std::string(mpi_test_pattern);

      ::testing::GTEST_FLAG(filter) = filter;
    }
  else
    {
      constexpr const char *mpi_test_pattern = "*.MPI_*";
      std::string           filter           = ::testing::GTEST_FLAG(filter);

      if (filter.empty() || filter == "*")
        filter = mpi_test_pattern;
      else if (const auto dash_pos = filter.find('-');
               dash_pos == std::string::npos)
        filter += ":" + std::string(mpi_test_pattern);
      else
        filter.insert(dash_pos, ":" + std::string(mpi_test_pattern));

      ::testing::GTEST_FLAG(filter) = filter;
    }

  ::testing::TestEventListeners &listeners =
    ::testing::UnitTest::GetInstance()->listeners();
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) != 0)
    {
      delete listeners.Release(listeners.default_result_printer());
    }
  return RUN_ALL_TESTS();
}
