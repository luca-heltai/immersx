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
#include <unistd.h>

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>

namespace
{
  constexpr char default_timeout_message[] =
    "GoogleTest exceeded the 120 second correctness-test budget.\n";
  std::atomic<const char *> timeout_message{default_timeout_message};
  static_assert(std::atomic<const char *>::is_always_lock_free,
                "The test watchdog requires a signal-safe atomic pointer.");

  void
  test_time_limit_exceeded(int)
  {
    // A signal handler cannot use streams or MPI. A failed rank also makes
    // the MPI launcher fail, rather than leaving a passing rank-zero report.
    const char *message = timeout_message.load(std::memory_order_acquire);
    const auto  written = ::write(STDERR_FILENO, message, std::strlen(message));
    (void)written;
    std::_Exit(EXIT_FAILURE);
  }

  class TestTimeLimit : public ::testing::EmptyTestEventListener
  {
  public:
    void
    OnTestStart(const ::testing::TestInfo &info) override
    {
      // Keep messages immutable and alive until program exit so the signal
      // handler can identify a slow nonzero MPI rank without consulting GTest.
      messages.emplace_back(
        "GoogleTest " + std::string(info.test_suite_name()) + "." +
        info.name() + " exceeded the 120 second correctness-test budget.\n");
      timeout_message.store(messages.back().c_str(), std::memory_order_release);
      ::alarm(120);
    }

    void
    OnTestEnd(const ::testing::TestInfo &) override
    {
      ::alarm(0);
      timeout_message.store(default_timeout_message, std::memory_order_release);
    }

  private:
    std::deque<std::string> messages;
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
  AssertThrow(std::signal(SIGALRM, test_time_limit_exceeded) != SIG_ERR,
              dealii::ExcMessage("Cannot install the GoogleTest time limit."));
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
