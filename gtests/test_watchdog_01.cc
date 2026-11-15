// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>

#include <optional>
#include <string>

#include "test_watchdog.h"

namespace
{
  class TestWatchdog : public ::testing::Test
  {
  public:
    void
    SetUp() override
    {
      if (const char *value = std::getenv("IMMERSX_TEST_TIMEOUT_SECONDS"))
        previous = value;
      ::unsetenv("IMMERSX_TEST_TIMEOUT_SECONDS");
    }

    void
    TearDown() override
    {
      if (previous)
        ::setenv("IMMERSX_TEST_TIMEOUT_SECONDS", previous->c_str(), 1);
      else
        ::unsetenv("IMMERSX_TEST_TIMEOUT_SECONDS");
    }

    std::optional<std::string> previous;
  };
} // namespace

TEST_F(TestWatchdog, DefaultCorrectnessBudget)
{
  EXPECT_EQ(ImmersX::TestWatchdog::case_timeout_seconds(), 115u);
}

TEST_F(TestWatchdog, NightlyBudget)
{
  ::setenv("IMMERSX_TEST_TIMEOUT_SECONDS", "3600", 1);
  EXPECT_EQ(ImmersX::TestWatchdog::case_timeout_seconds(), 3600u);
}

TEST_F(TestWatchdog, InvalidBudgetCannotDisableTheWatchdog)
{
  for (const auto *value : {"", "0", "-1", "3601", "abc", "115x"})
    {
      SCOPED_TRACE(value);
      ::setenv("IMMERSX_TEST_TIMEOUT_SECONDS", value, 1);
      EXPECT_THROW(ImmersX::TestWatchdog::case_timeout_seconds(),
                   std::invalid_argument);
    }
}
