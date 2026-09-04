// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <string>

#include "test_paths.h"

namespace
{
  std::string
  shell_quote(const std::string &value)
  {
    std::string result = "'";
    for (const char character : value)
      if (character == '\'')
        result += "'\\''";
      else
        result += character;
    result += "'";
    return result;
  }

  std::filesystem::path
  application_path(const std::string &name)
  {
#ifdef DEBUG
    return ImmersX::TestPaths::binary_path(name + "_debug");
#else
    return ImmersX::TestPaths::binary_path(name);
#endif
  }

  void
  run_application(const std::string &name,
                  const std::string &parameter_relative_path)
  {
    const auto executable = application_path(name);
    const auto parameter  = ImmersX::TestPaths::binary_path(
      "gtests/parameters/" + parameter_relative_path);
    const auto run_directory = ImmersX::TestPaths::output_path(
      "application-execution-working-directory");
    std::filesystem::create_directories(run_directory);

    const auto command = "cd " + shell_quote(run_directory.string()) + " && " +
                         shell_quote(executable.string()) + " " +
                         shell_quote(parameter.string());
    ASSERT_EQ(std::system(command.c_str()), 0) << command;
  }
} // namespace

TEST(ApplicationExecution, StandaloneAndAdapterStaticApplications)
{
  const auto poisson_output =
    ImmersX::TestPaths::output_path("poisson-execution");
  const auto elastic_output =
    ImmersX::TestPaths::output_path("elastic-static-execution");
  std::filesystem::remove_all(poisson_output);
  std::filesystem::remove_all(elastic_output);

  run_application("poisson", "poisson_execution_1d3.prm");
  EXPECT_TRUE(std::filesystem::exists(poisson_output / "solution.pvd"));
  run_application("poisson_adapter", "poisson_execution_1d3.prm");
  EXPECT_TRUE(std::filesystem::exists(poisson_output / "solution.pvd"));

  run_application("elastic_static", "elastic_static_execution_1d3.prm");
  EXPECT_TRUE(std::filesystem::exists(elastic_output / "elastic_static.pvd"));
  run_application("elastic_static_adapter", "elastic_static_execution_1d3.prm");
  EXPECT_TRUE(std::filesystem::exists(elastic_output / "elastic_static.pvd"));
}
