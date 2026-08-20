// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef IMMERSX_GTEST_TEST_PATHS_H
#define IMMERSX_GTEST_TEST_PATHS_H

#include <filesystem>
#include <string>

namespace ImmersX::TestPaths
{
  inline std::filesystem::path
  source_path(const std::string &relative_path)
  {
    return std::filesystem::path(IMMERSX_SOURCE_DIR) / relative_path;
  }

  inline std::filesystem::path
  binary_path(const std::string &relative_path)
  {
    return std::filesystem::path(IMMERSX_BINARY_DIR) / relative_path;
  }

  inline std::filesystem::path
  data_path(const std::string &relative_path)
  {
    return std::filesystem::path(TEST_DATA_DIR) / relative_path;
  }

  inline std::filesystem::path
  output_path(const std::string &relative_path)
  {
    return std::filesystem::path(TEST_OUTPUT_DIR) / relative_path;
  }

  inline std::string
  parameter_path(const std::string &relative_path)
  {
    return binary_path(relative_path).string();
  }

  inline std::string
  data_filename(const std::string &relative_path)
  {
    return data_path(relative_path).string();
  }

  inline std::string
  output_directory(const std::string &relative_path)
  {
    return output_path(relative_path).string();
  }

  inline std::string
  expand_configured_paths(std::string text)
  {
    const auto replace = [&text](const std::string &token,
                                 const std::string &value) {
      std::size_t position = 0;
      while ((position = text.find(token, position)) != std::string::npos)
        {
          text.replace(position, token.size(), value);
          position += value.size();
        }
    };

    replace("@IMMERSX_SOURCE_DIR@", IMMERSX_SOURCE_DIR);
    replace("@IMMERSX_BINARY_DIR@", IMMERSX_BINARY_DIR);
    replace("@TEST_DATA_DIR@", TEST_DATA_DIR);
    replace("@TEST_OUTPUT_DIR@", TEST_OUTPUT_DIR);
    return text;
  }
} // namespace ImmersX::TestPaths

#endif
