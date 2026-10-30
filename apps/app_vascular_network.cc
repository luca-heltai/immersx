// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <immersx/vascularization/application.h>

#include <filesystem>
#include <fstream>
#include <iostream>

int
main(int argc, char **argv)
{
  using namespace dealii;
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
      const auto                       comm = MPI_COMM_WORLD;
      const auto rank = Utilities::MPI::this_mpi_process(comm);
      if (argc == 2 && std::string(argv[1]) == "--help")
        {
          if (rank == 0)
            std::cout
              << "Usage: vascular_network FILE.prm\n       vascular_network --write-parameters NEW.prm\n";
          return 0;
        }
      ImmersX::Vascularization::ApplicationParameters p;
      if (argc == 3 && std::string(argv[1]) == "--write-parameters")
        {
          bool valid = true;
          if (rank == 0)
            try
              {
                AssertThrow(!std::filesystem::exists(argv[2]),
                            ExcMessage("Parameter output file must be new."));
                ParameterAcceptor::initialize("", argv[2]);
              }
            catch (const std::exception &e)
              {
                std::cerr << e.what() << '\n';
                valid = false;
              }
          return Utilities::MPI::min(static_cast<unsigned int>(valid), comm) ==
                     1 ?
                   0 :
                   1;
        }
      AssertThrow(argc == 2,
                  ExcMessage("Usage: vascular_network FILE.prm (or --help)."));
      bool valid = true;
      try
        {
          const auto    filename = std::filesystem::absolute(argv[1]);
          std::ifstream in(filename);
          AssertThrow(in, ExcMessage("Cannot open vascular parameter file."));
          ParameterAcceptor::initialize(in);
          const auto resolve = [&](std::string &path) {
            if (!path.empty())
              path =
                (filename.parent_path() / path).lexically_normal().string();
          };
          resolve(p.input_archive);
          resolve(p.output_directory);
          ParameterAcceptor::prm.enter_subsection("Vascularization");
          ParameterAcceptor::prm.enter_subsection("Input and ports");
          ParameterAcceptor::prm.set("Input archive", p.input_archive);
          ParameterAcceptor::prm.leave_subsection();
          ParameterAcceptor::prm.enter_subsection("Output");
          ParameterAcceptor::prm.set("Directory", p.output_directory);
          ParameterAcceptor::prm.leave_subsection();
          ParameterAcceptor::prm.leave_subsection();
        }
      catch (const std::exception &e)
        {
          std::cerr << "Rank " << rank << ": " << e.what() << '\n';
          valid = false;
        }
      AssertThrow(
        Utilities::MPI::min(static_cast<unsigned int>(valid), comm) == 1,
        ExcMessage("Vascular parameters could not be read on every process."));
      const auto result = ImmersX::Vascularization::run(p, comm);
      if (rank == 0)
        try
          {
            std::ofstream out;
            out.exceptions(std::ios::failbit | std::ios::badbit);
            out.open(std::filesystem::path(p.output_directory) /
                     "parameters.prm");
            ParameterAcceptor::prm.print_parameters(out,
                                                    ParameterHandler::ShortPRM);
            out.close();
            std::cout << "Vascular result "
                      << (result.validation.valid() ? "valid" : "invalid")
                      << "; reports: " << p.output_directory << '\n';
            for (const auto &issue : result.validation.issues)
              std::cerr << issue.message << '\n';
          }
        catch (const std::exception &e)
          {
            std::cerr << e.what() << '\n';
            valid = false;
          }
      return Utilities::MPI::min(static_cast<unsigned int>(
                                   valid && result.validation.valid()),
                                 comm) == 1 ?
               0 :
               1;
    }
  catch (const std::exception &e)
    {
      std::cerr << e.what() << '\n';
      return 1;
    }
}
