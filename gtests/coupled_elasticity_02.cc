#include <gtest/gtest.h>
#include <immersx/physics/coupled_elasticity.h>

using namespace ImmersX;
#include <immersx/io/utils.h>
#include <immersx/physics/elasticity_problem_parameters.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <regex>
#include <sstream>

#include "test_paths.h"

// Tests of CoupledElasticityProblem on a straight cylinder along x embedded in
// a cube: a comparison of point and tensor-product coupling, and one test per
// member function for both coupling types. The reduced grids and inclusion
// files are generated in the test output directory.
#ifdef DEAL_II_WITH_VTK

using namespace dealii;

namespace
{
  using Overrides = std::vector<std::pair<std::string, std::string>>;

  // Write `content` below the test output directory and return the file name.
  // Rank 0 writes a temporary file and renames it, so that test processes
  // running at the same time never read a partially written file.
  std::string
  input_file(const std::string &name, const std::string &content)
  {
    const auto filename =
      TestPaths::output_path("gtests/data/coupled_elasticity/" + name);
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::filesystem::create_directories(filename.parent_path());
        auto temporary = filename;
        temporary += "." + std::to_string(::getpid());
        std::ofstream(temporary) << content;
        std::filesystem::rename(temporary, filename);
      }
    MPI_Barrier(MPI_COMM_WORLD);
    return filename.string();
  }

  // Reduced grid on the x axis with one cell between consecutive `vertices`.
  // Cell i has vessel id vessel_ids[i], and every vertex has displacement 0.1.
  std::string
  reduced_grid(const std::string               &name,
               const std::vector<double>       &vertices,
               const std::vector<unsigned int> &vessel_ids)
  {
    AssertDimension(vertices.size(), vessel_ids.size() + 1);
    const auto         n_cells = vessel_ids.size();
    std::ostringstream vtk;
    vtk << "# vtk DataFile Version 3.0\n"
        << "reduced grid\n"
        << "ASCII\n"
        << "DATASET UNSTRUCTURED_GRID\n"
        << "POINTS " << vertices.size() << " double\n";
    for (const double x : vertices)
      vtk << x << " 0 0\n";
    vtk << "CELLS " << n_cells << " " << 3 * n_cells << "\n";
    for (unsigned int i = 0; i < n_cells; ++i)
      vtk << "2 " << i << " " << i + 1 << "\n";
    vtk << "CELL_TYPES " << n_cells << "\n";
    for (unsigned int i = 0; i < n_cells; ++i)
      vtk << "3\n";
    vtk << "POINT_DATA " << vertices.size() << "\n"
        << "SCALARS displacement float\n"
        << "LOOKUP_TABLE default\n";
    for (unsigned int i = 0; i < vertices.size(); ++i)
      vtk << "0.1\n";
    vtk << "CELL_DATA " << n_cells << "\n"
        << "SCALARS vessel_id int\n"
        << "LOOKUP_TABLE default\n";
    for (const auto id : vessel_ids)
      vtk << id << "\n";
    return input_file(name + ".vtk", vtk.str());
  }

  // One cell from x = 1 to x = -1 with vessel id 0.
  std::string
  single_cell_grid()
  {
    return reduced_grid("single_cell", {1., -1.}, {0});
  }


  // 50 point inclusions along x, centered at -0.98, -0.94, ..., 0.98 and split
  // into `n_vessels` consecutive vessels of equal size. Each line is cx, cy,
  // cz, dx, dy, dz, r, vessel id.
  std::string
  inclusions_file(const unsigned int n_vessels)
  {
    std::ostringstream text;
    for (unsigned int i = 0; i < 50; ++i)
      text << -0.98 + 0.04 * i << " 0 0 0.04 0 0 0.2 " << i * n_vessels / 50
           << "\n";
    return input_file("inclusions_" + std::to_string(n_vessels) + ".txt",
                      text.str());
  }

  // Parameters of the cylinder with point coupling, using the inclusions of
  // inclusions_file(1), or with tensor-product coupling. Each "set <key> = ..."
  // entry listed in `overrides` is replaced by "set <key> = <value>". The inner
  // displacement solve needs more than the default 100 iterations on locally
  // refined meshes and on two ranks.
  std::string
  parameters(const bool         point_coupling,
             const std::string &reduced_grid,
             const std::string &input_file_fields,
             const Overrides   &overrides = {})
  {
    const std::string name = point_coupling ? "point" : "tensor";
    std::string       text = R"(
      set dimension               = 3
      set space dimension         = 3
      set reduced dimension       = 1
      set cross section dimension = 2
      subsection Immersed Problem
        set Coupling type                      = @COUPLING@
        set Dirichlet boundary ids             = 2,3,4,5
        set FE degree                          = 1
        set Initial refinement                 = 3
        set Normal flux boundary ids           = 0,1
        set Output directory                   = @TEST_OUTPUT_DIR@/gtests/data/pressure_comparison_@NAME@
        set Output name                        = pressure_comparison_@NAME@
        set Output pressure                    = true
        set Output results also before solving = false
        set Pressure coupling                  = false
        set Weak Dirichlet penalty coefficient = 1000
        subsection Grid generation
          set Domain type              = generate
          set Grid generator           = hyper_cube
          set Grid generator arguments = -1:1:true
          set Triangulation type       = distributed
          set Grid scale               = 1
        end
        subsection Immersed inclusions
          set Bounding boxes extraction level   = 0
          set Cluster inclusions with segments  = @CLUSTER@
          set Inclusions file                   = @INCLUSIONS_FILE@
          set Inclusions refinement             = 100
          set Number of fourier coefficients    = 2
          set Reference inclusion data          = @REFERENCE_DATA@
          set Selection of Fourier coefficients = @SELECTION@
          subsection Boundary data
            set Function expression = 0; 0; 0
          end
        end
        subsection Material properties
          subsection default
            set Density        = 0
            set Lame lambda    = 1
            set Lame mu        = 1
            set Rayleigh alpha = 0
            set Rayleigh beta  = 0
            set Viscosity eta  = 0
          end
        end
        subsection Refinement and remeshing
          set Maximum number of cells     = 2000000
          set Number of refinement cycles = 1
          set Refinement fraction         = 1
          set Strategy                    = global
        end
      end
      subsection Tensor product coupling
        subsection Cross section
          set Inclusion type           = hyper_ball
          set Maximum inclusion degree = 1
          set Refinement level         = 1
          set Selected indices         = 3, 7
        end
        subsection Particle coupling
          set RTree extraction level = 1
        end
        subsection Representative domain
          set Finite element degree            = 1
          set Input file fields                = @FIELDS@
          set Number of quadrature points      = 3
          set Number of quadrature repetitions = @REPETITIONS@
          set Quadrature type                  = gauss
          set Reduced grid name                = @GRID@
          set Reduced right hand side          = 0.1; 0.1
          set Thickness                        = 0.2
        end
      end
      subsection Solvers
        subsection Displacement
          set Max steps = 1000
        end
      end
    )";

    const auto replace = [&text](const std::string &token,
                                 const std::string &value) {
      for (std::size_t pos = 0;
           (pos = text.find(token, pos)) != std::string::npos;
           pos += value.size())
        text.replace(pos, token.size(), value);
    };
    replace("@COUPLING@", point_coupling ? "Point" : "TensorProduct");
    replace("@NAME@", name);
    replace("@CLUSTER@", point_coupling ? "true" : "false");
    replace("@INCLUSIONS_FILE@", point_coupling ? inclusions_file(1) : "");
    replace("@REFERENCE_DATA@", point_coupling ? "0,0,0,0.1,0,0,0,0.1,0" : "");
    replace("@SELECTION@", point_coupling ? "3,7" : "");
    replace("@REPETITIONS@", point_coupling ? "4" : "8");
    replace("@FIELDS@", input_file_fields);
    replace("@GRID@", reduced_grid);

    for (const auto &[key, value] : overrides)
      {
        const std::regex entry("set " + key + R"(\s*=[^\n]*)");
        EXPECT_TRUE(std::regex_search(text, entry)) << key;
        text = std::regex_replace(text, entry, "set " + key + " = " + value);
      }
    return TestPaths::expand_configured_paths(text);
  }


  // Declare the parameters, read them from `text`, and pass the problem to
  // `test`. The problem is built before parsing, as in app_coupled_elasticity,
  // so that its Inclusions member reads its parameters too.
  void
  with_problem(const std::string                                        &text,
               const std::function<void(CoupledElasticityProblem<3> &)> &test)
  {
    ParameterAcceptor::clear();
    ElasticityProblemParameters<3> par;
    CoupledElasticityProblem<3>    problem(par);
    initialize_parameters();
    ParameterAcceptor::prm.parse_input_from_string(text);
    ParameterAcceptor::parse_all_parameters();
    test(problem);
  }


  // All entries of a distributed vector, on every rank.
  std::vector<double>
  gather(const TrilinosWrappers::MPI::Vector &v)
  {
    std::vector<double> local(v.size(), 0.);
    for (const auto i : v.locally_owned_elements())
      local[i] = v[i];
    std::vector<double> values(v.size(), 0.);
    Utilities::MPI::sum(local, MPI_COMM_WORLD, values);
    return values;
  }

} // namespace

// -----------------------------------------------------------------------------
// Point and tensor-product coupling
// -----------------------------------------------------------------------------

// A straight cylinder along x with constant radial displacement must give the
// same pressure and displacement whether the coupling is made through point
// inclusions or through the tensor-product space. The multipliers are not
// compared: they live in different spaces for the two couplings.
TEST(CoupledElasticity, BOTH_PointAndTensorProductAgree)
{
  // Pressure of the only vessel, and l1, l2 and linfty norms of the
  // displacement.
  const auto solve = [](const bool point_coupling) {
    std::vector<double> result;
    with_problem(
      parameters(point_coupling,
                 single_cell_grid(),
                 point_coupling ? "displacement" : "displacement, vessel_id"),
      [&](auto &problem) {
        EXPECT_EQ(problem.par.coupling_type,
                  point_coupling ? CouplingType::Point :
                                   CouplingType::TensorProduct);
        problem.run_timestep0();
        problem.run_timestep();
        EXPECT_EQ(problem.n_vessels(), 1u);

        const auto pressure = gather(problem.pressure_at_immersed_surface);
        ASSERT_EQ(pressure.size(), 1u);
        const auto &u = problem.solution.block(0);
        result = {pressure[0], u.l1_norm(), u.l2_norm(), u.linfty_norm()};
      });
    return result;
  };
  const auto point  = solve(true);
  const auto tensor = solve(false);

  const std::vector<std::string> names = {"pressure",
                                          "l1 norm of displacement",
                                          "l2 norm of displacement",
                                          "linfty norm of displacement"};
  ASSERT_EQ(point.size(), names.size());
  ASSERT_EQ(tensor.size(), names.size());
  for (unsigned int i = 0; i < names.size(); ++i)
    {
      std::cout << names[i] << ": point " << point[i] << ", tensor "
                << tensor[i] << std::endl;
      ASSERT_NE(point[i], 0.0) << names[i];
      EXPECT_NEAR(tensor[i], point[i], 0.05 * std::abs(point[i])) << names[i];
    }
}

#endif
