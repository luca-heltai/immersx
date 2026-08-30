// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// ---------------------------------------------------------------------

#ifndef immersx_linear_algebra_h
#define immersx_linear_algebra_h

#include <deal.II/lac/generic_linear_algebra.h>

#define FORCE_USE_OF_TRILINOS
namespace ImmersX
{
  namespace ImmersXLA
  {
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
    using namespace dealii::LinearAlgebraPETSc;
#  define IMMERSX_USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
    using namespace dealii::LinearAlgebraTrilinos;
    using SolverDirect = dealii::TrilinosWrappers::SolverDirect;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
  } // namespace ImmersXLA

} // namespace ImmersX

#endif // immersx_linear_algebra_h
