// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II library.
//
// ---------------------------------------------------------------------

#ifndef immersx_coral_ida_elastodynamics_h
#define immersx_coral_ida_elastodynamics_h

#include <memory>
#include <string>

namespace ImmersX::Coral
{
  /** Graph-facing execution seam for one 2D elastodynamics Problem. */
  class IDAElastodynamics2D
  {
  public:
    explicit IDAElastodynamics2D(const std::string &parameter_file);

    ~IDAElastodynamics2D();

    IDAElastodynamics2D(const IDAElastodynamics2D &) = delete;
    IDAElastodynamics2D &
    operator=(const IDAElastodynamics2D &)      = delete;
    IDAElastodynamics2D(IDAElastodynamics2D &&) = delete;
    IDAElastodynamics2D &
    operator=(IDAElastodynamics2D &&) = delete;

    void
    run();

    bool
    state_is_finite() const;

    double
    current_time() const;

  private:
    class Impl;
    std::unique_ptr<Impl> implementation;
  };
} // namespace ImmersX::Coral

#endif // immersx_coral_ida_elastodynamics_h
