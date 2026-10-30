// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_generation_pairs_h
#define immersx_internal_generation_pairs_h
#include <immersx/vascularization/generation.h>
namespace ImmersX::Vascularization::internal
{
  // Pair at the common territory's writer, independently of endpoint ownership.
  // Keep input beds active while deriving new beds from terminal flow loads,
  // then check the actual paired flow.
  void
  pair_territories(GenerationResult                &result,
                   const Parameters                &parameters,
                   MPI_Comm                         communicator,
                   const std::vector<TerminalPair> &retained_pairs = {});
} // namespace ImmersX::Vascularization::internal
#endif
