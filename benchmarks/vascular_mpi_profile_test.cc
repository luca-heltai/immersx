// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <mpi.h>

#include <vector>

#include "../source/vascularization/growth_mpi_scope.h"
using namespace ImmersX::Vascularization::internal;
int
main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  double             send[3] = {1., 2., 3.}, receive[3] = {};
  unsigned long long sent[3] = {1, 2, 3}, received[3] = {};
  double             one = 1., total = 0., asynchronous = 0.;
  GrowthMpiRun       run(0, 8);
  auto        initialization = run.phase(GrowthMpiPhase::initialization, 1);
  MPI_Request requests[2];
  MPI_Irecv(receive,
            3,
            MPI_DOUBLE,
            (rank + size - 1) % size,
            0,
            MPI_COMM_WORLD,
            &requests[0]);
  MPI_Isend(
    send, 3, MPI_DOUBLE, (rank + 1) % size, 0, MPI_COMM_WORLD, &requests[1]);
  MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
  initialization.finish();
  initialization.finish();
  {
    auto batch = run.phase(GrowthMpiPhase::batch, 4);
    MPI_Sendrecv(sent,
                 3,
                 MPI_UNSIGNED_LONG_LONG,
                 (rank + 1) % size,
                 1,
                 received,
                 3,
                 MPI_UNSIGNED_LONG_LONG,
                 (rank + size - 1) % size,
                 1,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    {
      auto        finalization = run.phase(GrowthMpiPhase::finalization, 8);
      MPI_Request barrier;
      MPI_Ibarrier(MPI_COMM_WORLD, &barrier);
      MPI_Wait(&barrier, MPI_STATUS_IGNORE);
    }
    MPI_Allreduce(&one, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Sendrecv(nullptr,
                 0,
                 MPI_BYTE,
                 (rank + 1) % size,
                 2,
                 nullptr,
                 0,
                 MPI_BYTE,
                 (rank + size - 1) % size,
                 2,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    MPI_Send(nullptr, 0, MPI_BYTE, MPI_PROC_NULL, 3, MPI_COMM_WORLD);
  }
  // These calls remain outside every growth phase.
  double broadcast[2] = {2., 3.};
  MPI_Bcast(broadcast, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  std::vector<int> gathered(2 * size), counts(size, 2), displacements(size);
  for (int i = 0; i < size; ++i)
    displacements[i] = 2 * i;
  gathered[2 * rank]     = rank;
  gathered[2 * rank + 1] = rank + 1;
  MPI_Allgatherv(MPI_IN_PLACE,
                 0,
                 MPI_DATATYPE_NULL,
                 gathered.data(),
                 counts.data(),
                 displacements.data(),
                 MPI_INT,
                 MPI_COMM_WORLD);
  std::vector<int> transposed(size), ones(size, 1), offsets(size);
  for (int i = 0; i < size; ++i)
    {
      transposed[i] = rank * size + i;
      offsets[i]    = i;
    }
  MPI_Alltoallv(MPI_IN_PLACE,
                nullptr,
                nullptr,
                MPI_DATATYPE_NULL,
                transposed.data(),
                ones.data(),
                offsets.data(),
                MPI_INT,
                MPI_COMM_WORLD);
  {
    auto        batch = run.phase(GrowthMpiPhase::batch, 7);
    MPI_Request reduction;
    MPI_Iallreduce(
      &one, &asynchronous, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &reduction);
    MPI_Wait(&reduction, MPI_STATUS_IGNORE);
  }
  bool valid = total == size && asynchronous == size && receive[2] == 3. &&
               received[2] == 3;
  for (int i = 0; i < size; ++i)
    valid = valid && gathered[2 * i] == i && gathered[2 * i + 1] == i + 1 &&
            transposed[i] == i * size + rank;
  MPI_Finalize();
  return valid ? 0 : 1;
}
