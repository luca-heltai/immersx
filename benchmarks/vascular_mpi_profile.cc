// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Benchmark-only standard MPI profiling interposition. The normal library and
// application do not link this translation unit.
#include <mpi.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
namespace
{
  enum class Function
  {
    Send,
    Ssend,
    Rsend,
    Isend,
    Issend,
    Recv,
    Irecv,
    Sendrecv,
    Allreduce,
    Scan,
    Exscan,
    Reduce,
    Reduce_scatter_block,
    Bcast,
    Allgather,
    Alltoall,
    Allgatherv,
    Alltoallv,
    Gather,
    Gatherv,
    Scatter,
    Scatterv,
    Barrier,
    Ibarrier,
    Wait,
    Waitall,
    Waitany,
    Test,
    Testall,
    Probe,
    Iprobe,
    Comm_dup,
    Comm_free,
    Comm_split,
    count
  };
  constexpr std::array<const char *, static_cast<unsigned int>(Function::count)>
    names{{"MPI_Send",
           "MPI_Ssend",
           "MPI_Rsend",
           "MPI_Isend",
           "MPI_Issend",
           "MPI_Recv",
           "MPI_Irecv",
           "MPI_Sendrecv",
           "MPI_Allreduce",
           "MPI_Scan",
           "MPI_Exscan",
           "MPI_Reduce",
           "MPI_Reduce_scatter_block",
           "MPI_Bcast",
           "MPI_Allgather",
           "MPI_Alltoall",
           "MPI_Allgatherv",
           "MPI_Alltoallv",
           "MPI_Gather",
           "MPI_Gatherv",
           "MPI_Scatter",
           "MPI_Scatterv",
           "MPI_Barrier",
           "MPI_Ibarrier",
           "MPI_Wait",
           "MPI_Waitall",
           "MPI_Waitany",
           "MPI_Test",
           "MPI_Testall",
           "MPI_Probe",
           "MPI_Iprobe",
           "MPI_Comm_dup",
           "MPI_Comm_free",
           "MPI_Comm_split"}};
  struct Measurement
  {
    std::atomic<std::uint64_t> calls{0}, sent{0}, collective{0}, nanoseconds{0};
  };
  std::array<Measurement, names.size()> measurements;
  std::atomic<bool>                     enabled{false};
  int                                   world_rank = 0;
  void
  start()
  {
    PMPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    const char *directory = std::getenv("IMMERSX_MPI_PROFILE_DIRECTORY");
    enabled.store(directory != nullptr && *directory != '\0');
  }
  std::uint64_t
  bytes(int count, MPI_Datatype type)
  {
    if (!enabled.load(std::memory_order_relaxed) || count <= 0)
      return 0;
    int width = 0;
    if (PMPI_Type_size(type, &width) != MPI_SUCCESS || width < 0)
      return 0;
    return static_cast<std::uint64_t>(count) *
           static_cast<std::uint64_t>(width);
  }
  int
  size(MPI_Comm comm)
  {
    if (!enabled.load(std::memory_order_relaxed))
      return 0;
    int n = 0;
    PMPI_Comm_size(comm, &n);
    return n;
  }
  int
  rank(MPI_Comm comm)
  {
    if (!enabled.load(std::memory_order_relaxed))
      return -1;
    int n = -1;
    PMPI_Comm_rank(comm, &n);
    return n;
  }
  std::uint64_t
  vector_bytes(const int *counts, MPI_Datatype type, MPI_Comm comm)
  {
    std::uint64_t total = 0;
    for (int i = 0; i < size(comm); ++i)
      total += bytes(counts[i], type);
    return total;
  }
  template <typename Payload, typename Call>
  int
  observe(Function function, Payload payload, Call call)
  {
    if (!enabled.load(std::memory_order_relaxed))
      return call();
    const auto [sent, collective] = payload();
    const auto begin              = std::chrono::steady_clock::now();
    const int  status             = call();
    const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - begin)
                            .count();
    if (status == MPI_SUCCESS)
      {
        auto &m = measurements[static_cast<unsigned int>(function)];
        m.calls.fetch_add(1, std::memory_order_relaxed);
        m.sent.fetch_add(sent, std::memory_order_relaxed);
        m.collective.fetch_add(collective, std::memory_order_relaxed);
        m.nanoseconds.fetch_add(duration, std::memory_order_relaxed);
      }
    return status;
  }
  void
  finish()
  {
    if (!enabled.exchange(false))
      return;
    const auto        directory = std::getenv("IMMERSX_MPI_PROFILE_DIRECTORY");
    const std::string filename =
      std::string(directory) + "/mpi-" + std::to_string(world_rank) + ".csv";
    auto *out = std::fopen(filename.c_str(), "wx");
    if (!out)
      {
        std::fprintf(stderr,
                     "Cannot write MPI profile: %s\n",
                     filename.c_str());
        return;
      }
    std::fprintf(
      out, "function,calls,posted_send_bytes,collective_input_bytes,seconds\n");
    for (unsigned int i = 0; i < names.size(); ++i)
      {
        const auto &m = measurements[i];
        std::fprintf(out,
                     "%s,%llu,%llu,%llu,%.17g\n",
                     names[i],
                     static_cast<unsigned long long>(m.calls.load()),
                     static_cast<unsigned long long>(m.sent.load()),
                     static_cast<unsigned long long>(m.collective.load()),
                     static_cast<double>(m.nanoseconds.load()) * 1.e-9);
      }
    if (std::fclose(out) != 0)
      std::fprintf(stderr, "Cannot close MPI profile: %s\n", filename.c_str());
  }
} // namespace
extern "C" int
MPI_Init(int *argc, char ***argv)
{
  const int result = PMPI_Init(argc, argv);
  if (result == MPI_SUCCESS)
    start();
  return result;
}
extern "C" int
MPI_Init_thread(int *argc, char ***argv, int required, int *provided)
{
  const int result = PMPI_Init_thread(argc, argv, required, provided);
  if (result == MPI_SUCCESS)
    start();
  return result;
}
extern "C" int
MPI_Finalize()
{
  finish();
  return PMPI_Finalize();
}
extern "C" int
MPI_Send(const void  *buf,
         int          count,
         MPI_Datatype type,
         int          dest,
         int          tag,
         MPI_Comm     comm)
{
  return observe(
    Function::Send,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{
        dest == MPI_PROC_NULL ? 0 : bytes(count, type), 0};
    },
    [&]() { return PMPI_Send(buf, count, type, dest, tag, comm); });
}
extern "C" int
MPI_Ssend(const void  *buf,
          int          count,
          MPI_Datatype type,
          int          dest,
          int          tag,
          MPI_Comm     comm)
{
  return observe(
    Function::Ssend,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{
        dest == MPI_PROC_NULL ? 0 : bytes(count, type), 0};
    },
    [&]() { return PMPI_Ssend(buf, count, type, dest, tag, comm); });
}
extern "C" int
MPI_Rsend(const void  *buf,
          int          count,
          MPI_Datatype type,
          int          dest,
          int          tag,
          MPI_Comm     comm)
{
  return observe(
    Function::Rsend,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{
        dest == MPI_PROC_NULL ? 0 : bytes(count, type), 0};
    },
    [&]() { return PMPI_Rsend(buf, count, type, dest, tag, comm); });
}
extern "C" int
MPI_Isend(const void  *buf,
          int          count,
          MPI_Datatype type,
          int          dest,
          int          tag,
          MPI_Comm     comm,
          MPI_Request *request)
{
  return observe(
    Function::Isend,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{
        dest == MPI_PROC_NULL ? 0 : bytes(count, type), 0};
    },
    [&]() { return PMPI_Isend(buf, count, type, dest, tag, comm, request); });
}
extern "C" int
MPI_Issend(const void  *buf,
           int          count,
           MPI_Datatype type,
           int          dest,
           int          tag,
           MPI_Comm     comm,
           MPI_Request *request)
{
  return observe(
    Function::Issend,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{
        dest == MPI_PROC_NULL ? 0 : bytes(count, type), 0};
    },
    [&]() { return PMPI_Issend(buf, count, type, dest, tag, comm, request); });
}
extern "C" int
MPI_Recv(void        *buf,
         int          count,
         MPI_Datatype type,
         int          source,
         int          tag,
         MPI_Comm     comm,
         MPI_Status  *status)
{
  return observe(
    Function::Recv,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{0, 0};
    },
    [&]() { return PMPI_Recv(buf, count, type, source, tag, comm, status); });
}
extern "C" int
MPI_Irecv(void        *buf,
          int          count,
          MPI_Datatype type,
          int          source,
          int          tag,
          MPI_Comm     comm,
          MPI_Request *request)
{
  return observe(
    Function::Irecv,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{0, 0};
    },
    [&]() { return PMPI_Irecv(buf, count, type, source, tag, comm, request); });
}
extern "C" int
MPI_Sendrecv(const void  *sendbuf,
             int          sendcount,
             MPI_Datatype sendtype,
             int          dest,
             int          sendtag,
             void        *recvbuf,
             int          recvcount,
             MPI_Datatype recvtype,
             int          source,
             int          recvtag,
             MPI_Comm     comm,
             MPI_Status  *status)
{
  return observe(
    Function::Sendrecv,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{
        dest == MPI_PROC_NULL ? 0 : bytes(sendcount, sendtype), 0};
    },
    [&]() {
      return PMPI_Sendrecv(sendbuf,
                           sendcount,
                           sendtype,
                           dest,
                           sendtag,
                           recvbuf,
                           recvcount,
                           recvtype,
                           source,
                           recvtag,
                           comm,
                           status);
    });
}
extern "C" int
MPI_Allreduce(const void  *sendbuf,
              void        *recvbuf,
              int          count,
              MPI_Datatype type,
              MPI_Op       op,
              MPI_Comm     comm)
{
  return observe(
    Function::Allreduce,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{0, bytes(count, type)};
    },
    [&]() { return PMPI_Allreduce(sendbuf, recvbuf, count, type, op, comm); });
}
extern "C" int
MPI_Scan(const void  *sendbuf,
         void        *recvbuf,
         int          count,
         MPI_Datatype type,
         MPI_Op       op,
         MPI_Comm     comm)
{
  return observe(
    Function::Scan,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{0, bytes(count, type)};
    },
    [&]() { return PMPI_Scan(sendbuf, recvbuf, count, type, op, comm); });
}
extern "C" int
MPI_Exscan(const void  *sendbuf,
           void        *recvbuf,
           int          count,
           MPI_Datatype type,
           MPI_Op       op,
           MPI_Comm     comm)
{
  return observe(
    Function::Exscan,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{0, bytes(count, type)};
    },
    [&]() { return PMPI_Exscan(sendbuf, recvbuf, count, type, op, comm); });
}
extern "C" int
MPI_Reduce(const void  *sendbuf,
           void        *recvbuf,
           int          count,
           MPI_Datatype type,
           MPI_Op       op,
           int          root,
           MPI_Comm     comm)
{
  return observe(
    Function::Reduce,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{0, bytes(count, type)};
    },
    [&]() {
      return PMPI_Reduce(sendbuf, recvbuf, count, type, op, root, comm);
    });
}
extern "C" int
MPI_Reduce_scatter_block(const void  *sendbuf,
                         void        *recvbuf,
                         int          recvcount,
                         MPI_Datatype type,
                         MPI_Op       op,
                         MPI_Comm     comm)
{
  return observe(
    Function::Reduce_scatter_block,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{0,
                                                     bytes(recvcount, type) *
                                                       size(comm)};
    },
    [&]() {
      return PMPI_Reduce_scatter_block(
        sendbuf, recvbuf, recvcount, type, op, comm);
    });
}
extern "C" int
MPI_Bcast(void *buf, int count, MPI_Datatype type, int root, MPI_Comm comm)
{
  return observe(
    Function::Bcast,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{
        0, rank(comm) == root ? bytes(count, type) : 0};
    },
    [&]() { return PMPI_Bcast(buf, count, type, root, comm); });
}
extern "C" int
MPI_Allgather(const void  *sendbuf,
              int          sendcount,
              MPI_Datatype sendtype,
              void        *recvbuf,
              int          recvcount,
              MPI_Datatype recvtype,
              MPI_Comm     comm)
{
  return observe(
    Function::Allgather,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{
        0,
        sendbuf == MPI_IN_PLACE ? bytes(recvcount, recvtype) :
                                  bytes(sendcount, sendtype)};
    },
    [&]() {
      return PMPI_Allgather(
        sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    });
}
extern "C" int
MPI_Alltoall(const void  *sendbuf,
             int          sendcount,
             MPI_Datatype sendtype,
             void        *recvbuf,
             int          recvcount,
             MPI_Datatype recvtype,
             MPI_Comm     comm)
{
  return observe(
    Function::Alltoall,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{
        0,
        sendbuf == MPI_IN_PLACE ? bytes(recvcount, recvtype) * size(comm) :
                                  bytes(sendcount, sendtype) * size(comm)};
    },
    [&]() {
      return PMPI_Alltoall(
        sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    });
}
extern "C" int
MPI_Allgatherv(const void  *sendbuf,
               int          sendcount,
               MPI_Datatype sendtype,
               void        *recvbuf,
               const int    recvcounts[],
               const int    displs[],
               MPI_Datatype recvtype,
               MPI_Comm     comm)
{
  return observe(
    Function::Allgatherv,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{
        0,
        sendbuf == MPI_IN_PLACE ? bytes(recvcounts[rank(comm)], recvtype) :
                                  bytes(sendcount, sendtype)};
    },
    [&]() {
      return PMPI_Allgatherv(sendbuf,
                             sendcount,
                             sendtype,
                             recvbuf,
                             recvcounts,
                             displs,
                             recvtype,
                             comm);
    });
}
extern "C" int
MPI_Alltoallv(const void  *sendbuf,
              const int    sendcounts[],
              const int    sdispls[],
              MPI_Datatype sendtype,
              void        *recvbuf,
              const int    recvcounts[],
              const int    rdispls[],
              MPI_Datatype recvtype,
              MPI_Comm     comm)
{
  return observe(
    Function::Alltoallv,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{
        0,
        sendbuf == MPI_IN_PLACE ? vector_bytes(recvcounts, recvtype, comm) :
                                  vector_bytes(sendcounts, sendtype, comm)};
    },
    [&]() {
      return PMPI_Alltoallv(sendbuf,
                            sendcounts,
                            sdispls,
                            sendtype,
                            recvbuf,
                            recvcounts,
                            rdispls,
                            recvtype,
                            comm);
    });
}
extern "C" int
MPI_Gather(const void  *sendbuf,
           int          sendcount,
           MPI_Datatype sendtype,
           void        *recvbuf,
           int          recvcount,
           MPI_Datatype recvtype,
           int          root,
           MPI_Comm     comm)
{
  return observe(
    Function::Gather,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{
        0,
        sendbuf == MPI_IN_PLACE ? bytes(recvcount, recvtype) :
                                  bytes(sendcount, sendtype)};
    },
    [&]() {
      return PMPI_Gather(
        sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
    });
}
extern "C" int
MPI_Gatherv(const void  *sendbuf,
            int          sendcount,
            MPI_Datatype sendtype,
            void        *recvbuf,
            const int    recvcounts[],
            const int    displs[],
            MPI_Datatype recvtype,
            int          root,
            MPI_Comm     comm)
{
  return observe(
    Function::Gatherv,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{
        0,
        sendbuf == MPI_IN_PLACE ? bytes(recvcounts[rank(comm)], recvtype) :
                                  bytes(sendcount, sendtype)};
    },
    [&]() {
      return PMPI_Gatherv(sendbuf,
                          sendcount,
                          sendtype,
                          recvbuf,
                          recvcounts,
                          displs,
                          recvtype,
                          root,
                          comm);
    });
}
extern "C" int
MPI_Scatter(const void  *sendbuf,
            int          sendcount,
            MPI_Datatype sendtype,
            void        *recvbuf,
            int          recvcount,
            MPI_Datatype recvtype,
            int          root,
            MPI_Comm     comm)
{
  return observe(
    Function::Scatter,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{
        0, rank(comm) == root ? bytes(sendcount, sendtype) * size(comm) : 0};
    },
    [&]() {
      return PMPI_Scatter(
        sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
    });
}
extern "C" int
MPI_Scatterv(const void  *sendbuf,
             const int    sendcounts[],
             const int    displs[],
             MPI_Datatype sendtype,
             void        *recvbuf,
             int          recvcount,
             MPI_Datatype recvtype,
             int          root,
             MPI_Comm     comm)
{
  return observe(
    Function::Scatterv,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{
        0, rank(comm) == root ? vector_bytes(sendcounts, sendtype, comm) : 0};
    },
    [&]() {
      return PMPI_Scatterv(sendbuf,
                           sendcounts,
                           displs,
                           sendtype,
                           recvbuf,
                           recvcount,
                           recvtype,
                           root,
                           comm);
    });
}
extern "C" int
MPI_Barrier(MPI_Comm comm)
{
  return observe(
    Function::Barrier,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{0, 0};
    },
    [&]() { return PMPI_Barrier(comm); });
}
extern "C" int
MPI_Ibarrier(MPI_Comm comm, MPI_Request *request)
{
  return observe(
    Function::Ibarrier,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{0, 0};
    },
    [&]() { return PMPI_Ibarrier(comm, request); });
}
extern "C" int
MPI_Wait(MPI_Request *request, MPI_Status *status)
{
  return observe(
    Function::Wait,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{0, 0};
    },
    [&]() { return PMPI_Wait(request, status); });
}
extern "C" int
MPI_Waitall(int count, MPI_Request requests[], MPI_Status statuses[])
{
  return observe(
    Function::Waitall,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{0, 0};
    },
    [&]() { return PMPI_Waitall(count, requests, statuses); });
}
extern "C" int
MPI_Waitany(int count, MPI_Request requests[], int *index, MPI_Status *status)
{
  return observe(
    Function::Waitany,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{0, 0};
    },
    [&]() { return PMPI_Waitany(count, requests, index, status); });
}
extern "C" int
MPI_Test(MPI_Request *request, int *flag, MPI_Status *status)
{
  return observe(
    Function::Test,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{0, 0};
    },
    [&]() { return PMPI_Test(request, flag, status); });
}
extern "C" int
MPI_Testall(int count, MPI_Request requests[], int *flag, MPI_Status statuses[])
{
  return observe(
    Function::Testall,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{0, 0};
    },
    [&]() { return PMPI_Testall(count, requests, flag, statuses); });
}
extern "C" int
MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status *status)
{
  return observe(
    Function::Probe,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{0, 0};
    },
    [&]() { return PMPI_Probe(source, tag, comm, status); });
}
extern "C" int
MPI_Iprobe(int source, int tag, MPI_Comm comm, int *flag, MPI_Status *status)
{
  return observe(
    Function::Iprobe,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{0, 0};
    },
    [&]() { return PMPI_Iprobe(source, tag, comm, flag, status); });
}
extern "C" int
MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm)
{
  return observe(
    Function::Comm_dup,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{0, 0};
    },
    [&]() { return PMPI_Comm_dup(comm, newcomm); });
}
extern "C" int
MPI_Comm_free(MPI_Comm *comm)
{
  return observe(
    Function::Comm_free,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{0, 0};
    },
    [&]() { return PMPI_Comm_free(comm); });
}
extern "C" int
MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm)
{
  return observe(
    Function::Comm_split,
    [&]() {
      return std::pair<std::uint64_t, std::uint64_t>{0, 0};
    },
    [&]() { return PMPI_Comm_split(comm, color, key, newcomm); });
}
