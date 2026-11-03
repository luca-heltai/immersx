// Shared only by the GoogleTest driver and its application launcher.
#pragma once

#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>

namespace ImmersX::TestWatchdog
{
  // The same mutex protects deadline changes, process-group registration, and
  // reaping. The watchdog therefore never signals a recycled process ID.
  class Watchdog
  {
  public:
    Watchdog()
      : worker([this]() { watch(); })
    {}

    ~Watchdog()
    {
      {
        std::lock_guard<std::mutex> lock(mutex);
        stopping = true;
        changed.notify_one();
      }
      worker.join();
    }

    void
    arm(const std::string &test)
    {
      std::lock_guard<std::mutex> lock(mutex);
      message = "GoogleTest " + test +
                " exceeded the 115 second correctness-test budget.\n";
      deadline = std::chrono::steady_clock::now() + std::chrono::seconds(115);
      armed    = true;
      ++generation;
      changed.notify_one();
    }

    void
    disarm()
    {
      std::lock_guard<std::mutex> lock(mutex);
      armed = false;
      ++generation;
      changed.notify_one();
    }

    std::mutex mutex;
    pid_t      owned_application_group = 0;

  private:
    void
    watch()
    {
      std::unique_lock<std::mutex> lock(mutex);
      while (!stopping)
        {
          if (!armed)
            {
              changed.wait(lock, [this]() { return stopping || armed; });
              continue;
            }
          const auto observed_generation = generation;
          if (changed.wait_until(lock, deadline, [this, observed_generation]() {
                return stopping || generation != observed_generation;
              }))
            continue;
          if (owned_application_group > 0)
            ::kill(-owned_application_group, SIGKILL);
          const auto written =
            ::write(STDERR_FILENO, message.data(), message.size());
          (void)written;
          std::_Exit(EXIT_FAILURE);
        }
    }

    std::condition_variable               changed;
    std::chrono::steady_clock::time_point deadline;
    std::string                           message;
    unsigned long                         generation = 0;
    bool                                  armed      = false;
    bool                                  stopping   = false;
    std::thread                           worker;
  };

  inline Watchdog &
  instance()
  {
    static Watchdog watchdog;
    return watchdog;
  }
  // Isolate the owned application and its descendants from the GoogleTest/MPI
  // process group. Registration and reaping share the watchdog's mutex.
  inline int
  run_owned_application(const std::string &command)
  {
    auto       &watchdog      = instance();
    const char *shell_command = command.c_str();
    pid_t       child;
    {
      std::lock_guard<std::mutex> lock(watchdog.mutex);
      child = ::fork();
      if (child == 0)
        {
          // Only async-signal-safe operations are allowed after fork in this
          // multithreaded process. Never touch the inherited locked mutex.
          if (::setpgid(0, 0) != 0)
            std::_Exit(127);
          ::execl(
            "/bin/sh", "sh", "-c", shell_command, static_cast<char *>(nullptr));
          std::_Exit(127);
        }
      if (child < 0)
        return -1;
      // The child also creates its group, so EACCES after exec is harmless if
      // its actual process group is already the expected child-owned group.
      if (::setpgid(child, child) != 0 && ::getpgid(child) != child)
        {
          ::kill(child, SIGKILL);
          int status;
          while (::waitpid(child, &status, 0) < 0 && errno == EINTR)
            {
            }
          return -1;
        }
      watchdog.owned_application_group = child;
    }
    for (;;)
      {
        {
          std::lock_guard<std::mutex> lock(watchdog.mutex);
          int                         status;
          const pid_t result = ::waitpid(child, &status, WNOHANG);
          if (result == child || (result < 0 && errno != EINTR))
            {
              watchdog.owned_application_group = 0;
              return result == child ? status : -1;
            }
        }
        const timespec pause{0, 10000000};
        ::nanosleep(&pause, nullptr);
      }
  }

} // namespace ImmersX::TestWatchdog
