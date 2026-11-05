#!/usr/bin/env python3
"""Plan, measure, archive and export performance history (standard library)."""
import argparse
import datetime
import hashlib
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tempfile

from history import (SCHEMA, expected_cases, archive_attempt, attempts, dumps, load,
                     parse_jsonl, pending, report)

HERE = Path(__file__).resolve().parent


def command(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def git(repo, *args):
    return command(["git", "-C", str(repo), *args])


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(data))


def is_ancestor(repo, ancestor, descendant):
    result = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                             ancestor, descendant], capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise ValueError("cannot verify source history: " + result.stderr.strip())
    return result.returncode == 0


def ancestry(repo, baseline, tip):
    chain = git(repo, "rev-list", "--first-parent", "--reverse", tip).splitlines()
    if baseline in chain:
        return chain[chain.index(baseline):]
    if not is_ancestor(repo, baseline, tip):
        raise ValueError("baseline is not an ancestor of the source revision")
    # A fast-forward can incorporate the previous master as merge parent two.
    # Keep its activation, excluding feature commits from before activation.
    descendants = set(git(repo, "rev-list", "--ancestry-path",
                          f"{baseline}..{tip}").splitlines())
    return [baseline, *(sha for sha in chain if sha in descendants)]


def merge_plan(old, new, repo=None):
    if old["baseline"] != new["baseline"]:
        raise ValueError("baseline conflict")
    if repo is None:
        # Legacy synthetic plans have no Git graph with which to prove ancestry.
        if new["commits"][:len(old["commits"])] == old["commits"]:
            return new
        if old["commits"][:len(new["commits"])] == new["commits"]:
            return old
        raise ValueError("non-fast-forward master history; manual review required")
    if old["source_sha"] == new["source_sha"] and old["commits"] == new["commits"]:
        return new
    if not is_ancestor(repo, old["source_sha"], new["source_sha"]):
        if is_ancestor(repo, new["source_sha"], old["source_sha"]):
            # A late run can have observed a different first-parent path.
            # Keep that coverage too, without moving the current source back.
            commits = list(dict.fromkeys([*old["commits"], *new["commits"]]))
            commits.remove(old["source_sha"])
            commits.append(old["source_sha"])
            return {**old, "commits": commits}
        raise ValueError("non-fast-forward master history; manual review required")
    # First-parent coverage can change without rewriting the Git history.
    # Retain every monitored revision, with this source remaining the endpoint.
    commits = list(dict.fromkeys([*old["commits"], *new["commits"]]))
    commits.remove(new["source_sha"])
    commits.append(new["source_sha"])
    return {**new, "commits": commits}


def plan(args):
    sha = git(args.repo, "rev-parse", args.ref)
    if args.preview:
        baseline, commits = sha, [sha]
    else:
        state = Path(args.archive) / "baseline.json"
        if state.exists():
            baseline = load(state)["sha"]
        else:
            # The activation commit, not CP8's old report SHA or all of Git history.
            activation = git(args.repo, "log", "--first-parent", "--reverse",
                             "--diff-filter=A", "--format=%H", sha, "--",
                             "scripts/performance/config.json").splitlines()
            if not activation:
                raise ValueError("performance monitoring has not been activated")
            baseline = activation[0]
        commits = ancestry(args.repo, baseline, sha)
        ledger = Path(args.archive) / "plan.json"
        if ledger.exists():
            combined = merge_plan(load(ledger), {"source_sha": sha,
                "baseline": baseline, "commits": commits}, args.repo)
            # A late run still measures its own source. Publication preserves
            # the newer ledger and appends its immutable attempts separately.
            if combined["source_sha"] == sha:
                commits = combined["commits"]
    todo = pending(commits, attempts(args.archive), args.retry)
    if sha in todo:
        todo.remove(sha)
        todo.insert(0, sha)
    if args.commit:
        if args.commit not in commits:
            raise ValueError("requested SHA is outside monitored history")
        current_pending = sha in todo
        todo = [args.commit]
        if current_pending and args.commit != sha:
            todo.insert(0, sha)
        if len(todo) > args.limit:
            raise ValueError("batch limit is too small for the current source and requested commit")
    selected = todo[:args.limit]
    config = load(HERE / "config.json")
    save(args.output, {"schema_version": SCHEMA, "source_sha": sha,
                      "harness_sha": git(args.repo, "rev-parse", "HEAD"),
                      "baseline": baseline, "commits": commits,
                      "selected": selected, "configuration": config})


def capture(args, output, cwd=None, timeout=1800, env=None):
    try:
        p = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout, env=env)
        output.write_text(p.stdout)
        return p.returncode
    except subprocess.TimeoutExpired as exc:
        content = exc.stdout or b""
        output.write_text((content.decode(errors="replace") if isinstance(content, bytes) else content)
                          + "\nHarness timeout\n")
        return 124


def cpu():
    if Path("/proc/cpuinfo").exists():
        return next((line.split(":", 1)[1].strip() for line in
                     Path("/proc/cpuinfo").read_text().splitlines()
                     if line.startswith("model name")), platform.machine())
    return command(["sysctl", "-n", "machdep.cpu.brand_string"])


def compiler_cache(source, build):
    """Normalize historical checkouts too, even before their own cache launcher."""
    if not shutil.which("ccache"):
        raise ValueError("ccache is required for benchmark compilation")
    directory = Path(os.environ.get("CCACHE_DIR", Path.home() / ".ccache")).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CCACHE_DIR=str(directory), CCACHE_BASEDIR=str(source),
               CCACHE_NOHASHDIR="true")
    disabled = subprocess.check_output(["ccache", "--get-config", "disable"],
                                       env=env, text=True).strip()
    if disabled != "false":
        raise ValueError("ccache is disabled in the effective build configuration")
    mappings = [f"-{flag}-prefix-map={path}=/immersx/{name}"
                for path, name in ((source, "source"), (build, "build"))
                for flag in ("ffile", "fdebug", "fmacro")]
    flags = ["-DIMMERSX_USE_CCACHE=ON", "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
             "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
             "-DCMAKE_CXX_FLAGS=" + " ".join(filter(None, [os.environ.get("CXXFLAGS", ""),
                 *(shlex.quote(mapping) for mapping in mappings)]))]
    return flags, env


def measure(args):
    p = load(args.plan)
    config = p["configuration"]
    run = os.environ.get("GITHUB_RUN_ID", "local")
    attempt_no = int(os.environ.get("GITHUB_RUN_ATTEMPT", "1"))
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    local_id = hashlib.sha256(started.encode()).hexdigest()[:12] if run == "local" else run
    for sha in p["selected"]:
        ident = f"{sha}-{local_id}-{attempt_no}"
        existing = Path(args.output) / "attempts" / (ident + ".json")
        if existing.exists():
            print(ident, "already recorded", flush=True)
            continue
        out = Path(args.output).resolve() / "raw" / ident
        out.mkdir(parents=True, exist_ok=True)
        result = {"schema_version": SCHEMA, "id": ident, "sha": sha,
                  "harness_sha": p["harness_sha"], "harness_version": config["harness_version"],
                  "run": run, "attempt": attempt_no, "started_at": started,
                  "run_url": f"https://github.com/{os.environ.get('GITHUB_REPOSITORY', 'luca-heltai/immersx')}/actions/runs/{run}",
                  "configuration": config,
                  "environment": {"id": args.environment, "image": args.image,
                      "diagnostic": not args.stable, "cpu": cpu(), "architecture": platform.machine(),
                      "os": platform.platform(), "threads": 1,
                      "rank_counts": config.get("particle_penalty_ranks", [1])},
                  "records": [], "errors": [],
                  "status": "failed", "raw_path": f"raw/{ident}/"}
        with tempfile.TemporaryDirectory(prefix="immersx-benchmark-") as tmp:
            source = Path(tmp) / "source"
            # Keep generated include paths under the launcher's CCACHE_BASEDIR.
            build = source / "build-performance"
            try:
                result["benchmark_sources"] = {
                    name: git(args.repo, "rev-parse", f"{sha}:benchmarks/{name}.cc")
                    for name in ("weak_term_assembly", "lifted_integration_support", "particle_penalty")
                    if git(args.repo, "ls-tree", sha, f"benchmarks/{name}.cc")}
                if "particle_penalty" in result["benchmark_sources"]:
                    for key, path in (("particle_penalty_native", "benchmarks/native_particle_penalty.h"),
                                      ("particle_penalty_coupling", "include/immersx/compatibility/dealii_9_9/dof_handler_coupling.h")):
                        result["benchmark_sources"][key] = git(args.repo, "rev-parse", f"{sha}:{path}")
                git(args.repo, "worktree", "add", "--detach", str(source), sha)
                cache_flags, build_env = compiler_cache(source, build)
                flags = ["-DCMAKE_BUILD_TYPE=" + config["build_type"],
                         "-DIMMERSX_BUILD_BENCHMARKS=ON", "-DENABLE_COUPLED_PROBLEMS=OFF",
                         "-DENABLE_GOOGLE_TESTING=OFF", "-DENABLE_DEAL_II_APP_TESTING=OFF",
                         "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", *cache_flags]
                capture(["ccache", "--show-config"], out / "ccache-config.log", env=build_env)
                capture(["ccache", "--show-stats"], out / "ccache-before.log", env=build_env)
                rc = capture(["cmake", "-S", str(source), "-B", str(build), *flags],
                             out / "configure.log", env=build_env)
                if rc:
                    raise ValueError(f"configure failed ({rc})")
                cache = (build / "CMakeCache.txt").read_text()
                (out / "CMakeCache.txt").write_text(cache)
                compiler = next(line.split("=", 1)[1] for line in cache.splitlines()
                                if line.startswith("CMAKE_CXX_COMPILER:FILEPATH="))
                compile_commands = load(build / "compile_commands.json")
                benchmark_commands = [e["command"].replace(str(build), "/build").replace(str(source), "/source")
                                      for e in compile_commands if "/benchmarks/" in e["file"]]
                save(out / "compile_commands.json", compile_commands)
                result["environment"].update({"compiler": command([compiler, "--version"]),
                    "compile_commands": benchmark_commands,
                    "cmake_options": [flag.replace(str(build), "/build").replace(str(source), "/source")
                                      for flag in flags],
                    "threads": 1, "rank_counts": config.get("particle_penalty_ranks", [1]),
                    "deal_ii_config": next((s for s in cache.splitlines() if s.startswith("deal.II_DIR:")), "unknown")})
                suffix = "_debug" if config["build_type"] == "Debug" else ""
                targets = [name + "_benchmark" + suffix for name in
                           ("weak_term_assembly", "lifted_integration_support", "particle_penalty")]
                for target in targets:
                    rc = capture(["cmake", "--build", str(build), "--target", target,
                                  "--verbose", "-j2"], out / (target + "-build.log"), env=build_env)
                    if rc:
                        result["errors"].append({"reason": f"{target} build failed ({rc})"})
                        continue
                    rank_counts = config.get("particle_penalty_ranks", [1]) if target.startswith("particle_penalty_") else [1]
                    for ranks in rank_counts:
                        stem = target + (f"-mpi{ranks}" if ranks > 1 else "")
                        cmd = [str(build / "benchmarks" / target),
                               f"--refinements={config['refinements']}", f"--repetitions={config['repetitions']}"]
                        if ranks > 1:
                            launcher = shutil.which("mpirun") or shutil.which("mpiexec")
                            if not launcher:
                                raise ValueError("MPI launcher unavailable")
                            result["environment"]["mpi_launcher_version"] = command([launcher, "--version"])
                            cmd = [launcher, "-np", str(ranks), *cmd]
                        env = dict(os.environ, OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
                        env.update({name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                           "MKL_NUM_THREADS", "DEAL_II_NUM_THREADS")})
                        stdout, stderr = out / (stem + ".jsonl"), out / (stem + ".stderr")
                        with stdout.open("w") as so, stderr.open("w") as se:
                            try:
                                rc = subprocess.run(cmd, stdout=so, stderr=se, cwd=tmp, env=env, timeout=600).returncode
                            except subprocess.TimeoutExpired:
                                rc = 124
                        records, errors = parse_jsonl(stdout.read_text(), config)
                        if any(r["ranks"] != ranks for r in records):
                            errors.append({"reason": "reported rank count differs from MPI launch"})
                        result["records"].extend(records)
                        result["errors"].extend({**error, "source": stem + ".jsonl"} for error in errors)
                        if rc:
                            result["errors"].append({"reason": f"{stem} execution failed ({rc})"})
                versions = sorted({r["raw"]["deal_ii_version"] for r in result["records"]
                                   if "deal_ii_version" in r["raw"]})
                result["environment"]["deal_ii_version"] = versions
                if not versions:
                    result["errors"].append({"reason": "deal.II version unavailable"})
                capture(["ccache", "--show-stats"], out / "ccache-after.log", env=build_env)
            except (ValueError, OSError, subprocess.CalledProcessError, StopIteration) as exc:
                result["errors"].append({"reason": str(exc)})
            finally:
                if source.exists():
                    git(args.repo, "worktree", "remove", "--force", str(source))
        present = {(r["case"], r["dimension"], r["ranks"]) for r in result["records"]}
        result["missing_cases"] = [list(k) for k in expected_cases(config) if k not in present]
        if not result["errors"] and not result["missing_cases"] and all(r["status"] == "passed" for r in result["records"]):
            result["status"] = "passed"
        result["raw_files"] = sorted(path.name for path in out.iterdir() if path.is_file())
        result["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        archive_attempt(args.output, result)
        print(ident, result["status"], flush=True)


def export(args):
    p = load(args.plan) if args.plan else {"source_sha": None, "baseline": None, "commits": []}
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    save(destination / "history.json", report(p, attempts(args.archive)))
    for name in ("raw", "attempts"):
        source = Path(args.archive) / name
        if source.exists():
            shutil.copytree(source, destination / name, dirs_exist_ok=True)


def merge(args):
    p = load(args.plan)
    state = Path(args.archive) / "baseline.json"
    if state.exists() and load(state)["sha"] != p["baseline"]:
        raise ValueError("baseline conflict")
    ledger = Path(args.archive) / "plan.json"
    combined = merge_plan(load(ledger), p, getattr(args, "repo", None)) if ledger.exists() else p
    for a in attempts(args.incoming):
        archive_attempt(args.archive, a)
    source = Path(args.incoming) / "raw"
    if source.exists():
        for path in source.rglob("*"):
            if path.is_file():
                dest = Path(args.archive) / path.relative_to(args.incoming)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists() and dest.read_bytes() != path.read_bytes():
                    raise ValueError("conflicting raw output " + str(dest))
                shutil.copyfile(path, dest)
    save(state, {"sha": p["baseline"], "policy": "first-parent activation, inclusive"})
    save(ledger, combined)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    q = sub.add_parser("plan")
    q.add_argument("--repo", default=".")
    q.add_argument("--ref", default="HEAD")
    q.add_argument("--archive", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--preview", action="store_true")
    q.add_argument("--retry", action="store_true")
    q.add_argument("--commit")
    q.add_argument("--limit", type=int, default=8)
    q = sub.add_parser("measure")
    q.add_argument("--repo", default=".")
    q.add_argument("--plan", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--environment", required=True)
    q.add_argument("--image", required=True)
    q.add_argument("--stable", action="store_true")
    q = sub.add_parser("merge")
    q.add_argument("--repo", help="Full source repository for Git ancestry validation")
    q.add_argument("--archive", required=True)
    q.add_argument("--incoming", required=True)
    q.add_argument("--plan", required=True)
    q = sub.add_parser("export")
    q.add_argument("--archive", required=True)
    q.add_argument("--plan")
    q.add_argument("--output", required=True)
    args = parser.parse_args()
    {"plan": plan, "measure": measure, "merge": merge, "export": export}[args.action](args)


if __name__ == "__main__":
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "DEAL_II_NUM_THREADS"):
        os.environ[name] = "1"
    main()
