#!/usr/bin/env bash
# Standalone launcher: no checkout or local ImmersX installation is needed.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_container.sh [OPTIONS] [APPLICATION [ARGUMENTS...]]
       run_container.sh [OPTIONS] --exec COMMAND [ARGUMENTS...]

  --runtime NAME  auto (default), docker, podman, or apptainer
  --image REF     OCI image tag or digest (IMMERSX_IMAGE overrides default)
  --ranks N       MPI ranks inside the container, on one node (default: 1)
  --workdir DIR   Host directory mounted at /work (default: current directory)
  --exec COMMAND  Run another installed program, e.g. bash or cmake
  --help          Show launcher help (use -- --help for image help)

Inputs and outputs must be under --workdir; use paths relative to this directory
inside parameter files, or absolute paths below --workdir. A runtime must already
be installed. No compiler is needed. With no application, show image help.
USAGE
}
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

runtime=auto
image=${IMMERSX_IMAGE:-ghcr.io/luca-heltai/immersx:master}
ranks=1
workdir=$PWD
program=/opt/immersx/bin/immersx
while [[ $# -gt 0 ]]; do
  case $1 in
    --help|-h) usage; exit 0 ;;
    --runtime|--image|--ranks|--workdir)
      [[ $# -ge 2 ]] || fail "Missing value for $1"
      case $1 in
        --runtime) runtime=$2 ;;
        --image) image=$2 ;;
        --ranks) ranks=$2 ;;
        --workdir) workdir=$2 ;;
      esac
      shift 2 ;;
    --exec)
      [[ $# -ge 2 ]] || fail 'Missing command after --exec'
      program=$2
      shift 2
      break ;;
    --) shift; break ;;
    -*) fail "Unknown option: $1" ;;
    *) break ;;
  esac
done
[[ $ranks =~ ^[1-9][0-9]*$ ]] || fail '--ranks must be a positive integer'
[[ -d $workdir ]] || fail "Work directory does not exist: $workdir"
workdir=$(cd -- "$workdir" && pwd -P)
case $workdir in
  *:*|*,*|*$'\n'*) fail 'Work directory cannot contain colons, commas, or newlines' ;;
esac
case $runtime in
  auto)
    runtime=
    for candidate in docker podman apptainer; do
      if command -v "$candidate" >/dev/null 2>&1; then runtime=$candidate; break; fi
    done
    [[ -n $runtime ]] || fail 'Install or load Docker, Podman, or Apptainer first' ;;
  docker|podman|apptainer) command -v "$runtime" >/dev/null 2>&1 || fail "Runtime unavailable: $runtime" ;;
  *) fail "Unsupported runtime: $runtime" ;;
esac
command=("$program" "$@")
if [[ $ranks -gt 1 ]]; then
  command=(mpirun --oversubscribe -n "$ranks" "${command[@]}")
fi
printf 'Running %s with %s (%s MPI rank(s))\n' "$image" "$runtime" "$ranks" >&2
case $runtime in
  docker|podman)
    options=(run --rm --user "$(id -u):$(id -g)" --workdir /work
      --mount "type=bind,source=$workdir,target=/work"
      --mount "type=bind,source=$workdir,target=$workdir"
      --env HOME=/tmp --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1
      --env OMPI_ALLOW_RUN_AS_ROOT=1 --env OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1)
    if [[ $runtime == podman && $(id -u) != 0 ]]; then options+=(--userns=keep-id); fi
    if [[ -t 0 && -t 1 ]]; then options+=(--interactive --tty); fi
    exec "$runtime" "${options[@]}" --entrypoint "${command[0]}" "$image" "${command[@]:1}" ;;
  apptainer)
    case $image in
      docker://*|oras://*|library://*|*.sif) apptainer_image=$image ;;
      *) apptainer_image="docker://$image" ;;
    esac
    exec apptainer exec --cleanenv --bind "$workdir:/work" --bind "$workdir:$workdir" --pwd /work \
      "$apptainer_image" "${command[@]}" ;;
esac
