#!/usr/bin/env bash
set -euo pipefail

bindir=$(cd -- "$(dirname -- "$0")" && pwd -P)

usage() {
  cat <<'USAGE'
ImmersX: run a precompiled application

Usage: immersx [APPLICATION [ARGUMENTS...]]
       immersx --help

Pass an application name followed by its arguments, usually a parameter file:
  docker run --rm -v "$PWD:/work" IMAGE poisson input.prm
  podman run --rm -v "$PWD:/work" IMAGE poisson input.prm
  apptainer run --bind "$PWD:/work" --pwd /work docker://IMAGE poisson input.prm

Inputs and outputs belong in the mounted working directory. With no arguments,
show this help. The standalone run_container.sh launcher also supports MPI.
Optional MetricFlowX, Coral, and legacy 1dsolver integrations are not included.

Installed applications:
USAGE
  for executable in "$bindir"/*; do
    [[ -f $executable && -x $executable && ${executable##*/} != immersx ]] || continue
    printf '  %s\n' "${executable##*/}"
  done
}

if [[ $# -eq 0 ]]; then
  usage
  exit 0
fi
case $1 in
  --help|-h|help) usage; exit 0 ;;
esac
application=$1
shift
if [[ $application == */* || $application == immersx || ! -f $bindir/$application || ! -x $bindir/$application ]]; then
  printf 'Unknown application: %s\nRun without arguments to list installed applications.\n' "$application" >&2
  exit 127
fi
exec "$bindir/$application" "$@"
