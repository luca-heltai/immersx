#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
SITE_DIR="${ROOT_DIR}/build/docs/site"
PORT="${1:-8000}"

if [ ! -d "${SITE_DIR}" ]; then
  echo "Documentation site not found in ${SITE_DIR}." >&2
  echo "Run ./scripts/build_doc.sh first." >&2
  exit 1
fi

cd "${SITE_DIR}"
echo "Serving documentation at http://127.0.0.1:${PORT}"
python3 -m http.server "${PORT}"
