#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
VENV_DIR="${ROOT_DIR}/env"
REQUIREMENTS_FILE="${ROOT_DIR}/doc/requirements.txt"
BUILD_DIR="${ROOT_DIR}/build/docs"
DOXYGEN_OUT="${BUILD_DIR}/doxygen"
SITE_DIR="${BUILD_DIR}/site"
DOXYFILE="${ROOT_DIR}/doc/Doxyfile"
DOCS_SOURCE_DIR="${ROOT_DIR}/doc"
API_STUB_DIR="${DOCS_SOURCE_DIR}/api"

if [ -d "$HOME/anaconda3/bin" ]; then
  echo "Anaconda3 found."
  PATH="$HOME/anaconda3/bin:$PATH"
  export PATH
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found in PATH." >&2
  exit 1
fi

if ! command -v doxygen >/dev/null 2>&1; then
  echo "doxygen not found in PATH." >&2
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  python3 -m pip install --user virtualenv
  python3 -m venv "${VENV_DIR}"
fi

. "${VENV_DIR}/bin/activate"

if [ ! -f "${REQUIREMENTS_FILE}" ]; then
  echo "${REQUIREMENTS_FILE} not found." >&2
  exit 1
fi

python3 -m pip install -r "${REQUIREMENTS_FILE}"

if ! command -v sphinx-build >/dev/null 2>&1; then
  echo "sphinx-build not found after installing documentation requirements." >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}" "${DOXYGEN_OUT}" "${API_STUB_DIR}"
rm -rf "${SITE_DIR}" "${API_STUB_DIR}"
mkdir -p "${API_STUB_DIR}"

tmp_doxyfile="$(mktemp "${TMPDIR:-/tmp}/rlm-doxygen.XXXXXX")"
trap 'rm -f "${tmp_doxyfile}"' EXIT

sed \
  -e "s|^OUTPUT_DIRECTORY[[:space:]]*=.*$|OUTPUT_DIRECTORY = ${DOXYGEN_OUT}|" \
  -e "s|^GENERATE_HTML[[:space:]]*=.*$|GENERATE_HTML = NO|" \
  -e "s|^GENERATE_XML[[:space:]]*=.*$|GENERATE_XML = YES|" \
  -e "s|^HTML_OUTPUT[[:space:]]*=.*$|HTML_OUTPUT = html|" \
  -e "s|^XML_OUTPUT[[:space:]]*=.*$|XML_OUTPUT = xml|" \
  "${DOXYFILE}" > "${tmp_doxyfile}"

(
  cd "${ROOT_DIR}"
  doxygen "${tmp_doxyfile}"
)

sphinx-build \
  -b html \
  -W \
  "${DOCS_SOURCE_DIR}" \
  "${SITE_DIR}"

echo "Documentation site generated in ${SITE_DIR}"
