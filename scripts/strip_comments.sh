#!/bin/sh
set -eu

usage() {
  cat <<'EOF'
Usage:
  strip_comments.sh [-i] <file>
  strip_comments.sh -i <file1> [file2 ...]

Removes shell-style '#' comments:
  - whole-line comments (optionally preceded by whitespace)
  - trailing comments after content

Notes:
  - A '\#' is treated as a literal '#'.
  - Without -i, writes the stripped file to stdout.
EOF
}

in_place=0
case "${1-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac

while [ $# -gt 0 ]; do
  case "$1" in
    -i|--in-place) in_place=1; shift ;;
    --) shift; break ;;
    -*) echo "strip_comments.sh: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) break ;;
  esac
done

if [ $# -lt 1 ]; then
  usage >&2
  exit 2
fi

if [ "$in_place" -eq 0 ] && [ $# -ne 1 ]; then
  echo "strip_comments.sh: without -i, pass exactly one file" >&2
  usage >&2
  exit 2
fi

awk_program='
function strip_comment(line,    i, c, prev, out) {
  out = "";
  prev = "";
  for (i = 1; i <= length(line); i++) {
    c = substr(line, i, 1);
    if (c == "#" && prev != "\\") {
      break;
    }
    out = out c;
    prev = c;
  }
  sub(/[[:space:]]+$/, "", out);
  return out;
}

{
  if ($0 ~ /^[[:space:]]*#/) {
    next;
  }
  print strip_comment($0);
}
'

strip_file_to_stdout() {
  awk "$awk_program" "$1"
}

strip_file_in_place() {
  file="$1"
  tmp="${file}.tmp.$$"
  strip_file_to_stdout "$file" > "$tmp"
  mv "$tmp" "$file"
}

if [ "$in_place" -eq 1 ]; then
  for f in "$@"; do
    strip_file_in_place "$f"
  done
else
  strip_file_to_stdout "$1"
fi
