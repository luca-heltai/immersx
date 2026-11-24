#!/usr/bin/env bash
set -euo pipefail

image=$1
tag=$2
revision=$3
repository=${image#ghcr.io/}
scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
status=$(curl --silent --show-error --retry 2 --connect-timeout 20 --max-time 60 \
  -o "$scratch/token.json" -w '%{http_code}' \
  "https://ghcr.io/token?service=ghcr.io&scope=repository:$repository:pull")
case "$status" in
  403|404) exit 0 ;;
  200) ;;
  *) printf 'Cannot resolve registry token (HTTP %s).\n' "$status" >&2; exit 1 ;;
esac
token=$(jq -er .token "$scratch/token.json")
status=$(curl --silent --show-error --retry 2 --connect-timeout 20 --max-time 60 \
  -H "Authorization: Bearer $token" \
  -H 'Accept: application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json' \
  -o "$scratch/manifest.json" -D "$scratch/headers" -w '%{http_code}' \
  "https://ghcr.io/v2/$repository/manifests/$tag")
case "$status" in
  404) exit 0 ;;
  200) ;;
  *) printf 'Cannot verify %s:%s (HTTP %s).\n' "$image" "$tag" "$status" >&2; exit 1 ;;
esac
jq -e '[.manifests[].platform | select(.os == "linux") | .architecture] |
  (index("amd64") != null and index("arm64") != null)' "$scratch/manifest.json" >/dev/null
# A retry can reuse a published version for this Git revision, never a development tag.
for arch in amd64 arm64; do
  manifest=$(jq -er --arg arch "$arch" \
    '.manifests[] | select(.platform.os == "linux" and .platform.architecture == $arch) | .digest' \
    "$scratch/manifest.json")
  config=$(curl --fail --silent --show-error --retry 2 --connect-timeout 20 --max-time 60 \
    -H "Authorization: Bearer $token" \
    -H 'Accept: application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json' \
    "https://ghcr.io/v2/$repository/manifests/$manifest" | jq -er .config.digest)
  built_revision=$(curl --location --fail --silent --show-error --retry 2 --connect-timeout 20 --max-time 60 \
    -H "Authorization: Bearer $token" \
    "https://ghcr.io/v2/$repository/blobs/$config" | jq -er '.config.Labels["org.opencontainers.image.revision"]')
  if [[ "$built_revision" != "$revision" ]]; then
    printf 'Published %s:%s is for revision %s.\n' "$image" "$tag" "$built_revision" >&2
    exit 1
  fi
done
digest=$(tr -d '\r' < "$scratch/headers" | awk 'tolower($1) == "docker-content-digest:" {print $2}')
[[ "$digest" =~ ^sha256:[0-9a-f]{64}$ ]]
printf '%s\n' "$digest"
