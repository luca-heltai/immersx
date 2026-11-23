#!/usr/bin/env python3
"""Resolve the highest stable, public GHCR tag to a multiarch index."""

import json
import re
import sys
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.request import Request, urlopen


REGISTRY = "https://ghcr.io"
STABLE = re.compile(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
INDEX_TYPES = (
    "application/vnd.oci.image.index.v1+json, "
    "application/vnd.docker.distribution.manifest.list.v2+json"
)


def get_json(url, headers=None):
    with urlopen(Request(url, headers=headers or {}), timeout=30) as response:
        return json.load(response), response.headers


def resolve(repository):
    query = urlencode({"service": "ghcr.io", "scope": f"repository:{repository}:pull"})
    token, _ = get_json(f"{REGISTRY}/token?{query}")
    auth = {"Authorization": f"Bearer {token['token']}"}

    path = f"/v2/{repository}/tags/list"
    url = f"{REGISTRY}{path}?n=100"
    seen = set()
    releases = []
    while url:
        if url in seen:
            raise ValueError("Repeated registry pagination link")
        seen.add(url)
        page, headers = get_json(url, auth)
        for tag in page["tags"] or []:
            match = STABLE.fullmatch(tag)
            if match:
                releases.append((tuple(map(int, match.groups())), tag))

        next_link = re.search(r'<([^>]+)>\s*;\s*rel="?next"?', headers.get("Link", ""))
        url = urljoin(url, next_link[1]) if next_link else ""
        if url:
            next_url = urlsplit(url)
            if (next_url.scheme, next_url.netloc, next_url.path) != ("https", "ghcr.io", path):
                raise ValueError("Unexpected registry pagination link")

    if not releases:
        raise ValueError("no published stable vX.Y.Z tag")
    _, tag = max(releases)
    index, headers = get_json(
        f"{REGISTRY}/v2/{repository}/manifests/{tag}",
        {**auth, "Accept": INDEX_TYPES},
    )
    platforms = {
        (descriptor.get("platform", {}).get("os"),
         descriptor.get("platform", {}).get("architecture"))
        for descriptor in index.get("manifests", [])
    }
    if not {("linux", "amd64"), ("linux", "arm64")} <= platforms:
        raise ValueError(f"{tag} does not include Linux AMD64 and ARM64")
    digest = headers.get("Docker-Content-Digest", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError(f"{tag} has no valid manifest digest")
    return f"ghcr.io/{repository}:{tag}@{digest}"


if __name__ == "__main__":
    try:
        print(resolve(sys.argv[1]))
    except (OSError, ValueError, KeyError) as error:
        sys.exit(f"Cannot resolve a stable container release for {sys.argv[1]}: {error}")
