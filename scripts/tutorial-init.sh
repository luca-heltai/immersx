#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SSH_DIR="$ROOT/docker/tutorial-ssh"
mkdir -p "$SSH_DIR" "$ROOT/docker/shared-data" "$ROOT/docker/platform-userdata"
chmod 700 "$SSH_DIR"

if [ ! -f "$SSH_DIR/id_ed25519" ]; then
  ssh-keygen -q -t ed25519 -N '' -f "$SSH_DIR/id_ed25519" -C dealiix-tutorial
fi
chmod 600 "$SSH_DIR/id_ed25519"
chmod 644 "$SSH_DIR/id_ed25519.pub"

echo "Tutorial stack initialized."
echo "Open http://localhost:6080 after running: docker compose pull && docker compose up -d"
