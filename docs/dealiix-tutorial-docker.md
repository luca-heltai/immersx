# dealiiX tutorial Docker stack

This stack is defined in the ImmersX repository. It runs Coral with the ImmersX
plugins, Coral Visualizer, and the dealiiX-platform Electron UI through noVNC.

## Start

```bash
./scripts/tutorial-init.sh
docker compose pull
docker compose up -d
```

Open <http://localhost:6080> and use the platform in the browser. The visualizer
is also available directly at <http://localhost:8008>.

The default platform configuration uses the 2D ImmersX plugin and connects to
`coral-ssh-slurm` over the internal Compose network. The Coral SSH port is also
published as `localhost:2222` for debugging.

## Stop

```bash
docker compose down
```

The mutable `dealiix-tutorial` images are rebuilt from the corresponding
`dealiix-tutorial` branches by GitHub Actions.
