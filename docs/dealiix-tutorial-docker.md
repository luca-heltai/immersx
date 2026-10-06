# dealiiX tutorial Docker stack

This stack is defined in the ImmersX repository. It runs Coral with the ImmersX
plugins and MetricFlowX, Coral Visualizer, and the dealiiX-platform Electron UI
through noVNC. Published images are available for both Linux AMD64 and ARM64.

## Start

```bash
./scripts/tutorial-init.sh
docker compose pull
docker compose up -d
```

Open <http://localhost:6080/vnc.html?autoconnect=1&resize=scale> and use the
platform in the browser. The container runs the production Electron build
through noVNC, not `npm run dev:vite`; the latter is only the renderer and does
not provide the Electron SSH/filesystem bridge. The visualizer is also available directly at <http://localhost:8008>.

The default platform configuration uses the 2D ImmersX plugin and connects to
`coral-ssh-slurm` over the internal Compose network. The Coral SSH port is also
published as `localhost:2222` for debugging. The compute image installs
MetricFlowX under `/opt/dealiix/metric-flow-x` and exposes the corresponding
ImmersX support when the graph uses it.

The latest mutable images are pulled by running:

```bash
git pull
docker compose pull
docker compose up -d
```

## Stop

```bash
docker compose down
```

The mutable `dealiix-tutorial` images are rebuilt by GitHub Actions from the
ImmersX `dealiix-tutorial` branch, the Coral and visualizer `dealiix-tutorial`
branches, the dealiiX-platform `dealiix-tutorial` branch, and the
MetricFlowX `dealiix-tutorial` branch.
