# dealiiX tutorial containers

The ImmersX tutorial backend includes the Debug and Release Coral CLI and
ImmersX plugins, both MetricFlowX libraries built with Trilinos, and Slurm/SSH.
The workflow publishes it as `ghcr.io/luca-heltai/immersx-tutorial:master`
for Linux AMD64 and ARM64. The standalone ImmersX applications use the
separate `ghcr.io/luca-heltai/immersx:master` image.

## Start the stack

```sh
./scripts/tutorial-init.sh
docker compose pull
docker compose up -d
```

The local ports are 2222 (SSH), 8008 (Coral Visualizer), and 6080
(dealiiX Platform through noVNC). Change them with `CORAL_SSH_PORT`,
`VISUALIZER_PORT`, and `PLATFORM_PORT`. The ports bind to loopback.
The platform and visualizer are pulled from their own GHCR packages.
Use `IMMERSX_TUTORIAL_IMAGE`, `CORAL_VISUALIZER_IMAGE`, and
`DEALIIX_PLATFORM_IMAGE` in Compose's `.env` to pin working image digests.
The complete native stack requires all three images for the host architecture.

Use this remote target in the platform settings:

| Setting | Value |
| --- | --- |
| Host and port | `coral-ssh-slurm`, `22` |
| Username and SSH key | `root`, `/run/tutorial-ssh/id_ed25519` |
| Coral command | `/usr/local/bin/coral-release` |
| 2D plugin | `/opt/dealiix/immersx/Release/lib/immersx/coral/libcoral_backend_immersx_2.so` |
| Work directory | `/app/shared-data` |
| MPI launcher | `srun` |

Use `coral-debug`, the `Debug` prefix, and the `_debug.so` plugin suffix
to select Debug. Example graphs are mounted read-only at
`/opt/dealiix/immersx/coral/examples`. Files written under
`/app/shared-data` are visible to Coral Visualizer under `/deploy/data`.

Platform publication and the shared GUI acceptance checks are tracked in
[Platform issue #259](https://github.com/2listic/dealiiX-platform/issues/259).
Native ARM64 Visualizer publication is tracked in
[Visualizer issue #47](https://github.com/2listic/coral-visualizer/issues/47).

## Build the backend

The workflow checks out the commits in `docker/tutorial-sources.env` and
passes those checkouts to BuildKit as the named contexts `coral` and
`metric_flow_x`. Build locally from the same sources:

```sh
. docker/tutorial-sources.env
git clone https://github.com/2listic/coral.git _deps/coral
git -C _deps/coral checkout "$CORAL_REF"
git -C _deps/coral submodule update --init --recursive
git clone https://github.com/luca-heltai/metric-flow-x.git _deps/metric-flow-x
git -C _deps/metric-flow-x checkout "$METRIC_FLOW_X_REF"
docker buildx build --load -f docker/coral.Dockerfile \
  --build-context coral=_deps/coral \
  --build-context metric_flow_x=_deps/metric-flow-x \
  -t immersx-tutorial:local .
IMMERSX_TUTORIAL_IMAGE=immersx-tutorial:local docker compose up -d coral-ssh-slurm
```

To consume published SDKs directly, pass `--build-arg CORAL_SDK=IMAGE@DIGEST`
and `--build-arg METRIC_FLOW_X_SDK=IMAGE@DIGEST`. The images need the same
deal.II base, both configurations, and the same SDK layout as the source
build. MetricFlowX must use Trilinos.
