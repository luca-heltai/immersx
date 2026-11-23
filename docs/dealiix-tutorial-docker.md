# ImmersX Coral plugin image

The backend workflow publishes `ghcr.io/luca-heltai/immersx-tutorial:main` for
Linux AMD64 and ARM64. The development image is built from the ImmersX default
branch, `master`, with `ghcr.io/2listic/coral:main` as its base. It consumes the
published Debug and Release Trilinos SDK from
`ghcr.io/luca-heltai/metric-flow-x:main`.

The image installs the ImmersX plugins and example graphs in both
configurations. The runtime includes SSH and Slurm for remote execution.
The Compose stack is maintained in the
[dealii-X integration repository](https://github.com/dealii-X/dealii-X).

## Build against published SDK tags

```sh
docker buildx build --load -f docker/coral.Dockerfile \
  --build-arg CORAL_IMAGE=ghcr.io/2listic/coral:main \
  --build-arg METRIC_FLOW_X_IMAGE=ghcr.io/luca-heltai/metric-flow-x:main \
  -t immersx-tutorial:local .
```

Choose the SDK image tags with the two build arguments or the manual workflow
inputs. Matching SDK tags must use a compatible deal.II base and include both
variants. The workflow refreshes the development tag on relevant pushes and
on its daily run. A manual build with custom SDK tags validates that combination
without updating the development image.

The Release CLI wrapper is `/usr/local/bin/coral-release`. The 2D plugin is
`/opt/dealiix/immersx/Release/lib/immersx/coral/libcoral_backend_immersx_2.so`.
For Debug, use `coral-debug`, the `Debug` installation prefix, and the
`_debug.so` plugin suffix. Examples are installed under
`/opt/dealiix/immersx/Release/share/immersx/coral/examples`.
