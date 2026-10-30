# Run with Docker, Podman, or Apptainer

The OCI image contains precompiled Release applications and the matching
deal.II libraries. Only a container runtime is needed on the host. The image
targets Linux AMD64 and ARM64 and retains the deal.II SDK, so the first download
is large (roughly 12 GB of image storage, plus simulation inputs and outputs).
Optional MetricFlowX, Coral, and legacy `1dsolver` integrations are not included.

## Select an application

Run the image without arguments to print help and list its installed
applications. Pass an application name to execute it:

```bash
docker run --rm ghcr.io/luca-heltai/immersx:master
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" ghcr.io/luca-heltai/immersx:master poisson input.prm

podman run --rm ghcr.io/luca-heltai/immersx:master
podman run --rm --userns=keep-id --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" ghcr.io/luca-heltai/immersx:master poisson input.prm

apptainer run docker://ghcr.io/luca-heltai/immersx:master
apptainer run --bind "$PWD:/work" --pwd /work \
  docker://ghcr.io/luca-heltai/immersx:master poisson input.prm
```

The image works in `/work`. Put `input.prm` and all files it references in the
mounted directory. Outputs are written directly into that directory. On hosts
with SELinux, configure an appropriate bind-mount label for Docker or Podman.
Applications receive all arguments after their name unchanged and return their
own exit status. An unknown application exits with status 127.

Use materialized `.prm` files: source `.prm.in` templates may contain CMake
tokens or build-tree paths that are unavailable in the image. The container
does not download datasets or rewrite parameter files.

## Standalone launcher and MPI

Download the launcher without cloning or compiling ImmersX:

```bash
curl --fail --location \
  https://raw.githubusercontent.com/luca-heltai/immersx/master/scripts/run_container.sh \
  --output run_container.sh
bash run_container.sh --help
bash run_container.sh
bash run_container.sh --workdir /path/to/case poisson input.prm
bash run_container.sh --runtime podman --workdir /path/to/case poisson input.prm
bash run_container.sh --runtime apptainer --workdir /path/to/case poisson input.prm
bash run_container.sh --ranks 2 --workdir /path/to/case poisson input.prm
```

The launcher checks for Docker, then Podman, then Apptainer. Detection checks
the executable; use `--runtime` to select a working alternative if the Docker
daemon is unavailable. It preserves the invoking user's output ownership and
mounts the working directory at `/work` and at its original absolute path.
All inputs and outputs must lie under that directory. Bind paths containing
colons, commas, or newlines are rejected.

MPI uses the bundled `mpirun` inside one container on one node. The selected
application and its solver must support the requested number of ranks.
Multinode runs and host MPI integration require separate configuration.

Use `--exec` for other installed commands, including a shell:

```bash
bash run_container.sh --exec bash
```

To reuse a local Apptainer SIF:

```bash
apptainer pull immersx.sif docker://ghcr.io/luca-heltai/immersx:master
bash run_container.sh --runtime apptainer --image ./immersx.sif poisson input.prm
```

`scripts/run_docker.sh` delegates to this launcher with Docker selected.

## Image versions and publication

The image name is `ghcr.io/luca-heltai/immersx`. `:master` follows development;
`:sha-COMMIT` identifies a built source revision. Release tags matching
`vMAJOR.MINOR.PATCH` produce `:vMAJOR.MINOR.PATCH`, `:MAJOR.MINOR.PATCH`,
`:MAJOR.MINOR`, `:MAJOR`, and `:latest`. The tag must match the CMake project
version. These image references become available after the container workflow
has published them; adding the workflow alone does not publish an image.

For reproducibility, use the immutable digest recorded in the workflow's
`container-image.txt` artifact:

```bash
bash run_container.sh --image ghcr.io/luca-heltai/immersx@sha256:DIGEST \
  --workdir /path/to/case poisson input.prm
```

Replace `DIGEST` with the published digest. `IMMERSX_IMAGE` overrides the
launcher's default image. The `container-launcher` workflow artifact also
contains a launcher with that digest embedded and checksums.

The workflow follows MetricFlowX's container pipeline: native AMD64 and ARM64
builds, a digest-pinned deal.II base, and a daily check at midnight in
`Europe/Rome`. Manual dispatch on `master` also checks for changed inputs.
The fingerprint includes tracked source, build files, installed resources,
entry point, launcher, and verification scripts. An existing tested fingerprint
reuses the published image. A release tag always builds its exact revision.
Pull requests build and test both architectures without publishing.

Before publishing the multi-architecture index, CI checks default help,
application dispatch, a nonzero-forcing Poisson solve with one and two MPI
ranks, and output ownership. AMD64 also runs Podman and Apptainer checks.
Rootless Podman depends on the host's user namespace configuration; CI checks
rootful Podman. GHCR package visibility must permit anonymous pulls; the
workflow verifies this after publication.

To build the image locally:

```bash
docker build --build-arg BUILD_JOBS=2 -t immersx:local .
bash scripts/run_container.sh --image immersx:local
bash scripts/run_container.sh --image immersx:local --workdir /path/to/case poisson input.prm
```
