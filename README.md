# ImmersX

![ImmersX logo](https://raw.githubusercontent.com/luca-heltai/immersx/master/doc/immersx-logo.png)

![GitHub CI](https://github.com/luca-heltai/immersx/actions/workflows/tests.yml/badge.svg)
![Documentation](https://github.com/luca-heltai/immersx/actions/workflows/doxygen.yml/badge.svg)
![Indent](https://github.com/luca-heltai/immersx/actions/workflows/indentation.yml/badge.svg)

ImmersX is a C++ framework for embedded and mixed-dimensional finite-element
simulations. It provides Poisson and elasticity problems, reduced multiplier
spaces, immersed coupling, and coupled 3D/1D workflows on top of
[deal.II](https://www.dealii.org) 9.7.1 or later.

Read the [documentation](https://luca-heltai.github.io/immersx/) for guided
tutorials, how-to guides, concepts, application reference, and API reference.

## Quick start

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=/path/to/deal.II
cmake --build build -j
./build/poisson path/to/input.prm
```

For a single-config build tree containing both variants, use
`-DCMAKE_BUILD_TYPE=DebugRelease`. This creates unsuffixed Release targets and
`_debug`-suffixed Debug targets; it is an ImmersX build mode, not a CMake
multi-config generator.

See the [getting-started guide](https://luca-heltai.github.io/immersx/getting-started/)
for dependencies, configuration, and the first runnable example.

## dealiiX tutorial Docker images

The `dealiix-tutorial` branch publishes a complete multi-architecture tutorial
stack containing the Coral runtime, the ImmersX Coral plugins, MetricFlowX,
the Coral Visualizer, and the dealiiX-platform interface. Docker automatically
selects the native `linux/amd64` or `linux/arm64` image for your machine.

Clone this branch and start the latest published images:

```bash
git clone --branch dealiix-tutorial \
  https://github.com/luca-heltai/immersx.git
cd immersx
./scripts/tutorial-init.sh
docker compose pull
docker compose up -d
```

Open the dealiiX-platform interface at
<http://localhost:6080/vnc.html?autoconnect=1&resize=scale>. It is the
Electron platform running in a browser through noVNC. The Coral Visualizer is
available directly at <http://localhost:8008>.

The Compose file pulls these mutable tutorial images:

```text
heltai/coral:dealiix-tutorial
heltai/coral-visualizer:dealiix-tutorial
heltai/dealiix-platform:dealiix-tutorial
```

The compute image includes the ImmersX 1D, 2D, and 3D Coral plugins and the
MetricFlowX installation. To retrieve newer images while staying on the same
branch, run:

```bash
git pull
docker compose pull
docker compose up -d
```

Stop the stack with:

```bash
docker compose down
```

See [`docs/dealiix-tutorial-docker.md`](docs/dealiix-tutorial-docker.md) for
service details and troubleshooting.

## References

- Giovanni Alzetta and Luca Heltai, *Multiscale modeling of fiber reinforced materials via non-matching immersed methods*, Computers & Structures, 239 (2020), 106334. DOI: <https://doi.org/10.1016/j.compstruc.2020.106334>
- Camilla Belponer, Alfonso Caiazzo, and Luca Heltai, *Mixed-dimensional modeling of vascular tissues with reduced Lagrange multipliers* (2025).
- Luca Heltai and Alfonso Caiazzo, *Multiscale modeling of vascularized tissues via nonmatching immersed methods*, International Journal for Numerical Methods in Biomedical Engineering, 35(12) (2019), e3264. DOI: <https://doi.org/10.1002/cnm.3264>
- Luca Heltai, Alfonso Caiazzo, and Lucas O. Muller, *Multiscale Coupling of One-dimensional Vascular Models and Elastic Tissues*, Annals of Biomedical Engineering, 49 (2021), 3243-3254. DOI: <https://doi.org/10.1007/s10439-021-02804-0>
- Luca Heltai and Paolo Zunino, *Reduced Lagrange multiplier approach for non-matching coupling of mixed-dimensional domains*, Mathematical Models and Methods in Applied Sciences, 33(12) (2023), 2425-2462. DOI: <https://doi.org/10.1142/S0218202523500525>
- Yashasvi Verma, Jakob Schattenfroh, Ingolf Sack, Silvia Budday, Paul Steinmann, and Luca Heltai, *Simulation Platform to Evaluate Inversion Techniques for Magnetic Resonance Elastography Data* (2026).

## License

See `LICENSE.md`.
