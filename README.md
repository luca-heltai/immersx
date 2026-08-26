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

See the [getting-started guide](https://luca-heltai.github.io/immersx/getting-started/)
for dependencies, configuration, and the first runnable example.

## References

- Giovanni Alzetta and Luca Heltai, *Multiscale modeling of fiber reinforced materials via non-matching immersed methods*, Computers & Structures, 239 (2020), 106334. DOI: <https://doi.org/10.1016/j.compstruc.2020.106334>
- Camilla Belponer, Alfonso Caiazzo, and Luca Heltai, *Mixed-dimensional modeling of vascular tissues with reduced Lagrange multipliers* (2025).
- Luca Heltai and Alfonso Caiazzo, *Multiscale modeling of vascularized tissues via nonmatching immersed methods*, International Journal for Numerical Methods in Biomedical Engineering, 35(12) (2019), e3264. DOI: <https://doi.org/10.1002/cnm.3264>
- Luca Heltai, Alfonso Caiazzo, and Lucas O. Muller, *Multiscale Coupling of One-dimensional Vascular Models and Elastic Tissues*, Annals of Biomedical Engineering, 49 (2021), 3243-3254. DOI: <https://doi.org/10.1007/s10439-021-02804-0>
- Luca Heltai and Paolo Zunino, *Reduced Lagrange multiplier approach for non-matching coupling of mixed-dimensional domains*, Mathematical Models and Methods in Applied Sciences, 33(12) (2023), 2425-2462. DOI: <https://doi.org/10.1142/S0218202523500525>
- Yashasvi Verma, Jakob Schattenfroh, Ingolf Sack, Silvia Budday, Paul Steinmann, and Luca Heltai, *Simulation Platform to Evaluate Inversion Techniques for Magnetic Resonance Elastography Data* (2026).

## License

See `LICENSE.md`.
