# Reduced Lagrange Multipliers

![GitHub CI](https://github.com/luca-heltai/reduced_lagrange_multipliers/actions/workflows/tests.yml/badge.svg)
![Documentation](https://github.com/luca-heltai/reduced_lagrange_multipliers/actions/workflows/doxygen.yml/badge.svg)
![Indent](https://github.com/luca-heltai/reduced_lagrange_multipliers/actions/workflows/indentation.yml/badge.svg)

This repository contains C++ implementations of reduced Lagrange multiplier methods for mixed-dimensional coupling problems, built on top of [deal.II](https://www.dealii.org). Required version: 9.7.1 or later.

The website documentation is available at <https://luca-heltai.github.io/reduced_lagrange_multipliers/>.

## Quick Start

```bash
mkdir -p build
cd build
cmake -DDEAL_II_DIR=/path/to/deal.II ..
cmake --build . -j
./elasticity path/to/input.prm # or any other app_*.cc executable
cd ..
python3 -m pip install -r doc/requirements.txt
./scripts/build_doc.sh
```

## Repository Layout

- `include/`, `source/`: library headers and implementations.
- `apps/`: executable entry points derived from `app_*.cc`.
- `tests/`, `gtests/`: regression and GoogleTest-based test suites.
- `doc/`: Doxygen and Sphinx source for the published documentation site.
- `scripts/`: helper scripts, including `scripts/build_doc.sh`.
- `prms/`, `benchmarks/`: parameter files, meshes, and benchmark assets.
- `blood/`: auxiliary 1D hemodynamics code and data used by coupled workflows.

## Notes

- This repository contains large benchmark/data files

## Coupled 3D/1D workflow

For setup/build/run notes specific to the coupled elasticity problem, see
`COUPLED_PROBLEM.md`.

## References

- Giovanni Alzetta and Luca Heltai, *Multiscale modeling of fiber reinforced materials via non-matching immersed methods*, Computers & Structures, 239 (2020), 106334. DOI: <https://doi.org/10.1016/j.compstruc.2020.106334>
- Camilla Belponer, Alfonso Caiazzo, and Luca Heltai, *Mixed-dimensional modeling of vascular tissues with reduced Lagrange multipliers* (2025). Local PDF: `doc/papers/2309.06797v2.pdf`
- Luca Heltai and Alfonso Caiazzo, *Multiscale modeling of vascularized tissues via nonmatching immersed methods*, International Journal for Numerical Methods in Biomedical Engineering, 35(12) (2019), e3264. DOI: <https://doi.org/10.1002/cnm.3264>
- Luca Heltai, Alfonso Caiazzo, and Lucas O. Muller, *Multiscale Coupling of One-dimensional Vascular Models and Elastic Tissues*, Annals of Biomedical Engineering, 49 (2021), 3243-3254. DOI: <https://doi.org/10.1007/s10439-021-02804-0>
- Luca Heltai and Paolo Zunino, *Reduced Lagrange multiplier approach for non-matching coupling of mixed-dimensional domains*, Mathematical Models and Methods in Applied Sciences, 33(12) (2023), 2425-2462. DOI: <https://doi.org/10.1142/S0218202523500525>
- Yashasvi Verma, Jakob Schattenfroh, Ingolf Sack, Silvia Budday, Paul Steinmann, and Luca Heltai, *Simulation Platform to Evaluate Inversion Techniques for Magnetic Resonance Elastography Data* (2026).

## License

See `LICENSE.md`.
