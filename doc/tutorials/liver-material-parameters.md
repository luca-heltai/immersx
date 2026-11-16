---
canonical: https://luca-heltai.github.io/immersx/tutorials/liver-material-parameters.html
---

# Healthy human liver: material parameters for elastodynamics

This page gives recommended values for the six `ElastodynamicsMaterialParameters`
coefficients when modelling **healthy human liver tissue** as a single-phase linear
viscoelastic solid.  It is a companion to the
[Elastodynamics tutorial](elastodynamics.md) and to the
[Vascularize a liver](liver-vascularization.md) tutorial.

## Model recap

Each named material block in `Elastodynamics/Material properties` accepts:

| Parameter | Symbol | Governing term |
|---|---|---|
| `Density` | ρ | Inertia — mass matrix |
| `Lame mu` | μ | Shear stiffness |
| `Lame lambda` | λ | Bulk/compressibility stiffness |
| `Rayleigh alpha` | α | Mass-proportional Rayleigh damping |
| `Rayleigh beta` | β | Stiffness-proportional Rayleigh damping |
| `Viscosity eta` | η | Kelvin–Voigt shear viscosity (no factor of 2) |

The damping bilinear form is

$$
C(v,w) = \alpha\rho(v,w)
        + \beta\bigl(2\mu(\varepsilon(v),\varepsilon(w))
               + \lambda(\nabla\cdot v,\nabla\cdot w)\bigr)
        + \eta(\varepsilon(v),\varepsilon(w)).
$$

## Recommended values

```prm
subsection Material properties
  subsection default
    # Bulk liver tissue density — soft-tissue value [1,5]
    set Density        = 1050.0   # kg m^-3

    # Shear modulus — in vivo MRE, healthy liver, 50–80 Hz [3,4]
    # Use 800 Pa for low-frequency (<10 Hz) or cardiac-scale problems.
    # Use 1500–2500 Pa for 50–80 Hz MRE-range excitation.
    set Lame mu        = 1200.0   # Pa

    # First Lamé parameter — near-incompressible (nu ~ 0.49) [5,6]
    # lambda = 2 mu nu / (1 - 2 nu).  For mu = 1200 Pa, nu = 0.49:
    # lambda ~ 58 800 Pa.  Rounded conservatively to avoid locking issues.
    set Lame lambda    = 59000.0  # Pa

    # Rayleigh damping — set to zero unless calibrating to a measured
    # damping ratio; the Voigt term eta already captures the dominant
    # loss mechanism at MRE frequencies [2].
    set Rayleigh alpha = 0
    set Rayleigh beta  = 0

    # Kelvin–Voigt shear viscosity — solid-viscosity component from
    # Biot-model fits to in vivo liver MRE dispersion [2].
    # Voigt-model fits give mu_1 ~ 0.5–1.6 Pa s [3]; the ImmersX
    # convention carries no factor of 2 before eta.
    set Viscosity eta  = 1.5      # Pa s
  end
end
```

## Parameter-by-parameter justification

### Density — 1050 kg m⁻³

Bulk density of liver parenchyma as a homogenised solid continuum, we call ρ the the density of the
material filling the domain.  Soft biological tissues are
approximately 70% water by mass, which places their bulk density close to that
of water (1000 kg m⁻³); liver parenchyma is consistently reported at
1050–1060 kg m⁻³ [1].  The anthropomorphic torso FEM study for liver MRE uses
the round value ρ = 1000 kg m⁻³ [5]; 1050 kg m⁻³ is the more precise
experimental value and is the safer default for wave-speed calculations, since
shear wave speed scales as √(μ/ρ).

### Shear modulus `Lame mu` — 800–2500 Pa (default 1200 Pa)

Liver is a frequency-dispersive material: its complex shear modulus follows a
power law across at least four frequency decades [7]:

$$
G^*(\omega) = \mu_\mathrm{SP}(i\omega)^\alpha,
\qquad \mu_\mathrm{SP} = 0.90\ \text{kPa},\ \alpha = 0.13
$$

(healthy, in vivo, 5–80 Hz [8]).  Picking μ at a single frequency:

| Frequency | Recommended `Lame mu` | Source |
|---|---|---|
| ~1 Hz (cardiac / iMRE) | 800 Pa | [8,9] |
| 50–60 Hz (clinical MRE) | 1200–1500 Pa | [3,4] |
| 60–80 Hz (clinical MRE) | 1500–2500 Pa | [3,4,5] |
| Cirrhotic liver (50–80 Hz) | 2500–4700 Pa | [3] |

The best-cited single-frequency benchmark is an in vivo MRE torso FEM at 60 Hz
using |G*| = 1.0 kPa, G′ = 0.6 kPa, G″ = 0.8 kPa [5].

### First Lamé parameter `Lame lambda` — ≈ 59 000 Pa (ν ≈ 0.49)

Liver is nearly incompressible.  Measured Poisson's ratios span 0.43–0.499
depending on method [6,10]:

| Source | ν | Context |
|---|---|---|
| Raghunathan et al. 2010 [6] | 0.49 | Porcine PVE, unconfined compression |
| Bonfiglio et al. 2010 [10] | 0.43–0.47 | Pig liver, direct measurement |
| Moran et al. 2012 [11] | 0.35 | Porcine avascular parenchyma |
| IOP FEM torso sim. 2022 [5] | 0.499 | Near-incompressible assumption |

The relation λ = 2μν/(1 − 2ν) gives:

| ν | λ (at μ = 1200 Pa) |
|---|---|
| 0.35 | 1800 Pa |
| 0.45 | 9900 Pa |
| 0.49 | 58 800 Pa |
| 0.499 | 598 800 Pa |

ν = 0.49 (λ ≈ 59 000 Pa) is the standard near-incompressible default for
in vivo liver modelling.  If volumetric locking is an issue on coarse meshes,
reduce to ν = 0.45–0.47.  Avoid ν = 0.499 unless a stabilised mixed formulation
is used.

### Rayleigh damping `alpha`, `beta`

The two Rayleigh coefficients and the Kelvin–Voigt viscosity η capture distinct
physical loss mechanisms and are **not interchangeable**.

| Coefficient | Damping contribution | Frequency dependence | Physical interpretation |
|---|---|---|---|
| α (mass) | ζ(ω) = α / (2ω) | Decreases with ω — dominates at low f | Displacement-proportional dissipation; approximates low-frequency fluid redistribution when porosity is ignored |
| β (stiffness) | ζ(ω) = β ω / 2 | Increases with ω | Strain-rate proportional; partially overlaps with η |
| η (Voigt) | G″ = η ω | Increases linearly with ω | Kelvin–Voigt dashpot in parallel with the elastic spring |

**Narrow-band / single-frequency problems** (e.g. harmonic MRE at a fixed
excitation frequency): setting α = β = 0 and tuning η alone is defensible,
because you are matching G″ at one frequency only.

**Broadband or time-domain problems** (cardiac cycle, breathing, impact,
wideband MRE): the Kelvin–Voigt model alone will underdamp low frequencies
and overdamp high ones relative to the measured power-law loss angle of liver.
In this case non-zero Rayleigh coefficients are physically motivated.  The
in vivo loss angle of healthy liver measured over 5–80 Hz follows approximately
φ(ω) ∝ ω^α (springpot behaviour, α ≈ 0.13 [8]): it is roughly 0.3–0.4 rad at
5 Hz and 0.2–0.3 rad at 60 Hz.  The corresponding damping ratio
ζ = tan(φ)/2 ranges from ~0.15–0.20 at low frequency to ~0.10–0.15 at
60 Hz [8].

To fit α and β to two target (ω, ζ) pairs — for example (2π·5, 0.18) and
(2π·60, 0.12) — solve the linear system:

$$
\begin{pmatrix} 1/(2\omega_1) & \omega_1/2 \\ 1/(2\omega_2) & \omega_2/2 \end{pmatrix}
\begin{pmatrix} \alpha \\ \beta \end{pmatrix}
=
\begin{pmatrix} \zeta_1 \\ \zeta_2 \end{pmatrix}.
$$

For the example values above this gives α ≈ 5.5 s⁻¹ and β ≈ 5 × 10⁻⁴ s.
Note that β overlaps with η, so set η = 0 when using non-zero β to avoid
double-counting stiffness-proportional damping.

**Default for a first simulation** without a specific frequency target: set
α = β = 0 and use η = 1.5 Pa s.  This is consistent with published Voigt-model
fits at 50–80 Hz [4] and avoids spurious low-frequency over-damping.

### Viscosity `Viscosity eta` — 1.5 Pa s

Corresponds to the solid-viscosity component μ₁ of the Kelvin–Voigt (Voigt)
model, fit to in vivo human liver MRE dispersion data.  Aichele & Catheline [2]
report μ₁ ≈ 0.5–0.6 Pa s from their poro-visco-elastic Biot fits (Table 2),
while standard Voigt-model fits to the same dispersion data require
μ₁ ≈ 1.6–4.3 Pa s to compensate for the absence of a fluid phase.
Chatelin et al. [4] and Parker [7] confirm the Voigt solid-viscosity range
of 1–5 Pa s for 50–80 Hz.

The ImmersX convention carries **no factor of 2** before `Viscosity eta`;
1.5 Pa s is therefore the direct entry corresponding to μ₁ in the literature.
If you switch to a purely elastic model (η = 0) the solution will be lossless;
if you observe excessive ringing, increase η toward 3–4 Pa s.

## Frequency-dependent calibration

For harmonic (single-frequency) problems use the springpot to pick μ at the
excitation frequency f:

$$
\mu(f) = \mu_\mathrm{SP}(2\pi f)^\alpha,
\qquad \mu_\mathrm{SP} = 0.90\ \text{kPa},\ \alpha = 0.13.
$$

| f | μ (Pa) |
|---|---|
| 1 Hz | 790 |
| 10 Hz | 998 |
| 25 Hz | 1130 |
| 50 Hz | 1270 |
| 60 Hz | 1310 |
| 80 Hz | 1390 |

Values for the springpot parameters are from wideband in vivo MRE of healthy
human liver (5–80 Hz) [8].

## What these values do not cover

- **Porosity and fluid-phase effects.**
- **Fibrosis and disease.**
- **Anisotropy.**  The values above assume isotropy.  Liver parenchyma shows mild
  transverse isotropy due to the lobular architecture, but isotropic models are
  standard in MRE inversion.
- **Large deformations.**  Liver is known to strain-stiffen under surgical
  manipulation; a hyperelastic solid phase is needed for deformations beyond
  ~10% strain.

## References

[1] Duck FA. *Physical Properties of Tissues: A Comprehensive Reference Book.*
Academic Press, London (1990). ISBN 978-0-12-222800-1.
(Liver parenchyma density: 1050–1070 kg m⁻³, Chapter 2.)

[2] Aichele J, Catheline S. Fluids alter elasticity measurements: porous wave
propagation accounts for shear wave dispersion in elastography.
*Front Phys* **9**, 697990 (2021).
<https://doi.org/10.3389/fphy.2021.697990>

[3] Huwart L, et al. Magnetic resonance elastography for the noninvasive staging
of liver fibrosis. *Gastroenterology* **135**(1), 32–40 (2008).
<https://doi.org/10.1053/j.gastro.2008.03.076>

[4] Chatelin S, et al. In vivo liver tissue mechanical properties by transient
elastography: comparison with dynamic mechanical analysis.
*Biorheology* **48**(2), 75–88 (2011).
<https://doi.org/10.3233/BIR-2011-0584>

[5] Ipek-Ugay S, et al. In silico evaluation and optimisation of magnetic
resonance elastography of the liver. *Phys Med Biol* **66**(24), 245007 (2022).
<https://doi.org/10.1088/1361-6560/ac3263>

[6] Raghunathan S, Evans D, Sparks JL. Poroviscoelastic modeling of liver
biomechanical response in unconfined compression.
*Ann Biomed Eng* **38**(6), 1789–1800 (2010).
<https://doi.org/10.1007/s10439-010-9957-x>

[7] Parker KJ. Towards a consensus on rheological models for shear waves in
soft tissues. *arXiv* 1906.08612 (2019).
<https://arxiv.org/abs/1906.08612>

[8] Lilaj L, et al. Ultra-wideband MRE of the human liver and spleen for
viscoelastic model identification in hepatic inflammation.
*arXiv* 2608.18867 (2025).
<https://arxiv.org/abs/2608.18867>

[9] Burman Ingeberg V, et al. Evaluating the performance and repeatability of
poroelastic and poroviscoelastic models in intrinsic MR elastography.
*NMR Biomed* (2025).
<https://doi.org/10.1002/nbm.70073>

[10] Bonfiglio A, et al. Mathematical modeling of the circulation in the liver
lobule. *J Biomech Eng* **132**(11), 111011 (2010).
<https://doi.org/10.1115/1.4002563>

[11] Moran EC, et al. Porohyperviscoelastic model simultaneously predicts
parenchymal fluid pressure and reaction force in perfused liver.
*J Biomech Eng* **134**(9), 091002 (2012).
<https://doi.org/10.1115/1.4007175>
