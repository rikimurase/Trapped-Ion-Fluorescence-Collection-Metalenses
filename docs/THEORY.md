# Theory and methods

Basis for every equation in the pipeline. The design follows standard metalens practice: a
hyperbolic phase profile realized by a sub-wavelength meta-atom library, placed under the local
periodic approximation, validated by full-wave FDTD, with collection efficiency from
reciprocity. Companion to DESIGN_PROCEDURE.md (the workflow). Conventions: e^(-iωt), SI units,
Z0 = 376.73 Ω, k = 2π/λ0 the vacuum wavenumber, λ0 = 866 nm.

## 1. Strategy

η is the fraction of ion emission coupled into the lens's collimated output mode. By reciprocity
it reduces to the collection mode's field at the ion, obtained from a short lens FDTD driven by
the collimated mode and propagated analytically to the ion, which avoids a meshed 80 µm dipole
domain.

## 2. Reciprocity collection efficiency

Lorentz reciprocity collapses the emitter-mode overlap onto the mode field at the emitter r0:

```
∫_{z=0} ( E_d × H_g* + E_g* × H_d ) · da = i ω0 p0  p̂ · E_g*(r0)
```

With both modes normalized to unit power, the collected fraction and the unit-power dipole
moment are:

```
η  = (1/16) ω0² p0² | p̂ · E_g*(r0) |²
p0 = sqrt( 12π / ( c² Z0 k⁴ ) )
```

E_g carries unit power, units V/m/√W. Reduction with ω0 = ck gives ω0²p0² = 12π/(Z0 k²), so

```
η = (3π) / (4 Z0 k²) · | p̂ · E_g(r0) |²                        (A)
```

exact algebra, no new approximation. The code never forms E_g in absolute units. Writing
E_g = E_focal/√P_mode with P_mode the mode power (§4), and P_mode = ½(n_mode/Z0)|E_inc|²A_ap,

```
η = (3π) / (2 k² n_mode A_ap) · | p̂ · E_focal(r0) |² / |E_inc|²    (B)
```

which is the code:

```python
pref = (3*np.pi) / (2 * k**2 * n_mode * A_ap)
eta  = pref * abs(np.vdot(phat, E_focal))**2 / E_inc**2
```

Form (B) is unit-robust: any global field scale cancels between numerator and denominator.

Transmission needs no separate factor. E_focal is the field that actually appears after the
real lens, so it already carries every reflection and scattering loss. A lossier lens gives a
smaller E_focal and smaller η automatically. P_mode is the incident mode power, the correct
reference.

Index placement. The ion radiates in air, so k in the equations above is the vacuum wavenumber.
The collimated mode lives in SiO2, so P_mode uses n_mode = n_SiO2 = 1.4528. k is set by where
the emitter sits, n_mode by where the mode power flows.

## 3. Near-to-far projection

Phase 2 records the near field on a plane just above the pillars (z = h + λ/2, in air) and
propagates it to the ion plane with `FieldProjectionCartesianMonitor` and
`far_field_approx=False`. In homogeneous air this is the exact Rayleigh-Sommerfeld integral.
`far_field_approx=False` is required because the Fresnel number is O(1 to 30), so the
Fraunhofer reduction would be wrong. Residual error lives in the FDTD near field (uniform grid
at P/36 ≈ 14 nm, about 25 cells per wavelength in the pillar), bounded by the Strehl check
(§12).

## 4. Mode power and incident field

Top-hat mode power. The collection mode is the collimated plane wave filling the aperture:

```
P_mode = ½ (n_mode / Z0) |E_inc|² A_ap ,   A_ap = π (D/2)²        (C)
```

using H = (n_mode/Z0)E for a plane wave and a uniform amplitude across the aperture, true by
construction since Phase 2 illuminates with a normal plane wave.

E_inc and the standing wave. E_inc is read from the incident SiO2 region of the x-z slice.
That region holds incident plus reflected waves, so raw |E|² is inflated. Averaging
|E_inc e^(ikz) + √R E_inc e^(-ikz)|² over a band longer than λ/2 kills the cross term and
leaves |E_inc|²(1+R). The code divides by (1+R_pow) with R_pow ≈ 1 - ⟨|t|²⟩ from the library.
This assumes the cross term averages out and R is uniform, bounded by the Strehl check: the
measured focal enhancement is 0.86 to 0.90 of the ideal A_ap/(λf), which requires E_inc correct
to about ±10%. A backward flux monitor would remove the assumption.

This computes collection into the lens's natural top-hat mode. An optimized Gaussian
collection-mode waist would be a separate calculation. For a free-space detector capturing the
whole collimated beam the top-hat is the relevant mode; for fiber or waveguide coupling the
Gaussian overlap matters instead.

## 5. Dipole orientations and channels

Quantization axis along x̂:

```
π  : p̂ = x̂
σ± : p̂ = (ŷ ± i ẑ) / √2
```

Linear readout. Phase 2 injects an x mode whose on-axis focal field is pure x̂ (Ey, Ez ≈ 0,
verified in output). So one linear channel collects π and is blind to on-axis σ. This is the
readout, not the device: the C4v pillars are polarization-independent, so σ light is collimated
equally but exits orthogonally polarized. A σ± dipole couples at half the π efficiency into a
single linear mode.

Two channels by symmetry. The y mode is the exact 90° rotation of the x mode (C4v), so its
focal field is E_ymode = (-Efy, Efx, Efz) with no extra FDTD. Both channels sum. The in-plane
half of σ then couples at η_π/2. The small x/y FWHM asymmetry at NA 0.50 (0.93 vs 0.88 µm) shows
C4v is slightly broken by discretization and the normal-incidence library, so the rotation holds
to a few percent.

## 6. Axial-dipole bound

The ẑ half of each σ radiates a node on axis and couples only to a radial mode the x/y readout
misses. Phase 3 bounds it by its geometric ceiling rather than simulating it:

```
η_axial ≤ ½ - ¾ c + ¼ c³ ,   c = cos( asin(NA) )                 (D)
```

the exact cone integral of the sin²ψ axial pattern. Reported as a range, not a point (e.g.
σ± = 3.16 to 3.79% at D=92). The term is 0.02% at NA 0.18 and 1.27% at NA 0.50.

## 7. Geometric ceilings

Isotropic. Fraction of isotropic emission into a cone of half-angle θ = atan((D/2)/f):

```
η_geom = (1 - cos θ) / 2                                         (E)
```

Repo D=30 at f=80 gives 0.86%, matching the code.

Perpendicular dipole. η_π is bounded not by (E) but by the higher perpendicular-dipole ceiling,
because an in-plane dipole radiates preferentially toward the on-axis aperture: 9.35% at
NA 0.50 versus isotropic 6.65%. Measured η_π (6.87% TiO2, 6.32% Si3N4) sits below the
perpendicular ceiling and above the isotropic one, as it must. The unpolarized average is
bounded by the isotropic (E), and 4.58%/4.21% fall below 6.65%.

## 8. Phase library

Local periodic approximation. Each library point is one FDTD of a single pillar in a periodic
cell reading the complex (0,0) transmission. Placing a varying array assumes each cell behaves
as if periodic, valid when parameters vary slowly. This is standard metalens practice and the
largest layout approximation, checked by the diffraction-limited lens-scale FDTD.

Phase span at P=500 nm, h=800 nm: TiO2 (n=2.40) 1.06×2π, mean|t| 0.97; Si3N4 (n=2.00) 0.66×2π,
mean|t| 1.00; ITO placeholder (n=1.51) 0.30×2π. Span scales as (n-1)·k·h·fill. The ITO row used
a placeholder index. Real ITO is n ≈ 1.90, k ≈ 0.014, so its true limit is absorption, not
contrast (DESIGN_PROCEDURE §5).

Normal incidence. The library is built at normal incidence, but edge rays leave at up to
asin(NA) ≈ 30°. The local periodic approximation degrades as NA rises and point-by-point
placement carries error for high-NA lenses, so the library is used only to place pillars and the
full lens-scale FDTD sets the actual performance. At NA 0.50 the FDTD stayed diffraction-limited
with only a small x/y FWHM asymmetry.

## 9. Hyperbolic mask and sampling

Stigmatic phase for a normal plane wave to (0, f) in air, realized mod 2π (targets wrapped into
[0, 2π)):

```
φ(ρ) = -k ( sqrt(ρ² + f²) - f )                                 (F)
```

Sampling. The maximum phase gradient is at the edge, |dφ/dρ|_max = k·NA, so the outermost 2π
zone is λ/NA wide. Nyquist sampling of the phase requires the pitch P ≤ λ/(2·NA), i.e. the
platform supports NA up to λ/(2P) = 0.87 at P = 500 nm, λ = 866 nm. Both designs (NA 0.18 and
0.50) satisfy this with margin; the NA 0.50 edge zone is 1.73 µm wide, sampled by about 3.5
pillars. The pitch is also sub-wavelength in the surrounding media (P < λ/n_SiO2 = 596 nm), so
no higher diffraction orders propagate into air or substrate.

## 10. Radius assignment

The focal field is a coherent sum E_focal ∝ Σ_j t_j e^(-iφ_target,j), so the on-axis amplitude
and hence η are set by the coherent Strehl:

```
S = | mean_j ( |t_j| e^( i(φ_realized,j - φ_target,j) ) ) |²      (G)
```

For a library spanning ≥ 2π every target is reachable and the answer is exact matching. For a
short library (Si3N4 0.66×2π, ITO) some targets are unreachable. Each cell adds
|t_j|cos(φ_realized - φ_target) to the focal phasor, so the optimal reachable pillar maximizes
that projection:

```
r* = argmax_r  |t(r)| cos( φ(r) - φ_target )                     (H)
```

This reduces to exact matching when reachable. It respects that phase is circular: a target near
2π is served by a pillar near 0, not by the library endpoint. Clipping the target to the library
range instead pins over-range targets to the maximum, near the worst choice on a circle:

| span | clip S | max-projection S |
|---|---|---|
| 0.30×2π | 6% | 31% |
| 0.66×2π | 67% | 88% |

The TiO2 designs use clip (nothing clips at span 1.06×2π); the Si3N4 designs use (H). Both are
confirmed within 1% by scalar propagation and FDTD (§14).

Where the loss goes. Lost coherent power scatters into other Fresnel-zone orders (other focal
planes, the zeroth order), not the spot, so the focus stays diffraction-limited and only the peak
drops. The penalty grows with zone count and hence NA: FDTD loss versus the TiO2 sibling is about
1% at NA 0.18 and 8% at NA 0.50.

## 11. Focal shift

For a converging wave of Fresnel number N = (D/2)²/(λf), the on-axis peak falls short of the
geometric focus when N ≲ 10. The D=30 lens (N=3.25) peaks near 0.88f; D=92 (N=30.5) shows no
shift. N depends only on geometry, so the shift is material-independent and the TiO2 correction
carries to Si3N4 unchanged (Si3N4 D=30 peaks at 78 µm with f_design=100, matching TiO2). The
correction f_design from the two-point fit peak ≈ 0.5·f_design + 30 is empirical and
geometry-specific, not a general law. Scalar propagation independently predicts a peak near 0.9f,
confirming the direction.

## 12. Focusing metrics

- FWHM: from |E|² cuts on the projected plane, versus the Airy value 0.514 λ/NA, half-max
  crossings interpolated.
- Focal length: peak of an axial scan of on-axis projection monitors.
- Focusing efficiency: power in a 3×FWHM disk over transmitted power, from
  ½Re(Ex Hy* - Ey Hx*). Standard metalens definition, distinct from collection η.
- Strehl check: measured enhancement |E_focal|/E_inc versus the ideal A_ap/(λf). The ratio
  (0.86 at D=30, 0.90 at D=92) is the lens Strehl and validates the absolute normalization of §4:
  a wrong E_inc or A_ap would give an unphysical value.

## 13. Assumptions

| # | Assumption | Status | Basis |
|---|---|---|---|
| 1 | Lorentz reciprocity | exact for lossless dielectrics | standard |
| 2 | Single frequency | Γ/ω0 ~ 1e-8 ≪ 1e-5 | narrow atomic linewidth |
| 3 | Unit-power dipole p0 | exact | standard |
| 4 | Compact form (A) | exact algebra | derived |
| 5 | Top-hat mode power (C) | exact for uniform plane wave | plane-wave Poynting |
| 6 | Standing-wave E_inc | approx, validated by Strehl 0.86-0.90 | this design |
| 7 | Exact projection | exact in air, low N keeps full kernel | standard |
| 8 | vacuum k, SiO2 n_mode | correct index placement | physical |
| 9 | π=x̂, σ±=(ŷ±iẑ)/√2 | exact convention | atomic physics |
| 10 | σ at η_π/2 | exact for C4v | symmetry |
| 11 | y = 90° rotation of x | exact for ideal C4v, few-% residual | symmetry |
| 12 | Axial bound (D) | conservative exact bound | this design |
| 13 | Isotropic ceiling (E) | exact solid angle | geometry |
| 14 | Local periodic | standard, FDTD-checked | standard practice |
| 15 | Normal-incidence library | approx to NA≈0.30, FDTD forgiving to 0.50 | standard practice |
| 16 | Hyperbolic mask (F), 2π wrap | exact thin lens | textbook |
| 17 | Nyquist sampling P ≤ λ/(2NA) | satisfied, NA_max 0.87 | sampling theorem |
| 18 | Max-projection (H) | optimal for objective (G) | this design |
| 19 | Focal-shift fit | standard N physics, empirical constant | scalar-confirmed |

## 14. Cross-checks

1. Scalar vs FDTD. The free scalar Strehl (88.4%) predicted Si3N4 D=92 η_π ≈ 6.32% before any
   cloud run. FDTD measured 6.316%.
2. Reciprocity self-consistency. `summarize_designs.py` recomputes all four designs from FDTD
   fields and reproduces the per-design outputs exactly (TiO2 D=92 η_π 6.873%, etc.).
3. Absolute normalization. Measured enhancement is 0.86 to 0.90 of A_ap/(λf), fixing E_inc and
   A_ap to ±10% independently.
