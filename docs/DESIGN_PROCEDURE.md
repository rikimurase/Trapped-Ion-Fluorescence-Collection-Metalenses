# Metalens design procedure

The workflow that built and validated four 866 nm Ca+ collection metalenses: two apertures
(30 and 92 µm) in two materials (TiO2, Si3N4), all at 80 µm working distance. This is the how.
The equations and their justification are in [THEORY.md](THEORY.md). Comparison figures are in
`figures/` (rebuild with `scripts/make_comparison_figures.py`).

## 1. Goal

Collect 866 nm fluorescence from a single 40Ca+ ion at an 80 µm working distance above a flat
chip-integrated metalens and collimate it for detection. The working distance is fixed, so the
aperture D sets the NA and the geometric collection ceiling. The figure of merit is the coupling
into the lens's collimated output mode, computed by reciprocity.

## 2. Method

Four phases. Phase 0 is done once per material and shared. Phases 1 to 3 are per design.

```
 Phase 0            Phase 1            Phase 2            Phase 3
 unit-cell   r(φ)   hyperbolic  (x,y,r)  lens-scale  field   collection
 sweep      -----> phase mask  ------->  FDTD +      ----->  efficiency
 (shared)           to pillars           near->far           (reciprocity)
```

Phase 0. One FDTD per pillar in a periodic cell reads the complex (0,0) transmission. Sweeping
radius builds the map r(φ) and |t|(r). The result is a smooth library saved to
`data/phase_library_<material>.npz`. Rebuild offline from the cache with
`scripts/regenerate_library.py <material>` at no cloud cost. Physics: THEORY §8.

Phase 1. Evaluate the hyperbolic phase φ(ρ) = -k(sqrt(ρ²+f²) - f) at each grid site inside the
aperture, wrap into [0, 2π), and assign each site the pillar radius that best realizes it. On a
full library (TiO2) that is clip-to-range exact matching. On a short library (Si3N4) it is
max-projection, the reachable pillar maximizing |t|cos(φ_realized - φ_target). Physics:
THEORY §9, §10.

Phase 2. Full-wave FDTD of a thin slab around the pillars, recording the near field above them
and projecting to the focus with the exact (non-Fraunhofer) integral (`far_field_approx=False`),
which avoids meshing the 80 µm gap. Quarter symmetry `(-1,1,0)`. Analysis reports spot size,
focal length, and focusing efficiency. Physics: THEORY §3, §12.

Phase 3. Collection efficiency from the same field by reciprocity, no separate dipole
simulation:

```
η = (3π)/(2 k² n_mode A_ap) · |p̂·E_focal|² / |E_inc|²
```

covering all dipole orientations (π, σ±) and both linear channels. Physics: THEORY §2, §5 to §7.

Phases 1 to 3 for any design run from one module:

```
python scripts/metalens_tidy3d.py <design> --layout            # Phase 1, free
python scripts/metalens_tidy3d.py <design> --lens              # Phase 2, cost estimate only
python scripts/metalens_tidy3d.py <design> --lens --submit     # Phase 2, run the FDTD (FlexCredits)
python scripts/metalens_tidy3d.py <design> --analyze --collect # Phase 2 metrics and Phase 3
```

where <design> is tio2_D30, tio2_D92, si3n4_D30, or si3n4_D92. Outputs land in
designs/<design>/data and designs/<design>/figures. The Lumerical port under `lumerical/` mirrors
this in a single script (see lumerical/README.md).

## 3. The four designs

All use P = 500 nm, h = 800 nm, 80 µm working distance. η_π is the single-dipole efficiency with
both linear channels detected. All are diffraction-limited and FDTD-validated.

| design | NA | cylinders | span | focus | FWHM/limit | enh | η_π | σ± | unpol | FDTD |
|---|---|---|---|---|---|---|---|---|---|---|
| tio2_D30 | 0.184 | 2821 | 1.06×2π | 81 µm | 2.6/2.42 | 8.8x | 0.68% | 0.34% | 0.46% | 1.2 FC |
| si3n4_D30 | 0.184 | 2821 | 0.66×2π | 78 µm | 2.48/2.42 | 8.8x | 0.67% | 0.34% | 0.45% | 0.94 FC |
| tio2_D92 | 0.498 | 26565 | 1.06×2π | 80 µm | 0.9/0.89 | 86x | 6.87% | 3.44% | 4.58% | 15.9 FC |
| si3n4_D92 | 0.498 | 26565 | 0.66×2π | 80 µm | 0.93/0.88 | 83x | 6.32% | 3.16% | 4.21% | 6.13 FC |

enh is the focal amplitude enhancement. Transmission above the lens is 86 to 89% across the four.
Collection is set by NA, not lens quality: the 10x jump from D=30 to D=92 is the solid-angle
scaling, while both focus at the diffraction limit. Si3N4 nearly matches TiO2
at each aperture despite its shorter library, losing ~1% at NA 0.18 and ~8% at NA 0.50, because
the short-library penalty grows with Fresnel-zone count (THEORY §10).

For the D=30 lenses the focal shift (THEORY §11) puts the peak short of the geometric focus, so
f_design = 100 µm is used to land the real peak near 80 µm. The D=92 lenses (high Fresnel
number) need no correction and use f_design = 80 µm.

## 4. Aperture and NA scaling

With the 80 µm working distance fixed, aperture D sets the NA and the geometric collection
ceiling η_geom = (1-cosθ)/2. FDTD cost scales as D².

| NA | D (µm) | η_geom | cylinders | ~FDTD cost |
|---|---|---|---|---|
| 0.18 | 30 | 0.86% | 2.8 k | 1.2 FC |
| 0.30 | 50 | 2.30% | 8 k | 3.4 FC |
| 0.40 | 70 | 4.17% | 15 k | 6.5 FC |
| 0.50 | 92 | 6.65% | 27 k | 16 FC |
| 0.60 | 120 | 10.0% | 45 k | 19 FC |

To collect more, enlarge the aperture or shorten the working distance. The actual single-mode
collection is computed per design in Phase 3, not from this table.

## 5. Materials

TiO2 (n=2.40) is the reference platform: at P=500 nm, h=800 nm it spans 1.06×2π. Si3N4 (n=2.00)
spans only 0.66×2π but still makes a good lens, because a full 2π is sufficient not necessary.
The figure of merit is the coherent Strehl, which stays at 88% on the Si3N4 library given the
max-projection assignment (THEORY §10). At each aperture the Si3N4 lens is within ~8% of TiO2.

ITO was recorded as a dead end for low contrast, but that used a lossless placeholder index
`permittivity=1.5147**2`. Real ITO at 866 nm is n ≈ 1.90 with k ≈ 0.014, contrast close to
Si3N4. Its true disqualifier is free-carrier absorption (~15% single-pass at 800 nm), not
contrast. The 96 cached ITO cells used the wrong index and cannot be rescaled, so
`regenerate_library.py` refuses ITO. A correct library needs a fresh lossy sweep. See THEORY §8.

## 6. Building a new design

Same platform (P=500 nm, h=800 nm), different aperture or focal length:

1. Add an entry to the DESIGNS dict at the top of `scripts/metalens_tidy3d.py` (mat, folder, D,
   f_work, f_design). Compute N = (D/2)²/(λf). If N ≲ 10 set f_design > f_work to pre-compensate
   the focal shift, else leave them equal.
2. Run `--layout` (free), then `scripts/scalar_focus_check.py` on the layout for a free
   Strehl/FWHM prediction.
3. Run `--lens` to see the cost estimate, then `--lens --submit` to run.
4. Run `--analyze --collect`.

A new material, wavelength, period, or height needs a new Phase 0 library.

## 7. Repository layout

```
data/          phase_library_<mat>.npz, sweep_data_<mat>.npz committed, cache ignored
scripts/       metalens_tidy3d (pipeline), regenerate_library, scalar_focus_check,
               summarize_designs, make_comparison_figures
notebooks/     Phase-0 unit-cell sweep and material comparison
figures/       phase-library and cross-design comparison PNGs
docs/          DESIGN_PROCEDURE (workflow), THEORY (equations)
designs/       one folder per design, each holding data/ (layout npz, FDTD hdf5) and figures/
lumerical/     single-file Lumerical FDTD port of the same pipeline
```

The whole Tidy3D pipeline is in one module, `scripts/metalens_tidy3d.py`, parameterized by
design. Large FDTD `.hdf5` files are gitignored and re-downloadable. The `.npz` artifacts and
figures are committed.
