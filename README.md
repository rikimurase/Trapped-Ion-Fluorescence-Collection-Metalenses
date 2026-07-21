# 866 nm Ca+ collection metalens

Cylindrical-pillar metalenses for collecting 866 nm fluorescence from a trapped 40Ca+ ion at an
80 um working distance, in TiO2 (n=2.4) and Si3N4 (n=2.0) on SiO2, at NA 0.18 and 0.50. Design by
the local periodic approximation over a radius-swept meta-atom library, validation by full-wave
Tidy3D FDTD with exact near-to-far projection, and collection efficiency by Lorentz reciprocity.
The pipeline (library, hyperbolic layout, lens FDTD, analysis) is one parameterized module,
`scripts/metalens_tidy3d.py`. A Lumerical FDTD port is in `lumerical/`.

- Method and per-design results: [docs/DESIGN_PROCEDURE.md](docs/DESIGN_PROCEDURE.md)
- Equations, normalization, and assumptions: [docs/THEORY.md](docs/THEORY.md)

## Requirements

Python 3.11+ with the packages in `requirements.txt` (numpy, scipy, matplotlib, tidy3d). Layout,
the scalar check, and figure generation run offline for free. A lens-scale FDTD needs a Tidy3D
account and FlexCredits; configure the key once with `tidy3d configure` and do not hardcode it.
The Lumerical port under `lumerical/` additionally needs an Ansys Lumerical FDTD install and its
bundled `lumapi` (see [lumerical/README.md](lumerical/README.md)).

## Quickstart

```
pip install -r requirements.txt

# free, offline (uses the committed phase-library .npz)
python scripts/metalens_tidy3d.py si3n4_D92 --layout
python scripts/scalar_focus_check.py designs/si3n4_D92um_f80um_866nm/data/metalens_layout_si3n4.npz

# lens-scale FDTD (spends FlexCredits); --lens alone only prints the cost estimate
python scripts/metalens_tidy3d.py si3n4_D92 --lens --submit
python scripts/metalens_tidy3d.py si3n4_D92 --analyze --collect
```

The FDTD field files (`.hdf5`) are not committed, so `summarize_designs.py` and
`make_comparison_figures.py` reproduce their tables and figures only after a lens run has written
them locally. The committed figures and the design table below already show the results.

## Layout

```
data/          phase_library_<mat>.npz, sweep_data_<mat>.npz (committed); cache/ (ignored)
scripts/       metalens_tidy3d (pipeline), regenerate_library, scalar_focus_check,
               summarize_designs, make_comparison_figures
notebooks/     Phase-0 unit-cell sweep and material comparison
figures/       phase_library_comparison + 2 cross-design comparison PNGs
docs/          DESIGN_PROCEDURE (how), THEORY (why)
lumerical/     single-file Lumerical FDTD port of the pipeline
designs/       one folder per design, each holding data/ and figures/:
                 tio2_D30um_f80um_866nm    D=30 µm, NA 0.184
                 si3n4_D30um_f80um_866nm   D=30 µm, NA 0.184
                 tio2_D92um_f80um_866nm    D=92 µm, NA 0.498
                 si3n4_D92um_f80um_866nm   D=92 µm, NA 0.498
```

Meta-atom platform: P = 500 nm, h = 800 nm, r in [40, 230] nm. TiO2 (n=2.4) spans 1.06·2π and
Si3N4 (n=2.0) spans 0.66·2π. A different wavelength, material, period, or height needs a new
Phase-0 library (re-run the unit-cell sweep and `regenerate_library.py`).

## Designs

Four designs, two apertures by two materials, all diffraction-limited and FDTD-validated.
Collection η_π is the single-dipole efficiency with both linear channels detected. See
`figures/design_comparison.png` and the design table in DESIGN_PROCEDURE.

| design | NA | library | η_π (unpol.) | FDTD |
|---|---|---|---|---|
| tio2_D30 | 0.184 | 1.06·2π | 0.68% (0.46%) | 1.2 FC |
| si3n4_D30 | 0.184 | 0.66·2π | 0.67% (0.45%) | 0.94 FC |
| tio2_D92 | 0.498 | 1.06·2π | 6.87% (4.58%) | 15.9 FC |
| si3n4_D92 | 0.498 | 0.66·2π | 6.32% (4.21%) | 6.13 FC |

At each aperture Si3N4 nearly matches TiO2 despite its shorter library, losing ~1% at NA 0.18
and ~8% at NA 0.50 (the short-library penalty grows with Fresnel-zone count, THEORY §10). The
Si3N4 D=92 FDTD (6.32%) matched its free scalar prediction (6.32%) to three digits. Collection
is set by NA, not lens quality: the 10x jump from D=30 to D=92 is solid-angle scaling.

## Materials note

The notebooks' original "ITO and Si3N4 are dead ends" conclusion is wrong on both counts
(THEORY §8, §10).

- Si3N4 is viable. A full 2π span is sufficient, not necessary. The figure of merit is the
  coherent Strehl, and a 0.66·2π library keeps 88% of it given the max-projection radius
  assignment (mapping unreachable target phases to the nearest point on the phase circle rather
  than clipping to the library endpoint).
- The ITO sweeps used a lossless placeholder index `permittivity=1.5147**2`. Real ITO at 866 nm
  is n ≈ 1.90, k ≈ 0.014, contrast close to Si3N4. Its real disqualifier is absorption, not
  contrast. The cached ITO cells used the wrong index and cannot be rescaled.

## Notes

- Configure your own Tidy3D API key with `tidy3d configure`. Do not hardcode it.
- Single-wavelength, non-dispersive design. Wideband validation is deferred.
- Before spending FlexCredits on a lens-scale FDTD, run `scripts/scalar_focus_check.py` on the
  layout for a free Strehl/FWHM/focus prediction.
- FDTD `.hdf5` field files are gitignored (large, and tied to a Tidy3D account). The committed
  `.npz` artifacts, figures, and result tables document the outputs. Reproducing the collection
  and focus numbers requires a lens FDTD run.
