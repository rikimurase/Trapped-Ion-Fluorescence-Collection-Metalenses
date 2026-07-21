# Lumerical FDTD implementation

Ansys Lumerical FDTD implementation of the metalens pipeline: unit-cell library, hyperbolic
layout, lens simulation, near-to-far projection, and reciprocity collection. Equations and
normalization are in ../docs/THEORY.md. The 80 um projection uses `farfieldexact3d` (exact
finite-distance, not the Fraunhofer `farfield3d`). Materials are constant index (SiO2 n=1.4528,
pillar 2.40/2.00, air). Everything is in `metalens_lumerical.py`, parameterized by design
(tio2_D30, tio2_D92, si3n4_D30, si3n4_D92).

## FDTD setup

| quantity | value |
|---|---|
| substrate | SiO2 box below z=0, air background |
| pillars | addcircle, radius from layout, z 0..H |
| source | Ex plane wave from z = -0.6 um, 866 nm |
| symmetry | x-min Anti-Symmetric, y-min Symmetric (quarter domain) |
| in-plane mesh | uniform dx=dy=dz = P/36 (~14 nm) over the active z-range |
| near-to-far | farfieldexact3d, near-field monitor at z = H + lambda/2 |
| E_inc | 2D Y-normal slice in SiO2, standing-wave corrected |
| transmission | transmission("nf") normalized to source power |

The layout, radius assignment, and reciprocity collection are engine-independent and reproduce
the Tidy3D reference to full precision (byte-identical layout arrays; eta_pi 6.316% / sigma
3.158% / unpol 4.211% on the si3n4_D92 focal field).

Symmetry: Lumerical's symmetry plane is at the center of the centered region with the condition
set at the min boundary. For an Ex normal-incidence source E is normal to the x=0 plane
(Anti-Symmetric) and tangential to the y=0 plane (Symmetric). `--full` disables symmetry (all
cylinders, PML on all sides) for a 4x cross-check.

## First-run checks

- Monitor wavelength. The lens run prints it; it should read 866.0 nm, otherwise adjust the
  source range in `_single_wavelength`.
- Symmetry and projection. `farfieldexact3d` accounts for symmetric BCs internally. Confirm on
  your build by comparing `--lens` and `--lens --full` on tio2_D30: focal field, FWHM, and eta_pi
  should agree to a percent.
- Materials. If pillars read index 1, set the object-defined dielectric material explicitly
  before index.
- lumapi requires Python 3.11 or 3.12.

## Workstation setup

1. Install Ansys Lumerical FDTD and confirm a license is reachable. `metalens_lumerical.py`
   auto-adds `C:\Program Files\Lumerical\v252\api\python` (and v242) to the path; edit those
   paths in `lumapi()` for a different version or location.

2. Python 3.11/3.12 environment with numpy, scipy, matplotlib:
   ```
   python -m venv .venv-lum
   .venv-lum\Scripts\pip install numpy scipy matplotlib
   .venv-lum\Scripts\python -c "import sys; sys.path.append(r'C:\Program Files\Lumerical\v252\api\python'); import lumapi"
   ```

3. The folder is self-contained (rebuilds its own library, writes to `data/` and `figures/`).

4. Run:
   ```
   python metalens_lumerical.py tio2_D30 --library    # 8 periodic unit-cell sims
   python metalens_lumerical.py tio2_D30 --layout      # instant
   python metalens_lumerical.py tio2_D30 --lens        # FDTD + projection + collection
   ```
   The library is per material. After a `--library` run, other designs of that material reuse it.
   `--all` chains all three. Inspect the saved `lens_<design>.fsp` in the GUI before large runs.

Acceptance: library span TiO2 ~1.06x2pi, Si3N4 ~0.66x2pi; D92 collection eta_pi ~6.9% (TiO2),
~6.3% (Si3N4).

## Compute

- Unit-cell library: trivial.
- D30 lens, symmetric: ~0.24 billion cells, tens of GB of RAM, hours. `--full` is 4x.
- D92 lens, symmetric: ~2 billion cells, hundreds of GB of RAM (cluster). `--full` not feasible.

## Note

The focusing efficiency uses H from the projected E via the angular spectrum (farfieldexact3d
returns E only), whereas the reference projects H directly. This affects only the
focusing-efficiency diagnostic, not the collection efficiency or the focal field.
