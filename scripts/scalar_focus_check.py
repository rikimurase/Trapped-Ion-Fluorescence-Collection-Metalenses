"""Scalar Rayleigh-Sommerfeld propagation of a metalens layout to its focal plane.

    python scripts/scalar_focus_check.py <path>/metalens_layout_<mat>.npz

Each cell is a point emitter t_j = |t_j| exp(i phi_j), propagated with the exact RS kernel in
homogeneous air above the pillars:

    E(X,Y,z) = (1/i*lda) sum_j t_j A_cell (z/R_j) exp(i k R_j)/R_j,  R_j = |(X,Y,z)-(x_j,y_j,0)|

Reports Strehl, FWHM (vs 0.514 lda/NA), and axial peak against an ideal-library reference and
compares the clip-to-range and max-projection radius-assignment rules. See docs/THEORY.md
section 10.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[1]


def rs_propagate(x, y, t, lda, Xg, Yg, z):
    """Rayleigh-Sommerfeld sum. Returns E on the (Xg, Yg) points at height z."""
    k = 2 * np.pi / lda
    E = np.zeros(Xg.shape, dtype=complex)
    # chunk over cells to bound memory
    for s in range(0, x.size, 4000):
        xs, ys, ts = x[s:s+4000], y[s:s+4000], t[s:s+4000]
        R = np.sqrt((Xg[..., None] - xs) ** 2 + (Yg[..., None] - ys) ** 2 + z ** 2)
        E += np.sum(ts * (z / R) * np.exp(1j * k * R) / R, axis=-1)
    return E / (1j * lda)


def main(layout_path):
    lay = np.load(layout_path, allow_pickle=True)
    x, y = np.asarray(lay["x_um"], float), np.asarray(lay["y_um"], float)
    phi_t = np.asarray(lay["phase_target_rad"], float)
    phi_r = np.asarray(lay["phase_realized_rad"], float)
    amp = np.asarray(lay["amplitude"], float)
    lda = float(lay["wavelength_um"]); D = float(lay["aperture_um"])
    f_work = float(lay["working_distance_um"]); NA = float(lay["NA"])
    label = str(lay["material"])

    lib = np.load(ROOT / "data" / f"phase_library_{label.lower()}.npz", allow_pickle=True)
    r_lib = np.asarray(lib["r_um"], float); phi_lib = np.asarray(lib["phase_rad"], float)
    amp_lib = np.asarray(lib["amplitude"], float)
    phi_of_r = CubicSpline(r_lib, phi_lib); amp_of_r = CubicSpline(r_lib, amp_lib)
    r_of_phi = CubicSpline(phi_lib, r_lib)
    span = (phi_lib[-1] - phi_lib[0]) / (2 * np.pi)
    print(f"{label}  D={D:.0f} um  NA={NA:.3f}  f_work={f_work:.0f} um  cells={x.size}")
    print(f"library span {span:.3f} x 2pi\n")

    # the three masks: ideal reference, this layout's max-projection, and clip-to-range
    r_clip = np.clip(r_of_phi(np.clip(phi_t, phi_lib[0], phi_lib[-1])), r_lib[0], r_lib[-1])
    masks = {
        "ideal library (>=2pi, reference)": np.exp(1j * phi_t),
        "max-projection (this layout)": amp * np.exp(1j * phi_r),
        "clip-to-range (TiO2 script rule)": amp_of_r(r_clip) * np.exp(1j * phi_of_r(r_clip)),
    }

    # on-axis focal amplitude -> Strehl
    zero = np.zeros((1, 1))
    E_ref = rs_propagate(x, y, masks["ideal library (>=2pi, reference)"], lda, zero, zero, f_work)
    I_ref = abs(E_ref[0, 0]) ** 2

    print(f"{'radius-assignment rule':<36} {'Strehl':>8} {'FWHM (um)':>11} {'peak z (um)':>12}")
    print("-" * 71)

    xs = np.linspace(-3.0, 3.0, 241)
    Xg, Yg = xs[None, :], np.zeros((1, xs.size))
    zs = np.linspace(f_work - 20, f_work + 20, 41)

    results = {}
    for name, t in masks.items():
        # focal-plane x-cut -> FWHM
        Ecut = rs_propagate(x, y, t, lda, Xg, Yg, f_work)[0]
        I = np.abs(Ecut) ** 2
        strehl = I.max() / I_ref  # peak on the cut (focus is on axis)
        half = I.max() / 2
        above = np.where(I >= half)[0]
        fwhm = xs[above[-1]] - xs[above[0]] if above.size > 1 else np.nan
        # axial scan -> true focal position
        Iax = np.array([abs(rs_propagate(x, y, t, lda, zero, zero, z)[0, 0]) ** 2 for z in zs])
        zpk = zs[np.argmax(Iax)]
        results[name] = (strehl, fwhm, zpk)
        print(f"{name:<36} {strehl*100:7.1f}% {fwhm:11.2f} {zpk:12.1f}")

    print("-" * 71)
    print(f"diffraction limit 0.514*lda/NA = {0.514*lda/NA:.2f} um")
    s_max = results["max-projection (this layout)"][0]
    s_clip = results["clip-to-range (TiO2 script rule)"][0]
    print(f"\nmax-projection buys {s_max/s_clip:.2f}x the focal intensity of clip-to-range.")
    print(f"Predicted collection: eta ~ {s_max*100:.0f}% of the same lens built on a full-2pi library.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("layout", type=Path, help="metalens_layout_*.npz from Phase 1")
    main(ap.parse_args().layout)
