"""Rebuild a meta-atom phase library from the local Tidy3D cache. Run from the repo root.

    python scripts/regenerate_library.py tio2
    python scripts/regenerate_library.py si3n4

Offline and free, reads the cached unit-cell sweeps in data/cache. ITO is absent by design:
its 96 cached cells used a lossless placeholder n=1.5147, whereas real ITO at 866 nm is
n~1.90, k~0.014, so they cannot be rescaled and a correct ITO library needs a fresh sweep.
"""
import argparse
import glob
import warnings

import numpy as np
import tidy3d as td
from scipy.interpolate import CubicSpline

warnings.filterwarnings("ignore")

LDA0 = 0.866
FREQ0 = td.C_0 / LDA0
P_LIST = np.array([0.30, 0.40, 0.50])
H_LIST = np.array([0.20, 0.40, 0.60, 0.80])

# tag -> (label, n, locked (P, h)). Cached-index trustworthy for both.
MATERIALS = {
    "tio2": ("TiO2", 2.4, (0.500, 0.800)),
    "si3n4": ("Si3N4", 2.0, (0.500, 0.800)),
}


def r_list_for_P(P_val, n=8, gap=0.02):
    return np.linspace(0.04, P_val / 2 - gap, n)


def classify(sim):
    """(period, radius, height, eps) of a unit-cell sim, or None."""
    tooth = next((s for s in sim.structures if type(s.geometry).__name__ == "Cylinder"), None)
    if tooth is None or getattr(tooth.medium, "permittivity", None) is None:
        return None
    return (float(sim.size[0]), float(tooth.geometry.radius),
            float(tooth.geometry.length), float(tooth.medium.permittivity))


def main(tag):
    label, n_mat, (P_lock, h_lock) = MATERIALS[tag]
    eps_mat = n_mat ** 2

    # Index the cache: (P, h) -> {r: t00}.
    found = {}
    files = sorted(glob.glob("data/cache/fdve-*.hdf5"))
    print(f"scanning {len(files)} cached files for {label} (eps={eps_mat:.3f}) ...")
    for f in files:
        try:
            sd = td.SimulationData.from_file(f)
        except Exception:
            continue
        if "t_orders" not in [m.name for m in sd.simulation.monitors]:
            continue
        info = classify(sd.simulation)
        if info is None or abs(info[3] - eps_mat) > 1e-2:
            continue
        try:
            t00 = complex(sd["t_orders"].amps.sel(
                f=FREQ0, polarization="p", orders_x=0, orders_y=0).values)
        except Exception:
            continue
        P_snap = float(P_LIST[np.argmin(np.abs(P_LIST - info[0]))])
        h_snap = float(H_LIST[np.argmin(np.abs(H_LIST - info[2]))])
        found.setdefault((P_snap, h_snap), {})[round(info[1], 4)] = t00

    data_per_P = {}
    missing = 0
    for P_val in P_LIST:
        r_vals = r_list_for_P(P_val)
        t = np.full((len(r_vals), len(H_LIST)), np.nan, dtype=complex)
        for i, r in enumerate(r_vals):
            for j, h in enumerate(H_LIST):
                cell = found.get((float(P_val), float(h)), {})
                key = next((rk for rk in cell if abs(rk - r) < 1e-3), None)
                if key is None:
                    missing += 1
                else:
                    t[i, j] = cell[key]
        phase = np.unwrap(np.angle(t), axis=0)
        phase -= phase[0:1, :]
        data_per_P[float(P_val)] = dict(r=r_vals, t=t, phase=phase)

    print(f"missing cells: {missing}" if missing else f"all 96 {label} cells recovered")
    np.savez(f"data/sweep_data_{tag}.npz", data_per_P=data_per_P)
    print(f"saved -> data/sweep_data_{tag}.npz")

    print("\nphase span (x2pi) per (P, h):")
    for P_val in P_LIST:
        d = data_per_P[float(P_val)]
        spans = " ".join(
            f"h{h*1e3:.0f}={((d['phase'][-1, j] - d['phase'][0, j]) / (2*np.pi)):.2f}"
            for j, h in enumerate(H_LIST))
        print(f"  P={P_val*1e3:.0f} nm:  {spans}")

    # Inverse library at the locked point.
    d = data_per_P[P_lock]
    j = int(np.argmin(np.abs(H_LIST - h_lock)))
    r_data, phi_data, amp_data = d["r"], d["phase"][:, j], np.abs(d["t"][:, j])
    assert np.all(np.diff(phi_data) > 0), "phase must be monotonic in r to invert"
    probe = np.linspace(r_data[0] * 1.01, r_data[-1] * 0.99, 50)
    rt_err = np.max(np.abs(
        probe - CubicSpline(phi_data, r_data)(CubicSpline(r_data, phi_data)(probe)))) * 1e3
    span = (phi_data[-1] - phi_data[0]) / (2 * np.pi)
    print(f"\nlocked point P={P_lock*1e3:.0f} nm, h={h_lock*1e3:.0f} nm: "
          f"span {span:.3f}x2pi, mean |t|={amp_data.mean():.3f}, round-trip err {rt_err:.3f} nm")
    if span < 1.0:
        print(f"  NOTE: span < 2pi. Targets outside [0, {span:.3f}x2pi) are unreachable; the "
              f"layout must map them to the nearest reachable phase ON THE CIRCLE (not clip to "
              f"the range endpoint). See designs/*/metalens_layout.py.")

    np.savez(f"data/phase_library_{tag}.npz",
             material=label, n_material=n_mat, wavelength_um=LDA0,
             period_um=P_lock, height_um=h_lock,
             r_um=r_data, phase_rad=phi_data, amplitude=amp_data, complex_t=d["t"][:, j],
             sweep_source="regenerate_library.py (offline cache extraction)")
    print(f"saved -> data/phase_library_{tag}.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("material", choices=sorted(MATERIALS), help="meta-atom material tag")
    main(ap.parse_args().material)
