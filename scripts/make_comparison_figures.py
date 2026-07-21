"""Cross-design comparison graphics.

    python scripts/make_comparison_figures.py

Reads every design under designs/ (layout .npz + lens_scale .hdf5) and writes two figures:
  design_comparison.png     collection efficiency by channel and design, with ceilings
  compare_focal_spots.png   focal-plane |E|^2 gallery and spot cuts, per aperture

Free: reads cached FDTD results, no cloud cost.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
COL = {"TiO2": "#2c7fb8", "Si3N4": "#d95f0e"}


def cone_fractions(NA):
    c = np.cos(np.arcsin(NA))
    return (1 - c) / 2, (3 / 8) * (4 / 3 - c - c ** 3 / 3)   # isotropic, perpendicular-dipole


def load(folder):
    lays = list(folder.glob("data/metalens_layout_*.npz"))
    if not lays:
        return None
    lay = np.load(lays[0], allow_pickle=True)
    tag = lays[0].stem.replace("metalens_layout_", "")
    hdf5 = folder / "data" / f"lens_scale_{tag}.hdf5"
    d = dict(tag=tag, label=str(lay["material"]), D=float(lay["aperture_um"]), NA=float(lay["NA"]),
             lda0=float(lay["wavelength_um"]), n_mode=float(lay["n_SiO2"]),
             amp=np.asarray(lay["amplitude"], float), has_fdtd=hdf5.exists())
    if not d["has_fdtd"]:
        return d

    freq0 = td.C_0 / d["lda0"]; k = 2 * np.pi / (d["lda0"] * 1e-6)
    A_ap = np.pi * ((d["D"] / 2) * 1e-6) ** 2
    sd = td.SimulationData.from_file(str(hdf5))
    proj = sd["focal_plane_proj"].fields_cartesian
    zc = float(proj.z.values[0])
    Ex, Ey, Ez = (proj[c].sel(f=freq0, z=zc).values for c in ("Ex", "Ey", "Ez"))
    xs, ys = proj.x.values, proj.y.values
    I = np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2
    ip, jp = np.unravel_index(np.argmax(I), I.shape)
    d.update(xs=xs, ys=ys, I2d=I, ip=ip, jp=jp,
             cutx=I[:, jp] / I.max(), cuty=I[ip, :] / I.max())

    def fwhm(coord, prof):
        prof = prof / prof.max(); a = np.where(prof >= 0.5)[0]
        if a.size < 2:
            return np.nan
        lo, hi = a[0], a[-1]
        L = np.interp(0.5, [prof[max(lo - 1, 0)], prof[lo]], [coord[max(lo - 1, 0)], coord[lo]])
        R = np.interp(0.5, [prof[min(hi + 1, len(coord) - 1)], prof[hi]],
                      [coord[min(hi + 1, len(coord) - 1)], coord[hi]])
        return abs(R - L)

    d["fwhm_x"], d["fwhm_y"] = fwhm(xs, I[:, jp]), fwhm(ys, I[ip, :])
    d["dl"] = 0.514 * d["lda0"] / d["NA"]

    # reciprocity collection
    Efx, Efy, Efz = (complex(proj[c].sel(f=freq0, z=zc).interp(x=0.0, y=0.0).values)
                     for c in ("Ex", "Ey", "Ez"))
    band = sd["xz"].Ex.sel(f=freq0).sel(x=slice(-d["D"] / 2, d["D"] / 2), z=slice(-0.55, -0.03))
    R_pow = max(0.0, 1.0 - float(np.mean(d["amp"] ** 2)))
    E_inc = np.sqrt(np.mean(np.abs(band.values) ** 2) / (1.0 + R_pow))
    pref = (3 * np.pi) / (2 * k ** 2 * d["n_mode"] * A_ap)
    Exm = np.array([Efx, Efy, Efz]); Eym = np.array([-Efy, Efx, Efz])
    into = lambda mE, ph: pref * abs(np.vdot(ph, mE)) ** 2 / E_inc ** 2
    pi_hat = np.array([1.0, 0, 0]); sig = np.array([0, 1, 1j]) / np.sqrt(2)
    d["eta_pi"] = into(Exm, pi_hat) + into(Eym, pi_hat)
    d["eta_sig"] = into(Exm, sig) + into(Eym, sig)
    d["eta_unpol"] = (d["eta_pi"] + 2 * d["eta_sig"]) / 3
    d["enh"] = np.sqrt(abs(Efx) ** 2 + abs(Efy) ** 2 + abs(Efz) ** 2) / E_inc
    d["iso"], d["perp"] = cone_fractions(d["NA"])
    return d


folders = sorted(p for p in (ROOT / "designs").iterdir() if p.is_dir())
D = [d for d in (load(f) for f in folders) if d and d["has_fdtd"]]
D.sort(key=lambda r: (r["D"], r["label"]))
apertures = sorted({r["D"] for r in D})
print(f"loaded {len(D)} designs with FDTD: " + ", ".join(f"{r['label']}-D{r['D']:.0f}" for r in D))


# collection efficiency by channel and design
fig, axes = plt.subplots(1, len(apertures), figsize=(6.4 * len(apertures), 5.2), squeeze=False)
axes = axes[0]
for ax, Dap in zip(axes, apertures):
    grp = [r for r in D if r["D"] == Dap]
    chans = ["eta_pi", "eta_sig", "eta_unpol"]; clabels = ["π (x̂)", "σ± both", "unpolarized"]
    x = np.arange(len(chans)); w = 0.8 / len(grp)
    for gi, r in enumerate(grp):
        vals = [r[c] * 100 for c in chans]
        bars = ax.bar(x + (gi - (len(grp) - 1) / 2) * w, vals, w,
                      color=COL.get(r["label"]), label=f"{r['label']}")
        for xi, v in zip(x + (gi - (len(grp) - 1) / 2) * w, vals):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    r0 = grp[0]
    ax.axhline(r0["iso"] * 100, ls=":", color="k", lw=1.2,
               label=f"isotropic ceiling {r0['iso']*100:.2f}%")
    ax.axhline(r0["perp"] * 100, ls="--", color="gray", lw=1.2,
               label=f"⊥-dipole ceiling {r0['perp']*100:.2f}%")
    ax.set_xticks(x); ax.set_xticklabels(clabels)
    ax.set(ylabel="collection efficiency (%)",
           title=f"D = {Dap:.0f} µm   (NA {r0['NA']:.2f})")
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8, loc="upper right")
fig.suptitle("Collection efficiency by material, aperture, and dipole channel\n"
             "866 nm Ca+, reciprocity method",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG / "design_comparison.png", dpi=160, bbox_inches="tight", facecolor="white")
print("saved -> figures/design_comparison.png")
plt.close(fig)


# focal-spot gallery and cuts
fig, axes = plt.subplots(len(apertures), len(D) // len(apertures) + 1,
                         figsize=(4.6 * (len(D) // len(apertures) + 1), 4.4 * len(apertures)))
for row, Dap in enumerate(apertures):
    grp = [r for r in D if r["D"] == Dap]
    for col, r in enumerate(grp):
        ax = axes[row, col]
        im = ax.pcolormesh(r["ys"], r["xs"], r["I2d"], cmap="magma", shading="auto")
        ax.set(aspect="equal", xlabel="y (µm)", ylabel="x (µm)",
               title=f"{r['label']}  D={r['D']:.0f} µm\n|E|² @ focus, enh {r['enh']:.0f}×")
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        plt.colorbar(im, ax=ax, fraction=0.046)
    axc = axes[row, -1]
    for r in grp:
        axc.plot(r["xs"], r["cutx"], color=COL.get(r["label"]),
                 label=f"{r['label']} x  FWHM {r['fwhm_x']:.2f}")
        axc.plot(r["ys"], r["cuty"], color=COL.get(r["label"]), ls=":",
                 label=f"{r['label']} y  FWHM {r['fwhm_y']:.2f}")
    axc.axhline(0.5, ls=":", color="gray", lw=0.8)
    axc.set(xlabel="position (µm)", ylabel="normalized |E|²",
            title=f"spot cuts (limit {grp[0]['dl']:.2f} µm)", xlim=(-4, 4))
    axc.legend(fontsize=7); axc.grid(alpha=0.3)
fig.suptitle("Focal spots, all designs diffraction-limited, round, on-axis",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG / "compare_focal_spots.png", dpi=160, bbox_inches="tight", facecolor="white")
print("saved -> figures/compare_focal_spots.png")
plt.close(fig)


print("\ncomparison figures written to figures/")
