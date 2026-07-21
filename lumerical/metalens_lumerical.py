"""866 nm collection metalens pipeline in Lumerical FDTD.

Builds the meta-atom phase library from a unit-cell sweep, places pillars from a hyperbolic
phase mask, simulates the assembled lens, projects the near field to the focus with
farfieldexact3d, and computes collection efficiency by reciprocity.

    python metalens_lumerical.py tio2_D30 --library      # unit-cell sweep -> phase library
    python metalens_lumerical.py tio2_D30 --layout       # pillar layout (no FDTD)
    python metalens_lumerical.py tio2_D30 --lens         # lens FDTD, projection, analysis
    python metalens_lumerical.py tio2_D30 --lens --full  # no symmetry (4x cost)
    python metalens_lumerical.py tio2_D30 --all

Requires Ansys Lumerical FDTD with lumapi (tested on Python 3.11/3.12). The lens run prints the
monitor wavelength for a quick sanity check.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"; FIG = HERE / "figures"; DATA.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
UM = 1e-6; Z0 = 376.730313668; C0 = 299792458.0
LDA0 = 0.866; P = 0.500; H = 0.800; N_SIO2 = 1.4528
NMAT = {"tio2": 2.40, "si3n4": 2.00}
DESIGNS = {
    "tio2_D30":  dict(mat="tio2",  D=30.0, f_work=80.0, f_design=100.0),
    "tio2_D92":  dict(mat="tio2",  D=92.0, f_work=80.0, f_design=80.0),
    "si3n4_D30": dict(mat="si3n4", D=30.0, f_work=80.0, f_design=100.0),
    "si3n4_D92": dict(mat="si3n4", D=92.0, f_work=80.0, f_design=80.0),
}


def lumapi():
    for p in (r"C:\Program Files\Lumerical\v252\api\python",
              r"C:\Program Files\Lumerical\v242\api\python"):
        if Path(p).is_dir() and p not in sys.path:
            sys.path.append(p)
    import lumapi
    return lumapi


def _single_wavelength(fdtd):
    # broadband pulse centered in frequency at LDA0; monitors record the single center point
    fdtd.setglobalsource("wavelength start", (LDA0 / 1.1) * UM)
    fdtd.setglobalsource("wavelength stop", (LDA0 / 0.9) * UM)
    fdtd.setglobalmonitor("use source limits", 1)
    fdtd.setglobalmonitor("frequency points", 1)


def _dielectric(fdtd, name, index):
    fdtd.setnamed(name, "material", "<Object defined dielectric>")
    fdtd.setnamed(name, "index", float(index))


def r_list():
    return np.linspace(0.04, P / 2 - 0.02, 8)


# unit-cell phase library
def _unit_cell(fdtd, r, n_pillar):
    fdtd.switchtolayout(); fdtd.deleteall()
    zsrc, znf = -0.6, H + 0.5 * LDA0
    fdtd.addfdtd(dimension="3D", x=0, y=0, x_span=P * UM, y_span=P * UM,
                 z_min=(zsrc - LDA0) * UM, z_max=(znf + LDA0) * UM, mesh_accuracy=3)
    fdtd.setnamed("FDTD", "x min bc", "Periodic"); fdtd.setnamed("FDTD", "y min bc", "Periodic")
    fdtd.setnamed("FDTD", "z min bc", "PML"); fdtd.setnamed("FDTD", "z max bc", "PML")
    fdtd.addrect(name="sub", x=0, y=0, x_span=2 * P * UM, y_span=2 * P * UM,
                 z_min=(zsrc - LDA0) * UM, z_max=0.0); _dielectric(fdtd, "sub", N_SIO2)
    fdtd.addcircle(name="pillar", x=0, y=0, radius=r * UM, z_min=0.0, z_max=H * UM)
    _dielectric(fdtd, "pillar", n_pillar)
    fdtd.addplane(name="src", injection_axis="z-axis", direction="forward", x=0, y=0,
                  x_span=2 * P * UM, y_span=2 * P * UM, z=zsrc * UM)
    fdtd.setnamed("src", "polarization angle", 0.0)
    fdtd.addpower(name="T", monitor_type="2D Z-normal", x=0, y=0, x_span=P * UM, y_span=P * UM, z=znf * UM)
    fdtd.addmesh(name="mo", x=0, y=0, x_span=P * UM, y_span=P * UM,
                 z_min=(zsrc - 0.1) * UM, z_max=(znf + 0.1) * UM,
                 dx=P / 36 * UM, dy=P / 36 * UM, dz=P / 36 * UM)
    _single_wavelength(fdtd)


def run_library(mat):
    lm = lumapi(); fdtd = lm.FDTD(hide=True)
    rs = r_list(); t = np.zeros(len(rs), complex)
    for i, r in enumerate(rs):
        _unit_cell(fdtd, r, NMAT[mat]); fdtd.run()
        Tp = float(np.real(fdtd.transmission("T")).ravel()[-1])
        Ex = np.asarray(fdtd.getdata("T", "Ex")).ravel()          # (0,0) order = spatial mean
        t[i] = np.sqrt(max(Tp, 0.0)) * np.exp(1j * np.angle(np.mean(Ex)))
        fdtd.switchtolayout()
    fdtd.close()
    phase = np.unwrap(np.angle(t)); phase -= phase[0]; amp = np.abs(t)
    if not np.all(np.diff(phase) > 0):
        print("WARN: phase not monotonic in r; inverse library may be unreliable")
    span = (phase[-1] - phase[0]) / (2 * np.pi)
    np.savez(DATA / f"phase_library_{mat}.npz", material=mat.replace("si3n4", "Si3N4").replace("tio2", "TiO2"),
             n_material=NMAT[mat], wavelength_um=LDA0, period_um=P, height_um=H,
             r_um=rs, phase_rad=phase, amplitude=amp, complex_t=t, sweep_source="lumerical")
    print(f"[library {mat}] span {span:.3f}x2pi  mean|t| {amp.mean():.3f}  -> phase_library_{mat}.npz")


# pillar layout
def build_layout(design_key):
    d = DESIGNS[design_key]; mat = d["mat"]
    lib = np.load(DATA / f"phase_library_{mat}.npz", allow_pickle=True)
    r_lib = np.asarray(lib["r_um"], float); phi_lib = np.asarray(lib["phase_rad"], float)
    amp_lib = np.asarray(lib["amplitude"], float); label = str(lib["material"])
    phi_of_r = CubicSpline(r_lib, phi_lib); amp_of_r = CubicSpline(r_lib, amp_lib)
    k0 = 2 * np.pi / LDA0; span = (phi_lib[-1] - phi_lib[0]) / (2 * np.pi)
    D, f_work, f_design = d["D"], d["f_work"], d["f_design"]; r_ap = D / 2
    ns = int(np.floor(r_ap / P)); coords = np.arange(-ns, ns + 1) * P
    X, Y = np.meshgrid(coords, coords); rho = np.sqrt(X ** 2 + Y ** 2); m = rho <= r_ap
    xc, yc, rho = X[m], Y[m], rho[m]
    phi_t = np.mod(-k0 * (np.sqrt(rho ** 2 + f_design ** 2) - f_design), 2 * np.pi)
    if mat == "tio2":  # full library: clip to range
        r_of_phi = CubicSpline(phi_lib, r_lib)
        r_as = np.clip(r_of_phi(np.clip(phi_t, phi_lib[0], phi_lib[-1])), r_lib[0], r_lib[-1])
    else:              # short library: max-projection
        rg = np.linspace(r_lib[0], r_lib[-1], 2001)
        proj = amp_of_r(rg)[None, :] * np.cos(phi_of_r(rg)[None, :] - phi_t[:, None])
        r_as = rg[np.argmax(proj, axis=1)]
    phi_r = phi_of_r(r_as); amp_r = amp_of_r(r_as)
    err = np.angle(np.exp(1j * (phi_r - phi_t))); strehl = abs(np.mean(amp_r * np.exp(1j * err))) ** 2
    NA = np.sin(np.arctan(r_ap / f_work))
    np.savez(DATA / f"metalens_layout_{design_key}.npz", material=label, wavelength_um=LDA0, period_um=P,
             height_um=H, n_material=NMAT[mat], n_SiO2=N_SIO2, aperture_um=D, focal_length_um=f_design,
             working_distance_um=f_work, NA=NA, x_um=xc, y_um=yc, rho_um=rho, radius_um=r_as,
             phase_target_rad=phi_t, phase_realized_rad=phi_r, amplitude=amp_r,
             library_span_x2pi=span, strehl_scalar=strehl)
    print(f"[layout {design_key}] N={xc.size} NA={NA:.3f} span={span:.2f}x2pi Strehl={strehl*100:.1f}%")


# lens FDTD, projection, focus metrics
def _H_from_E(Ex, Ey, Ez, dx, dy, k):
    kx = 2 * np.pi * np.fft.fftfreq(Ex.shape[0], dx); ky = 2 * np.pi * np.fft.fftfreq(Ex.shape[1], dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij"); KZ = np.sqrt((k ** 2 - KX ** 2 - KY ** 2).astype(complex))
    F = np.fft.fft2; iF = np.fft.ifft2; Exf, Eyf, Ezf = F(Ex), F(Ey), F(Ez); pref = 1.0 / (Z0 * k)
    Hx = iF(pref * (KY * Ezf - KZ * Eyf)); Hy = iF(pref * (KZ * Exf - KX * Ezf))
    return Hx, Hy


def _fwhm(coord, prof):
    prof = np.asarray(prof) / np.max(prof); a = np.where(prof >= 0.5)[0]
    if a.size < 2:
        return np.nan
    lo, hi = a[0], a[-1]
    L = np.interp(0.5, [prof[max(lo - 1, 0)], prof[lo]], [coord[max(lo - 1, 0)], coord[lo]])
    R = np.interp(0.5, [prof[min(hi + 1, len(coord) - 1)], prof[hi]], [coord[min(hi + 1, len(coord) - 1)], coord[hi]])
    return abs(R - L)


def _project(fdtd, mon, xs_um, ys_um, z_rel_um):
    # coordinates are relative to the monitor center; symmetry BCs are handled internally
    x = np.asarray(xs_um, float) * UM; y = np.asarray(ys_um, float) * UM
    z = np.atleast_1d(z_rel_um).astype(float) * UM
    return np.asarray(fdtd.farfieldexact3d(mon, x, y, z, {"field": "E"}))  # (Nx, Ny, Nz, 3)


def run_lens(design_key, full=False):
    lay = np.load(DATA / f"metalens_layout_{design_key}.npz", allow_pickle=True)
    x_all = np.asarray(lay["x_um"], float); y_all = np.asarray(lay["y_um"], float)
    r_all = np.asarray(lay["radius_um"], float); D = float(lay["aperture_um"])
    f_work = float(lay["working_distance_um"]); f_design = float(lay["focal_length_um"])
    n_mat = float(lay["n_material"]); NA = float(lay["NA"]); amp = np.asarray(lay["amplitude"], float)
    label = str(lay["material"])
    if full:                                   # all cylinders, no symmetry
        xq, yq, rq = x_all, y_all, r_all
    else:                                       # quarter domain, symmetry completes the lens
        q = (x_all >= -1e-9) & (y_all >= -1e-9); xq, yq, rq = x_all[q], y_all[q], r_all[q]
    zsrc, znf = -0.6, H + 0.5 * LDA0; half = D / 2 + 1.0

    lm = lumapi(); fdtd = lm.FDTD(hide=True); fdtd.deleteall()
    fdtd.addfdtd(dimension="3D", x=0, y=0, x_span=2 * half * UM, y_span=2 * half * UM,
                 z_min=(zsrc - LDA0) * UM, z_max=(znf + LDA0) * UM, mesh_accuracy=2)
    if full:
        for b in ("x min bc", "y min bc"):
            fdtd.setnamed("FDTD", b, "PML")
    else:
        # Ex source: E normal to x=0 (anti-symmetric), tangential to y=0 (symmetric)
        fdtd.setnamed("FDTD", "x min bc", "Anti-Symmetric"); fdtd.setnamed("FDTD", "y min bc", "Symmetric")
    fdtd.setnamed("FDTD", "x max bc", "PML"); fdtd.setnamed("FDTD", "y max bc", "PML")
    fdtd.setnamed("FDTD", "z min bc", "PML"); fdtd.setnamed("FDTD", "z max bc", "PML")
    fdtd.addrect(name="sub", x=0, y=0, x_span=4 * half * UM, y_span=4 * half * UM,
                 z_min=(zsrc - LDA0) * UM, z_max=0.0); _dielectric(fdtd, "sub", N_SIO2)
    fdtd.redrawoff()
    for i, (x, y, r) in enumerate(zip(xq, yq, rq)):
        fdtd.addcircle(name=f"p{i}", x=float(x) * UM, y=float(y) * UM, radius=float(r) * UM,
                       z_min=0.0, z_max=H * UM, index=float(n_mat))
    fdtd.redrawon()
    fdtd.addplane(name="src", injection_axis="z-axis", direction="forward", x=0, y=0,
                  x_span=4 * half * UM, y_span=4 * half * UM, z=zsrc * UM)
    fdtd.setnamed("src", "polarization angle", 0.0)
    fdtd.addpower(name="nf", monitor_type="2D Z-normal", x=0, y=0, x_span=2 * half * UM,
                  y_span=2 * half * UM, z=znf * UM)                       # near field for projection
    fdtd.addpower(name="xz", monitor_type="2D Y-normal", x=0, y=0, x_span=2 * half * UM,
                  z_min=(zsrc - LDA0) * UM, z_max=(znf + LDA0) * UM)       # incident-field slice for E_inc
    fdtd.addmesh(name="mo", x=0, y=0, x_span=2 * half * UM, y_span=2 * half * UM,
                 z_min=(zsrc - 0.1) * UM, z_max=(znf + 0.1) * UM,
                 dx=P / 36 * UM, dy=P / 36 * UM, dz=P / 36 * UM)
    _single_wavelength(fdtd)
    fdtd.save(str(DATA / f"lens_{design_key}.fsp")); fdtd.run()
    lam = C0 / float(np.asarray(fdtd.getdata("nf", "f")).ravel()[0])
    print(f"[lens {design_key}] {'full' if full else 'symmetric'}  cylinders {x_all.size}->{xq.size}  "
          f"monitor wavelength {lam*1e9:.1f} nm")

    xs = np.linspace(-8.0, 8.0, 161); z_rel = (H + f_work) - znf
    Ef = _project(fdtd, "nf", xs, xs, z_rel)[:, :, 0, :]
    Ex, Ey, Ez = Ef[..., 0], Ef[..., 1], Ef[..., 2]
    I = np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2
    ip, jp = np.unravel_index(np.argmax(I), I.shape)
    fwx, fwy = _fwhm(xs, I[:, jp]), _fwhm(xs, I[ip, :]); dl = 0.514 * LDA0 / NA
    zscan = np.linspace(f_work - 18.0, f_work + 18.0, 37)
    Eax = _project(fdtd, "nf", [0.0], [0.0], (H + zscan) - znf)[0, 0, :, :]
    Iax = np.sum(np.abs(Eax) ** 2, axis=1); z_true = float(zscan[np.argmax(Iax)])
    P_trans = float(np.real(fdtd.transmission("nf")).ravel()[-1])
    dx = (xs[1] - xs[0]) * UM; Hx, Hy = _H_from_E(Ex, Ey, Ez, dx, dx, 2 * np.pi / (LDA0 * UM))
    Sz = 0.5 * np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))
    XX, YY = np.meshgrid(xs - xs[ip], xs - xs[jp], indexing="ij"); Rspot = 3.0 * 0.5 * np.nanmean([fwx, fwy])
    eff = float(np.sum(Sz[(XX ** 2 + YY ** 2) <= Rspot ** 2]) * dx * dx) / P_trans if P_trans else np.nan

    E0 = _project(fdtd, "nf", [0.0], [0.0], [z_rel])[0, 0, 0, :]
    xz_x = np.asarray(fdtd.getdata("xz", "x")).ravel() / UM
    xz_z = np.asarray(fdtd.getdata("xz", "z")).ravel() / UM
    xz_Ex = np.asarray(fdtd.getdata("xz", "Ex")).squeeze()
    fdtd.close()

    zb = (xz_z >= -0.55) & (xz_z <= -0.03); xb = (np.abs(xz_x) <= D / 2)
    band = xz_Ex[np.ix_(xb, zb)] if xz_Ex.ndim == 2 else xz_Ex
    R_pow = max(0.0, 1.0 - float(np.mean(amp ** 2)))
    E_inc = np.sqrt(np.mean(np.abs(band) ** 2) / (1.0 + R_pow))
    enh = np.sqrt(np.sum(np.abs(E0) ** 2)) / E_inc

    eta = _collection(E0, E_inc, D, f_work, NA)
    np.savez(DATA / f"lens_results_{design_key}.npz", xs=xs, I=I, cutx=I[:, jp] / I.max(),
             cuty=I[ip, :] / I.max(), zscan=zscan, Iax=Iax / Iax.max(), z_true=z_true, fwx=fwx, fwy=fwy,
             dl=dl, P_trans=P_trans, eff=eff, enh=enh, E_focal=E0, E_inc=E_inc, NA=NA, D=D, label=label,
             f_work=f_work, f_design=f_design, **eta)
    print(f"   FWHM {fwx:.2f}/{fwy:.2f} (limit {dl:.2f})  focus {z_true:.1f}um  T {P_trans*100:.0f}%  "
          f"eff {eff*100:.0f}%  enh {enh:.1f}x")
    print(f"   eta_pi {eta['eta_pi']*100:.3f}%  sigma+/- {eta['eta_sig']*100:.3f}%  unpol {eta['eta_unpol']*100:.3f}%")
    _figures(design_key)


# reciprocity collection
def _collection(E_focal, E_inc, D, f_work, NA):
    k = 2 * np.pi / (LDA0 * UM); A_ap = np.pi * ((D / 2) * UM) ** 2
    pref = (3 * np.pi) / (2 * k ** 2 * N_SIO2 * A_ap)
    Efx, Efy, Efz = E_focal
    Ex = np.array([Efx, Efy, Efz]); Ey = np.array([-Efy, Efx, Efz])
    into = lambda mE, ph: pref * abs(np.vdot(ph, mE)) ** 2 / E_inc ** 2
    pi_hat = np.array([1.0, 0, 0]); sig = np.array([0, 1, 1j]) / np.sqrt(2)
    eta_pi = into(Ex, pi_hat) + into(Ey, pi_hat); eta_sig = into(Ex, sig) + into(Ey, sig)
    c = np.cos(np.arcsin(NA)); axial = 0.5 - 0.75 * c + 0.25 * c ** 3
    return dict(eta_pi=eta_pi, eta_sig=eta_sig, eta_unpol=(eta_pi + 2 * eta_sig) / 3,
                eta_axial=axial, eta_geom=(1 - c) / 2)


# figures
def _figures(design_key):
    r = np.load(DATA / f"lens_results_{design_key}.npz", allow_pickle=True)
    xs = r["xs"]; label = str(r["label"]); D = float(r["D"]); NA = float(r["NA"])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    im = ax[0].pcolormesh(xs, xs, r["I"], cmap="magma", shading="auto"); ax[0].set_aspect("equal")
    ax[0].set(xlabel="y (um)", ylabel="x (um)", title=f"{label} focal |E|^2  enh {float(r['enh']):.0f}x")
    plt.colorbar(im, ax=ax[0])
    ax[1].plot(xs, r["cutx"], label=f"x FWHM {float(r['fwx']):.2f}"); ax[1].plot(xs, r["cuty"], label=f"y FWHM {float(r['fwy']):.2f}")
    ax[1].axhline(0.5, ls=":", c="gray"); ax[1].set(xlabel="pos (um)", ylabel="norm |E|^2",
              title=f"cuts (limit {float(r['dl']):.2f} um)", xlim=(-6, 6)); ax[1].legend(); ax[1].grid(alpha=0.3)
    ax[2].plot(r["zscan"], r["Iax"], "-o", ms=3); ax[2].axvline(r["f_work"], ls="--", c="r", label=f"ion {float(r['f_work']):.0f}")
    ax[2].axvline(r["z_true"], ls="-", c="g", label=f"peak {float(r['z_true']):.0f}")
    ax[2].set(xlabel="z above lens (um)", ylabel="on-axis |E|^2", title="axial scan"); ax[2].legend(); ax[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(FIG / f"lens_focus_{design_key}.png", dpi=140, bbox_inches="tight", facecolor="white"); plt.close(fig)

    ep, es = float(r["eta_pi"]), float(r["eta_sig"]); eu = float(r["eta_unpol"])
    v = np.array([ep, es, es, eu]) * 100
    fig, a = plt.subplots(figsize=(7, 4.4)); a.bar(["pi", "sigma+", "sigma-", "unpol"], v, color="#e83")
    for i, val in enumerate(v):
        a.text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    a.axhline(float(r["eta_geom"]) * 100, ls=":", c="k", label=f"isotropic ceiling {float(r['eta_geom'])*100:.2f}%")
    a.set(ylabel="collection (%)", title=f"{label} collection  D={D:.0f}um NA={NA:.2f}"); a.legend(); a.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(FIG / f"collection_{design_key}.png", dpi=140, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"   figures -> lens_focus_{design_key}.png, collection_{design_key}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design", choices=sorted(DESIGNS))
    ap.add_argument("--library", action="store_true"); ap.add_argument("--layout", action="store_true")
    ap.add_argument("--lens", action="store_true"); ap.add_argument("--all", action="store_true")
    ap.add_argument("--full", action="store_true", help="no symmetry (4x cost), for cross-checking")
    a = ap.parse_args(); mat = DESIGNS[a.design]["mat"]
    if a.all or a.library:
        run_library(mat)
    if a.all or a.layout or a.lens:
        build_layout(a.design)
    if a.all or a.lens:
        run_lens(a.design, full=a.full)


if __name__ == "__main__":
    main()
