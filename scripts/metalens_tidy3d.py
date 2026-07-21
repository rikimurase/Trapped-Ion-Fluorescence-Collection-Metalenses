"""Tidy3D metalens pipeline for every design in one module.

    python scripts/metalens_tidy3d.py si3n4_D92 --layout          # Phase 1, free
    python scripts/metalens_tidy3d.py si3n4_D92 --lens            # Phase 2, cost estimate only
    python scripts/metalens_tidy3d.py si3n4_D92 --lens --submit   # Phase 2, run the FDTD (FlexCredits)
    python scripts/metalens_tidy3d.py si3n4_D92 --analyze --collect
    python scripts/metalens_tidy3d.py si3n4_D92 --all --submit

Phase 0 (the shared meta-atom library) is built by scripts/regenerate_library.py or the
unit-cell notebook. Outputs go to designs/<folder>/data and designs/<folder>/figures. tio2
uses clip-to-range assignment, si3n4 uses max-projection (see docs/THEORY.md section 10).
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[1]
DES = ROOT / "designs"
LDA0 = 0.866; P = 0.500; H = 0.800; N_SIO2 = 1.4528; Z0 = 376.730313668
DESIGNS = {
    "tio2_D30":  dict(mat="tio2",  folder="tio2_D30um_f80um_866nm",  D=30.0, f_work=80.0, f_design=100.0),
    "tio2_D92":  dict(mat="tio2",  folder="tio2_D92um_f80um_866nm",  D=92.0, f_work=80.0, f_design=80.0),
    "si3n4_D30": dict(mat="si3n4", folder="si3n4_D30um_f80um_866nm", D=30.0, f_work=80.0, f_design=100.0),
    "si3n4_D92": dict(mat="si3n4", folder="si3n4_D92um_f80um_866nm", D=92.0, f_work=80.0, f_design=80.0),
}


def _dirs(key):
    d = DESIGNS[key]; folder = DES / d["folder"]
    data = folder / "data"; fig = folder / "figures"
    data.mkdir(parents=True, exist_ok=True); fig.mkdir(exist_ok=True)
    return d, data, fig


def build_layout(key):
    d, data, fig = _dirs(key); mat = d["mat"]
    lib = np.load(ROOT / "data" / f"phase_library_{mat}.npz", allow_pickle=True)
    r_lib = np.asarray(lib["r_um"], float); phi_lib = np.asarray(lib["phase_rad"], float)
    amp_lib = np.asarray(lib["amplitude"], float); n_mat = float(lib["n_material"])
    label = str(lib["material"])
    phi_of_r = CubicSpline(r_lib, phi_lib); amp_of_r = CubicSpline(r_lib, amp_lib)
    k0 = 2 * np.pi / LDA0; span = (phi_lib[-1] - phi_lib[0]) / (2 * np.pi)
    D, f_work, f_design = d["D"], d["f_work"], d["f_design"]; r_ap = D / 2
    ns = int(np.floor(r_ap / P)); coords = np.arange(-ns, ns + 1) * P
    X, Y = np.meshgrid(coords, coords); rho = np.sqrt(X ** 2 + Y ** 2); m = rho <= r_ap
    xc, yc, rho = X[m], Y[m], rho[m]
    phi_t = np.mod(-k0 * (np.sqrt(rho ** 2 + f_design ** 2) - f_design), 2 * np.pi)
    if mat == "tio2":
        r_of_phi = CubicSpline(phi_lib, r_lib)
        r_as = np.clip(r_of_phi(np.clip(phi_t, phi_lib[0], phi_lib[-1])), r_lib[0], r_lib[-1])
    else:
        rg = np.linspace(r_lib[0], r_lib[-1], 2001)
        proj = amp_of_r(rg)[None, :] * np.cos(phi_of_r(rg)[None, :] - phi_t[:, None])
        r_as = rg[np.argmax(proj, axis=1)]
    phi_r = phi_of_r(r_as); amp_r = amp_of_r(r_as)
    err = np.angle(np.exp(1j * (phi_r - phi_t))); strehl = abs(np.mean(amp_r * np.exp(1j * err))) ** 2
    NA = np.sin(np.arctan(r_ap / f_work))
    np.savez(data / f"metalens_layout_{mat}.npz", material=label, wavelength_um=LDA0, period_um=P,
             height_um=H, n_material=n_mat, n_SiO2=N_SIO2, aperture_um=D, focal_length_um=f_design,
             working_distance_um=f_work, NA=NA, x_um=xc, y_um=yc, rho_um=rho, radius_um=r_as,
             phase_target_rad=phi_t, phase_realized_rad=phi_r, amplitude=amp_r,
             library_span_x2pi=span, strehl_scalar=strehl)
    print(f"[layout {key}] N={xc.size} NA={NA:.3f} span={span:.2f}x2pi Strehl={strehl*100:.1f}% "
          f"radii {r_as.min()*1e3:.0f}-{r_as.max()*1e3:.0f} nm")
    axs = plt.subplots(2, 2, figsize=(12, 10))[1]
    s = axs[0, 0].scatter(xc, yc, c=r_as * 1e3, s=6, cmap="viridis")
    axs[0, 0].add_artist(Circle((0, 0), r_ap, color="r", fill=False, lw=1))
    axs[0, 0].set(aspect="equal", xlabel="x (um)", ylabel="y (um)", title=f"radius (N={xc.size})")
    plt.colorbar(s, ax=axs[0, 0], label="radius (nm)")
    s = axs[0, 1].scatter(xc, yc, c=np.degrees(err), s=6, cmap="coolwarm", vmin=-180, vmax=180)
    axs[0, 1].set(aspect="equal", xlabel="x (um)", ylabel="y (um)", title="phase error (deg)")
    plt.colorbar(s, ax=axs[0, 1])
    o = np.argsort(rho)
    axs[1, 0].plot(rho[o], phi_t[o], ".", ms=3, alpha=0.4, label="target")
    axs[1, 0].plot(rho[o], phi_r[o], ".", ms=3, alpha=0.4, label="realized")
    axs[1, 0].set(xlabel="rho (um)", ylabel="phase (rad)", ylim=(-0.3, 2 * np.pi + 0.3)); axs[1, 0].legend()
    s = axs[1, 1].scatter(xc, yc, c=amp_r, s=6, cmap="magma", vmin=0.8, vmax=1.05)
    axs[1, 1].set(aspect="equal", xlabel="x (um)", ylabel="y (um)", title=f"|t| (mean {amp_r.mean():.3f})")
    plt.colorbar(s, ax=axs[1, 1])
    plt.suptitle(f"{label} D={D:.0f} um NA={NA:.2f} span {span:.2f}x2pi Strehl {strehl*100:.0f}%",
                 fontweight="bold")
    plt.tight_layout(); plt.savefig(fig / f"metalens_layout_{mat}.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def build_lens(key, submit):
    import tidy3d as td, tidy3d.web as web
    d, data, fig = _dirs(key); mat = d["mat"]
    lay = np.load(data / f"metalens_layout_{mat}.npz", allow_pickle=True)
    x_all, y_all, r_all = (np.asarray(lay[k], float) for k in ("x_um", "y_um", "radius_um"))
    n_mat = float(lay["n_material"]); D = d["D"]; f_work = d["f_work"]; label = str(lay["material"])
    freq0 = td.C_0 / LDA0; pillar = td.Medium(permittivity=n_mat ** 2); SiO2 = td.Medium(permittivity=N_SIO2 ** 2)
    q = (x_all >= -1e-9) & (y_all >= -1e-9); xq, yq, rq = x_all[q], y_all[q], r_all[q]
    z_teeth_top = H; z_source = -0.6; z_nf = z_teeth_top + 0.5 * LDA0; buf = LDA0
    z_min, z_max = z_source - buf, z_nf + buf; z_ctr = 0.5 * (z_min + z_max); half_xy = D / 2 + 1.0
    structures = [
        td.Structure(geometry=td.Box(center=(0, 0, -2.5), size=(td.inf, td.inf, 5.0)), medium=SiO2, name="SiO2_substrate"),
        td.Structure(geometry=td.GeometryGroup(geometries=[
            td.Cylinder(center=(float(x), float(y), H / 2), radius=float(r), length=H, axis=2)
            for x, y, r in zip(xq, yq, rq)]), medium=pillar, name=f"{label}_metalens")]
    source = td.PlaneWave(source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10),
                          size=(td.inf, td.inf, 0), center=(0, 0, z_source), direction="+", pol_angle=0.0, name="plane_wave")
    xs_far = np.linspace(-8.0, 8.0, 161)
    mon_proj = td.FieldProjectionCartesianMonitor(
        center=(0, 0, z_nf), size=(td.inf, td.inf, 0), normal_dir="+", freqs=[freq0], name="focal_plane_proj",
        proj_axis=2, proj_distance=(z_teeth_top + f_work) - z_nf, x=list(xs_far), y=list(xs_far),
        custom_origin=(0, 0, z_nf), far_field_approx=False)
    axial_mons = [td.FieldProjectionCartesianMonitor(
        center=(0, 0, z_nf), size=(td.inf, td.inf, 0), normal_dir="+", freqs=[freq0], name=f"axial_{i:02d}",
        proj_axis=2, proj_distance=(z_teeth_top + zf) - z_nf, x=[0.0], y=[0.0], custom_origin=(0, 0, z_nf),
        far_field_approx=False) for i, zf in enumerate(np.linspace(f_work - 18.0, f_work + 18.0, 37))]
    flux_above = td.FluxMonitor(center=(0, 0, z_nf), size=(td.inf, td.inf, 0), freqs=[freq0], name="flux_above")
    flux_below = td.FluxMonitor(center=(0, 0, -0.3), size=(td.inf, td.inf, 0), freqs=[freq0], name="flux_below")
    mon_xz = td.FieldMonitor(center=(0, 0, z_ctr), size=(td.inf, 0, td.inf), freqs=[freq0], fields=["Ex", "Ey", "Ez"], name="xz")
    dl = P / 36
    grid_spec = td.GridSpec(wavelength=LDA0, grid_x=td.UniformGrid(dl=dl), grid_y=td.UniformGrid(dl=dl),
                            grid_z=td.AutoGrid(min_steps_per_wvl=25), override_structures=[td.Structure(
                                geometry=td.Box.from_bounds(rmin=(-td.inf, -td.inf, 0.0), rmax=(td.inf, td.inf, H)), medium=pillar)])
    sim = td.Simulation(center=(0, 0, z_ctr), size=(2 * half_xy, 2 * half_xy, z_max - z_min), grid_spec=grid_spec,
                        structures=structures, sources=[source], monitors=[mon_proj, flux_above, flux_below, mon_xz] + axial_mons,
                        run_time=td.RunTimeSpec(quality_factor=1), symmetry=(-1, 1, 0),
                        boundary_spec=td.BoundarySpec(x=td.Boundary.absorber(), y=td.Boundary.absorber(), z=td.Boundary.pml()))
    ax = plt.subplots(1, 3, figsize=(15, 5))[1]
    sim.plot(y=0, ax=ax[0]); sim.plot(x=0, ax=ax[1]); sim.plot(z=H / 2, ax=ax[2])
    plt.tight_layout(); plt.savefig(fig / "lens_scale_sim_setup.png", dpi=130, bbox_inches="tight", facecolor="white"); plt.close()
    job = web.Job(simulation=sim, task_name=f"metalens_lens_scale_{mat}", verbose=False)
    print(f"[lens {key}] cylinders {x_all.size}->{xq.size}  grid {sim.grid.num_cells}  est max cost {job.estimate_cost()} FC")
    if submit:
        job.run(path=str(data / f"lens_scale_{mat}.hdf5")); print(f"   run complete -> lens_scale_{mat}.hdf5")
    else:
        print("   estimate only (pass --submit to run)")


def _fwhm(coord, prof):
    prof = np.asarray(prof) / np.max(prof); a = np.where(prof >= 0.5)[0]
    if a.size < 2:
        return np.nan
    lo, hi = a[0], a[-1]
    L = np.interp(0.5, [prof[max(lo - 1, 0)], prof[lo]], [coord[max(lo - 1, 0)], coord[lo]])
    R = np.interp(0.5, [prof[min(hi + 1, len(coord) - 1)], prof[hi]], [coord[min(hi + 1, len(coord) - 1)], coord[hi]])
    return abs(R - L)


def analyze(key):
    import tidy3d as td
    d, data, fig = _dirs(key); mat = d["mat"]
    lay = np.load(data / f"metalens_layout_{mat}.npz", allow_pickle=True)
    NA = float(lay["NA"]); f_work = float(lay["working_distance_um"]); f_design = float(lay["focal_length_um"])
    label = str(lay["material"]); freq0 = td.C_0 / LDA0; z_teeth_top = H; z_nf = z_teeth_top + 0.5 * LDA0
    sd = td.SimulationData.from_file(str(data / f"lens_scale_{mat}.hdf5"))
    proj = sd["focal_plane_proj"].fields_cartesian; zc = float(proj.z.values[0])
    Ex, Ey, Ez, Hx, Hy = (proj[c].sel(f=freq0, z=zc).values for c in ("Ex", "Ey", "Ez", "Hx", "Hy"))
    xs, ys = proj.x.values, proj.y.values
    I = np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2; ip, jp = np.unravel_index(np.argmax(I), I.shape)
    fwx, fwy = _fwhm(xs, I[:, jp]), _fwhm(ys, I[ip, :]); dl = 0.514 * LDA0 / NA
    axl = []
    for mon in sd.simulation.monitors:
        if mon.name.startswith("axial_"):
            za = (mon.proj_distance + z_nf) - z_teeth_top
            e2 = sum(np.abs(sd[mon.name].fields_cartesian[c].sel(f=freq0).values) ** 2 for c in ("Ex", "Ey", "Ez"))
            axl.append((za, float(np.squeeze(e2))))
    axl.sort(); zsc = np.array([a[0] for a in axl]); Iax = np.array([a[1] for a in axl])
    z_true = float(zsc[np.argmax(Iax)]) if Iax.size else np.nan
    P_trans = float(sd["flux_above"].flux.sel(f=freq0).values)
    Sz = 0.5 * np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx)); dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    XX, YY = np.meshgrid(xs - xs[ip], ys - ys[jp], indexing="ij"); Rspot = 3.0 * 0.5 * np.nanmean([fwx, fwy])
    eff = float(np.sum(Sz[(XX ** 2 + YY ** 2) <= Rspot ** 2]) * dx * dy) / P_trans if P_trans else np.nan
    print(f"[focus {key} {label}] FWHM {fwx:.2f}/{fwy:.2f} (limit {dl:.2f})  focus {z_true:.1f}um "
          f"(target {f_work:.0f}, geom {f_design:.0f})  T {P_trans*100:.0f}%  eff {eff*100:.0f}%")
    ax = plt.subplots(1, 3, figsize=(16, 4.6))[1]
    im = ax[0].pcolormesh(ys, xs, I, cmap="magma", shading="auto")
    ax[0].add_artist(Circle((ys[jp], xs[ip]), Rspot, fill=False, color="cyan", lw=1))
    ax[0].set(aspect="equal", xlabel="y (um)", ylabel="x (um)", title=f"focal |E|^2 @ z={z_teeth_top+f_work:.0f} um")
    plt.colorbar(im, ax=ax[0])
    ax[1].plot(xs, I[:, jp] / I.max(), label=f"x FWHM {fwx:.2f}"); ax[1].plot(ys, I[ip, :] / I.max(), label=f"y FWHM {fwy:.2f}")
    ax[1].axhline(0.5, ls=":", color="gray"); ax[1].set(xlabel="pos (um)", ylabel="norm |E|^2", xlim=(-6, 6)); ax[1].legend(); ax[1].grid(alpha=0.3)
    if Iax.size:
        ax[2].plot(zsc, Iax / Iax.max(), "-o", ms=4); ax[2].axvline(f_work, ls="--", color="red", label=f"target {f_work:.0f}")
        ax[2].axvline(z_true, ls="-", color="green", lw=1, label=f"actual {z_true:.0f}")
    ax[2].set(xlabel="z above lens (um)", ylabel="on-axis |E|^2", title="axial scan"); ax[2].legend(); ax[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(fig / "lens_scale_focus.png", dpi=140, bbox_inches="tight", facecolor="white"); plt.close()


def collect(key):
    import tidy3d as td
    d, data, fig = _dirs(key); mat = d["mat"]
    lay = np.load(data / f"metalens_layout_{mat}.npz", allow_pickle=True)
    D = float(lay["aperture_um"]); f_work = float(lay["working_distance_um"]); n_mode = float(lay["n_SiO2"])
    amp = np.asarray(lay["amplitude"], float); label = str(lay["material"]); freq0 = td.C_0 / LDA0
    k = 2 * np.pi / (LDA0 * 1e-6); A_ap = np.pi * ((D / 2) * 1e-6) ** 2; NA = np.sin(np.arctan((D / 2) / f_work))
    sd = td.SimulationData.from_file(str(data / f"lens_scale_{mat}.hdf5"))
    proj = sd["focal_plane_proj"].fields_cartesian; zc = float(proj.z.values[0])
    Efx, Efy, Efz = (complex(proj[c].sel(f=freq0, z=zc).interp(x=0.0, y=0.0).values) for c in ("Ex", "Ey", "Ez"))
    band = sd["xz"].Ex.sel(f=freq0).sel(x=slice(-D / 2, D / 2), z=slice(-0.55, -0.03))
    R_pow = max(0.0, 1.0 - float(np.mean(amp ** 2))); E_inc = np.sqrt(np.mean(np.abs(band.values) ** 2) / (1.0 + R_pow))
    pref = (3 * np.pi) / (2 * k ** 2 * n_mode * A_ap)
    Ex = np.array([Efx, Efy, Efz]); Ey = np.array([-Efy, Efx, Efz])
    dip = {"pi": np.array([1.0, 0, 0]), "sigma+": np.array([0, 1, 1j]) / np.sqrt(2), "sigma-": np.array([0, 1, -1j]) / np.sqrt(2)}
    into = lambda mE, ph: pref * abs(np.vdot(ph, mE)) ** 2 / E_inc ** 2
    ex = {p: into(Ex, v) for p, v in dip.items()}; both = {p: ex[p] + into(Ey, v) for p, v in dip.items()}
    unpol_x = sum(ex.values()) / 3; unpol_both = sum(both.values()) / 3
    c = np.cos(np.arcsin(NA)); axial = 0.5 - 0.75 * c + 0.25 * c ** 3
    enh = np.sqrt(abs(Efx) ** 2 + abs(Efy) ** 2 + abs(Efz) ** 2) / E_inc
    print(f"[collect {key} {label}] D={D:.0f}um NA={NA:.3f}  enh {enh:.1f}x")
    print(f"   both channels: pi {both['pi']*100:.3f}%  sigma+/- {both['sigma+']*100:.3f}%  unpol {unpol_both*100:.3f}%"
          f"  (+axial <= {axial*100:.3f}%)")
    labels = ["pi", "sigma+", "sigma-", "unpol"]
    v1 = np.array([ex["pi"], ex["sigma+"], ex["sigma-"], unpol_x]) * 100
    v2 = np.array([both["pi"], both["sigma+"], both["sigma-"], unpol_both]) * 100
    xp = np.arange(4); w = 0.38; ax = plt.subplots(figsize=(7.2, 4.6))[1]
    ax.bar(xp - w / 2, v1, w, color="#26c", label="single linear channel")
    ax.bar(xp + w / 2, v2, w, color="#e83", label="both channels")
    for x, a, b in zip(xp, v1, v2):
        ax.text(x - w / 2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(x + w / 2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xp); ax.set_xticklabels(labels)
    ax.set(ylabel="collection (%)", title=f"{label} collection  D={D:.0f}um NA={NA:.2f}"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(fig / "collection_reciprocity.png", dpi=140, bbox_inches="tight", facecolor="white"); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design", choices=sorted(DESIGNS))
    ap.add_argument("--layout", action="store_true"); ap.add_argument("--lens", action="store_true")
    ap.add_argument("--analyze", action="store_true"); ap.add_argument("--collect", action="store_true")
    ap.add_argument("--all", action="store_true"); ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    if a.all or a.layout:
        build_layout(a.design)
    if a.all or a.lens:
        build_lens(a.design, a.submit)
    if a.all or a.analyze:
        analyze(a.design)
    if a.all or a.collect:
        collect(a.design)


if __name__ == "__main__":
    main()
