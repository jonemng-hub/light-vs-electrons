"""
Simulate it yourself: copper wire + silicon waveguide + manufacturing variation
==============================================================================

Everything here is simulated in Python. No circuit simulator to install, no
commercial software.

    pip install numpy scipy matplotlib femwell
    python simulate_from_scratch.py

(Linux only: femwell's mesher also needs `sudo apt install libglu1-mesa`.)
Runs in about two minutes on a laptop. Results go to ./output/.

PART 1  Copper wire. The wire is modelled as 200 small resistor-capacitor
        sections and solved exactly with linear algebra. This is the same
        circuit an ngspice simulation solves, written out by hand.

PART 2  Silicon waveguide. A finite-element mode solver (femwell) finds how
        light travels in a 220 nm thick silicon strip. From it we get the
        group index (speed of light in the waveguide) and how far a ring
        resonator's wavelength shifts when the width, thickness, or
        temperature is off.

PART 3  Manufacturing variation. 10,000 virtual chips, each with randomly
        wrong waveguide widths and thicknesses (spread measured across a
        real wafer) and a random operating temperature. Heaters must pull
        each ring back on wavelength. That heater power decides where light
        starts to beat copper, so every chip gets its own crossover length.
        The result is a PROBABILITY that light wins at each length instead
        of a single number.

The link-energy formulas come from crossover_model.py (same folder).
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import OrderedDict
from dataclasses import replace

import numpy as np
from scipy.optimize import brentq

from crossover_model import (Electrical, Optical, repeater_design, crossover_mm,
                             optical_delay_ps, electrical_energy_fJ)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
RNG_SEED = 1

# ---------------------------------------------------------------------------
# Inputs. Each one is either a design choice or has a source.
# ---------------------------------------------------------------------------
WAVELENGTH_UM = 1.55
WG_THICKNESS_UM = 0.220      # standard silicon-on-insulator layer
WG_WIDTH_UM = 0.500          # design choice: common single-mode strip width
RING_RADIUS_UM = 5.0         # design choice: typical microring modulator size
N_RINGS = 2                  # modulator ring + receiver filter ring

# Manufacturing spread measured across a wafer (Su et al., OFC 2014)
SIGMA_WIDTH_NM = 5.52
SIGMA_THICKNESS_NM = 1.35
# Run-time temperature range of a processor (Manipatruni et al. 2012: 20 K)
TEMP_RANGE_K = 20.0

# Thermo-optic coefficients at 1550 nm (Frey et al. 2006; Leviton & Frey 2008)
DNDT_SI = 1.86e-4
DNDT_OX = 8.57e-6

# Optical receiver energy measured by Samanta et al. (191 fJ/bit at 25 Gb/s).
# Replaces the 60 fJ/bit default, which their Table I lists under the copper link.
E_RX_FJ = 191.0

N_CHIPS = 10_000


# ---------------------------------------------------------------------------
# PART 1: copper wire as an RC ladder, solved exactly
# ---------------------------------------------------------------------------

def wire_step_delay_ps(r_ohm_per_mm, c_fF_per_mm, L_mm, r_drv_ohm, c_drv_fF, c_load_fF,
                       n_sections=200):
    """50% delay of a driver -> distributed RC wire -> load, for a step input.

    Node equations:  C dv/dt = -G v + (input through the driver resistor).
    The circuit is linear, so the answer is a sum of decaying exponentials;
    we get them exactly from an eigen-decomposition instead of time-stepping.
    Units: ohm x fF = fs.
    """
    N = n_sections
    rs, cs = r_ohm_per_mm * L_mm / N, c_fF_per_mm * L_mm / N
    C = np.full(N + 1, cs)                    # pi-sections: cs/2 from each side
    C[0], C[-1] = cs / 2 + c_drv_fF, cs / 2 + c_load_fF
    G = np.zeros((N + 1, N + 1))
    g = 1.0 / rs
    for i in range(N):
        G[i, i] += g; G[i + 1, i + 1] += g
        G[i, i + 1] -= g; G[i + 1, i] -= g
    G[0, 0] += 1.0 / r_drv_ohm
    # u = v - Vdd decays to 0 from u(0) = -Vdd.  Symmetrise with C^(1/2).
    s = 1.0 / np.sqrt(C)
    lam, Q = np.linalg.eigh(s[:, None] * G * s[None, :])
    coef = Q.T @ (np.sqrt(C) * -1.0)          # Vdd = 1
    row = s[-1] * Q[-1, :]
    v_end = lambda t_fs: 1.0 + row @ (np.exp(-lam * t_fs) * coef)
    t_hi = 50.0 / lam.min()
    return brentq(lambda t: v_end(t) - 0.5, 0.0, t_hi, xtol=1e-6) / 1000.0


def part1_wire(e: Electrical):
    d = repeater_design(e)
    x = d["x_fF"]
    r_drv = e.tau_ps / x * 1000.0             # repeater output resistance, ohm
    c_drv = e.gamma * x                       # repeater output capacitance, fF
    rows = []
    for name, L in [("one repeater segment", d["l_mm"]), ("1 mm, no repeaters", 1.0),
                    ("2 mm, no repeaters", 2.0), ("5 mm, no repeaters", 5.0),
                    ("10 mm, no repeaters", 10.0)]:
        t0 = time.perf_counter()
        dly = wire_step_delay_ps(e.r_ohm_per_mm, e.c_fF_per_mm, L, r_drv, c_drv, x)
        rows.append({"case": name, "length_mm": round(L, 3), "delay_ps": round(dly, 2),
                     "runtime_ms": round((time.perf_counter() - t0) * 1e3, 1)})
    seg = rows[0]["delay_ps"]
    return {"rows": rows, "segment_mm": d["l_mm"], "sim_ps_per_mm": seg / d["l_mm"],
            "formula_ps_per_mm": d["delay_ps_per_mm"]}


# ---------------------------------------------------------------------------
# PART 2: silicon waveguide mode solver
# ---------------------------------------------------------------------------

def n_si(lam_um):
    """Silicon refractive index (Salzberg & Villa 1957 Sellmeier, 1.36-11 um)."""
    l2 = lam_um ** 2
    return math.sqrt(1 + 10.6684293 * l2 / (l2 - 0.301516485 ** 2)
                     + 0.0030434748 * l2 / (l2 - 1.13475115 ** 2)
                     + 1.54133408 * l2 / (l2 - 1104.0 ** 2))


def n_ox(lam_um):
    """Silica refractive index (Malitson 1965 Sellmeier)."""
    l2 = lam_um ** 2
    return math.sqrt(1 + 0.6961663 * l2 / (l2 - 0.0684043 ** 2)
                     + 0.4079426 * l2 / (l2 - 0.1162414 ** 2)
                     + 0.8974794 * l2 / (l2 - 9.896161 ** 2))


def neff_te0(w_um, t_um, lam_um, dT=0.0):
    """Effective index of the fundamental TE mode of an oxide-clad silicon strip."""
    from shapely.geometry import box
    from skfem import Basis, ElementTriP0
    from skfem.io.meshio import from_meshio
    from femwell.maxwell.waveguide import compute_modes
    from femwell.mesh import mesh_from_OrderedDict

    polys = OrderedDict(core=box(-w_um / 2, 0, w_um / 2, t_um),
                        clad=box(-2.0, -1.5, 2.0, 1.5 + t_um))
    mesh = from_meshio(mesh_from_OrderedDict(
        polys, {"core": {"resolution": 0.012, "distance": 0.4}}, default_resolution_max=0.3))
    basis = Basis(mesh, ElementTriP0())
    eps = basis.zeros()
    eps[basis.get_dofs(elements="core")] = (n_si(lam_um) + DNDT_SI * dT) ** 2
    eps[basis.get_dofs(elements="clad")] = (n_ox(lam_um) + DNDT_OX * dT) ** 2
    modes = compute_modes(basis, eps, wavelength=lam_um, num_modes=3, order=2)
    te = [m for m in modes if m.te_fraction > 0.5]
    return float(np.real(max(te, key=lambda m: np.real(m.n_eff)).n_eff))


def waveguide_properties(w_um, t_um, lam_um=WAVELENGTH_UM):
    """Group index and ring-resonance sensitivities, by central differences.
    For a ring, d(lambda)/lambda = d(n_eff)/n_g, so each sensitivity is
    (lambda / n_g) x d(n_eff)/dx."""
    dl, dw, dt, dT = 0.01, 0.010, 0.004, 10.0
    n0 = neff_te0(w_um, t_um, lam_um)
    ng = n0 - lam_um * (neff_te0(w_um, t_um, lam_um + dl) - neff_te0(w_um, t_um, lam_um - dl)) / (2 * dl)
    dn_dw = (neff_te0(w_um + dw, t_um, lam_um) - neff_te0(w_um - dw, t_um, lam_um)) / (2 * dw * 1000)
    dn_dt = (neff_te0(w_um, t_um + dt, lam_um) - neff_te0(w_um, t_um - dt, lam_um)) / (2 * dt * 1000)
    dn_dT = (neff_te0(w_um, t_um, lam_um, dT) - neff_te0(w_um, t_um, lam_um, -dT)) / (2 * dT)
    k = lam_um * 1000 / ng                    # nm of wavelength per unit of n_eff
    return {"width_nm": w_um * 1000, "thickness_nm": t_um * 1000, "n_eff": n0, "n_g": ng,
            "dlam_dwidth_nm_per_nm": k * dn_dw, "dlam_dthick_nm_per_nm": k * dn_dt,
            "dlam_dT_nm_per_K": k * dn_dT}


# ---------------------------------------------------------------------------
# PART 3: Monte Carlo over manufacturing variation
# ---------------------------------------------------------------------------

def part3_monte_carlo(wg: dict, e: Electrical, o: Optical):
    rng = np.random.default_rng(RNG_SEED)
    fsr_nm = (WAVELENGTH_UM * 1000) ** 2 / (wg["n_g"] * 2 * math.pi * RING_RADIUS_UM * 1000)
    # Each ring gets its own width and thickness error; the chip has one temperature.
    dw = rng.normal(0, SIGMA_WIDTH_NM, (N_CHIPS, N_RINGS))
    dt = rng.normal(0, SIGMA_THICKNESS_NM, (N_CHIPS, N_RINGS))
    dT = rng.uniform(-TEMP_RANGE_K / 2, TEMP_RANGE_K / 2, (N_CHIPS, 1))
    dlam = (wg["dlam_dwidth_nm_per_nm"] * dw + wg["dlam_dthick_nm_per_nm"] * dt
            + wg["dlam_dT_nm_per_K"] * dT)
    # Heaters only warm a ring, which only moves it to longer wavelength.
    # A ring that lands too long must be pushed to the next resonance.
    tune_heat_only = np.mod(-dlam, fsr_nm).sum(axis=1)
    # Lower bound: as if each ring could be pushed either way.
    tune_either_way = np.abs(dlam).sum(axis=1)

    o1 = replace(o, n_tuned_rings=1)
    def crossovers(tune_nm):
        return np.array([crossover_mm("energy", e, replace(o1, tuning_nm=float(x))) for x in tune_nm])
    x_heat, x_either = crossovers(tune_heat_only), crossovers(tune_either_way)
    x_fixed = crossover_mm("energy", e, o)    # the usual fixed 2 nm per ring assumption
    return {"fsr_nm": fsr_nm, "tune_heat_only": tune_heat_only, "tune_either_way": tune_either_way,
            "x_heat_only": x_heat, "x_either_way": x_either, "x_fixed_assumption": x_fixed}


def summarize(x):
    f = x[np.isfinite(x)]
    return {"median_mm": float(np.median(f)), "p5_mm": float(np.percentile(f, 5)),
            "p95_mm": float(np.percentile(f, 95)), "never_wins_pct": 100 * (1 - len(f) / len(x))}


def plot_probability(mc: dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                         "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    L = np.linspace(0, 20, 401)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for key, color, label in [("x_heat_only", "#2a78d6", "Heaters can only warm (realistic)"),
                              ("x_either_way", "#1baf7a", "Rings pushed either way (best case)")]:
        x = mc[key]
        ax.plot(L, [100 * np.mean(x < l) for l in L], color=color, lw=2, label=label)
    xf = mc["x_fixed_assumption"]
    ax.axvline(xf, color=MUTED, lw=1)
    ax.annotate(f"usual fixed\n2 nm assumption\n{xf:.1f} mm", (xf, 8), xytext=(-6, 0),
                textcoords="offset points", ha="right", color=INK2, fontsize=8.5)
    ax.set_xlabel("Link length (mm)")
    ax.set_ylabel("Chips where light uses less energy (%)")
    ax.set_title(f"Probability light beats copper, {N_CHIPS:,} simulated chips", loc="left", color=INK)
    ax.set_xlim(0, 12); ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    e = Electrical()
    o = replace(Optical(), e_rx_fJ=E_RX_FJ)

    print("PART 1  Copper wire, simulated as 200 RC sections")
    p1 = part1_wire(e)
    for r in p1["rows"]:
        print(f"  {r['case']:<24}{r['delay_ps']:8.1f} ps   ({r['runtime_ms']:.0f} ms)")
    print(f"  repeated wire: {p1['sim_ps_per_mm']:.1f} ps/mm simulated "
          f"vs {p1['formula_ps_per_mm']:.1f} ps/mm from the Elmore formula")

    print("PART 2  Silicon waveguide, finite-element mode solver")
    t0 = time.perf_counter()
    wg = waveguide_properties(WG_WIDTH_UM, WG_THICKNESS_UM)
    su = waveguide_properties(0.400, WG_THICKNESS_UM)      # Su et al.'s geometry, for checking
    print(f"  ({time.perf_counter() - t0:.0f} s)")
    for tag, w in (("design 500 nm", wg), ("check 400 nm", su)):
        print(f"  {tag}: n_eff {w['n_eff']:.3f}, n_g {w['n_g']:.2f}, ring shift "
              f"{w['dlam_dwidth_nm_per_nm']:.2f} nm per nm width, "
              f"{w['dlam_dthick_nm_per_nm']:.2f} nm per nm thickness, "
              f"{w['dlam_dT_nm_per_K'] * 1000:.0f} pm per K")

    # The delay crossover with the simulated group index and simulated wire delay
    o_sim = replace(o, group_index=wg["n_g"])
    xd_sim = o_sim.conversion_delay_ps / (p1["sim_ps_per_mm"] - wg["n_g"] / 0.299792458)

    print(f"PART 3  Monte Carlo, {N_CHIPS:,} chips")
    t0 = time.perf_counter()
    mc = part3_monte_carlo(wg, e, o_sim)
    print(f"  ({time.perf_counter() - t0:.0f} s)  ring FSR {mc['fsr_nm']:.1f} nm")
    s_heat, s_either = summarize(mc["x_heat_only"]), summarize(mc["x_either_way"])
    print(f"  heater tuning per chip (median): heat-only {np.median(mc['tune_heat_only']):.1f} nm, "
          f"either-way {np.median(mc['tune_either_way']):.1f} nm  (usual assumption: 4.0 nm)")
    print(f"  energy crossover, usual fixed assumption: {mc['x_fixed_assumption']:.2f} mm")
    for tag, s in (("heat-only", s_heat), ("either-way", s_either)):
        print(f"  energy crossover, {tag:<10}: median {s['median_mm']:.2f} mm, "
              f"90% of chips between {s['p5_mm']:.2f} and {s['p95_mm']:.2f} mm")
    print(f"  delay crossover with simulated wire + waveguide: {xd_sim:.2f} mm")

    plot_probability(mc, os.path.join(OUT, "fig5_probability_light_wins.png"))
    np.savetxt(os.path.join(OUT, "monte_carlo_chips.csv"),
               np.column_stack([mc["tune_heat_only"], mc["x_heat_only"],
                                mc["tune_either_way"], mc["x_either_way"]]),
               delimiter=",", fmt="%.4f",
               header="tuning_heat_only_nm,crossover_heat_only_mm,tuning_either_way_nm,crossover_either_way_mm",
               comments="")
    with open(os.path.join(OUT, "simulated_results.json"), "w") as f:
        json.dump({"wire": p1, "waveguide_design": wg, "waveguide_check_400nm": su,
                   "fsr_nm": mc["fsr_nm"], "crossover_fixed_mm": mc["x_fixed_assumption"],
                   "crossover_heat_only": s_heat, "crossover_either_way": s_either,
                   "delay_crossover_simulated_mm": xd_sim}, f, indent=2)
    print(f"Wrote results to {OUT}")


# ---------------------------------------------------------------------------
# Sources for the inputs above
# ---------------------------------------------------------------------------
# Su, Z. et al., "Reduced Wafer-Scale Frequency Variation in Adiabatic Microring
#   Resonators," OFC 2014: width sigma 5.52 nm, thickness sigma 1.35 nm; measured
#   ring sensitivities 0.894 nm/nm (width) and 1.367 nm/nm (thickness) for a
#   400 nm x 220 nm, 3 um radius ring -- used to check Part 2.
# Manipatruni, Lipson & Young, arXiv:1207.6819: 20 K run-time temperature range,
#   100 uW/nm heaters.
# Frey, Leviton & Madison 2006 (Si) and Leviton & Frey 2008 (SiO2): thermo-optic
#   coefficients, as cited in arXiv:1710.03605.
# Salzberg & Villa 1957 (Si) and Malitson 1965 (SiO2): refractive index formulas.
# Samanta et al., arXiv:2510.03943: optical receiver 191 fJ/bit (Sec. IV).
# femwell: open-source finite-element Maxwell solver, https://github.com/HelgeGehring/femwell

if __name__ == "__main__":
    main()
