"""
Heater strategies for a multi-wavelength light link
==================================================

Real light links send several wavelengths ("channels") down one waveguide.
Each channel has a sending ring and a receiving ring, and every ring must be
heated onto a channel. This script compares three ways of doing that:

  fixed         each ring is heated onto its own assigned channel
  barrel shift  all rings in a bank move over by the same number of channels
                (a rotation), whichever rotation needs the least heating
  reassign      each ring takes whichever channel is cheapest to reach,
                one ring per channel (the best possible one-to-one matching)

and two ring designs: a simple 5 um ring, and a ring sized so its free
spectral range equals the whole channel span (Krishnamoorthy et al. 2011).

Heaters can only warm a ring, which only moves it to longer wavelengths, so
every strategy above is "heat only".

For each of 10,000 simulated chips the script finds the heater power for each
strategy, then the length at which light starts to use less energy than copper.

Run simulate_from_scratch.py once first; this script reads the waveguide
results it saved. Then:
    python tuning_strategies.py          (about a minute)

Every heater assumption is in ASSUMPTIONS below, with its source and the range
tested. The script writes them to output/heater_assumptions.csv, ready for the
methods section of a paper.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import replace

import numpy as np
from scipy.optimize import linear_sum_assignment

from crossover_model import (Electrical, Optical, crossover_mm, optical_energy_fJ,
                             electrical_energy_fJ)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
C_NM_GHZ = 299792458.0  # speed of light in nm x GHz (c = 2.998e17 nm/s)

# ---------------------------------------------------------------------------
# Every assumption about heating, in one place
# ---------------------------------------------------------------------------
ASSUMPTIONS = [
    # key, value, unit, what it means, source, range tested
    ("heater_uW_per_nm", 100.0, "uW per nm", "Heater power to shift one ring by 1 nm",
     "Manipatruni et al. 2012 (target); 225 demonstrated (Dong et al. 2010, cited there)", (50.0, 225.0)),
    ("sigma_width_nm", 5.52, "nm", "Waveguide width spread across a wafer (1 sigma)",
     "Su et al., OFC 2014 (measured)", (2.0, 10.0)),
    ("sigma_thickness_nm", 1.35, "nm", "Silicon thickness spread across a wafer (1 sigma)",
     "Su et al., OFC 2014 (measured)", (0.5, 3.0)),
    ("temp_range_K", 20.0, "K", "Range of chip operating temperature",
     "Manipatruni et al. 2012", (0.0, 40.0)),
    ("sigma_local_nm", 0.27, "nm", "Extra ring-to-ring spread within one chip (1 sigma)",
     "Mirza et al., IEEE TCAD 2021: 0.17, 0.25, 0.38 nm measured for 3 designs on one chip (mean 0.27); "
     "Beausoleil et al. (cited there): 0.5 nm within-die", (0.17, 0.5)),
    ("n_channels", 8, "", "Wavelengths per waveguide",
     "Design choice; matches the 1x8 array of Krishnamoorthy et al. 2011", (4, 16)),
    ("channel_spacing_GHz", 200.0, "GHz", "Spacing between wavelengths",
     "Design choice; 200 GHz grid as in Krishnamoorthy et al. 2011", (100.0, 400.0)),
    ("ring_radius_um", 5.0, "um", "Ring radius for the simple design (sets the free spectral range)",
     "Design choice", (3.0, 10.0)),
    ("heat_only", True, "", "Heaters can only warm, so rings only move to longer wavelength",
     "Physics: silicon's refractive index rises with temperature", None),
    ("heater_driver_energy", 0.0, "", "Energy of the circuits that drive and control heaters",
     "Not modelled (limitation)", None),
    ("thermal_crosstalk", 0.0, "", "Heat leaking from one ring's heater to its neighbours",
     "Not modelled (limitation)", None),
]
A = {k: v for k, v, *_ in ASSUMPTIONS}

N_CHIPS = 10_000
N_CHIPS_SWEEP = 2_000
RNG_SEED = 2


def load_waveguide():
    path = os.path.join(OUT, "simulated_results.json")
    if not os.path.exists(path):
        raise SystemExit("Run simulate_from_scratch.py first (it saves the waveguide results).")
    with open(path) as f:
        r = json.load(f)
    return r["waveguide_design"]


def fsr_nm(wg, radius_um):
    return 1550.0 ** 2 / (wg["n_g"] * 2 * math.pi * radius_um * 1000)


def simulate_errors(rng, wg, n_chips, a):
    """Wavelength error (nm) of every ring: shape (chips, 2 banks, channels).
    One shared part per chip (wafer-scale width and thickness error plus chip
    temperature) and one local part per ring."""
    n = int(a["n_channels"])
    shared = (wg["dlam_dwidth_nm_per_nm"] * rng.normal(0, a["sigma_width_nm"], (n_chips, 1, 1))
              + wg["dlam_dthick_nm_per_nm"] * rng.normal(0, a["sigma_thickness_nm"], (n_chips, 1, 1))
              + wg["dlam_dT_nm_per_K"] * rng.uniform(-a["temp_range_K"] / 2, a["temp_range_K"] / 2,
                                                       (n_chips, 1, 1)))
    local = rng.normal(0, a["sigma_local_nm"], (n_chips, 2, n))
    return shared + local


def heating_needed(err, wg, a):
    """Total heating (nm) per chip for each strategy, shape (chips,)."""
    n = int(a["n_channels"])
    spacing = (1550.0 ** 2) * a["channel_spacing_GHz"] / C_NM_GHZ      # GHz -> nm
    fsr = fsr_nm(wg, a["ring_radius_um"])
    ch = np.arange(n) * spacing                                          # channel wavelengths
    out = {"fixed": [], "barrel shift": [], "reassign": []}
    for chip in err:
        tot = {k: 0.0 for k in out}
        for bank in chip:
            ring = ch + bank                                             # where each ring landed
            # cost[i, j]: heat ring i up to channel j (wrapping to its next resonance if needed)
            cost = np.mod(ch[None, :] - ring[:, None], fsr)
            tot["fixed"] += np.trace(cost)
            tot["barrel shift"] += min(cost[np.arange(n), (np.arange(n) + k) % n].sum() for k in range(n))
            r, c = linear_sum_assignment(cost)
            tot["reassign"] += cost[r, c].sum()
        for k in out:
            out[k].append(tot[k])
    return {k: np.array(v) for k, v in out.items()}


def crossovers(heating_total, a, e, o):
    """Energy crossover per chip. Heating is shared over all channels, so each
    channel's bits pay (total heating / channels) worth of heater power.

    Heater power only adds a constant to the light link's energy per bit, so
    instead of searching again for every chip we compute the energy gap
    (light minus copper, without heaters) once on a fine length grid and look
    up where each chip's heater cost closes it. Same answer as crossover_mm,
    about 1000x faster."""
    o0 = replace(o, n_tuned_rings=1, tuning_nm=0.0)
    L = np.linspace(0.0, 200.0, 200_001)
    gap = optical_energy_fJ(L, o0) - electrical_energy_fJ(L, e)
    lowest_so_far = np.minimum.accumulate(gap)
    heater_fJ = (a["heater_uW_per_nm"] * 1e-6 * heating_total / a["n_channels"]
                 / (o.bit_rate_gbps * 1e9) * 1e15)
    idx = np.searchsorted(-lowest_so_far, heater_fJ, side="right")   # first L where gap + heater < 0
    out = np.full(len(heater_fJ), np.inf)
    ok = idx < len(L)
    i = np.maximum(idx[ok], 1)
    g0, g1 = gap[i - 1] + heater_fJ[ok], gap[i] + heater_fJ[ok]
    out[ok] = L[i - 1] + (L[i] - L[i - 1]) * g0 / (g0 - g1)
    out[ok & (idx == 0)] = 0.0
    return out


def run(a, wg, n_chips, seed, e, o):
    rng = np.random.default_rng(seed)
    err = simulate_errors(rng, wg, n_chips, a)
    heat = heating_needed(err, wg, a)
    return heat, {k: crossovers(v, a, e, o) for k, v in heat.items()}


def stats(x):
    f = x[np.isfinite(x)]
    return float(np.median(f)), float(np.percentile(f, 5)), float(np.percentile(f, 95))


def matched_radius(wg, a):
    """Ring radius whose free spectral range equals the whole channel span
    (channels x spacing), the design rule of Krishnamoorthy et al. 2011. Then a
    ring that misses one channel lands exactly one channel-step from the next."""
    span = a["n_channels"] * 1550.0 ** 2 * a["channel_spacing_GHz"] / C_NM_GHZ
    return 1550.0 ** 2 / (wg["n_g"] * 2 * math.pi * span) / 1000


CASES = [  # (label, ring design, strategy, colour)
    ("Fixed channels, 5 um ring", "simple", "fixed", "#2a78d6"),
    ("Reassign rings, 5 um ring", "simple", "reassign", "#4a3aa7"),
    ("Reassign rings, ring matched to channel span", "matched", "reassign", "#1baf7a"),
]


def plot(xs_by_design, x_fixed_assumption, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                         "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    L = np.linspace(0, 12, 481)
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    for label, design, strat, color in CASES:
        x = xs_by_design[design][strat]
        ax.plot(L, [100 * np.mean(x < l) for l in L], color=color, lw=2, label=label)
    ax.axvline(x_fixed_assumption, color=MUTED, lw=1)
    ax.annotate(f"usual 2 nm\nassumption\n{x_fixed_assumption:.1f} mm", (x_fixed_assumption, 60),
                xytext=(-26, 0), textcoords="offset points", ha="right", color=INK2, fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlabel("Link length (mm)")
    ax.set_ylabel("Chips where light uses less energy (%)")
    ax.set_title(f"Heater strategy decides where light wins ({N_CHIPS:,} chips, 8 channels)",
                 loc="left", color=INK, fontsize=10)
    ax.set_xlim(0, 12); ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def run_design(a, design, wg, n_chips, seed, e, o):
    a2 = dict(a)
    if design == "matched":
        a2["ring_radius_um"] = matched_radius(wg, a2)
    return run(a2, wg, n_chips, seed, e, o), a2["ring_radius_um"]


def main():
    os.makedirs(OUT, exist_ok=True)
    wg = load_waveguide()
    e = Electrical()
    o = replace(Optical(), e_rx_fJ=191.0, group_index=wg["n_g"])
    x_fixed_assumption = crossover_mm("energy", e, o)
    spacing = 1550.0 ** 2 * A["channel_spacing_GHz"] / C_NM_GHZ
    rings = 2 * A["n_channels"]
    print(f"{int(A['n_channels'])} channels, {spacing:.2f} nm apart (span {A['n_channels'] * spacing:.1f} nm)")

    rows, xs_by_design = [], {}
    print(f"{'ring design':<36}{'strategy':<14}{'heating/ring':>13}{'median':>10}{'90% of chips':>16}")
    for design in ("simple", "matched"):
        (heat, xs), radius = run_design(A, design, wg, N_CHIPS, RNG_SEED, e, o)
        xs_by_design[design] = xs
        name = f"{radius:.2f} um ring, FSR {fsr_nm(wg, radius):.1f} nm" + (" (matched)" if design == "matched" else "")
        for k in ("fixed", "barrel shift", "reassign"):
            per_ring = heat[k] / rings
            med, lo, hi = stats(xs[k])
            finite = xs[k][np.isfinite(xs[k])]
            rows.append({"ring_design": name, "strategy": k,
                         "heating_per_ring_mean_nm": round(float(per_ring.mean()), 3),
                         "crossover_median_mm": round(med, 3), "crossover_p5_mm": round(lo, 3),
                         "crossover_p95_mm": round(hi, 3),
                         "crossover_mean_mm": round(float(finite.mean()), 3)})
            print(f"{name:<36}{k:<14}{per_ring.mean():10.2f} nm{med:7.2f} mm{lo:8.2f}-{hi:.2f} mm")
    print(f"usual fixed assumption (2 nm per ring): {x_fixed_assumption:.2f} mm")
    print("published check: Krishnamoorthy et al. 2011, 1x8 array, 200 GHz grid, matched comb: "
          "< 1.2 nm per ring on average (measured); 1.72 nm covers 99% (their Monte Carlo)")

    with open(os.path.join(OUT, "strategy_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    plot(xs_by_design, x_fixed_assumption, os.path.join(OUT, "fig6_heater_strategies.png"))

    # Every chip's crossover, so any statistic in the paper can be recomputed.
    cols = [(d, k) for d in ("simple", "matched") for k in ("fixed", "barrel shift", "reassign")]
    with open(os.path.join(OUT, "chip_crossovers.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"{'ring5um' if d == 'simple' else 'matched'}_{k.replace(' ', '_')}_mm" for d, k in cols])
        for i in range(N_CHIPS):
            w.writerow([round(float(xs_by_design[d][k][i]), 4) for d, k in cols])
    # Fixed channels split the chips into two groups: rings that land on the short-wavelength
    # side of their channel need little heat, rings on the long side need almost a full FSR.
    xf = xs_by_design["simple"]["fixed"]
    xf = xf[np.isfinite(xf)]
    print(f"fixed channels, 5 um ring: mean {xf.mean():.2f} mm; {100 * np.mean(xf < 6.0):.0f}% of chips "
          f"cross before 6 mm, {100 * np.mean(xf > 7.5):.0f}% only beyond 7.5 mm")

    # Sensitivity: move one assumption at a time to the ends of its range.
    # Columns: fixed channels with the 5 um ring, and reassignment with the matched ring.
    print("SENSITIVITY (median crossover in mm: fixed + 5 um ring / reassign + matched ring)")
    sens = []
    for key, val, unit, desc, src, rng_ in ASSUMPTIONS:
        if rng_ is None:
            continue
        row = {"assumption": desc, "default": val, "low": rng_[0], "high": rng_[1]}
        for tag, v in (("low", rng_[0]), ("high", rng_[1])):
            a2 = dict(A); a2[key] = v
            (_, xf), _ = run_design(a2, "simple", wg, N_CHIPS_SWEEP, RNG_SEED, e, o)
            row[f"fixed_simple_{tag}_mm"] = round(stats(xf["fixed"])[0], 2)
            if key == "ring_radius_um":       # the matched design sets its own radius
                row[f"reassign_matched_{tag}_mm"] = ""
            else:
                (_, xm), _ = run_design(a2, "matched", wg, N_CHIPS_SWEEP, RNG_SEED, e, o)
                row[f"reassign_matched_{tag}_mm"] = round(stats(xm["reassign"])[0], 2)
        sens.append(row)
        rm = (f"{row['reassign_matched_low_mm']:5.2f}-{row['reassign_matched_high_mm']:5.2f}"
              if row["reassign_matched_low_mm"] != "" else "  (n/a)")
        print(f"  {desc:<52} {row['fixed_simple_low_mm']:5.2f}-{row['fixed_simple_high_mm']:5.2f}  /  {rm}")
    with open(os.path.join(OUT, "heater_sensitivity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sens[0])); w.writeheader(); w.writerows(sens)

    with open(os.path.join(OUT, "heater_assumptions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["assumption", "value", "unit", "source", "range tested"])
        for key, val, unit, desc, src, rng_ in ASSUMPTIONS:
            w.writerow([desc, val, unit, src, "" if rng_ is None else f"{rng_[0]} to {rng_[1]}"])
        w.writerow(["Matched ring radius (FSR = channels x spacing)", round(matched_radius(wg, A), 3), "um",
                    "Design rule from Krishnamoorthy et al. 2011; radius computed from simulated group index", ""])
        w.writerow(["Ring shift per nm of width error", round(wg["dlam_dwidth_nm_per_nm"], 3), "nm per nm",
                    "Simulated (simulate_from_scratch.py, femwell)", ""])
        w.writerow(["Ring shift per nm of thickness error", round(wg["dlam_dthick_nm_per_nm"], 3), "nm per nm",
                    "Simulated (simulate_from_scratch.py, femwell)", ""])
        w.writerow(["Ring shift per kelvin", round(wg["dlam_dT_nm_per_K"], 4), "nm per K",
                    "Simulated (simulate_from_scratch.py, femwell)", ""])
    print(f"Wrote results to {OUT}")


if __name__ == "__main__":
    main()
