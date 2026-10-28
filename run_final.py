"""
Final results: everything rerun with real SKY130 copper data
===========================================================

One command:
    python3 run_final.py

It reuses the results of the other scripts, and runs them first if needed:
  - simulate_from_scratch.py  (waveguide simulation)  -> output/simulated_results.json
  - sky130_copper.py          (SKY130 wire + gates)   -> output/sky130_results.json

Then it reruns the heater-strategy comparison (10,000 simulated chips) for four
copper cases, because two copper assumptions are still uncertain:

  supply voltage   1.8 V is SKY130's own supply. 0.8 V matches the newer chips
                   the light link's electronics numbers come from.
  neighbour wires  "isolated" uses the kit's capacitance to ground only.
                   "packed" adds coupling to a neighbouring wire on each side at
                   minimum spacing (estimate explained in neighbour_capacitance()).

Outputs (in output/):
  final_results.csv         crossover for every copper case x heater case
  final_assumptions.csv     every input, its value and its source (Table 1)
  fig7_final_sky130.png     2 x 2 panels, one per copper case
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np

import tuning_strategies as T
from crossover_model import Electrical, Optical, repeater_design, crossover_mm, C_LIGHT_MM_PER_PS
from sky130_copper import wire_per_mm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

# Coupling estimate inputs
EPS0_FF_PER_UM = 8.854e-3     # vacuum permittivity in fF/um
EPS_R_OXIDE = 3.9             # silicon dioxide (textbook value)
MET4_THICKNESS_UM = 0.8       # SKY130 docs, "Criteria & Assumptions" (met4, S8P* flows)

# Light-link conversion delay for a 130 nm-era chip: Friedman, Table I, 90 nm column
# (closest node listed): modulator driver 37.3 + modulator 40.0 + detector 2.5 + amplifier 34.0 ps
CONV_DELAY_90NM_PS = 37.3 + 40.0 + 2.5 + 34.0

HEATER_CASES = [  # (label, ring design, strategy, colour) - same as fig6
    ("Fixed channels, 5 um ring", "simple", "fixed", "#2a78d6"),
    ("Reassign rings, 5 um ring", "simple", "reassign", "#4a3aa7"),
    ("Reassign rings, matched ring", "matched", "reassign", "#1baf7a"),
]


def ensure(path, script):
    if not os.path.exists(path):
        print(f"{os.path.basename(path)} not found, running {script} first...\n")
        subprocess.run([sys.executable, os.path.join(HERE, script)], check=True, cwd=HERE)


def neighbour_capacitance(layer):
    """Side-to-side capacitance to one neighbouring wire at minimum spacing,
    treated as two parallel metal plates: C = eps0 * eps_r * thickness / spacing.
    Ignores fringing at the corners, so it is a simple estimate, not an extraction."""
    t = layer.get("thickness_um") or MET4_THICKNESS_UM
    s = layer.get("spacing_um") or 0.3
    return EPS0_FF_PER_UM * EPS_R_OXIDE * t / s * 1000.0, t, s     # fF/mm per neighbour


def stats(x):
    f = x[np.isfinite(x)]
    if len(f) == 0:
        return math.inf, math.inf, math.inf
    return float(np.median(f)), float(np.percentile(f, 5)), float(np.percentile(f, 95))


def fmt(x):
    return f"{x:.2f}" if math.isfinite(x) else "never"


def plot(panels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                         "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.6), sharey=True)
    for ax, p in zip(axes.flat, panels):
        finite = [v for k in p["xs"] for v in p["xs"][k][np.isfinite(p["xs"][k])]]
        xmax = max(1.0, np.percentile(finite, 99.5) * 1.25) if finite else 20
        L = np.linspace(0, xmax, 400)
        for label, design, strat, color in HEATER_CASES:
            x = p["xs"][(design, strat)]
            ax.plot(L, [100 * np.mean(x < l) for l in L], color=color, lw=2, label=label)
        ax.axvline(p["x_usual"], color=MUTED, lw=1)
        ax.set_title(p["title"], loc="left", color=INK, fontsize=9.5)
        ax.set_xlim(0, xmax); ax.set_ylim(0, 100)
        ax.set_xlabel("Link length (mm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Chips where light uses less energy (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=MUTED, lw=1)); labels.append("Usual 2 nm per ring assumption")
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle("Where light beats copper, with SKY130 copper data (10,000 chips, 8 channels)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=200); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    wg_path, sky_path = os.path.join(OUT, "simulated_results.json"), os.path.join(OUT, "sky130_results.json")
    ensure(wg_path, "simulate_from_scratch.py")
    ensure(sky_path, "sky130_copper.py")
    # Refresh the quick scripts so every output uses the same updated inputs
    for script in ("crossover_model.py", "tuning_strategies.py"):
        print(f"Rerunning {script} with the updated inputs...")
        subprocess.run([sys.executable, os.path.join(HERE, script)], check=True, cwd=HERE,
                       stdout=subprocess.DEVNULL)
    print()
    wg = json.load(open(wg_path))["waveguide_design"]
    sky = json.load(open(sky_path))
    inv, layer = sky["inverter"], sky["metal_layers"]["met4"]

    r_mm, c_ground = wire_per_mm(layer)
    c_side, t_used, s_used = neighbour_capacitance(layer)
    c_packed = c_ground + 2 * c_side
    print(f"SKY130 met4: {r_mm:.1f} ohm/mm; capacitance {c_ground:.1f} fF/mm isolated, "
          f"{c_packed:.1f} fF/mm packed (+{c_side:.1f} per neighbour, t={t_used} um, s={s_used} um)")
    print(f"SKY130 inverter: gate speed {inv['tau_ps']:.2f} ps, output/input cap {inv['gamma']:.2f}")

    o = replace(Optical(), group_index=wg["n_g"])            # receiver 191 fJ/bit is now the default
    copper = [
        ("1.8 V, isolated wire", 1.8, c_ground),
        ("1.8 V, packed wire", 1.8, c_packed),
        ("0.8 V, isolated wire", 0.8, c_ground),
        ("0.8 V, packed wire", 0.8, c_packed),
    ]
    rows, panels = [], []
    print(f"\n{'copper case':<22}{'heater case':<32}{'median':>9}{'90% of chips':>18}")
    for name, vdd, c in copper:
        e = Electrical(r_ohm_per_mm=r_mm, c_fF_per_mm=c, tau_ps=inv["tau_ps"], gamma=inv["gamma"], vdd=vdd)
        d = repeater_design(e)
        e_mm = e.activity * d["cap_fF_per_mm"] * vdd ** 2
        x_usual = crossover_mm("energy", e, o)
        x_delay = CONV_DELAY_90NM_PS / (d["delay_ps_per_mm"] - wg["n_g"] / C_LIGHT_MM_PER_PS)
        xs = {}
        for label, design, strat, _ in HEATER_CASES:
            if (design, strat) not in xs:
                (heat, res), _r = T.run_design(T.A, design, wg, T.N_CHIPS, T.RNG_SEED, e, o)
                for k in res:
                    xs[(design, k)] = res[k]
            med, lo, hi = stats(xs[(design, strat)])
            rows.append({"copper_case": name, "vdd_V": vdd, "c_fF_per_mm": round(c, 1),
                         "r_ohm_per_mm": round(r_mm, 1), "copper_ps_per_mm": round(d["delay_ps_per_mm"], 2),
                         "copper_fJ_per_mm": round(e_mm, 1), "heater_case": label,
                         "energy_crossover_median_mm": round(med, 3), "energy_crossover_p5_mm": round(lo, 3),
                         "energy_crossover_p95_mm": round(hi, 3),
                         "energy_crossover_usual_2nm_mm": round(x_usual, 3),
                         "delay_crossover_mm": round(x_delay, 3)})
            print(f"{name:<22}{label:<32}{fmt(med):>6} mm{fmt(lo):>9}-{fmt(hi)} mm")
        print(f"{'':<22}{'usual 2 nm assumption':<32}{fmt(x_usual):>6} mm   "
              f"(copper {e_mm:.0f} fJ/mm, {d['delay_ps_per_mm']:.1f} ps/mm; light faster past {x_delay:.2f} mm)")
        panels.append({"title": f"{name} ({c:.0f} fF/mm)", "xs": xs, "x_usual": x_usual})

    with open(os.path.join(OUT, "final_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    plot(panels, os.path.join(OUT, "fig7_final_sky130.png"))

    # Table 1: every input with its source
    A = T.A
    table = [
        ("Copper wire resistance (met4, min width)", f"{r_mm:.1f}", "ohm/mm", "SKY130 kit tech file (sheet resistance / width)"),
        ("Copper wire capacitance to ground", f"{c_ground:.1f}", "fF/mm", "SKY130 kit tech file (area + edge capacitance)"),
        ("Coupling to one neighbouring wire", f"{c_side:.1f}", "fF/mm",
         f"Estimate: parallel plates, eps_r {EPS_R_OXIDE} (SiO2), thickness {t_used} um (SKY130 docs), spacing {s_used} um (kit)"),
        ("Gate speed R0 x Cin", f"{inv['tau_ps']:.2f}", "ps", "SKY130 inverter (inv_1 sizes) simulated in ngspice, typical corner"),
        ("Gate output / input capacitance", f"{inv['gamma']:.2f}", "", "Same ngspice simulation"),
        ("Supply voltage", "1.8 and 0.8", "V", "1.8 V: SKY130 core supply. 0.8 V: matches the newer chips behind the optical numbers"),
        ("Data activity", "0.25", "", "Random data: a wire charges on 1 bit in 4 (textbook)"),
        ("Optical receiver sensitivity", f"{o.rx_sensitivity_dBm}", "dBm", "Samanta et al. 2025, Sec. IV (measured at 25 Gb/s)"),
        ("Coupler / modulator loss", f"{o.coupler_loss_dB} / {o.modulator_loss_dB}", "dB", "Samanta et al. 2025, Table I"),
        ("Waveguide loss", f"{o.wg_loss_dB_per_cm}", "dB/cm", "Samanta et al. 2025, Table I"),
        ("Laser wall-plug efficiency", f"{o.laser_wpe}", "", "Samanta et al. 2025, Table I"),
        ("Modulator driver energy", f"{o.e_mod_fJ}", "fJ/bit", "Samanta et al. 2025, Table I"),
        ("Receiver energy", f"{o.e_rx_fJ}", "fJ/bit", "Samanta et al. 2025, Sec. IV (measured)"),
        ("Bit rate", f"{o.bit_rate_gbps}", "Gb/s", "Design choice"),
        ("Waveguide group index", f"{wg['n_g']:.2f}", "", "Simulated (femwell); published 4.15-4.37"),
        ("Conversion delay (130 nm-era)", f"{CONV_DELAY_90NM_PS:.1f}", "ps", "Friedman, Table I, 90 nm column"),
    ]
    for key, val, unit, desc, src, rng in T.ASSUMPTIONS:
        table.append((desc, str(val), unit, src + ("" if rng is None else f"; range tested {rng[0]} to {rng[1]}")))
    table += [
        ("Ring shift per nm of width error", f"{wg['dlam_dwidth_nm_per_nm']:.3f}", "nm/nm", "Simulated (femwell)"),
        ("Ring shift per nm of thickness error", f"{wg['dlam_dthick_nm_per_nm']:.3f}", "nm/nm", "Simulated (femwell); Su et al. measured 1.367"),
        ("Ring shift per kelvin", f"{wg['dlam_dT_nm_per_K']:.4f}", "nm/K", "Simulated (femwell)"),
    ]
    with open(os.path.join(OUT, "final_assumptions.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["input", "value", "unit", "source"]); w.writerows(table)
    print(f"\nWrote final_results.csv, final_assumptions.csv and fig7_final_sky130.png to {OUT}")


if __name__ == "__main__":
    main()
