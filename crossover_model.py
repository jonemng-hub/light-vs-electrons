"""
Light vs. electrons: at what length does an on-chip optical link beat a copper wire?
=================================================================================

A first-order model comparing the ENERGY PER BIT and DELAY of two ways to send
one bit across a chip:

  1. ELECTRICAL: a copper wire broken into segments by repeaters (inverters).
     Its parasitic resistance r and capacitance c are what slow it down and
     cost energy -- the same parasitics from the internship.

  2. OPTICAL: an off-chip laser feeds light into the chip, a ring modulator
     writes the bits onto the light, a silicon waveguide carries it, and a
     photodetector + receiver turn it back into an electrical signal.

The wire's energy grows with length. The optical link pays a large fixed cost
(laser, modulator, receiver, heater tuning) but its cost barely grows with
length. Where the two curves cross is the "crossover" (also called the
partition length or critical length).

The project question this script is built to answer:
    How far do real-world imperfections (waveguide roughness, coupler loss,
    manufacturing variation that must be fixed with heaters, ...) push the
    crossover length?

Run it:
    python crossover_model.py
It prints a summary and writes CSV data and PNG figures to ./output/.

Every default value below has a source in SOURCES at the bottom of this file.
Change a value, re-run, and you have a new data point.

Units used throughout (chosen so the numbers stay readable):
    length  mm      resistance  ohm/mm      capacitance  fF/mm or fF
    time    ps      energy      fJ (femtojoules) per bit
    power   W       bit rate    Gb/s
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, replace, asdict

import numpy as np

# 1 ohm x 1 fF = 1 fs = 0.001 ps. Used to keep r*c products in picoseconds.
OHM_FF_TO_PS = 1e-3
C_LIGHT_MM_PER_PS = 0.299792458  # speed of light in vacuum


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class Electrical:
    """Repeated copper wire. See SOURCES['electrical']."""
    r_ohm_per_mm: float = 100.0   # wire resistance per mm (global wire)
    c_fF_per_mm: float = 200.0    # wire capacitance per mm (~0.2 fF/um)
    tau_ps: float = 3.0           # R0*C0 of a minimum inverter (gate speed)
    gamma: float = 1.0            # inverter output cap / input cap
    vdd: float = 0.8              # supply voltage, V
    activity: float = 0.25        # chance a bit charges the wire (random data = 1/4)
    repeaters: str = "delay"      # "delay" = fastest repeaters, "energy" = within 5% of fastest, lower energy


@dataclass
class Optical:
    """On-chip silicon photonic link with an off-chip laser. See SOURCES['optical']."""
    bit_rate_gbps: float = 10.0
    rx_sensitivity_dBm: float = -17.0   # optical power the receiver needs
    coupler_loss_dB: float = 3.0        # laser-to-chip coupler
    n_couplers: int = 1
    modulator_loss_dB: float = 1.0      # insertion loss of the modulator
    margin_dB: float = 2.0              # link margin
    wg_loss_dB_per_cm: float = 1.0      # waveguide propagation loss (sidewall roughness)
    laser_wpe: float = 0.30             # laser wall-plug efficiency (optical out / electrical in)
    e_mod_fJ: float = 50.0              # modulator driver energy per bit
    e_rx_fJ: float = 191.0              # receiver energy per bit, measured (Samanta et al. Sec. IV);
                                        # was 60, which their Table I lists under the copper link
    n_tuned_rings: int = 2              # modulator ring + receiver filter ring
    tuning_uW_per_nm: float = 100.0     # heater power to shift a ring by 1 nm
    tuning_nm: float = 2.0              # wavelength error to correct (fab variation + temperature)
    serdes_fJ_per_order: float = 10.0   # serializer cost, only when bit rate > 2 x clock
    clock_ghz: float = 5.0
    group_index: float = 4.2            # silicon strip waveguide, ~4.2 at 1550 nm
    conversion_delay_ps: float = 49.0   # modulator driver + modulator + detector + amplifier


# ---------------------------------------------------------------------------
# Electrical wire model (Elmore-based repeated RC line)
# ---------------------------------------------------------------------------
#
# One repeater segment of length l (mm), driven by an inverter whose input
# capacitance is x (fF). Its output resistance is tau/x and its output
# capacitance is gamma*x. Using the switch-resistor model and the distributed
# RC line result (0.69 for lumped RC, 0.38 for a distributed line):
#
#   t_seg = 0.69*(tau/x)*(gamma*x + c*l + x) + 0.69*(r*l)*x + 0.38*(r*l)*(c*l)
#
# Dividing by l gives delay per mm. Minimising it gives the classic
# delay-optimal repeater size and spacing (Bakoglu):
#
#   x_opt = sqrt(tau*c / r)        l_opt = sqrt(0.69*tau*(1+gamma) / (0.38*r*c))

def _seg_delay_per_mm(e: Electrical, x_fF: float, l_mm: float) -> float:
    k = OHM_FF_TO_PS
    r, c, tau, g = e.r_ohm_per_mm, e.c_fF_per_mm, e.tau_ps, e.gamma
    return (0.69 * tau * (1 + g) / l_mm
            + 0.69 * tau * c / x_fF
            + 0.69 * k * r * x_fF
            + 0.38 * k * r * c * l_mm)


def _cap_per_mm(e: Electrical, x_fF: float, l_mm: float) -> float:
    """Wire capacitance plus repeater input+output capacitance, per mm."""
    return e.c_fF_per_mm + (1 + e.gamma) * x_fF / l_mm


def repeater_design(e: Electrical) -> dict:
    """Return repeater size x (fF), spacing l (mm), delay per mm, cap per mm."""
    k = OHM_FF_TO_PS
    r, c, tau, g = e.r_ohm_per_mm, e.c_fF_per_mm, e.tau_ps, e.gamma
    x_opt = math.sqrt(tau * c / (k * r))
    l_opt = math.sqrt(0.69 * tau * (1 + g) / (0.38 * k * r * c))
    d_opt = _seg_delay_per_mm(e, x_opt, l_opt)
    x, l = x_opt, l_opt
    if e.repeaters == "energy":
        # Smaller, sparser repeaters: lowest capacitance whose delay stays
        # within 5% of the fastest design (power-optimal repeater insertion).
        best = None
        for kx in np.linspace(0.10, 1.00, 181):
            for kl in np.linspace(1.00, 4.00, 301):
                d = _seg_delay_per_mm(e, kx * x_opt, kl * l_opt)
                if d <= 1.05 * d_opt:
                    cap = _cap_per_mm(e, kx * x_opt, kl * l_opt)
                    if best is None or cap < best[0]:
                        best = (cap, kx, kl)
        _, kx, kl = best
        x, l = kx * x_opt, kl * l_opt
    return {
        "x_fF": x,
        "l_mm": l,
        "delay_ps_per_mm": _seg_delay_per_mm(e, x, l),
        "cap_fF_per_mm": _cap_per_mm(e, x, l),
        "delay_optimal_ps_per_mm": d_opt,
    }


def electrical_energy_fJ(L_mm, e: Electrical, design: dict | None = None):
    d = design or repeater_design(e)
    return e.activity * d["cap_fF_per_mm"] * e.vdd ** 2 * np.asarray(L_mm, float)


def electrical_delay_ps(L_mm, e: Electrical, design: dict | None = None):
    d = design or repeater_design(e)
    return d["delay_ps_per_mm"] * np.asarray(L_mm, float)


def unrepeated_delay_ps(L_mm, e: Electrical, design: dict | None = None):
    """Same wire with no repeaters: one driver at the start, one load at the end.
    The 0.38*r*c*L^2 term makes delay grow with the SQUARE of length."""
    d = design or repeater_design(e)
    L = np.asarray(L_mm, float)
    x = d["x_fF"]
    k = OHM_FF_TO_PS
    r, c, tau, g = e.r_ohm_per_mm, e.c_fF_per_mm, e.tau_ps, e.gamma
    return (0.69 * (tau / x) * (g * x + c * L + x)
            + 0.69 * k * r * L * x
            + 0.38 * k * r * c * L ** 2)


# ---------------------------------------------------------------------------
# Optical link model
# ---------------------------------------------------------------------------
#
# Work backwards from the receiver: it needs P_rx of light. Every dB of loss
# between laser and receiver multiplies the laser power needed. The laser
# turns only a fraction (wall-plug efficiency) of its electrical power into
# light. A continuous laser burns power whether or not bits are sent, so its
# energy per bit is power / bit rate. Heaters that hold the rings on
# wavelength work the same way.

def optical_breakdown_fJ(L_mm, o: Optical) -> dict:
    L = np.asarray(L_mm, float)
    B = o.bit_rate_gbps * 1e9
    p_rx_W = 1e-3 * 10 ** (o.rx_sensitivity_dBm / 10)
    loss_dB = (o.coupler_loss_dB * o.n_couplers + o.modulator_loss_dB + o.margin_dB
               + o.wg_loss_dB_per_cm * L / 10.0)
    p_laser_elec_W = p_rx_W * 10 ** (loss_dB / 10) / o.laser_wpe
    tuning_W = o.n_tuned_rings * o.tuning_uW_per_nm * 1e-6 * o.tuning_nm
    serdes = (o.serdes_fJ_per_order * o.bit_rate_gbps / (2 * o.clock_ghz)
              if o.bit_rate_gbps > 2 * o.clock_ghz else 0.0)
    ones = np.ones_like(L)
    return {
        "laser": p_laser_elec_W / B * 1e15,
        "modulator": o.e_mod_fJ * ones,
        "receiver": o.e_rx_fJ * ones,
        "tuning": tuning_W / B * 1e15 * ones,
        "serdes": serdes * ones,
    }


def optical_energy_fJ(L_mm, o: Optical):
    return sum(optical_breakdown_fJ(L_mm, o).values())


def optical_delay_ps(L_mm, o: Optical):
    L = np.asarray(L_mm, float)
    return o.conversion_delay_ps + o.group_index * L / C_LIGHT_MM_PER_PS


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

def crossover_mm(metric: str, e: Electrical, o: Optical, L_max: float = 200.0) -> float:
    """Shortest length (mm) beyond which the optical link wins on `metric`
    ("energy" or "delay"). Returns math.inf if optical never wins by L_max."""
    d = repeater_design(e)
    if metric == "energy":
        diff = lambda L: optical_energy_fJ(L, o) - electrical_energy_fJ(L, e, d)
    elif metric == "delay":
        diff = lambda L: optical_delay_ps(L, o) - electrical_delay_ps(L, e, d)
    else:
        raise ValueError(metric)
    grid = np.linspace(0.0, L_max, 4001)
    vals = np.asarray(diff(grid), float)
    idx = np.where(vals < 0)[0]
    if len(idx) == 0:
        return math.inf
    i = idx[0]
    if i == 0:
        return 0.0
    lo, hi = grid[i - 1], grid[i]
    for _ in range(60):  # bisection
        mid = 0.5 * (lo + hi)
        if float(diff(mid)) < 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Sensitivity ("what moves the crossover?")
# ---------------------------------------------------------------------------
# Each row: (label, side, field, value that helps optical, value that hurts optical, kind)
# kind: "imperfection" = set by physics or manufacturing quality,
#       "design" = a choice the engineer makes, or a technology level.
SWEEPS = [
    ("Waveguide loss (dB/cm)",            "o", "wg_loss_dB_per_cm", 0.5,   6.0,   "imperfection"),
    ("Coupler loss (dB)",                 "o", "coupler_loss_dB",   1.0,   5.0,   "imperfection"),
    ("Wavelength error to fix (nm)",      "o", "tuning_nm",         0.5,   5.0,   "imperfection"),
    ("Heater efficiency (uW per nm)",     "o", "tuning_uW_per_nm",  50.0,  225.0, "imperfection"),
    ("Laser efficiency",                  "o", "laser_wpe",         0.40,  0.10,  "imperfection"),
    ("Receiver sensitivity (dBm)",        "o", "rx_sensitivity_dBm", -24.0, -12.0, "design"),
    ("Modulator energy (fJ/bit)",         "o", "e_mod_fJ",          10.0,  100.0, "design"),
    ("Receiver energy (fJ/bit)",          "o", "e_rx_fJ",           20.0,  200.0, "design"),
    ("Bit rate (Gb/s)",                   "o", "bit_rate_gbps",     25.0,  5.0,   "design"),
    ("Wire activity",                     "e", "activity",          0.50,  0.10,  "design"),
    ("Wire capacitance (fF/mm)",          "e", "c_fF_per_mm",       250.0, 150.0, "design"),
]


def sensitivity(e: Electrical, o: Optical) -> list[dict]:
    base = crossover_mm("energy", e, o)
    rows = []
    for label, side, field, good, bad, kind in SWEEPS:
        res = {}
        for tag, val in (("good", good), ("bad", bad)):
            e2 = replace(e, **{field: val}) if side == "e" else e
            o2 = replace(o, **{field: val}) if side == "o" else o
            res[tag] = crossover_mm("energy", e2, o2)
        rows.append({"label": label, "kind": kind, "field": field,
                     "good_value": good, "bad_value": bad,
                     "crossover_good_mm": res["good"], "crossover_bad_mm": res["bad"],
                     "baseline_mm": base})
    rows.sort(key=lambda r: abs(min(r["crossover_bad_mm"], 200) - r["crossover_good_mm"]),
              reverse=True)
    return rows


def combined_imperfections(e: Electrical, o: Optical) -> list[dict]:
    """One-at-a-time sweeps miss interactions. Put EVERY imperfection at its
    worst value together, then fix them one at a time to see which matters."""
    imp = [s for s in SWEEPS if s[5] == "imperfection"]
    best = {s[2]: s[3] for s in imp}
    worst = {s[2]: s[4] for s in imp}
    rows = [
        {"case": "all imperfections at best value", "crossover_mm": crossover_mm("energy", e, replace(o, **best))},
        {"case": "baseline (published defaults)", "crossover_mm": crossover_mm("energy", e, o)},
        {"case": "all imperfections at worst value", "crossover_mm": crossover_mm("energy", e, replace(o, **worst))},
    ]
    for label, _, field, *_ in imp:
        w = dict(worst)
        w[field] = getattr(o, field)          # put this one back to baseline
        rows.append({"case": f"all worst, but {label} at baseline",
                     "crossover_mm": crossover_mm("energy", e, replace(o, **w))})
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def make_figures(e: Electrical, o: Optical, outdir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    LIGHT, COPPER = "#2a78d6", "#eb6834"     # optical link = blue, copper wire = orange
    INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    })
    d = repeater_design(e)
    L = np.linspace(0, 20, 401)

    # 1. Energy per bit vs length
    xe = crossover_mm("energy", e, o)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(L, electrical_energy_fJ(L, e, d), color=COPPER, lw=2, label="Copper wire (repeated)")
    ax.plot(L, optical_energy_fJ(L, o), color=LIGHT, lw=2, label="Optical link")
    if math.isfinite(xe):
        ax.axvline(xe, color=MUTED, lw=1)
        ax.annotate(f"crossover {xe:.1f} mm", (xe, float(optical_energy_fJ(xe, o))),
                    xytext=(8, 10), textcoords="offset points", color=INK)
    ax.set_xlabel("Link length (mm)")
    ax.set_ylabel("Energy per bit (fJ)")
    ax.set_title("Energy per bit vs. length", loc="left", color=INK)
    ax.set_xlim(0, 20); ax.set_ylim(0, None)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig1_energy_vs_length.png"), dpi=200)
    plt.close(fig)

    # 2. Delay vs length (with the no-repeater wire for contrast)
    xd = crossover_mm("delay", e, o)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(L, electrical_delay_ps(L, e, d), color=COPPER, lw=2, label="Copper wire (repeated)")
    ax.plot(L, unrepeated_delay_ps(L, e, d), color=COPPER, lw=1, alpha=0.45,
            label="Copper wire (no repeaters)")
    ax.plot(L, optical_delay_ps(L, o), color=LIGHT, lw=2, label="Optical link")
    if math.isfinite(xd):
        ax.axvline(xd, color=MUTED, lw=1)
        ax.annotate(f"crossover {xd:.1f} mm", (xd, float(optical_delay_ps(xd, o))),
                    xytext=(8, -16), textcoords="offset points", color=INK)
    ax.set_xlabel("Link length (mm)")
    ax.set_ylabel("Delay (ps)")
    ax.set_title("Delay vs. length", loc="left", color=INK)
    ax.set_xlim(0, 20); ax.set_ylim(0, 600)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig2_delay_vs_length.png"), dpi=200)
    plt.close(fig)

    # 3. Sensitivity: how far each assumption moves the energy crossover
    rows = sensitivity(e, o)
    base = rows[0]["baseline_mm"]
    finite = [r["crossover_bad_mm"] for r in rows if math.isfinite(r["crossover_bad_mm"])]
    cap = math.ceil(max(finite + [base]) + 1.0)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for i, r in enumerate(reversed(rows)):
        g, b = r["crossover_good_mm"], min(r["crossover_bad_mm"], cap)
        lo, hi = min(g, b), max(g, b)
        color = LIGHT if r["kind"] == "imperfection" else MUTED
        ax.barh(i, hi - lo, left=lo, height=0.55, color=color)
        ax.text(hi + 0.3, i, f"{r['crossover_bad_mm']:.1f}" if r["crossover_bad_mm"] < cap
                else f">{cap:.0f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in reversed(rows)])
    ax.axvline(base, color=INK, lw=1)
    ax.set_xlabel("Energy crossover length (mm)  -  bar spans best case to worst case")
    ax.set_title(f"What moves the energy crossover? Baseline {base:.1f} mm\n"
                 "blue = imperfection, gray = design choice",
                 loc="left", color=INK, fontsize=10)
    ax.set_xlim(0, cap + 1)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig3_sensitivity.png"), dpi=200)
    plt.close(fig)

    # 4. Crossover vs bit rate (laser + heater power is shared over more bits)
    rates = np.linspace(2, 50, 97)
    xs = [crossover_mm("energy", e, replace(o, bit_rate_gbps=float(b))) for b in rates]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(rates, xs, color=LIGHT, lw=2)
    knee = 2 * o.clock_ghz
    ax.axvline(knee, color=MUTED, lw=1)
    ax.annotate(f"serializer needed above {knee:g} Gb/s\n(2 x {o.clock_ghz:g} GHz clock)",
                (knee, max(x for x in xs if math.isfinite(x)) * 0.8),
                xytext=(8, 0), textcoords="offset points", color=INK2, fontsize=9)
    ax.set_xlabel("Optical bit rate (Gb/s)")
    ax.set_ylabel("Energy crossover length (mm)")
    ax.set_title("Faster optical links win sooner (until serialization costs bite)",
                 loc="left", color=INK, fontsize=10)
    ax.set_ylim(0, None)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig4_crossover_vs_bitrate.png"), dpi=200)
    plt.close(fig)


def main() -> None:
    e, o = Electrical(), Optical()
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(outdir, exist_ok=True)

    d = repeater_design(e)
    print("ELECTRICAL (repeated copper wire)")
    print(f"  repeater every {d['l_mm']:.2f} mm, input cap {d['x_fF']:.1f} fF")
    print(f"  delay  {d['delay_ps_per_mm']:.1f} ps/mm")
    print(f"  capacitance incl. repeaters {d['cap_fF_per_mm']:.0f} fF/mm")
    print(f"  energy {float(electrical_energy_fJ(1, e, d)):.1f} fJ/bit per mm")

    br = {k: float(v) for k, v in optical_breakdown_fJ(0.0, o).items()}
    print("OPTICAL (fixed cost at zero length)")
    for k, v in br.items():
        print(f"  {k:<10}{v:7.1f} fJ/bit")
    print(f"  total     {sum(br.values()):7.1f} fJ/bit")
    print(f"  delay     {o.conversion_delay_ps:.0f} ps + {o.group_index / C_LIGHT_MM_PER_PS:.1f} ps/mm")

    xe, xd = crossover_mm("energy", e, o), crossover_mm("delay", e, o)
    print(f"CROSSOVER  energy {xe:.2f} mm   delay {xd:.2f} mm")
    xe2 = crossover_mm("energy", replace(e, repeaters="energy"), o)
    d2 = repeater_design(replace(e, repeaters="energy"))
    print(f"  with energy-saving repeaters ({d2['delay_ps_per_mm']:.1f} ps/mm, "
          f"{d2['cap_fF_per_mm']:.0f} fF/mm): energy crossover {xe2:.2f} mm")

    # Sweep data (one row per length)
    L = np.round(np.linspace(0, 30, 301), 3)
    br_L = optical_breakdown_fJ(L, o)
    rows = []
    for i, Li in enumerate(L):
        rows.append({
            "length_mm": Li,
            "electrical_energy_fJ": round(float(electrical_energy_fJ(Li, e, d)), 3),
            "optical_energy_fJ": round(float(sum(v[i] for v in br_L.values())), 3),
            "optical_laser_fJ": round(float(br_L["laser"][i]), 3),
            "optical_tuning_fJ": round(float(br_L["tuning"][i]), 3),
            "electrical_delay_ps": round(float(electrical_delay_ps(Li, e, d)), 3),
            "electrical_unrepeated_delay_ps": round(float(unrepeated_delay_ps(Li, e, d)), 3),
            "optical_delay_ps": round(float(optical_delay_ps(Li, o)), 3),
        })
    write_csv(os.path.join(outdir, "sweep_vs_length.csv"), rows)

    sens = sensitivity(e, o)
    write_csv(os.path.join(outdir, "sensitivity.csv"), sens)
    print("SENSITIVITY (energy crossover, best case -> worst case)")
    for r in sens:
        bad = r["crossover_bad_mm"]
        bad_s = f"{bad:6.2f} mm" if math.isfinite(bad) else " never"
        print(f"  [{r['kind']:<12}] {r['label']:<32}{r['crossover_good_mm']:6.2f} -> {bad_s}")

    combo = combined_imperfections(e, o)
    write_csv(os.path.join(outdir, "combined_imperfections.csv"), combo)
    print("COMBINED IMPERFECTIONS (energy crossover)")
    for r in combo:
        x = r["crossover_mm"]
        print(f"  {r['case']:<62}{'never' if not math.isfinite(x) else f'{x:.2f} mm'}")

    make_figures(e, o, outdir)
    print(f"Wrote CSVs and figures to {outdir}")


# ---------------------------------------------------------------------------
# Where the default numbers come from
# ---------------------------------------------------------------------------
SOURCES = {
    "electrical": [
        "c = 200 fF/mm: typical wire capacitance ~0.2 fF/um (UT Austin EE382M "
        "interconnect lecture notes; MIT 6.884 L04 gives 100-160 fF/mm).",
        "r = 100 ohm/mm and tau = 3 ps are representative global-wire and gate "
        "values chosen so the delay-optimal repeated wire lands at ~22 ps/mm, "
        "matching the 20-25 ps/mm reported for delay-optimized global wires from "
        "90 nm to 22 nm (Haurylau et al., 'On-chip Optical Interconnect Roadmap'; "
        "Friedman, 'Predictions, Challenges, and Opportunities in CMOS Compatible "
        "On-Chip Optical Interconnect').",
        "Check: the model gives ~0.41 pF per repeated mm, inside the 0.35-0.45 "
        "pF/mm quoted in UT Austin EE382N lecture notes.",
        "activity = 0.25: for random data a wire is charged (0->1) on 1/4 of bits.",
    ],
    "optical": [
        "Coupler 3 dB, modulator loss 1 dB, waveguide 1 dB/cm, laser wall-plug "
        "efficiency 30%, modulator driver 50 fJ/bit, link margin 2 dB: Samanta et "
        "al., '3D Electronic-Photonic Heterogenous Interconnect Platforms...', "
        "arXiv:2510.03943, Table I and Sec. IV. Receiver 191 fJ/bit: their measured "
        "optical receiver (Sec. IV).",
        "Receiver sensitivity -17 dBm: demonstrated at 25 Gb/s in the same work "
        "(their ref. [31]); -24 dBm is their projected value.",
        "Tuning: 2 rings x 2 nm x 100 uW/nm, and serializer cost model: "
        "Manipatruni, Lipson & Young, 'Device Considerations for Nanophotonic CMOS "
        "Global Interconnects', arXiv:1207.6819 (one demonstrated heater: 225 uW/nm).",
        "Group index 4.2: silicon strip waveguide, 220 nm thick, ~450-500 nm wide "
        "at 1550 nm (reported values 4.15-4.37).",
        "Conversion delay 49 ps: modulator driver 5.2 + modulator 40 + detector 0.4 "
        "+ amplifier 3.4 ps at the 22 nm node (Friedman, Table I).",
    ],
}


if __name__ == "__main__":
    main()
