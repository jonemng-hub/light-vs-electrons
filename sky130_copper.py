"""
Real copper-wire numbers from the SKY130 process kit
====================================================

The copper side of the model used two hand-picked numbers: wire resistance
per mm and gate speed. This script replaces them with data from SkyWater's
SKY130 kit, installed with volare:

  1. Wire: reads each metal layer's sheet resistance and capacitance from the
     kit's technology file (the .tlef file), and converts them to ohms per mm
     and femtofarads per mm.
  2. Gates: simulates a SKY130 inverter in ngspice with the kit's transistor
     models, driving several loads, and fits its output resistance and its
     input and output capacitance. That gives the gate speed.

Then it reruns the light-vs-copper crossover with those numbers.

Needs: ngspice, and SKY130 installed with volare (in ~/.volare).
Run:   python3 sky130_copper.py
       (or python3 sky130_copper.py --pdk /path/to/sky130A if it can't find the kit)

You do not need to make a .spiceinit file; this script writes its own.
Results go to output/sky130_results.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import replace

import numpy as np

from crossover_model import Electrical, Optical, repeater_design, crossover_mm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
SUPPLY_V = 1.8            # SKY130 core supply voltage
WN_UM, WP_UM, L_UM = 0.65, 1.0, 0.15   # same sizes as the kit's smallest inverter (inv_1)
LOADS_FF = [0.0, 5.0, 10.0, 20.0, 40.0]
SPICEINIT = "set ngbehavior=hsa\nset ng_nomodcheck\n"   # settings the kit recommends


# ---------------------------------------------------------------------------
# Find the kit
# ---------------------------------------------------------------------------

def find_pdk(arg):
    if arg:
        return arg
    if os.environ.get("PDK_ROOT"):
        hits = glob.glob(os.path.join(os.environ["PDK_ROOT"], "**", "sky130A"), recursive=True)
        if hits:
            return sorted(hits)[-1]
    hits = glob.glob(os.path.expanduser("~/.volare/**/sky130A"), recursive=True)
    hits = [h for h in hits if os.path.isdir(os.path.join(h, "libs.tech"))]
    if not hits:
        sys.exit("Could not find SKY130. Install it with volare, or pass --pdk /path/to/sky130A")
    return sorted(hits)[-1]


def find_models(pdk):
    p = os.path.join(pdk, "libs.tech", "ngspice", "sky130.lib.spice")
    if not os.path.exists(p):
        sys.exit(f"Transistor models not found at {p}")
    return p


def find_tlef(pdk):
    pref = glob.glob(os.path.join(pdk, "libs.ref", "sky130_fd_sc_hd", "techlef", "*nom*.tlef"))
    anyt = glob.glob(os.path.join(pdk, "libs.ref", "**", "*.tlef"), recursive=True)
    hits = pref or sorted(anyt)
    if not hits:
        sys.exit(f"No technology (.tlef) file found under {pdk}/libs.ref")
    return hits[0]


# ---------------------------------------------------------------------------
# Part 1: wire resistance and capacitance per mm, from the kit's .tlef file
# ---------------------------------------------------------------------------

def read_metal_layers(tlef_path):
    """LEF units: RPERSQ in ohm/square, CPERSQDIST in pF/um^2,
    EDGECAPACITANCE in pF/um, WIDTH and SPACING in um."""
    text = open(tlef_path).read()
    layers = {}
    # Each layer block starts with a line "LAYER name" and ends with a line "END name".
    # Anchoring both to whole lines skips the "LAYER LEF58_..." property definitions.
    for m in re.finditer(r"^[ \t]*LAYER[ \t]+(\S+)[ \t]*\r?$(.*?)^[ \t]*END[ \t]+\1[ \t]*\r?$",
                         text, re.S | re.M):
        name, body = m.group(1), m.group(2)
        if not re.search(r"TYPE\s+ROUTING", body):
            continue
        def num(pattern):
            mm = re.search(pattern, body)
            return float(mm.group(1)) if mm else None
        layers[name] = {
            "width_um": num(r"\n\s*WIDTH\s+([0-9.eE+-]+)\s*;"),
            "spacing_um": num(r"\n\s*SPACING\s+([0-9.eE+-]+)\s*;"),
            "rsheet_ohm_per_sq": num(r"RESISTANCE\s+RPERSQ\s+([0-9.eE+-]+)"),
            "c_area_pF_per_um2": num(r"CAPACITANCE\s+CPERSQDIST\s+([0-9.eE+-]+)"),
            "c_edge_pF_per_um": num(r"EDGECAPACITANCE\s+([0-9.eE+-]+)"),
            "thickness_um": num(r"\n\s*THICKNESS\s+([0-9.eE+-]+)\s*;"),
        }
    return layers


def wire_per_mm(layer, width_um=None):
    w = width_um or layer["width_um"]
    r = layer["rsheet_ohm_per_sq"] / w * 1000.0                                 # ohm/mm
    c = (layer["c_area_pF_per_um2"] * w + 2 * layer["c_edge_pF_per_um"]) * 1e6   # fF/mm
    return r, c


# ---------------------------------------------------------------------------
# Part 2: inverter in ngspice with the kit's transistor models
# ---------------------------------------------------------------------------

def inverter_netlist(models, unit_style):
    """One inverter per load, all in one run so the slow model file loads once.
    unit_style 'scaled': sizes in microns (the kit normally sets scale = 1u).
    unit_style 'si': sizes with an explicit 'u' suffix, in case scaling is off."""
    def geom(w):
        if unit_style == "scaled":
            return (f"W={w} L={L_UM} ad={w * 0.29:.4f} as={w * 0.29:.4f} "
                    f"pd={2 * (w + 0.29):.4f} ps={2 * (w + 0.29):.4f}")
        return (f"W={w}u L={L_UM}u ad={w * 0.29:.4f}p as={w * 0.29:.4f}p "
                f"pd={2 * (w + 0.29):.4f}u ps={2 * (w + 0.29):.4f}u")
    v = SUPPLY_V
    lines = ["* SKY130 inverter characterization", f'.lib "{models}" tt',
             f"Vdd vdd 0 {v}"]
    for k, cl in enumerate(LOADS_FF):
        lines += [
            f"Vin{k} in{k} 0 PULSE(0 {v} 100p 10p 10p 1n 2n)",
            f"XMP{k} out{k} in{k} vdd vdd sky130_fd_pr__pfet_01v8 {geom(WP_UM)}",
            f"XMN{k} out{k} in{k} 0 0 sky130_fd_pr__nfet_01v8 {geom(WN_UM)}",
        ]
        if cl > 0:
            lines.append(f"CL{k} out{k} 0 {cl}f")
    lines.append(".tran 0.5p 2.2n")
    for k in range(len(LOADS_FF)):
        lines += [
            f".meas tran tphl{k} TRIG v(in{k}) VAL={v / 2} RISE=1 TARG v(out{k}) VAL={v / 2} FALL=1",
            f".meas tran tplh{k} TRIG v(in{k}) VAL={v / 2} FALL=1 TARG v(out{k}) VAL={v / 2} RISE=1",
            f".meas tran qin{k} INTEG i(Vin{k}) FROM=90p TO=1000p",
        ]
    lines.append(".end")
    return "\n".join(lines)


def run_ngspice(netlist):
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, ".spiceinit"), "w") as f:
            f.write(SPICEINIT)
        path = os.path.join(tmp, "inv.cir")
        with open(path, "w") as f:
            f.write(netlist)
        res = subprocess.run(["ngspice", "-b", path], cwd=tmp, capture_output=True, text=True)
    out = res.stdout + res.stderr
    vals = {m.group(1).lower(): float(m.group(2))
            for m in re.finditer(r"^\s*(\w+)\s*=\s*([-+0-9.eE]+)", out, re.M)}
    return vals, out


def characterize_inverter(models):
    for style in ("scaled", "si"):
        vals, log = run_ngspice(inverter_netlist(models, style))
        tpd = []
        for k in range(len(LOADS_FF)):
            a, b = vals.get(f"tphl{k}"), vals.get(f"tplh{k}")
            if a is None or b is None or not (0 < a < 1e-9 and 0 < b < 1e-9):
                tpd = None
                break
            tpd.append(0.5 * (a + b) * 1e12)                                   # ps
        if tpd:
            break
    else:
        print("ngspice did not return delays. Last lines of its output:\n")
        print("\n".join(log.strip().splitlines()[-40:]))
        sys.exit(1)
    # Switch model: delay = 0.69 * R0 * (Cout + Cload)
    slope, intercept = np.polyfit(LOADS_FF, tpd, 1)                           # ps/fF, ps
    r0_ohm = slope / 0.69 * 1000.0
    cout_ff = intercept / slope
    cin_ff = abs(vals["qin0"]) / SUPPLY_V * 1e15                               # charge into the gate
    return {"unit_style": style, "loads_fF": LOADS_FF, "delays_ps": [round(t, 3) for t in tpd],
            "R0_ohm": r0_ohm, "Cin_fF": cin_ff, "Cout_fF": cout_ff,
            "tau_ps": r0_ohm * cin_ff / 1000.0, "gamma": cout_ff / cin_ff}


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdk", help="path to the sky130A folder")
    ap.add_argument("--layer", default="met4", help="metal layer for the long wire (default met4)")
    args = ap.parse_args()

    pdk = find_pdk(args.pdk)
    models, tlef = find_models(pdk), find_tlef(pdk)
    print(f"SKY130 kit: {pdk}\n  models: {models}\n  tech file: {tlef}\n")

    layers = read_metal_layers(tlef)
    for v in layers.values():
        if v["c_edge_pF_per_um"] is None and v["c_area_pF_per_um2"] is not None:
            v["c_edge_pF_per_um"] = 0.0
            v["edge_cap_missing"] = True
    usable = {k: v for k, v in layers.items()
              if all(v[f] is not None for f in ("width_um", "rsheet_ohm_per_sq", "c_area_pF_per_um2"))}
    if any(v.get("edge_cap_missing") for v in usable.values()):
        print("  note: some layers have no edge capacitance in the tech file; using area capacitance only")
    if args.layer not in usable:
        found = {k: {f: v[f] for f in ("width_um", "rsheet_ohm_per_sq", "c_area_pF_per_um2")}
                 for k, v in layers.items()}
        print(f"Layer {args.layer} is missing values in {tlef}.")
        print(f"Routing layers found and their values: {found}")
        m = re.search(rf"^[ \t]*LAYER[ \t]+{args.layer}[ \t]*$(.*?)^[ \t]*END[ \t]+{args.layer}",
                      open(tlef).read(), re.S | re.M)
        if m:
            print(f"\nThe {args.layer} section of the tech file, for debugging:\n{m.group(0)[:3000]}")
        sys.exit(1)
    print("PART 1  Wire values from the kit (minimum width, capacitance to ground only)")
    print(f"  {'layer':<7}{'width um':>9}{'ohm/sq':>9}{'ohm/mm':>9}{'fF/mm':>8}")
    for name, L in usable.items():
        r, c = wire_per_mm(L)
        print(f"  {name:<7}{L['width_um']:9.3f}{L['rsheet_ohm_per_sq']:9.4f}{r:9.1f}{c:8.1f}")
    r_mm, c_mm = wire_per_mm(usable[args.layer])

    print("\nPART 2  SKY130 inverter in ngspice (typical corner, 1.8 V)")
    inv = characterize_inverter(models)
    for cl, t in zip(inv["loads_fF"], inv["delays_ps"]):
        print(f"  load {cl:5.1f} fF -> delay {t:6.2f} ps")
    print(f"  output resistance {inv['R0_ohm']:.0f} ohm, input cap {inv['Cin_fF']:.2f} fF, "
          f"output cap {inv['Cout_fF']:.2f} fF")
    print(f"  gate speed (R0 x Cin) {inv['tau_ps']:.2f} ps, output/input cap ratio {inv['gamma']:.2f}"
          f"  [size units: {inv['unit_style']}]")

    print(f"\nPART 3  Crossover with SKY130 copper ({args.layer}) vs. the light link")
    wg_path = os.path.join(OUT, "simulated_results.json")
    ng = json.load(open(wg_path))["waveguide_design"]["n_g"] if os.path.exists(wg_path) else 4.2
    o = replace(Optical(), e_rx_fJ=191.0, group_index=ng)
    cases = {
        "hand-picked (previous)": Electrical(),
        "SKY130 wire + gates, 1.8 V": Electrical(r_ohm_per_mm=r_mm, c_fF_per_mm=c_mm,
                                                tau_ps=inv["tau_ps"], gamma=inv["gamma"], vdd=SUPPLY_V),
        "SKY130 wire + gates, 0.8 V energy": Electrical(r_ohm_per_mm=r_mm, c_fF_per_mm=c_mm,
                                                       tau_ps=inv["tau_ps"], gamma=inv["gamma"], vdd=0.8),
    }
    rows = []
    print(f"  {'case':<36}{'ohm/mm':>8}{'fF/mm':>7}{'ps/mm':>7}{'fJ/mm':>7}  light wins past (energy / delay)")
    for name, e in cases.items():
        d = repeater_design(e)
        e_mm = e.activity * d["cap_fF_per_mm"] * e.vdd ** 2
        xe, xd = crossover_mm("energy", e, o), crossover_mm("delay", e, o)
        rows.append({"case": name, "r_ohm_per_mm": e.r_ohm_per_mm, "c_fF_per_mm": e.c_fF_per_mm,
                     "tau_ps": e.tau_ps, "gamma": e.gamma, "vdd": e.vdd,
                     "delay_ps_per_mm": d["delay_ps_per_mm"], "energy_fJ_per_mm": e_mm,
                     "repeater_spacing_mm": d["l_mm"], "energy_crossover_mm": xe, "delay_crossover_mm": xd})
        f = lambda x: f"{x:.2f}" if math.isfinite(x) else "never"
        print(f"  {name:<36}{e.r_ohm_per_mm:8.1f}{e.c_fF_per_mm:7.1f}{d['delay_ps_per_mm']:7.1f}"
              f"{e_mm:7.1f}  {f(xe):>6} mm / {f(xd)} mm")

    print("\nNotes")
    print("  - Capacitance is to ground only; neighbouring wires add more, so fF/mm is a lower bound.")
    print("  - The light link's electronics numbers come from newer 12-45 nm chips, so the 1.8 V row")
    print("    favours light. The 0.8 V row keeps SKY130's wire and gate speed but uses a modern supply.")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "sky130_results.json"), "w") as f:
        json.dump({"pdk": pdk, "tech_file": tlef, "layer": args.layer,
                   "metal_layers": usable, "inverter": inv, "crossovers": rows}, f, indent=2,
                  default=lambda x: None)
    print(f"\nWrote {os.path.join(OUT, 'sky130_results.json')}")


if __name__ == "__main__":
    main()
