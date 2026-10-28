"""
Check the wire-delay formula in crossover_model.py against a circuit simulator.

The crossover model uses a quick formula (Elmore delay with the 0.69 / 0.38
factors) for how long a signal takes to cross a copper wire. This script builds
the same wire as a detailed circuit -- 200 small resistor-capacitor sections --
simulates it in ngspice, and compares the two. It also times both, which is the
same accuracy-vs-runtime question as the internship.

Needs ngspice installed (free):  macOS: brew install ngspice
                                 Ubuntu/WSL: sudo apt install ngspice
Run:  python validate_wire_model.py
"""

import os
import re
import subprocess
import tempfile
import time

from crossover_model import Electrical, repeater_design, OHM_FF_TO_PS

N_SECTIONS = 200


def formula_delay_ps(e: Electrical, x_fF: float, L_mm: float) -> float:
    """Driver of input cap x drives L mm of wire into a load of input cap x."""
    k = OHM_FF_TO_PS
    r, c, tau, g = e.r_ohm_per_mm, e.c_fF_per_mm, e.tau_ps, e.gamma
    return (0.69 * (tau / x_fF) * (g * x_fF + c * L_mm + x_fF)
            + 0.69 * k * r * L_mm * x_fF
            + 0.38 * k * r * c * L_mm ** 2)


def netlist(e: Electrical, x_fF: float, L_mm: float, vdd: float) -> str:
    r_drv = e.tau_ps / x_fF * 1000.0          # ps/fF -> ohm
    c_out = e.gamma * x_fF                     # driver output cap, fF
    r_sec = e.r_ohm_per_mm * L_mm / N_SECTIONS
    c_sec = e.c_fF_per_mm * L_mm / N_SECTIONS
    lines = [
        "* driver + distributed RC wire + load",
        f"Vin in 0 PULSE(0 {vdd} 1p 0.01p 0.01p 10n 20n)",
        f"Rdrv in n0 {r_drv:.6g}",
        f"Cout n0 0 {c_out:.6g}f",
    ]
    for i in range(N_SECTIONS):          # pi-sections: C/2 - R - C/2
        lines.append(f"Ca{i} n{i} 0 {c_sec / 2:.6g}f")
        lines.append(f"R{i} n{i} n{i + 1} {r_sec:.6g}")
        lines.append(f"Cb{i} n{i + 1} 0 {c_sec / 2:.6g}f")
    lines.append(f"Cload n{N_SECTIONS} 0 {x_fF:.6g}f")
    t_stop = max(5.0, 4 * formula_delay_ps(e, x_fF, L_mm)) * 1e-12
    lines += [
        f".tran {t_stop / 4000:.4g} {t_stop:.4g}",
        f".meas tran tpd TRIG v(in) VAL={vdd / 2} RISE=1 TARG v(n{N_SECTIONS}) VAL={vdd / 2} RISE=1",
        ".end",
    ]
    return "\n".join(lines)


def spice_delay_ps(e: Electrical, x_fF: float, L_mm: float) -> float:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "wire.cir")
        with open(path, "w") as f:
            f.write(netlist(e, x_fF, L_mm, e.vdd))
        out = subprocess.run(["ngspice", "-b", path], capture_output=True, text=True).stdout
    m = re.search(r"tpd\s*=\s*([-+0-9.eE]+)", out)
    if not m:
        raise RuntimeError("ngspice did not report a delay:\n" + out[-2000:])
    return float(m.group(1)) * 1e12


def main() -> None:
    e = Electrical()
    d = repeater_design(e)
    x = d["x_fF"]
    cases = [("one repeater segment", d["l_mm"])] + [
        (f"{L:g} mm, no repeaters", L) for L in (1, 2, 5, 10)]

    print(f"{'case':<26}{'formula ps':>12}{'ngspice ps':>12}{'error':>9}"
          f"{'formula time':>15}{'ngspice time':>15}")
    for name, L in cases:
        t0 = time.perf_counter()
        f = formula_delay_ps(e, x, L)
        t_f = time.perf_counter() - t0
        t0 = time.perf_counter()
        s = spice_delay_ps(e, x, L)
        t_s = time.perf_counter() - t0
        print(f"{name:<26}{f:12.1f}{s:12.1f}{(f - s) / s * 100:8.1f}%"
              f"{t_f * 1e6:12.1f} us{t_s * 1e3:12.1f} ms")
    print(f"\nRepeated wire: {d['delay_ps_per_mm']:.1f} ps/mm from the formula; "
          f"ngspice segment gives {spice_delay_ps(e, x, d['l_mm']) / d['l_mm']:.1f} ps/mm")


if __name__ == "__main__":
    main()
