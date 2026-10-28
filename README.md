# Where Does Light Beat Copper?

Code and data for the paper *Where Does Light Beat Copper? How Manufacturing Variation and
Heater Strategy Set the Crossover Length of On-Chip Optical Links* (J. Ng, 2026).

**Question:** at what length does an on-chip optical link use less energy per bit than a copper
wire, and how do manufacturing variation and the heater (ring-tuning) strategy move that length?

## Main results

| | Energy crossover |
|---|---|
| Reference link, conventional 2 nm of tuning per ring | 4.73 mm |
| 10,000 chips, fixed channels, 5 µm rings (median, 90% of chips) | 6.93 mm (4.41–9.44) |
| 10,000 chips, reassigned channels, matched 7.11 µm rings | 4.49 mm (4.28–4.68) |

With fixed channels the chips split into two groups: 43% cross before 6 mm and 47% only beyond
7.5 mm, because heaters can only warm a ring, so a ring that lands on the wrong side of its
channel must be heated almost a full free spectral range.

With SKY130 copper (median of 10,000 chips, Table II of the paper):

| Copper case | Copper energy (fJ/bit/mm) | Matched rings + reassign | Fixed channels, 5 µm ring | Conventional 2 nm |
|---|---|---|---|---|
| 1.8 V, packed wire | 443.0 | 0.66 mm | 1.02 mm | 0.70 mm |
| 1.8 V, isolated wire | 129.3 | 2.26 mm | 3.50 mm | 2.39 mm |
| 0.8 V, packed wire | 87.5 | 3.35 mm | 5.18 mm | 3.54 mm |
| 0.8 V, isolated wire | 25.5 | 11.73 mm | 18.17 mm | 12.38 mm |

Fixed channels push the crossover 1.55 times further than matched, reassigned rings in every
copper case. The paper explains why: the crossover is roughly the optical link's fixed energy
divided by copper's energy per mm, so the copper value cancels in the ratio.

## Where each figure, table and number in the paper comes from

| In the paper | File | Made by |
|---|---|---|
| Fig. 1 | `output/fig1_energy_vs_length.png` | `crossover_model.py` |
| Fig. 2 | `output/fig6_heater_strategies.png` | `tuning_strategies.py` |
| Table I (and every other input, with sources) | `output/final_assumptions.csv`, `output/heater_assumptions.csv` | `run_final.py`, `tuning_strategies.py` |
| Table II | `output/final_results.csv` | `run_final.py` |
| Sec. II-C, wire validation | printed output | `validate_wire_model.py` |
| Sec. II-C, waveguide and ring sensitivities | `output/simulated_results.json` | `simulate_from_scratch.py` |
| Sec. III-A, one-at-a-time sensitivity | `output/sensitivity.csv` | `crossover_model.py` |
| Sec. III-B, strategy results and robustness | `output/strategy_results.csv`, `output/heater_sensitivity.csv` | `tuning_strategies.py` |
| Sec. III-B, crossover of every simulated chip | `output/chip_crossovers.csv` | `tuning_strategies.py` |

Supplementary figures, not in the paper: `fig2_delay_vs_length.png`, `fig3_sensitivity.png`,
`fig4_crossover_vs_bitrate.png` and `fig7_final_sky130.png` (all four SKY130 copper cases).
`fig5_probability_light_wins.png` comes from an earlier single-ring version of the tuning model.

## How to reproduce

Python 3.10 or newer. The random seeds are fixed, so every run gives the same numbers.

```bash
pip install -r requirements.txt          # Linux also needs: sudo apt install libglu1-mesa
python3 simulate_from_scratch.py         # waveguide mode solver + Monte Carlo, about 2 minutes
python3 crossover_model.py               # Fig. 1, sensitivity analysis
python3 tuning_strategies.py             # Fig. 2, heater strategies, about 10 seconds
```

For the SKY130 results (Tables I and II), install [ngspice](https://ngspice.sourceforge.io/) and
the SKY130 process design kit with [volare](https://github.com/efabless/volare)
(`pip install volare`, then `volare enable --pdk sky130 <version>`), then:

```bash
python3 sky130_copper.py                 # reads met4 R and C from the kit, simulates an inverter in ngspice
python3 run_final.py                     # Table I, Table II, fig7
python3 validate_wire_model.py           # optional: checks the wire delay formula against ngspice
```

The kit version used is recorded in `output/sky130_results.json`.

## Files

- `crossover_model.py`: link energy and delay models for the copper wire and the optical link,
  plus the one-at-a-time sensitivity analysis.
- `simulate_from_scratch.py`: simulates the copper wire as an RC ladder, the silicon waveguide
  with the femwell finite-element mode solver (group index and ring shift per nm of width,
  thickness and per kelvin), and a single-ring Monte Carlo.
- `tuning_strategies.py`: 8-channel link on 10,000 simulated chips. Compares fixed channels,
  barrel shift and full reassignment, for a 5 µm ring and a ring matched to the channel span.
- `sky130_copper.py`: SKY130 wire resistance and capacitance from the kit's technology file and
  inverter characteristics from ngspice.
- `run_final.py`: reruns the heater comparison for four SKY130 copper cases (1.8 V or 0.8 V,
  isolated or packed wire).
- `validate_wire_model.py`: compares the analytical wire delay with a 200-section ngspice circuit.

## Limitations

The copper side uses SKY130 (130 nm) while the optical electronics come from newer processes;
the 1.8 V and 0.8 V cases bracket this. Heater driver energy and heat leaking between rings are
not modelled, packed-wire coupling is a parallel-plate estimate, and the study covers a single
point-to-point link rather than a full network.

## License and citation

Code under the MIT License (see `LICENSE`). To cite, use `CITATION.cff` or the paper.


