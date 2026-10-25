# Causality of spliced GW + local self-energies — code and data

Code and data for "Splicing an exact local self-energy into $GW$: tests on small Hubbard clusters" (M. O. Atambo).

## Requirements
- Python ≥ 3.8 with numpy and matplotlib
- `causality-check.py` (the exact-diagonalization + GW code) placed in this top folder


## Main scan (Table 1, Figs. 1–3)
```bash
python code/splice_scan.py --code causality-check.py --jobs 4
```
Writes `splice_scan.jsonl` (all 624 grid points at cutoff factors ×1 and ×2)
and prints the summary table. A small-cluster-only run finishes in a few minutes;
the 4-site ring adds a few tens of minutes on 2 cores. The run can be interrupted
and restarted, because finished points are skipped.

To reprint the summary from existing data:
```bash
python code/splice_scan.py --summary --out data/splice_scan.jsonl
```

## Figures
```bash
python code/make_figures.py --code causality-check.py --data splice_scan.jsonl --outdir figs
```
Use `--data data/splice_scan.jsonl` to plot the shipped data instead of a fresh run.

## Validation (App. D, Tables 2–3)
```bash
#  imaginary-time implementation vs main code
python code/indep_check.py

# full Pick test with the independent implementation 
python code/indep_pick.py --clusters dimer chain3 ring3 ring4 --jobs 2 --out indep_pick.jsonl

# polarization at U=0, screening, Dyson-G test, Pick test with the main code
python code/validation.py --code causality-check.py --data splice_scan.jsonl --part A B D E --out validation.json
```
Pick-test counts (Table 1, last two columns), from the shipped or regenerated data:
```bash
cd data && python ../code/pick_summary.py
```
(`pick_summary.py` reads `splice_scan.jsonl` and `indep_pick.jsonl` from the current folder.)

## Cutoff convergence (older uncorrected-bubble study)
```bash
python code/converge_splice.py --code causality-check.py --factors 1 2 4 8
```

## Contents
| Folder | Files |
|---|---|
| `code/` | `splice_scan.py` (main scan, tail-corrected GW), `make_figures.py`, `indep_check.py` and `indep_pick.py` (independent implementation), `validation.py`, `pick_summary.py`, `converge_splice.py` |
| `data/` | `splice_scan.jsonl`, `indep_pick.jsonl`, `validation.json` |

## Note
`splice_scan.py` evaluates GW with an analytic tail correction of the
polarization bubble (App. C of the paper). The `sigma_gw()` routine in
`causality-check.py` does not have it, so use `splice_scan.py` to reproduce
the numbers in the paper.
