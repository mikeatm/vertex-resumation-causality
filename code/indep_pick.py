#!/usr/bin/env python3
"""Pick criterion on the lowest N_P Matsubara frequencies, computed from the independent implementation
(indep_check.py), for the exact, GW and spliced self-energies and for the Dyson Green's function of the splice.
Usage: python indep_pick.py --clusters dimer chain3 ring3 --out indep_pick_small.jsonl"""
import argparse, json, os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import indep_check as ic

NP = (2, 3, 4, 6, 8)
BETAS = (10.0, 20.0); US = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0); MUS = tuple(np.round(np.arange(-3, 3.01, 0.5), 2))


def pick(S, w, scaled=True):
    n = len(w); ns = S.shape[1]
    P = np.zeros((n * ns, n * ns), complex)
    for a in range(n):
        for b in range(n):
            P[a*ns:(a+1)*ns, b*ns:(b+1)*ns] = (S[b].conj().T - S[a]) / (1j * (w[a] + w[b]))
    P = 0.5 * (P + P.conj().T)
    raw = float(np.linalg.eigvalsh(P).min())
    dg = np.real(np.diag(P))
    if not scaled or (dg <= 0).any():
        return raw, raw
    s = 1 / np.sqrt(dg)
    return float(np.linalg.eigvalsh(s[:, None] * P * s[None, :]).min()), raw


def job(t):
    cl, b, U, mu = t
    ns, bonds = ic.CL[cl]
    r = ic.run(ns, bonds, b, U, mu, nw=max(NP), return_sigma=True)
    w = r["w"]; I = np.eye(ns)
    Gd = np.array([np.linalg.inv(1j * wi * I - r["h"] - S) for wi, S in zip(w, r["Sgd"])])
    out = dict(cluster=cl, beta=b, U=U, mu=mu)
    for lab, S in (("ex", r["Sex"]), ("gw", r["Sgw"]), ("gd", r["Sgd"]), ("Ggd", Gd)):
        res = [pick(S[:n], w[:n]) for n in NP]
        out[lab] = [x[0] for x in res]; out[lab + "_raw"] = [x[1] for x in res]
    out["lam_gd"] = [float(np.linalg.eigvalsh(-(S - S.conj().T) / 2j).min()) for S in r["Sgd"]]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", nargs="+", default=["dimer", "chain3", "ring3"])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="indep_pick.jsonl")
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l); done.add((r["cluster"], r["beta"], r["U"], r["mu"]))
    tasks = [(cl, b, U, float(mu)) for cl in a.clusters for b in BETAS for U in US for mu in MUS if (cl, b, U, float(mu)) not in done]
    with open(a.out, "a") as fh:
        if a.jobs > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(a.jobs) as pool:
                for r in pool.imap_unordered(job, tasks):
                    fh.write(json.dumps(r) + "\n"); fh.flush()
        else:
            for t in tasks:
                fh.write(json.dumps(job(t)) + "\n"); fh.flush()
    print("done", len(tasks))
