#!/usr/bin/env python3
"""
Validation of the corrected implementation (numbers quoted in the validation table).
  A  non-interacting polarisation vs the exact density response (Lehmann, imaginary-time quadrature)
  B  screening: P(i nu) <= 0, eigenvalues of 1 - U P >= 1, and what the flipped sign would give
  D  Dyson Green's function of the spliced self-energy: pointwise test on G itself
  E  full Pick (Nevanlinna) criterion on the lowest N_P Matsubara frequencies
Usage: python validation.py --code causality-check.py --data splice_scan.jsonl [--part A B D E] [--clusters ...]
"""
import argparse, json, os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import converge_splice as cs
import splice_scan as ss
import indep_check as ic

TOL = 1e-6


def part_A(mod):
    """P from the main code (with/without tail fix) vs exact chi from an independent ED, at U = 0."""
    rows = []
    for cl, mu in [("ring3", 0.0), ("chain3", 0.7), ("dimer", 0.3)]:
        ns, bonds = ic.CL[cl]; beta = 10.0
        H, c, h = ic.hamiltonian(ns, bonds, 1.0, 0.0, mu)
        E, V = np.linalg.eigh(H); E -= E.min(); Z = np.exp(-beta * E).sum()
        n_site = [V.T @ (c[2 * i].T @ c[2 * i] + c[2 * i + 1].T @ c[2 * i + 1]) @ V for i in range(ns)]
        tau, wt = ic.gl_nodes(beta, 400)
        nav = np.array([np.exp(-beta * E) @ np.diag(n) / Z for n in n_site])
        corr = np.empty((len(tau), ns, ns))
        for k, t in enumerate(tau):
            A = np.exp(-(beta - t) * E)[:, None] * np.exp(-t * E)[None, :]
            corr[k] = np.einsum("imn,jnm,mn->ij", np.array(n_site), np.array(n_site), A) / Z - np.outer(nav, nav)
        m = np.arange(6); nu = 2 * np.pi * m / beta
        chi = -np.einsum("k,km,kij->mij", wt, np.cos(np.outer(tau, nu)), corr)
        d = mod.build(ns, cs.CLUSTERS[cl][1], 1.0, 0.0, mu, beta)
        Nc = 160
        Kn = 2 * np.arange(-Nc, Nc + 1) + 1; wn = Kn * np.pi / beta
        Gn = mod.G(Kn, d)
        out = {}
        for tf in (False, True):
            P = []
            for k in 2 * m:
                p = (2 / beta) * np.einsum("nij,nji->ij", mod.G(Kn + k, d), Gn)
                if tf:
                    p = p - (2 / beta) * np.sum(1 / ((1j * wn + 1j * k * np.pi / beta) * (1j * wn))) * np.eye(ns)
                    if k == 0:
                        p = p + 2 * (-beta / 4) * np.eye(ns)
                P.append(p.real)
            out[tf] = np.abs(np.array(P) - chi).max()
        rows.append(dict(cluster=cl, mu=mu, beta=beta, chi0_norm=float(np.abs(chi[0]).max()),
                         err_raw=float(out[False]), err_tailfix=float(out[True]),
                         err_flipped=float(np.abs(-np.array(chi) - chi).max()),
                         chi_max_eig=float(max(np.linalg.eigvalsh(x).max() for x in chi))))
    return rows


def bubble(mod, d, Nc, Mlist):
    ns, beta = d["ns"], d["beta"]; I = np.eye(ns)
    Kn = 2 * np.arange(-Nc, Nc + 1) + 1; wn = Kn * np.pi / beta
    Gn = mod.G(Kn, d); out = []
    for k in Mlist:
        p = (2 / beta) * np.einsum("nij,nji->ij", mod.G(Kn + k, d), Gn)
        p = p - (2 / beta) * np.sum(1 / ((1j * wn + 1j * k * np.pi / beta) * (1j * wn))) * I
        if k == 0:
            p = p + 2 * (-beta / 4) * I
        out.append(p)
    return np.array(out)


def part_B(mod, clusters):
    """over the whole grid: max eig P(i nu_m), min eig (1 - U P), and min eig (1 + U P(0)) for the flipped sign."""
    res = []
    for cl in clusters:
        for b in cs.GRID_BETAS:
            for U in cs.GRID_US:
                for mu in cs.GRID_MUS:
                    ns, bonds = cs.CLUSTERS[cl]
                    d = mod.build(ns, bonds, 1.0, U, float(mu), b)
                    if not (0.1 < d["N"] < 2 * ns - 0.1):
                        continue
                    P = bubble(mod, d, int(16 * b), [0, 2, 4, 8, 16]).real
                    I = np.eye(ns)
                    res.append(dict(cluster=cl, beta=b, U=U, mu=float(mu),
                                    maxP=float(max(np.linalg.eigvalsh(p).max() for p in P)),
                                    min_scr=float(min(np.linalg.eigvalsh(I - U * p).min() for p in P)),
                                    W0_over_U=float(np.linalg.eigvalsh(np.linalg.solve(I - U * P[0], I)).max()),
                                    flipped=float(np.linalg.eigvalsh(I + U * P[0]).min())))
    return res


def sigma_all(mod, cl, b, U, mu, K, f):
    ns, bonds = cs.CLUSTERS[cl]
    d = mod.build(ns, bonds, 1.0, U, mu, b)
    Sg = ss.sigma_gw_tc(mod, d, K, int(16 * b * f), int(8 * b * f))
    Se = mod.sigma_exact(np.asarray(K), d, mod.G(np.asarray(K), d))
    return d, Sg, Se, mod.splice(Sg, Se)


def part_D(mod, data):
    """Dyson G of the spliced self-energy at violating points: is G itself pointwise non-causal?"""
    R = [json.loads(l) for l in open(data)]
    V = [r for r in R if r["factor"] == 2 and r["v_gd"] > TOL and 0.1 < r["N"] < 2 * cs.CLUSTERS[r["cluster"]][0] - 0.1]
    K = 2 * np.arange(ss.NW) + 1
    out = []
    for r in V:
        d, Sg, Se, Sd = sigma_all(mod, r["cluster"], r["beta"], r["U"], r["mu"], K, 1)
        w = K * np.pi / r["beta"]; I = np.eye(d["ns"])
        Gd = np.array([np.linalg.inv(1j * wi * I - d["h"] - S) for wi, S in zip(w, Sd)])
        gG = [float(np.linalg.eigvalsh(-(g - g.conj().T) / 2j).min()) for g in Gd]
        out.append(dict(cluster=r["cluster"], beta=r["beta"], U=r["U"], mu=r["mu"], v=r["v_gd"],
                        Gmin=min(gG), Gbad=[i + 1 for i, x in enumerate(gG) if x < -TOL]))
    return out


def pick_min(S, w):
    """smallest eigenvalue of the block Pick matrix of F = -Sigma, scaled by its diagonal blocks."""
    n = len(w); ns = S.shape[1]
    P = np.zeros((n * ns, n * ns), complex)
    for a in range(n):
        for b in range(n):
            P[a*ns:(a+1)*ns, b*ns:(b+1)*ns] = (S[b].conj().T - S[a]) / (1j * (w[a] + w[b]))
    P = 0.5 * (P + P.conj().T)
    dg = np.real(np.diag(P))
    if (dg <= 0).any():
        return float(np.linalg.eigvalsh(P).min())       # diagonal already non-positive
    s = 1 / np.sqrt(dg)
    return float(np.linalg.eigvalsh(s[:, None] * P * s[None, :]).min())


def part_E(mod, clusters, NP=(2, 3, 4, 6, 8)):
    out = []
    K = 2 * np.arange(max(NP)) + 1
    for cl in clusters:
        for b in cs.GRID_BETAS:
            for U in cs.GRID_US:
                for mu in cs.GRID_MUS:
                    ns = cs.CLUSTERS[cl][0]
                    rec = dict(cluster=cl, beta=b, U=U, mu=float(mu))
                    for f in (1, 2):
                        d, Sg, Se, Sd = sigma_all(mod, cl, b, U, float(mu), K, f)
                        rec["N"] = float(d["N"])
                        w = K * np.pi / b
                        for lab, S in (("gd", Sd), ("gw", Sg), ("ex", Se)):
                            rec[f"{lab}_f{f}"] = [pick_min(S[:n], w[:n]) for n in NP]
                    if 0.1 < rec["N"] < 2 * ns - 0.1:
                        out.append(rec)
    return out, list(NP)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="causality-check.py")
    ap.add_argument("--data", default="splice_scan.jsonl")
    ap.add_argument("--part", nargs="+", default=["A", "B", "D", "E"])
    ap.add_argument("--clusters", nargs="+", default=["dimer", "chain3", "ring3", "ring4"])
    ap.add_argument("--out", default="validation.json")
    a = ap.parse_args()
    mod = cs.load(a.code)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    if "A" in a.part:
        res["A"] = part_A(mod); print("A", json.dumps(res["A"], indent=0))
    if "B" in a.part:
        res.setdefault("B", [])
        res["B"] = [r for r in res["B"] if r["cluster"] not in a.clusters] + part_B(mod, a.clusters)
        B = res["B"]
        print(f"B: points {len(B)}  max eig P {max(r['maxP'] for r in B):.2e}  min eig(1-UP) {min(r['min_scr'] for r in B):.6f}"
              f"  max W(0)/U {max(r['W0_over_U'] for r in B):.4f}  flipped sign: min eig(1+UP(0)) {min(r['flipped'] for r in B):.3f},"
              f" non-positive at {sum(r['flipped'] <= 0 for r in B)} points")
    if "D" in a.part:
        res["D"] = part_D(mod, a.data); D = res["D"]
        print(f"D: violating points {len(D)}; Dyson G pointwise non-causal at {sum(1 for r in D if r['Gbad'])}")
    if "E" in a.part:
        res.setdefault("E", [])
        E, NP = part_E(mod, a.clusters)
        res["E"] = [r for r in res["E"] if r["cluster"] not in a.clusters] + E; res["NP"] = NP
    json.dump(res, open(a.out, "w"))
