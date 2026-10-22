#!/usr/bin/env python3
"""
Causality of the spliced self-energy  Sigma_GD = Sigma_GW - diag(Sigma_GW) + diag(Sigma_exact)
on small Hubbard clusters (exact diagonalisation), with a tail-corrected GW polarisation.

Imports the ED machinery (build, sigma_exact, splice) from causality-check.py and the fast
Lehmann G from converge_splice.py.  The GW self-energy is re-evaluated here with ONE change
relative to sigma_gw() in causality-check.py: the truncated fermionic sum in the bubble

    P_ij(i nu) = (2/beta) sum_{|n|<=Nc} G_ij(i w_n + i nu) G_ji(i w_n)

is corrected by subtracting the leading tail delta_ij / ((i w_n + i nu)(i w_n)) inside the
window and adding back its exact infinite Matsubara sum (0 for nu != 0, -beta/4 per spin
for nu = 0).  Without this the truncation leaves a constant O(beta/Nc) offset in P at all
bosonic frequencies, W - U does not decay, and the high-frequency tail of Sigma_GW is wrong.
With tailfix=False the routine reproduces sigma_gw() of causality-check.py to machine
precision (checked at start-up).

For every (cluster, beta, U, mu) it records, at the positive Matsubara frequencies
w_1 ... w_NW and at two tail frequencies:
    lam_ex, lam_gw, lam_gd : minimum eigenvalue of Gamma(i w_n) = -(Sigma - Sigma^dag)/2i
                             (causality requires Gamma(i w_n) >= 0 for w_n > 0)
    gloc_ex, gloc_gw       : site-averaged local damping Gamma_ii
    C_*                    : 1/(i w) moments, C ~ w Gamma at large w
Negative frequencies follow from Sigma(-i w) = Sigma(i w)^dag (real Hamiltonian).

Usage
  python splice_scan.py --code causality-check.py --jobs 4                 # full grid, cutoff x1 and x2
  python splice_scan.py --summary                                          # print the tables used in the paper
"""
import argparse, json, os, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import converge_splice as cs           # load(), fast_G, CLUSTERS, grid definitions

NW = 50                 # positive Matsubara frequencies checked
K_TAIL = (801, 1601)    # fermionic indices used for the 1/(i w) moment
TOL = 1e-6


def sigma_gw_tc(mod, d, Klist, Nc, Mc, tailfix=True):
    ns, U, beta = d["ns"], d["U"], d["beta"]; I = np.eye(ns)
    Kn = 2 * np.arange(-Nc, Nc + 1) + 1; Km = 2 * np.arange(-Mc, Mc + 1); Kw = np.asarray(Klist)
    allK = np.unique(np.concatenate([Kn, (Kn[:, None] + Km).ravel(), (Kw[:, None] + Km).ravel()]))
    Gall = mod.G(allK, d); pos = {k: i for i, k in enumerate(allK)}
    g = lambda K: Gall[[pos[k] for k in K]]
    Gn = g(Kn); wn = Kn * np.pi / beta
    Wm = np.empty((len(Km), ns, ns), complex)
    for a, k in enumerate(Km):
        P = (2 / beta) * np.einsum("nij,nji->ij", g(Kn + k), Gn)
        if tailfix:
            nu = k * np.pi / beta
            P = P - (2 / beta) * np.sum(1.0 / ((1j * wn + 1j * nu) * (1j * wn))) * I
            if k == 0:
                P = P + 2 * (-beta / 4) * I
        Wm[a] = np.linalg.solve(I - U * P, U * I) - U * I
    return np.array([-1 / beta * np.einsum("kij,kij->ij", g(K + Km), Wm) + U * d["dbar"] * I for K in Kw])


def gamma(S):
    """damping matrix at positive Matsubara frequency"""
    return -(S - S.conj().T) / 2j


def self_test(mod):
    d = mod.build(3, cs.CLUSTERS["ring3"][1], 1.0, 3.0, 0.7, 10.0)
    wn, Sg, _ = mod.sigma_gw(d, Nw=3, Nc=160, Mc=80)
    err = np.abs(sigma_gw_tc(mod, d, wn, 160, 80, tailfix=False) - Sg).max()
    if err > 1e-10:
        sys.exit(f"GW routine (tailfix=False) disagrees with sigma_gw() of your script: {err:.2e}")
    print(f"self-test: tailfix=False reproduces your sigma_gw() to {err:.1e}")


def run_point(task):
    code, cl, beta, U, mu, f = task
    mod = cs.load(code)
    ns, bonds = cs.CLUSTERS[cl]
    t0 = time.time()
    d = mod.build(ns, bonds, 1.0, U, mu, beta)
    Kpos = 2 * np.arange(NW) + 1
    K = np.concatenate([Kpos, K_TAIL])
    Sg = sigma_gw_tc(mod, d, K, int(16 * beta * f), int(8 * beta * f))
    Se = mod.sigma_exact(K, d, mod.G(K, d))
    Sd = mod.splice(Sg, Se)
    Gg, Ge, Gd = [np.array([gamma(s) for s in S]) for S in (Sg, Se, Sd)]
    lam = lambda Gm: [float(np.linalg.eigvalsh(0.5 * (x + x.conj().T)).min()) for x in Gm]
    lam_ex, lam_gw, lam_gd = lam(Ge), lam(Gg), lam(Gd)
    gloc_ex = [float(np.mean(np.diag(x).real)) for x in Ge]
    gloc_gw = [float(np.mean(np.diag(x).real)) for x in Gg]
    # sufficient condition, site by site, at every checked frequency
    suff = bool(all((np.diag(Ge[i]).real >= np.diag(Gg[i]).real - 1e-12).all() for i in range(NW)))
    w = K * np.pi / beta
    Cmat = lambda Gm: w[-1] * Gm[-1]
    sym = lambda A: 0.5 * (A + A.conj().T)
    Cg, Ce, Cd = Cmat(Gg), Cmat(Ge), Cmat(Gd)
    offd = lambda A: float(np.abs(A - np.diag(np.diag(A))).max())
    n = NW
    return dict(cluster=cl, beta=beta, U=U, mu=mu, factor=f, N=float(d["N"]),
                nsig=float(d["nsig"]),
                lam_ex=lam_ex[:n], lam_gw=lam_gw[:n], lam_gd=lam_gd[:n],
                gloc_ex=gloc_ex[:n], gloc_gw=gloc_gw[:n],
                v_ex=max(0.0, -min(lam_ex[:n])), v_gw=max(0.0, -min(lam_gw[:n])), v_gd=max(0.0, -min(lam_gd[:n])),
                bad_n=[i + 1 for i in range(n) if lam_gd[i] < -TOL],        # n such that w = (2n-1) pi / beta
                suff=suff,
                C_ex_diag=[float(x) for x in np.diag(Ce).real], C_gw_diag=[float(x) for x in np.diag(Cg).real],
                C_ex_min=float(np.linalg.eigvalsh(sym(Ce)).min()), C_gw_min=float(np.linalg.eigvalsh(sym(Cg)).min()),
                C_gd_min=float(np.linalg.eigvalsh(sym(Cd)).min()),
                C_gw_offd=offd(Cg), C_gd_offd=offd(Cd),
                C_gd_tailchange=float(np.abs(w[-2] * Gd[-2] - Cd).max()),
                seconds=round(time.time() - t0, 1))


def key(r):
    return (r["cluster"], float(r["beta"]), float(r["U"]), float(r["mu"]), int(r["factor"]))


def read(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line); out[key(r)] = r
    return out


def grid(clusters):
    return [(cl, b, U, float(mu)) for cl in clusters for b in cs.GRID_BETAS for U in cs.GRID_US for mu in cs.GRID_MUS]


def run(code, clusters, factors, out, jobs):
    done = read(out)
    todo = [(code, *g, f) for g in grid(clusters) for f in factors if (g[0], g[1], g[2], g[3], f) not in done]
    print(f"{len(todo)} calculations to run ({len(done)} already in {out})")
    todo.sort(key=lambda t: (cs.CLUSTERS[t[1]][0], t[5]))
    with open(out, "a") as fh:
        if jobs > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(jobs, initializer=cs.load, initargs=(code,)) as pool:
                for i, r in enumerate(pool.imap_unordered(run_point, todo), 1):
                    fh.write(json.dumps(r) + "\n"); fh.flush()
                    if i % 50 == 0 or i == len(todo):
                        print(f"  {i}/{len(todo)}", flush=True)
        else:
            for i, t in enumerate(todo, 1):
                fh.write(json.dumps(run_point(t)) + "\n"); fh.flush()
                if i % 50 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}", flush=True)


def summary(out, f_lo=1, f_hi=2):
    R = read(out)
    pts = sorted({k[:4] for k in R})
    ns = {k: v[0] for k, v in cs.CLUSTERS.items()}
    inwin = [p for p in pts if (p + (f_hi,)) in R and 0.1 < R[p + (f_hi,)]["N"] < 2 * ns[p[0]] - 0.1]
    print(f"grid points: {len(pts)}, inside 0.1 < <N> < 2Ns-0.1: {len(inwin)}")
    tot = dict(viol=0)
    print(f"{'cluster':7s} {'pts':>4s} {'viol':>4s} {'maxv':>7s} {'drift':>8s} {'GWviol':>6s} {'suff':>5s} {'suff&viol':>9s} {'mu=0 viol':>9s} {'ratio range':>12s} {'bad n':>6s} {'U_min':>5s}")
    for cl in cs.CLUSTERS:
        P = [p for p in inwin if p[0] == cl]
        if not P:
            continue
        hi = [R[p + (f_hi,)] for p in P]; lo = [R[p + (f_lo,)] for p in P if p + (f_lo,) in R]
        viol = [r for r in hi if r["v_gd"] > TOL]
        drift = max([abs(R[key(r)[:4] + (f_lo,)]["v_gd"] - r["v_gd"]) for r in hi if key(r)[:4] + (f_lo,) in R], default=0)
        ratio = [r["gloc_gw"][0] / r["gloc_ex"][0] for r in viol]
        badn = max([max(r["bad_n"]) for r in viol], default=0)
        print(f"{cl:7s} {len(P):4d} {len(viol):4d} {max([r['v_gd'] for r in viol], default=0):7.4f} {drift:8.1e} "
              f"{sum(r['v_gw'] > TOL for r in hi):6d} {sum(r['suff'] for r in hi):5d} {sum(r['suff'] and r['v_gd'] > TOL for r in hi):9d} "
              f"{sum(1 for r in viol if r['mu'] == 0):9d} {(f'{min(ratio):.2f}-{max(ratio):.2f}' if ratio else '-'):>12s} {badn:6d} "
              f"{min([r['U'] for r in viol], default=0):5.1f}")
        tot["viol"] += len(viol)
    allhi = [R[p + (f_hi,)] for p in inwin]
    V = [r for r in allhi if r["v_gd"] > TOL]
    print(f"total splice-violating points: {tot['viol']}")
    print(f"GW violations anywhere in window (any of {NW} frequencies): {sum(r['v_gw'] > TOL for r in allhi)};"
          f" exact: {sum(r['v_ex'] > TOL for r in allhi)}")
    print(f"min eigenvalue of C_GW over window: {min(r['C_gw_min'] for r in allhi):.4f};"
          f" max |offdiag C_GW|: {max(r['C_gw_offd'] for r in allhi):.2e}")
    print(f"violating points with PSD spliced tail: {sum(r['C_gd_min'] > 0 for r in V)} of {len(V)};"
          f" max |C_GD - diag C_ex| offdiag: {max(r['C_gd_offd'] for r in V):.2e}")
    print(f"max violation-cutoff drift x{f_lo}->x{f_hi}: "
          f"{max(abs(R[key(r)[:4] + (f_lo,)]['v_gd'] - r['v_gd']) for r in allhi if key(r)[:4] + (f_lo,) in R):.2e}")
    flips = [key(r)[:4] for r in allhi if key(r)[:4] + (f_lo,) in R and ((R[key(r)[:4] + (f_lo,)]['v_gd'] > TOL) != (r['v_gd'] > TOL))]
    print(f"points whose violating/causal status changes between x{f_lo} and x{f_hi}: {len(flips)} {flips[:6]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", default="causality-check.py")
    ap.add_argument("--clusters", nargs="+", default=["dimer", "chain3", "ring3", "ring4"])
    ap.add_argument("--factors", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="splice_scan.jsonl")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if not a.summary:
        mod = cs.load(a.code)
        cs.self_test(mod); self_test(mod)
        run(a.code, a.clusters, a.factors, a.out, a.jobs)
    summary(a.out, min(a.factors), max(a.factors))
