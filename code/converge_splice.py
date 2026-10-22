#!/usr/bin/env python3
"""
Cutoff-convergence test for causality violations of the spliced self-energy
    Sigma_GD = Sigma_GW (off-diagonal) + Sigma_exact (diagonal)
on small Hubbard clusters, built on top of the corrected causality-check.py.

Physics is NOT re-implemented here: build(), sigma_gw(), splice(), sigma_exact()
are imported from your script. The only change is a faster, memory-safe G()
(same Lehmann sum, but negligible transitions pruned and frequencies chunked),
which is checked against your G() at start-up.

For every case it runs the Matsubara cutoffs (Nc, Mc) = factor * (16*beta, 8*beta)
for each factor in --factors, and records the worst causality violation
    v = max eigenvalue of  sign(w_n) (Sigma - Sigma^dag)/2i   (must be <= 0)
plus the Richardson estimate v_inf = 2 v(f_last) - v(f_prev), assuming O(1/N)
truncation error under cutoff doubling.

Verdicts
    CONVERGED      v_last > tol and |v_last - v_prev| <= 10% of v_last and v_inf > tol
    ARTIFACT       v_last <= tol, or v_inf <= tol (heading to zero)
    UNSETTLED      anything else -> rerun that case with larger --factors

Results are appended to a JSONL file one case at a time, so the run can be
interrupted and restarted; finished (case, factor) pairs are skipped.

Usage
  python converge_splice.py --code causality-check.py                      # the 37 hits, factors 1 2 4
  python converge_splice.py --code causality-check.py --factors 1 2 4 8    # push further
  python converge_splice.py --code causality-check.py --mode scan --clusters ring4 --jobs 4
          # base-cutoff scan of ring4 over the same grid, then convergence on its hits
  python converge_splice.py --summary --out splice_convergence_v2.jsonl      # just print the table
"""
import argparse, importlib.util, json, os, sys, time

# one BLAS thread per worker; must be set before numpy is imported
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np

CLUSTERS = {
    "dimer":  (2, [(0, 1)]),
    "chain3": (3, [(0, 1), (1, 2)]),
    "ring3":  (3, [(0, 1), (1, 2), (2, 0)]),
    "ring4":  (4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
    "chain4": (4, [(0, 1), (1, 2), (2, 3)]),
}
# the 37 splice-violating cases from the base-cutoff scan (cluster, beta, U, mu)
HITS = [
    ("dimer", 10, 0.5, -1.5), ("dimer", 10, 0.5, 1.5), ("dimer", 10, 3, -2.5), ("dimer", 10, 3, 2.5),
    ("dimer", 10, 4, -3.0), ("dimer", 10, 4, 3.0), ("dimer", 20, 1, -1.0), ("dimer", 20, 1, 1.0),
    ("dimer", 20, 2, -2.0), ("dimer", 20, 2, 2.0), ("dimer", 20, 3, -2.5), ("dimer", 20, 3, 2.5),
    ("dimer", 20, 4, -3.0), ("dimer", 20, 4, 3.0),
    ("chain3", 10, 1, -2.0), ("chain3", 10, 1, 2.0), ("chain3", 10, 2, -2.5), ("chain3", 10, 2, 2.5),
    ("chain3", 10, 3, -3.0), ("chain3", 10, 3, 3.0), ("chain3", 20, 1, -2.0), ("chain3", 20, 1, -1.5),
    ("chain3", 20, 1, 1.5), ("chain3", 20, 1, 2.0), ("chain3", 20, 2, -2.5), ("chain3", 20, 2, 2.5),
    ("chain3", 20, 3, -3.0), ("chain3", 20, 3, 3.0), ("chain3", 20, 4, -2.5), ("chain3", 20, 4, 2.5),
    ("ring3", 10, 0.5, -2.5), ("ring3", 10, 0.5, -2.0), ("ring3", 10, 1, -2.0), ("ring3", 10, 4, -3.0),
    ("ring3", 20, 0.5, -2.0), ("ring3", 20, 2, -3.0), ("ring3", 20, 4, -3.0),
]
GRID_BETAS = (10.0, 20.0)
GRID_US = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)
GRID_MUS = tuple(np.round(np.arange(-3, 3.01, 0.5), 2))
TOL = 1e-6
NW = 6           # violations sit at the lowest 1-2 Matsubara frequencies; 6 positive w_n is plenty

_MOD = None


class _NumpyShim:
    """numpy, except einsum defaults to optimize=True (identical results, contraction order only)."""
    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def einsum(*args, **kw):
        kw.setdefault("optimize", True)
        return np.einsum(*args, **kw)


# ------------------------------------------------------------------ load user code
def load(code_path):
    global _MOD
    if _MOD is not None:
        return _MOD
    spec = importlib.util.spec_from_file_location("cgw", os.path.abspath(code_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # the __main__ block does not run
    mod.np = _NumpyShim()                 # einsum(optimize=True): build() on 4 sites goes O(dim^4) -> O(dim^3)
    mod._G_original = mod.G
    mod.G = lambda K, d: fast_G(K, d)     # sigma_gw() looks G up at call time
    _MOD = mod
    return mod


def fast_G(K, d, thr=1e-13, chunk=256):
    """Same Lehmann sum as G() in causality-check.py, pruned + chunked."""
    if "_pairs" not in d:
        wt = d["p"][:, None] + d["p"][None, :]
        M = np.einsum("imn,jnm,mn->ijmn", d["ce"], d["cde"], wt)
        keep = np.abs(M).max(axis=(0, 1)) > thr
        m, n = np.nonzero(keep)
        d["_pairs"] = (M[:, :, m, n], d["E"][m] - d["E"][n])
    Mp, dE = d["_pairs"]
    w = np.asarray(K) * np.pi / d["beta"]
    out = np.empty((len(w), d["ns"], d["ns"]), complex)
    for s in range(0, len(w), chunk):
        den = 1.0 / (1j * w[s:s + chunk, None] + dE[None, :])
        out[s:s + chunk] = np.einsum("ijp,kp->kij", Mp, den)
    return out


def sigma_gw_at(mod, d, Klist, Nc, Mc):
    """Sigma_GW at arbitrary fermionic indices K (w_K = K pi/beta), same formula and
    conventions as sigma_gw() in causality-check.py (P sign +1, convention -1).
    Used for the high-frequency tail, which sigma_gw() does not reach."""
    ns, U, beta = d["ns"], d["U"], d["beta"]; I = np.eye(ns)
    Kn = 2 * np.arange(-Nc, Nc + 1) + 1; Km = 2 * np.arange(-Mc, Mc + 1); Kw = np.asarray(Klist)
    allK = np.unique(np.concatenate([Kn, (Kn[:, None] + Km).ravel(), (Kw[:, None] + Km).ravel()]))
    Gall = mod.G(allK, d); pos = {k: i for i, k in enumerate(allK)}
    g = lambda K: Gall[[pos[k] for k in K]]
    Gn = g(Kn)
    Wm = np.array([np.linalg.solve(I - U * (2 / beta) * np.einsum("nij,nji->ij", g(Kn + k), Gn), U * I) - U * I
                   for k in Km])
    return np.array([-1 / beta * np.einsum("kij,kij->ij", g(K + Km), Wm) + U * d["dbar"] * I for K in Kw])


def self_test(mod):
    d = mod.build(3, CLUSTERS["ring3"][1], 1.0, 3.0, 0.7, 10.0)
    K = np.array([-41, -3, -1, 1, 5, 99])
    err = np.abs(fast_G(K, dict(d)) - mod._G_original(K, d)).max()
    if err > 1e-10:
        sys.exit(f"fast_G disagrees with your G() (max err {err:.2e}); aborting")
    print(f"self-test: fast G matches your G() to {err:.1e}")
    wn, Sg, _ = mod.sigma_gw(d, Nw=3, Nc=160, Mc=80)
    err2 = np.abs(sigma_gw_at(mod, d, wn, 160, 80) - Sg).max()
    if err2 > 1e-10:
        sys.exit(f"tail GW routine disagrees with your sigma_gw() (max err {err2:.2e}); aborting")
    print(f"self-test: tail GW routine matches your sigma_gw() to {err2:.1e}")


def tail_moment(mod, d, Nc, Mc, K_tail=(801, 1601)):
    """C ~ w * Gamma(i w) at two large w_n; returns the larger-w estimate and the change between them.
    Gamma = -(Sigma - Sigma^dag)/2i at w_n > 0;  Sigma ~ Sigma_inf + C/(i w) => w Gamma -> C."""
    Kt = np.array(K_tail)
    Sg = sigma_gw_at(mod, d, Kt, Nc, Mc)
    Se = mod.sigma_exact(Kt, d, mod.G(Kt, d))
    Sd = mod.splice(Sg, Se)
    w = Kt * np.pi / d["beta"]
    C = lambda S: np.array([wi * (-(Si - Si.conj().T) / 2j) for wi, Si in zip(w, S)])
    Cg, Ce, Cd = C(Sg), C(Se), C(Sd)
    sym = lambda A: 0.5 * (A + A.conj().T)
    mineig = lambda A: float(np.linalg.eigvalsh(sym(A)).min())
    offd = lambda A: float(np.abs(A - np.diag(np.diag(A))).max())
    return dict(
        C_exact_diag=[float(x) for x in np.diag(Ce[-1]).real],
        C_exact_mineig=mineig(Ce[-1]), C_gw_mineig=mineig(Cg[-1]), C_gd_mineig=mineig(Cd[-1]),
        C_gw_offdiag=offd(Cg[-1]), C_gd_offdiag=offd(Cd[-1]),
        C_gd_change=float(np.abs(Cd[-1] - Cd[0]).max()),          # tail-extraction error estimate
        U2n1n=float(d["U"] ** 2 * d["nsig"] * (1 - d["nsig"])),
    )


# ------------------------------------------------------------------ one calculation
def damping_maxeig(wn, S):
    """worst eigenvalue of sign(w)(S - S^dag)/2i per frequency (causal => <= 0)."""
    return np.array([np.linalg.eigvalsh(np.sign(K) * (Si - Si.conj().T) / 2j).max() for K, Si in zip(wn, S)])


def run_one(task):
    code, cl, beta, U, mu, f = task
    mod = load(code)
    ns, bonds = CLUSTERS[cl]
    t0 = time.time()
    d = mod.build(ns, bonds, 1.0, U, mu, beta)
    wn, Sg, Se = mod.sigma_gw(d, Nw=NW, Nc=int(16 * beta * f), Mc=int(8 * beta * f))
    Sd = mod.splice(Sg, Se)
    eg, ee, ed = damping_maxeig(wn, Sg), damping_maxeig(wn, Se), damping_maxeig(wn, Sd)
    # local damping Gamma_ii = -sign(w) Im Sigma_ii ; sufficient condition Gamma_ii^exact >= Gamma_ii^GW
    gam = lambda S: np.array([-np.sign(K) * np.diag(Si).imag for K, Si in zip(wn, S)])
    suff = bool((gam(Se) >= gam(Sg) - 1e-12).all())
    i0 = list(wn).index(1)
    ratio = float(np.mean(gam(Sg)[i0] / np.maximum(gam(Se)[i0], 1e-14)))
    bad = [int(K) for K, v in zip(wn, ed) if v > TOL and K > 0]
    tail = tail_moment(mod, d, int(16 * beta * f), int(8 * beta * f))
    return dict(cluster=cl, beta=beta, U=U, mu=mu, factor=f, N=float(d["N"]),
                v_gd=float(max(ed.max(), 0.0)), v_gw=float(max(eg.max(), 0.0)), v_ex=float(max(ee.max(), 0.0)),
                bad_K=bad, suff=suff, gw_over_exact_damping_w0=ratio, **tail,
                seconds=round(time.time() - t0, 1))


# ------------------------------------------------------------------ bookkeeping
def key(r):
    return (r["cluster"], float(r["beta"]), float(r["U"]), float(r["mu"]), int(r["factor"]))


def read_done(path):
    done = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                r = json.loads(line)
                done[key(r)] = r
    return done


def run_tasks(tasks, code, out, jobs):
    done = read_done(out)
    todo = [t for t in tasks if (t[1], float(t[2]), float(t[3]), float(t[4]), int(t[5])) not in done]
    print(f"{len(tasks)} calculations requested, {len(tasks) - len(todo)} already in {out}, {len(todo)} to run")
    if not todo:
        return
    # cheapest first, so partial runs give the most coverage
    todo.sort(key=lambda t: (CLUSTERS[t[1]][0], t[5], t[2]))
    with open(out, "a") as fh:
        if jobs > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(jobs, initializer=load, initargs=(code,)) as pool:
                for i, r in enumerate(pool.imap_unordered(run_one, todo), 1):
                    fh.write(json.dumps(r) + "\n"); fh.flush()
                    print(f"  [{i}/{len(todo)}] {r['cluster']:6s} b={r['beta']:4.0f} U={r['U']:3} mu={r['mu']:+4.1f} x{r['factor']}"
                          f"  v_GD={r['v_gd']:.5f}  ({r['seconds']}s)", flush=True)
        else:
            for i, t in enumerate(todo, 1):
                r = run_one(t)
                fh.write(json.dumps(r) + "\n"); fh.flush()
                print(f"  [{i}/{len(todo)}] {r['cluster']:6s} b={r['beta']:4.0f} U={r['U']:3} mu={r['mu']:+4.1f} x{r['factor']}"
                      f"  v_GD={r['v_gd']:.5f}  ({r['seconds']}s)", flush=True)


def summarize(out, factors=None):
    done = read_done(out)
    cases = {}
    for (cl, b, U, mu, f), r in done.items():
        cases.setdefault((cl, b, U, mu), {})[f] = r
    rows = []
    for c, byf in sorted(cases.items()):
        fs = sorted(byf) if factors is None else [f for f in sorted(factors) if f in byf]
        if len(fs) < 2:
            continue
        v = [byf[f]["v_gd"] for f in fs]
        vl, vp = v[-1], v[-2]
        vinf = 2 * vl - vp
        if vl <= TOL or vinf <= TOL:
            verdict = "ARTIFACT"
        elif abs(vl - vp) <= 0.10 * vl:
            verdict = "CONVERGED"
        else:
            verdict = "UNSETTLED"
        last = byf[fs[-1]]
        rows.append((c, fs, v, vinf, verdict, last))
    if not rows:
        print("no case has >= 2 cutoff factors yet"); return
    # tail columns: minimum eigenvalue of the spliced 1/(i w) moment C_GD, and C_GD's largest
    # off-diagonal element; a PSD tail (min eig >= 0) means the moment test is passed.
    tl = lambda r, k: (f"{r[k]:8.4f}" if k in r else f"{'-':>8s}")
    print(f"\n{'cluster':7s} {'beta':>4s} {'U':>4s} {'mu':>5s} {'<N>':>5s}  " +
          "  ".join(f"{'v(x'+str(f)+')':>9s}" for f in rows[0][1]) +
          f"  {'v_inf':>8s}  {'GW':>3s} {'suff':>4s} {'G/ex':>5s}  {'C_GDmin':>8s} {'C_GDoff':>8s}  verdict")
    for (cl, b, U, mu), fs, v, vinf, verdict, last in rows:
        print(f"{cl:7s} {b:4.0f} {U:4.1f} {mu:+5.1f} {last['N']:5.2f}  " +
              "  ".join(f"{x:9.5f}" for x in v) +
              f"  {vinf:8.5f}  {'ok' if last['v_gw'] <= TOL else 'BAD':>3s} {'y' if last['suff'] else 'n':>4s}"
              f" {last['gw_over_exact_damping_w0']:5.2f}  {tl(last, 'C_gd_mineig')} {tl(last, 'C_gd_offdiag')}  {verdict}")
    from collections import Counter
    cnt = Counter(r[4] for r in rows)
    print("\n" + ", ".join(f"{k}: {n}" for k, n in sorted(cnt.items())) + f"   (of {len(rows)} cases)")
    n_suff_viol = sum(1 for r in rows if r[4] == "CONVERGED" and r[5]["suff"])
    print(f"converged violations where the sufficient condition holds (should be 0): {n_suff_viol}")
    n_gw = sum(1 for r in rows if r[5]["v_gw"] > TOL)
    print(f"cases with a GW causality violation at the largest cutoff (should be 0): {n_gw}")
    conv = [r for r in rows if r[4] == "CONVERGED" and "C_gd_mineig" in r[5]]
    if conv:
        psd = [r for r in conv if r[5]["C_gd_mineig"] >= -1e-3]
        dev = max(abs(r[5]["C_gd_mineig"] - min(r[5]["C_exact_diag"])) for r in conv)
        print(f"converged violations whose spliced tail C_GD is PSD (moment test passes, causality fails): "
              f"{len(psd)} of {len(conv)}")
        print(f"  largest off-diagonal |C_GD_ij|: {max(r[5]['C_gd_offdiag'] for r in conv):.2e}"
              f"   largest |min eig C_GD - min C_exact,ii|: {dev:.2e}"
              f"   largest tail-extraction change: {max(r[5]['C_gd_change'] for r in conv):.2e}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", default="causality-check.py", help="path to your corrected causality-check.py")
    ap.add_argument("--mode", choices=["hits", "scan"], default="hits")
    ap.add_argument("--clusters", nargs="+", default=["ring4"], help="clusters for --mode scan")
    ap.add_argument("--factors", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--out", default="splice_convergence_v2.jsonl")
    ap.add_argument("--summary", action="store_true", help="only print the table from --out")
    a = ap.parse_args()

    if a.summary:
        summarize(a.out, a.factors if "--factors" in sys.argv else None); return

    mod = load(a.code)
    self_test(mod)

    if a.mode == "hits":
        cases = [(cl, float(b), float(U), float(mu)) for cl, b, U, mu in HITS]
    else:
        # step 1: base-cutoff scan over the grid
        f0 = min(a.factors)
        grid = [(cl, b, U, float(mu)) for cl in a.clusters for b in GRID_BETAS for U in GRID_US for mu in GRID_MUS]
        print(f"scan: {len(grid)} grid points at cutoff factor x{f0}")
        run_tasks([(a.code, *g, f0) for g in grid], a.code, a.out, a.jobs)
        done = read_done(a.out)
        cases = []
        for g in grid:
            r = done.get((g[0], float(g[1]), float(g[2]), float(g[3]), f0))
            if r and 0.1 < r["N"] < 2 * CLUSTERS[g[0]][0] - 0.1 and r["v_gd"] > TOL:
                cases.append(g)
        n_gw = sum(1 for g in grid if (r := done.get((g[0], float(g[1]), float(g[2]), float(g[3]), f0))) and r["v_gw"] > TOL)
        print(f"scan done: {len(cases)} splice-violating points, {n_gw} GW-violating points")

    tasks = [(a.code, cl, b, U, mu, f) for cl, b, U, mu in cases for f in a.factors]
    run_tasks(tasks, a.code, a.out, a.jobs)
    summarize(a.out, a.factors)


if __name__ == "__main__":
    main()
