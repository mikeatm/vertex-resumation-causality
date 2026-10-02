"""
Reproducibility script: GW and one-shot GW+DMFT causality on small Hubbard clusters (exact diagonalisation).

Model   H = -t sum_{<ij>,s} (c+_is c_js + h.c.) + U sum_i (n_i,up - 1/2)(n_i,dn - 1/2) - mu N
Checks  1. polarisation sign fixed against the *exact* U=0 density response (no theorem needed)
        2. exact / GW / GW+DMFT Matsubara causality, basis-independent:
              sign(w_n) * (Sigma - Sigma^dagger)/(2i)  must be negative semidefinite
        3. exact high-frequency tail  w*(-Im Sigma_ii) -> U^2 n_s (1-n_s)   (standard Hubbard result)
        4. small scan showing where the one-shot splice (GW off-diagonal + exact diagonal) can fail

Conventions (the ones that matter):
        P_ij(i nu) = +(2/beta) sum_n G_ij(i w+i nu) G_ji(i w)     (screening: P(0) <= 0, I-UP >= 1)
        W          = (I - U P)^-1 U
        Sigma_GW   = -(1/beta) sum_nu G(i w+i nu) o [W - U I]  + U*dbar*I     (o = element-wise product)
Run:    python cluster_gw_causality_check.py          (about 1-2 min on a laptop)
"""
import numpy as np

# ---------------------------------------------------------------- ED machinery
def popcount(x): return bin(x).count("1")

def fermion_ops(nm):
    dim = 1 << nm
    cd = np.zeros((nm, dim, dim))
    for s in range(dim):
        for p in range(nm):
            if not (s >> p) & 1:
                cd[p, s | (1 << p), s] = -1.0 if popcount(s & ((1 << p) - 1)) % 2 else 1.0
    return cd.swapaxes(1, 2).copy(), cd          # c, c^dagger  (modes 0..ns-1 = up, ns..2ns-1 = down)

def build(ns, bonds, t, U, mu, beta):
    nm = 2 * ns; c, cd = fermion_ops(nm); dim = 1 << nm; I = np.eye(dim)
    H = np.zeros((dim, dim)); h = np.zeros((ns, ns))
    for a, b in bonds:
        for s in (0, ns):                          # BOTH spin species hop
            H += -t * (cd[a + s] @ c[b + s] + cd[b + s] @ c[a + s])
        h[a, b] = h[b, a] = -t
    n = [cd[p] @ c[p] for p in range(nm)]
    for i in range(ns):
        H += U * (n[i] - 0.5 * I) @ (n[i + ns] - 0.5 * I)
    H -= mu * sum(n)
    E, V = np.linalg.eigh(H)
    p = np.exp(-beta * (E - E.min())); p /= p.sum()
    tr = lambda O: np.einsum('am,oab,bn->omn', V, O, V)          # to eigenbasis (H real)
    return dict(ns=ns, E=E, p=p, beta=beta, U=U, h=h - mu * np.eye(ns),
                ce=tr(c[:ns]), cde=tr(cd[:ns]),
                nme=tr(np.array([n[i] + n[i + ns] for i in range(ns)])),
                dbar=np.sum(p * np.diag(V.T @ n[ns] @ V)) - 0.5,
                nsig=np.mean([np.sum(p * np.diag(V.T @ n[i] @ V)) for i in range(ns)]),
                N=np.sum(p * np.diag(V.T @ sum(n) @ V)))

def G(K, d):
    """Exact cluster G_ij(i w_K), w_K = K*pi/beta, Lehmann form. K: integer array."""
    w = K * np.pi / d['beta']; E = d['E']
    wt = d['p'][:, None] + d['p'][None, :]
    M = np.einsum('imn,jnm,mn->ijmn', d['ce'], d['cde'], wt)
    return np.einsum('ijmn,kmn->kij', M, 1.0 / (1j * w[:, None, None] + E[None, :, None] - E[None, None, :]))

def sigma_exact(K, d, Gk):
    w = K * np.pi / d['beta']
    return np.array([1j * wi * np.eye(d['ns']) - d['h'] - np.linalg.inv(g) for wi, g in zip(w, Gk)])

#  GW (one shot on the exact G)
def sigma_gw(d, Nw=16, Nc=160, Mc=80, psign=+1, csign=-1):
    ns, U, beta = d['ns'], d['U'], d['beta']; I = np.eye(ns)
    Kn = 2 * np.arange(-Nc, Nc + 1) + 1; Km = 2 * np.arange(-Mc, Mc + 1); wn = 2 * np.arange(-Nw, Nw + 1) + 1
    allK = np.unique(np.concatenate([Kn, (Kn[:, None] + Km).ravel(), (wn[:, None] + Km).ravel()]))
    Gall = G(allK, d); pos = {k: i for i, k in enumerate(allK)}
    g = lambda K: Gall[[pos[k] for k in K]]
    Gn = g(Kn); W = {}
    for k in Km:
        P = psign * (2 / beta) * np.einsum('nij,nji->ij', g(Kn + k), Gn)
        W[k] = np.linalg.solve(I - U * P, U * I)
    S = np.array([csign / beta * sum(g([K + k])[0] * (W[k] - U * I) for k in Km) + U * d['dbar'] * I for K in wn])
    return wn, S, sigma_exact(wn, d, g(wn))

# diagnostics
def violations(wn, S, tol=1e-9):
    """count / size of causality violations; basis independent."""
    cnt, worst = 0, -np.inf
    for K, Si in zip(wn, S):
        ev = np.linalg.eigvalsh(np.sign(K) * (Si - Si.conj().T) / 2j)
        cnt += int((ev > tol).sum()); worst = max(worst, ev.max())
    return cnt, worst

def splice(Sg, Se):
    Sd = Sg.copy()
    for i in range(Sg.shape[1]): Sd[:, i, i] = Se[:, i, i]
    return Sd

#  polarisation sign vs exact density response
def check_polarisation_sign(beta=10.0, Nc=4000, Nq=2000):
    d = build(3, [(0, 1), (1, 2), (2, 0)], 1.0, 0.0, 0.0, beta)     # U = 0 ring
    E, p, A = d['E'], d['p'], d['nme']
    tau = (np.arange(Nq) + .5) * beta / Nq
    chi = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            integ = sum(np.sum(p[:, None] * np.exp(t * (E[:, None] - E[None, :])) * A[i] * A[j].T) for t in tau) * beta / Nq
            ni, nj = np.sum(p * np.diag(A[i])), np.sum(p * np.diag(A[j]))
            chi[i, j] = -(integ - beta * ni * nj)                    # chi(i nu=0) = -int <dn_i(tau) dn_j>
    Kn = 2 * np.arange(-Nc, Nc + 1) + 1; Gn = G(Kn, d)
    P = (2 / beta) * np.einsum('nij,nji->ij', Gn, Gn)
    print("exact U=0 static density response chi(0):\n", np.round(chi, 3))
    print("P(0) with  +(2/beta) sum G_ij G_ji       :\n", np.round(P.real, 3))
    print("max |chi - P| =", np.abs(chi - P.real).max(), "  '+' sign is the physical (screening) one\n")

#  the ring at U = 4 : all four sign combinations
def check_sign_table(U=4.0, beta=10.0):
    d = build(3, [(0, 1), (1, 2), (2, 0)], 1.0, U, 0.0, beta)
    print(f"3-site ring, t=1, U={U}, beta={beta}, mu=0  (<N> = {d['N']:.3f}; N=3 would be half filling)")
    for ps, cs in [(+1, -1), (-1, -1), (+1, +1), (-1, +1)]:
        wn, Sg, Se = sigma_gw(d, Nw=32, Nc=128, Mc=64, psign=ps, csign=cs)
        tag = "  physical" if (ps, cs) == (+1, -1) else ("   the sign error " if (ps, cs) == (-1, -1) else "")
        print(f"  P{ps:+d} conv{cs:+d}:  exact {violations(wn, Se)[0]:3d} | GW {violations(wn, Sg)[0]:3d} | GW+DMFT {violations(wn, splice(Sg, Se))[0]:3d}{tag}")
    print()

# tail
def check_tail(U=4.0, beta=10.0):
    d = build(3, [(0, 1), (1, 2), (2, 0)], 1.0, U, 0.0, beta)
    K = np.array([2 * 4000 + 1]); S = sigma_exact(K, d, G(K, d)); w = K[0] * np.pi / beta
    print("exact tail  w*(-Im Sigma_ii) =", np.round(w * -np.diag(S[0]).imag, 4),
          "  U^2 n(1-n) =", round(U**2 * d['nsig'] * (1 - d['nsig']), 4), "\n")

# scan
def scan(betas=(10.0, 20.0), Us=(1.0, 2.0, 3.0, 4.0, 6.0), mus=(0.0, 0.7, 1.5, 2.5)):
    clusters = {'dimer': (2, [(0, 1)]), 'chain3': (3, [(0, 1), (1, 2)]), 'ring3': (3, [(0, 1), (1, 2), (2, 0)])}
    print("scan, (dimer+ring+U)")
    hits = 0
    for name, (ns, b) in clusters.items():
        for beta in betas:
            for U in Us:
                for mu in mus:
                    d = build(ns, b, 1.0, U, mu, beta)
                    wn, Sg, Se = sigma_gw(d, Nw=16, Nc=int(12 * beta), Mc=int(6 * beta))
                    ve, vg, vd = violations(wn, Se), violations(wn, Sg), violations(wn, splice(Sg, Se))
                    if (ve[0] or vg[0] or vd[0]) and 0.05 < d['N'] < 2 * ns - 0.05:
                        hits += 1
                        print(f"  {name:6s} beta={beta:4.0f} U={U:3.1f} mu={mu:4.1f} <N>={d['N']:.2f} | exact {ve[0]} GW {vg[0]} GW+DMFT {vd[0]} (max {vd[1]:.4f})")
    print("  hits:", hits, )

if __name__ == "__main__":
    check_polarisation_sign()
    check_sign_table()
    check_tail()
    scan()
