#!/usr/bin/env python3
"""
Independent re-implementation of the exact, GW and spliced damping matrices.
Shares NO code with causality-check.py / splice_scan.py.

Differences from the main implementation
  * ED: own Jordan-Wigner construction with interleaved mode order (i,s) -> 2i+s
  * Green's function in imaginary time, G_ij(tau) = -Tr[e^{-(beta-tau)H} c_i e^{-tau H} c_j^+]/Z
  * polarisation in imaginary time, P_ij(tau) = -2 G_ij(tau) G_ji(beta-tau), and P(i nu) by quadrature:
    no fermionic Matsubara sum, hence no truncation of the bubble at all
  * screened interaction W - U on bosonic frequencies, back-transformed to tau with an analytic
    1/nu^2 tail;  Sigma_c(tau) = -G(tau) o [W-U](tau) (element-wise), Fourier transformed by quadrature
  * exact self-energy from G(i w) obtained by quadrature of G(tau)
Usage: python indep_check.py   (prints lambda_min of Gamma at the lowest three Matsubara frequencies)
"""
import itertools, json, sys
import numpy as np


def fock_ops(ns):
    nm = 2 * ns; dim = 1 << nm
    c = np.zeros((nm, dim, dim))
    for s in range(dim):
        for p in range(nm):
            if (s >> p) & 1:
                sign = (-1) ** bin(s & ((1 << p) - 1)).count("1")
                c[p, s ^ (1 << p), s] = sign
    return c                                    # annihilators; mode p = 2*i + spin


def hamiltonian(ns, bonds, t, U, mu):
    c = fock_ops(ns); cd = c.transpose(0, 2, 1)
    dim = c.shape[1]; I = np.eye(dim)
    H = np.zeros((dim, dim))
    for a, b in bonds:
        for s in (0, 1):
            p, q = 2 * a + s, 2 * b + s
            H -= t * (cd[p] @ c[q] + cd[q] @ c[p])
    n = cd @ c
    for i in range(ns):
        H += U * (n[2 * i] - 0.5 * I) @ (n[2 * i + 1] - 0.5 * I)
    H -= mu * n.sum(0)
    h = np.zeros((ns, ns))
    for a, b in bonds:
        h[a, b] = h[b, a] = -t
    return H, c, h - mu * np.eye(ns)


def gl_nodes(beta, panels, order=16):
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(0, beta, panels + 1)
    T, Wt = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        T.append(0.5 * (b - a) * x + 0.5 * (b + a)); Wt.append(0.5 * (b - a) * w)
    return np.concatenate(T), np.concatenate(Wt)       # symmetric under tau -> beta - tau


def run(ns, bonds, beta, U, mu, nw=3, panels=None, M=300, return_sigma=False):
    H, c, h = hamiltonian(ns, bonds, 1.0, U, mu)
    E, V = np.linalg.eigh(H); E = E - E.min()
    Z = np.exp(-beta * E).sum()
    up = [2 * i for i in range(ns)]                        # spin-up annihilators
    C = np.array([V.T @ c[p] @ V for p in up])              # <m|c_i|n>
    panels = panels or int(20 * beta)
    tau, wt = gl_nodes(beta, panels)
    # G_ij(tau) = -(1/Z) sum_mn e^{-(beta-tau)E_m} e^{-tau E_n} C_i[m,n] C_j[m,n]   (C^dagger_j[n,m] = C_j[m,n])
    G = np.empty((len(tau), ns, ns))
    for k, tk in enumerate(tau):
        A = np.exp(-(beta - tk) * E)[:, None] * np.exp(-tk * E)[None, :]
        G[k] = -np.einsum("imn,jmn,mn->ij", C, C, A) / Z
    Grev = G[::-1]                                          # G(beta - tau) on the same nodes
    P_tau = -2 * G * Grev.transpose(0, 2, 1)               # P_ij(tau) = -2 G_ij(tau) G_ji(beta-tau)
    nu = 2 * np.pi * np.arange(M + 1) / beta
    P_nu = np.einsum("k,km,kij->mij", wt, np.cos(np.outer(tau, nu)), P_tau)   # real, even in nu
    Iden = np.eye(ns)
    X_nu = np.array([np.linalg.solve(Iden - U * P, U * Iden) - U * Iden for P in P_nu])
    # X(tau) = (1/beta)[X0 + 2 sum_{m=1..M} X_m cos(nu_m tau) + 2 c sum_{m>M} cos(nu_m tau)/nu_m^2]
    cM = X_nu[-1] * nu[-1] ** 2
    xx = 2 * np.pi * tau / beta
    full = (beta / (2 * np.pi)) ** 2 * (np.pi ** 2 / 6 - np.pi * xx / 2 + xx ** 2 / 4)
    part = np.cos(np.outer(tau, nu[1:])) @ (1.0 / nu[1:] ** 2)
    X_tau = (X_nu[0][None] + 2 * np.einsum("km,mij->kij", np.cos(np.outer(tau, nu[1:])), X_nu[1:])
             + 2 * (full - part)[:, None, None] * cM[None]) / beta
    S_tau = -G * X_tau
    w = (2 * np.arange(1, nw + 1) - 1) * np.pi / beta
    ph = np.exp(1j * np.outer(w, tau)) * wt[None, :]
    Giw = np.einsum("nk,kij->nij", ph, G)
    Sgw = np.einsum("nk,kij->nij", ph, S_tau)               # correlation part; static parts drop out of Gamma
    Sex = np.array([1j * wn * Iden - h - np.linalg.inv(g) for wn, g in zip(w, Giw)])
    Sgd = Sgw.copy()
    for i in range(ns):
        Sgd[:, i, i] = Sex[:, i, i]
    gam = lambda S: -(S - S.conj().transpose(0, 2, 1)) / 2j
    lam = lambda S: [float(np.linalg.eigvalsh(g).min()) for g in gam(S)]
    Pmin = float(min(np.linalg.eigvalsh(P).max() for P in P_nu))      # P(i nu) <= 0 required
    screen = float(min(np.linalg.eigvalsh(Iden - U * P).min() for P in P_nu))
    W0 = np.linalg.eigvalsh(X_nu[0] + U * Iden)
    if return_sigma:
        return dict(w=w, Sex=Sex, Sgw=Sgw, Sgd=Sgd, h=h, N=None)
    return dict(lam_ex=lam(Sex), lam_gw=lam(Sgw), lam_gd=lam(Sgd),
                g11_ex=float(gam(Sex)[0][0, 0].real), g11_gw=float(gam(Sgw)[0][0, 0].real),
                g12_gw=float(gam(Sgw)[0][0, 1].real), maxeigP=Pmin, min_screen=screen, W0=W0.tolist(),
                N=float(np.exp(-beta * E) @ np.einsum("pmm->m", np.array([V.T @ (c[p].T @ c[p]) @ V for p in range(2 * ns)])) / Z))


CL = {"dimer": (2, [(0, 1)]), "chain3": (3, [(0, 1), (1, 2)]), "ring3": (3, [(0, 1), (1, 2), (2, 0)]),
      "ring4": (4, [(0, 1), (1, 2), (2, 3), (3, 0)])}
CASES = [("dimer", 20.0, 3.0, 2.5), ("chain3", 10.0, 2.0, 2.5), ("ring3", 20.0, 4.0, -3.0),
         ("dimer", 20.0, 3.0, 1.0), ("chain3", 10.0, 2.0, 0.0), ("ring3", 10.0, 4.0, 0.0)]

if __name__ == "__main__":
    out = []
    for cl, b, U, mu in CASES:
        ns, bonds = CL[cl]
        r = run(ns, bonds, b, U, mu)
        r2 = run(ns, bonds, b, U, mu, panels=int(40 * b), M=600)          # refinement check
        r.update(cluster=cl, beta=b, U=U, mu=mu, refine_dev=float(max(abs(np.array(r["lam_gd"]) - np.array(r2["lam_gd"])))))
        out.append(r)
        print(f"{cl:6s} b={b:4.0f} U={U} mu={mu:+.1f} N={r['N']:.3f}  lam_gd(w1..3)={np.round(r['lam_gd'],6)}  "
              f"lam_gw={np.round(r['lam_gw'],6)}  refine={r['refine_dev']:.1e}")
    json.dump(out, open("indep_check.json", "w"), indent=1)
