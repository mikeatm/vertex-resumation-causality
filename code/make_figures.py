#!/usr/bin/env python3
"""Figures for the manuscript, from splice_scan.jsonl (+ one direct calculation for Fig. 2).
Usage: python make_figures.py --code causality-check.py --data splice_scan.jsonl --outdir figs
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import converge_splice as cs
import splice_scan as ss

# validated palette (dataviz reference, light mode): slots 1-3, all-pairs CVD-safe
C_EX, C_GW, C_GD = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "lines.linewidth": 1.4,
    "font.family": "serif", "mathtext.fontset": "cm", "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
TOL = 1e-6
NAMES = {"dimer": "dimer", "chain3": "3-chain", "ring3": "3-ring", "ring4": "4-ring"}


def load(path, factor):
    R = [json.loads(l) for l in open(path) if l.strip()]
    return [r for r in R if r["factor"] == factor]


def inwin(r):
    return 0.1 < r["N"] < 2 * cs.CLUSTERS[r["cluster"]][0] - 0.1


def fig_maps(R, out):
    clusters = ["dimer", "chain3", "ring3", "ring4"]
    betas = sorted({r["beta"] for r in R}); Us = sorted({r["U"] for r in R}); mus = sorted({r["mu"] for r in R})
    vmax = max(r["v_gd"] for r in R if inwin(r))
    norm = LogNorm(vmin=3e-4, vmax=vmax)
    cmap = ListedColormap(SEQ[2:])
    fig, axes = plt.subplots(len(betas), 4, figsize=(7.0, 3.3), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.08, hspace=0.25))
    for bi, b in enumerate(betas):
        for ci, cl in enumerate(clusters):
            ax = axes[bi, ci]
            ax.set_facecolor("#fcfcfb")
            for r in R:
                if r["cluster"] != cl or r["beta"] != b:
                    continue
                x, y = mus.index(r["mu"]), Us.index(r["U"])
                if not inwin(r):
                    ax.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, fc="#efeeea", ec="white", lw=1, hatch="////"))
                elif r["v_gd"] > TOL:
                    ax.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, fc=cmap(norm(max(r["v_gd"], 3e-4))), ec="white", lw=1))
                else:
                    ax.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, fc="#f3f3f1", ec="white", lw=1))
            ax.axvline(mus.index(0.0), color=INK2, lw=0.6, ls=":")
            ax.set_xlim(-.5, len(mus) - .5); ax.set_ylim(-.5, len(Us) - .5)
            n = sum(1 for r in R if r["cluster"] == cl and r["beta"] == b and inwin(r) and r["v_gd"] > TOL)
            ax.set_title(f"{NAMES[cl]}, $\\beta t={b:.0f}$  ({n})", color=INK, pad=3)
            ax.spines[:].set_visible(False); ax.tick_params(length=0)
            if bi == len(betas) - 1:
                ax.set_xticks(range(0, len(mus), 2)); ax.set_xticklabels([f"{mus[i]:g}" for i in range(0, len(mus), 2)])
                ax.set_xlabel(r"$\mu/t$")
            if ci == 0:
                ax.set_yticks(range(len(Us))); ax.set_yticklabels([f"{u:g}" for u in Us]); ax.set_ylabel(r"$U/t$")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.015)
    cb.set_label(r"violation $v=-\min_n\lambda_{\min}[\Gamma^{\rm GD}(i\omega_n)]$", color=INK)
    cb.outline.set_visible(False)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_mechanism(code, out, cl="dimer", beta=20.0, U=3.0, mu=2.5, nmax=160):
    mod = cs.load(code)
    ns, bonds = cs.CLUSTERS[cl]
    d = mod.build(ns, bonds, 1.0, U, mu, beta)
    K = np.unique(np.round(np.geomspace(1, 2 * nmax - 1, 60) / 2).astype(int) * 2 + 1)
    Sg = ss.sigma_gw_tc(mod, d, K, int(16 * beta * 2), int(8 * beta * 2))
    Se = mod.sigma_exact(K, d, mod.G(K, d)); Sd = mod.splice(Sg, Se)
    w = K * np.pi / beta
    Gg = np.array([ss.gamma(s) for s in Sg]); Ge = np.array([ss.gamma(s) for s in Se]); Gd = np.array([ss.gamma(s) for s in Sd])
    lmin = lambda G: np.array([np.linalg.eigvalsh(0.5 * (x + x.conj().T)).min() for x in G])
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.4, 4.3), sharex=True, gridspec_kw=dict(hspace=0.12))
    a.plot(w, Ge[:, 0, 0].real, color=C_EX, marker="o", ms=3, label=r"$\Gamma^{\rm ex}_{11}$ (exact local)")
    a.plot(w, Gg[:, 0, 0].real, color=C_GW, marker="s", ms=3, ls="--", label=r"$\Gamma^{GW}_{11}$ ($GW$ local)")
    a.plot(w, np.abs(Gg[:, 0, 1]), color=C_GD, marker="^", ms=3, ls="-.", label=r"$|\Gamma^{GW}_{12}|$ ($GW$ non-local)")
    bad = Ge[:, 0, 0].real < np.abs(Gg[:, 0, 1])
    a.set_xscale("log"); a.set_yscale("log"); a.set_ylabel(r"damping at $i\omega_n$")
    if bad.any():
        a.axvspan(w[0] * 0.8, w[bad][-1] * 1.25, color="#ececea", zorder=0)
        a.annotate(r"$\Gamma^{\rm ex}_{11}<|\Gamma^{GW}_{12}|$", xy=(w[0], Ge[0, 0, 0].real), xytext=(w[0] * 1.3, 2e-4),
                   color=INK2, fontsize=7, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    a.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1); a.grid(color=GRID, lw=0.4)
    a.text(0.98, 0.95, "(a)", transform=a.transAxes, ha="right", va="top", color=INK)
    for G, col, mk, ls, lab in [(Ge, C_EX, "o", "-", "exact"), (Gg, C_GW, "s", "--", r"$GW$"), (Gd, C_GD, "^", "-.", "spliced")]:
        b.plot(w, w * lmin(G), color=col, marker=mk, ms=3, ls=ls, label=lab)
    Cex = U ** 2 * d["nsig"] * (1 - d["nsig"])
    b.axhline(0, color=INK2, lw=0.6)
    b.axhline(Cex, color=INK2, lw=0.6, ls=":")
    b.text(w[-1], Cex * 1.08, r"$U^2n_\sigma(1-n_\sigma)$", ha="right", va="bottom", color=INK2, fontsize=7)
    b.set_yscale("symlog", linthresh=1e-2); b.set_xscale("log")
    b.set_xlabel(r"$\omega_n/t$"); b.set_ylabel(r"$\omega_n\,\lambda_{\min}[\Gamma(i\omega_n)]$")
    b.legend(loc="lower right"); b.grid(color=GRID, lw=0.4)
    b.text(0.02, 0.95, "(b)", transform=b.transAxes, ha="left", va="top", color=INK)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return dict(w=w.tolist(), lam_gd=lmin(Gd).tolist(), Cex=Cex, g11ex=Ge[0, 0, 0].real, g11gw=Gg[0, 0, 0].real, g12=abs(Gg[0, 0, 1]))


def fig_criterion(R, out):
    W = [r for r in R if inwin(r)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw=dict(wspace=0.3))
    mk = {"dimer": "o", "chain3": "s", "ring3": "^", "ring4": "D"}
    floor = 1e-5
    for cl in mk:
        P = [r for r in W if r["cluster"] == cl]
        x = np.array([r["gloc_gw"][0] / r["gloc_ex"][0] for r in P])
        y = np.array([max(r["v_gd"], floor) for r in P])
        v = y > floor
        a.scatter(x[~v], y[~v], marker=mk[cl], s=10, fc="none", ec=GRID, lw=0.7)
        a.scatter(x[v], y[v], marker=mk[cl], s=14, fc=C_GD, ec="white", lw=0.4, label=NAMES[cl])
    a.axvline(1, color=INK2, lw=0.6, ls=":")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(r"$\Gamma^{GW}_{\rm loc}/\Gamma^{\rm ex}_{\rm loc}$ at $\omega_1$"); a.set_ylabel(r"violation $v$ (causal points at floor)")
    a.legend(loc="upper left", title="violating:", title_fontsize=7); a.grid(color=GRID, lw=0.4)
    a.text(0.98, 0.04, "(a)", transform=a.transAxes, ha="right", color=INK)
    # (b) iff criterion for translation-invariant clusters
    nneg = 0
    xs, ys = [], []
    for cl in ["dimer", "ring3", "ring4"]:
        P = [r for r in W if r["cluster"] == cl]
        x = np.array([r["gloc_gw"][0] - r["gloc_ex"][0] for r in P])
        y = np.array([r["lam_gw"][0] for r in P])
        v = np.array([r["lam_gd"][0] < -TOL for r in P])
        nneg += int((x <= 0).sum())
        p = x > 0
        b.scatter(x[p & ~v], y[p & ~v], marker=mk[cl], s=10, fc="none", ec=C_EX, lw=0.7, alpha=0.8)
        b.scatter(x[p & v], y[p & v], marker=mk[cl], s=14, fc=C_GD, ec="white", lw=0.4)
        xs += list(x[p]); ys += list(y[p])
    lo = min(min(xs), min(ys)) / 2; hi = max(max(xs), max(ys)) * 2
    b.plot([lo, hi], [lo, hi], color=INK2, lw=0.8)
    b.set_xlim(lo, hi); b.set_ylim(lo, hi)
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel(r"local damping deficit $\Gamma^{GW}_{\rm loc}-\Gamma^{\rm ex}_{\rm loc}$ at $\omega_1$")
    b.set_ylabel(r"$\min_k \Gamma^{GW}(k,i\omega_1)$")
    b.text(0.04, 0.95, "causal (open)", transform=b.transAxes, va="top", color=C_EX, fontsize=7)
    b.text(0.96, 0.05, "non-causal (filled)", transform=b.transAxes, ha="right", color=INK2, fontsize=7)
    b.text(0.96, 0.13, f"+{nneg} points with deficit $\\leq 0$ (all causal)", transform=b.transAxes, ha="right", color=INK2, fontsize=6.5)
    b.grid(color=GRID, lw=0.4)
    b.text(0.98, 0.98, "(b)", transform=b.transAxes, ha="right", va="top", color=INK)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="causality-check.py")
    ap.add_argument("--data", default="splice_scan.jsonl")
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--outdir", default="figs")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    R = load(a.data, a.factor)
    fig_maps(R, os.path.join(a.outdir, "violation_maps.pdf"))
    fig_criterion(R, os.path.join(a.outdir, "criterion.pdf"))
    info = fig_mechanism(a.code, os.path.join(a.outdir, "mechanism_dimer.pdf"))
    print(json.dumps({k: (v if not isinstance(v, list) else v[:3]) for k, v in info.items()}))
