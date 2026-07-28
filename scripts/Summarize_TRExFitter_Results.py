#!/usr/bin/env python3
"""Collect the expected limits and fit diagnostics from every TRExFitter output
directory into one table, and draw the MVA-method comparison.

Writes trexfitter/results/summary.csv and trexfitter/results/limit_comparison.png.
"""

import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "trexfitter", "results")

METHODS = ["XGB", "DNN", "MLP", "PNN", "GNN"]
ERAS = ["run2", "run3", "combined", "stat-comb"]


def parse_name(config):
    """hh1l2tau_run3_XGB -> (1l2tau, run3, XGB); combine_1l2tau_XGB -> (1l2tau, stat-comb, XGB)."""
    m = re.match(r"hh(\dl2tau)_(run2|run3|combined)_(\w+)$", config)
    if m:
        return m.group(1), m.group(2), m.group(3)
    m = re.match(r"combine_(\dl2tau)_(\w+)$", config)
    if m:
        return m.group(1), "stat-comb", m.group(2)
    return None, None, None


def read_limits(config):
    path = os.path.join(RESULTS, config, "Limits/Asymptotics/myLimit.root")
    if not os.path.isfile(path):
        return {}
    t = uproot.open(path)["stats"].arrays(library="np")
    return {k: float(v[0]) for k, v in t.items()}


def read_err_decomp(config):
    """TOT/STAT/MCSTAT breakdown of the mu_XS_hh uncertainty."""
    path = os.path.join(RESULTS, config, "Fits", f"{config}_errDecomp_mu_XS_hh.txt")
    out = {}
    if not os.path.isfile(path):
        return out
    for line in open(path):
        w = line.split()
        if len(w) == 4 and w[0].endswith("_ERROR"):
            out[w[0].lower()] = float(w[1])
            out[w[0].lower() + "_up"] = float(w[2])
            out[w[0].lower() + "_dn"] = float(w[3])
    return out


def read_sr_shape(config):
    """Pre-fit signal/background in the fitted SR, plus the binned Asimov significance."""
    files = glob.glob(os.path.join(RESULTS, config, "Plots", "SR__*_prefit.yaml"))
    if not files:
        return {}
    y = yaml.safe_load(open(files[0]))
    sig = bkg = None
    for s in y["Samples"]:
        v = np.array(s["Yield"], dtype=float)
        if s["Name"].startswith("HH"):
            sig = v if sig is None else sig + v
        else:
            bkg = v if bkg is None else bkg + v
    safe = np.maximum(bkg, 1e-9)
    # sum in quadrature of the per-bin Asimov discovery significance
    z = np.sqrt(np.sum(2 * ((sig + bkg) * np.log(1 + sig / safe) - sig)))
    return {
        "n_bins": len(sig),
        "S": sig.sum(),
        "B": bkg.sum(),
        "SB_last_bin": sig[-1] / safe[-1],
        "Z_asimov": z,
    }


def collect():
    rows = []
    for config in sorted(os.listdir(RESULTS)):
        if not os.path.isdir(os.path.join(RESULTS, config)):
            continue
        channel, era, method = parse_name(config)
        if channel is None:
            continue
        lim = read_limits(config)
        if not lim:
            continue
        row = {"config": config, "channel": channel, "era": era, "method": method}
        row.update(
            exp=lim["exp_upperlimit"],
            exp_m2=lim["exp_upperlimit_minus2"],
            exp_m1=lim["exp_upperlimit_minus1"],
            exp_p1=lim["exp_upperlimit_plus1"],
            exp_p2=lim["exp_upperlimit_plus2"],
            obs=lim["obs_upperlimit"],
        )
        row.update(read_err_decomp(config))
        row.update(read_sr_shape(config))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("exp").reset_index(drop=True)


def plot(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    x = np.arange(len(ERAS))
    for ax, channel in zip(axes, ["1l2tau", "2l2tau"]):
        sub = df[df.channel == channel]
        for i, method in enumerate(METHODS):
            m = sub[sub.method == method].set_index("era")
            vals = [m.exp.get(e, np.nan) for e in ERAS]
            lo = [m.exp.get(e, np.nan) - m.exp_m1.get(e, np.nan) for e in ERAS]
            hi = [m.exp_p1.get(e, np.nan) - m.exp.get(e, np.nan) for e in ERAS]
            off = (i - 2) * 0.14
            ax.errorbar(x + off, vals, yerr=[lo, hi], fmt="o", capsize=3, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels(ERAS)
        ax.set_title(f"HH $\\to$ {channel}")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel(r"expected 95% CL upper limit on $\mu_{HH}$")
    axes[0].legend(title="MVA")
    fig.suptitle("TRExFitter expected limits (blinded, MC-stat only)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("wrote", path)


if __name__ == "__main__":
    df = collect()
    csv = os.path.join(RESULTS, "summary.csv")
    df.to_csv(csv, index=False)
    print("wrote", csv, f"({len(df)} configs)")
    plot(df, os.path.join(RESULTS, "limit_comparison.png"))

    print("\nBest MVA per channel/era (expected UL on mu):")
    best = df.loc[df.groupby(["channel", "era"]).exp.idxmin()]
    print(best[["channel", "era", "method", "exp"]].to_string(index=False))
