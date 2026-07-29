"""Plot the TRExFitter limit/fit results, including the run2+run3 statistical
combinations (`trexfitter/results/combine_<channel>_<MODEL>/`).

Why this exists
---------------
The `combine_*` jobs are TRExFitter **MultiFit** jobs: they combine the two
per-run *workspaces*, never the ntuples. MultiFit therefore writes no
`Histograms/`, no `Plots/` and no `Tables/` — the `n`/`d`/`p` letters in
`trex-fitter mnwdfpl` are no-ops in `m` mode. Its own comparison figures
(`Compare*`) were switched off in the generated configs, so the combinations
came back with exactly one plot (the error breakdown) and no limit figure at
all.

This script draws the missing figures from what the jobs *did* write —
`Limits/Asymptotics/myLimit.root` (expected limit + bands),
`Fits/<job>.txt` (fitted POI and its uncertainty) and
`Fits/<job>_group_errDecomp_mu_XS_hh.yaml` (stat / MC-stat breakdown) — so no
re-run on lxplus is needed. Reading, not refitting: every number here is
already on disk.

Blinding: every SR is `DataType: ASIMOV` with `LimitBlind: True`, so
`obs_upperlimit` is the -1 sentinel and only *expected* limits are plotted.
The fitted POI is 1 by construction; its uncertainty is the physics content.

The SR distribution figure
--------------------------
`SRDistribution_<channel>_<MODEL>.png` is the SR / MVA-score plot for the
*combination*, which no `combine_*` directory contains. Two facts make it
drawable here rather than only on lxplus:

1. **Post-fit yields are identical to pre-fit yields in every one of the 40
   jobs.** The SR is `DataType: ASIMOV` at mu=1, so the Asimov dataset IS the
   prediction and the best fit is exactly it: every fitted NP in every job
   comes back 1.000 (largest deviation across all 40: 4e-6). A "post-fit" SR
   plot here differs from the pre-fit one only in its uncertainty band - the
   stack is the same by construction. That is the fit working as a closure
   test, not a bug.
2. **The band is reconstructible from the fit result.** With
   N_i = gamma_i (B_i + mu S_i), the only free parameters are the per-bin
   MC-stat gammas and mu, so
       var(N_i) = (N_i sg_i)^2 + (S_i sm)^2 + 2 N_i S_i rho_i sg_i sm
   with sg/sm the symmetrised HESSE errors and rho from the fit's correlation
   matrix. Verified against TRExFitter's own `UncertaintyUp` on three
   independent jobs: agreement better than 0.5% per bin. Feed it the
   *combined* fit's covariance and you get the combined-fit band on the same
   stack - the one thing the combination actually changes.

The lower panel shows that shrink directly: relative band size for the
per-run fit vs for the combination.

Outputs (trexfitter/results/plots/):
  LimitComparison_<channel>.png/pdf     run2 / run3 / stat-combination per model
  CombinedLimit_ModelComparison.png     the headline: best model per channel
  ErrorBreakdown_Combination.png        sigma(mu) split into stat vs MC-stat
  StatComb_vs_CombinedTraining.png      the two different "combined" things
  SRDistribution_<channel>_<MODEL>.png  SR score stack + combined-fit band
  trexfitter_limit_summary.csv          every number in the figures

Usage:  .venv/bin/python Plot_TRExFitter_Results.py
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "trexfitter" / "results"
PLOTS = RESULTS / "plots"

CHANNELS = ("1l2tau", "2l2tau")
MODELS = ("XGB", "DNN", "MLP", "GNN", "PNN")

CHANNEL_LABEL = {"1l2tau": "1$\\ell$2$\\tau$", "2l2tau": "2$\\ell$2$\\tau$"}

# The four things called a "track" in this repo, kept distinct on purpose:
#  run2/run3      one fit, that run's own model
#  stat-comb      MultiFit: the two per-run workspaces in one likelihood
#  comb-training  one fit of the run2+run3-trained model on the pooled events
TRACKS = (
    ("run2", "Run 2"),
    ("run3", "Run 3"),
    ("statcomb", "Run 2 + Run 3"),
)

# Categorical slots 1-5 of the reference palette, in fixed order, one per
# model. Identity is carried by the row label as well as the hue, and every
# mark is direct-labelled, so the low-contrast slots are fine here.
MODEL_COLOR = {
    "XGB": "#2a78d6",
    "DNN": "#eb6834",
    "MLP": "#1baf7a",
    "GNN": "#eda100",
    "PNN": "#e87ba4",
}

# HEP convention for the uncertainty bands - deliberately NOT the categorical
# palette: readers of a limit plot expect yellow outside green.
BAND_2SIG = "#f6dd6b"
BAND_1SIG = "#6cba6c"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d9d9d6"


def job_dir(channel: str, track: str, model: str) -> Path:
    """Directory a given (channel, track, model) job wrote its results to."""
    if track == "statcomb":
        return RESULTS / f"combine_{channel}_{model}"
    return RESULTS / f"hh{channel}_{track}_{model}"


def read_limit(d: Path) -> dict:
    """Expected 95% CL limit on mu and its +-1/2 sigma bands.

    TRExFitter's asymptotic limit output is a one-entry TTree named `stats`.
    `obs_upperlimit` is -1 when the fit was run blind, which is the case for
    every job here - it is read only so the assert below can prove that.
    """
    with uproot.open(d / "Limits" / "Asymptotics" / "myLimit.root") as f:
        t = f["stats"].arrays(library="np")
    out = {k: float(t[k][0]) for k in
           ("obs_upperlimit", "exp_upperlimit", "exp_upperlimit_minus2",
            "exp_upperlimit_minus1", "exp_upperlimit_plus1",
            "exp_upperlimit_plus2")}
    assert out["obs_upperlimit"] < 0, (
        f"{d.name}: obs_upperlimit={out['obs_upperlimit']} is not the blind "
        "sentinel - the SR may have been unblinded, do not plot this as expected"
    )
    return out


def read_poi(d: Path) -> dict:
    """Fitted POI and its asymmetric uncertainty from Fits/<job>.txt."""
    txt = (d / "Fits" / f"{d.name}.txt").read_text().splitlines()
    for line in txt:
        parts = line.split()
        if parts and parts[0] == "mu_XS_hh":
            return {"mu": float(parts[1]),
                    "mu_up": float(parts[2]),
                    "mu_down": abs(float(parts[3]))}
    raise AssertionError(f"{d.name}: no mu_XS_hh line in the fit result")


def read_breakdown(d: Path) -> dict:
    """Grouped error decomposition (stat / MC stat / total syst) on mu.

    Hand-parsed rather than via PyYAML: the file is four fixed keys per block
    and the repo's .venv does not carry a yaml dependency for the fitting
    side. Impacts are *not* additive - they are quadrature contributions.
    """
    path = d / "Fits" / f"{d.name}_group_errDecomp_mu_XS_hh.yaml"
    out, cat = {}, None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("- Category:"):
            cat = line.split(":", 1)[1].strip()
        elif line.startswith("Impact:") and cat:
            out[cat] = float(line.split(":", 1)[1])
    return out


def collect() -> pd.DataFrame:
    """Every number the figures need, one row per (channel, track, model)."""
    rows = []
    for channel in CHANNELS:
        for track in ("run2", "run3", "combined", "statcomb"):
            for model in MODELS:
                d = job_dir(channel, track, model)
                assert d.is_dir(), f"missing job directory: {d}"
                row = {"channel": channel, "track": track, "model": model,
                       "job": d.name}
                row.update(read_limit(d))
                row.update(read_poi(d))
                bd = read_breakdown(d)
                row["err_stat"] = bd.get("Stat unc.", np.nan)
                row["err_mcstat"] = bd.get("MC stat.", np.nan)
                rows.append(row)
    df = pd.DataFrame(rows)
    # The combination must not be worse than either input: a MultiFit that
    # silently combined the wrong workspaces would show up right here.
    for channel in CHANNELS:
        for model in MODELS:
            sel = df[(df.channel == channel) & (df.model == model)]
            comb = sel[sel.track == "statcomb"].exp_upperlimit.iloc[0]
            single = sel[sel.track.isin(["run2", "run3"])].exp_upperlimit.min()
            assert comb <= single * 1.001, (
                f"{channel}/{model}: stat combination ({comb:.2f}) is weaker "
                f"than the best single run ({single:.2f})"
            )
    return df


def style_axes(ax, xlabel=None):
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.grid(axis="y", visible=False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, length=0)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=10)


def brazil_row(ax, y, lim, height=0.34):
    """One expected-limit row: +-2 sigma box, +-1 sigma box, median tick."""
    ax.barh(y, lim["exp_upperlimit_plus2"] - lim["exp_upperlimit_minus2"],
            left=lim["exp_upperlimit_minus2"], height=height * 2,
            color=BAND_2SIG, zorder=2)
    ax.barh(y, lim["exp_upperlimit_plus1"] - lim["exp_upperlimit_minus1"],
            left=lim["exp_upperlimit_minus1"], height=height * 2,
            color=BAND_1SIG, zorder=3)
    ax.plot([lim["exp_upperlimit"]] * 2, [y - height, y + height],
            color=INK, lw=2, ls=(0, (4, 2)), zorder=4)


def fig_limit_comparison(df: pd.DataFrame, channel: str):
    """The figure MultiFit's CompareLimits would have drawn, for all 5 models."""
    sel = df[df.channel == channel]
    order = (sel[sel.track == "statcomb"]
             .sort_values("exp_upperlimit").model.tolist())

    ypos, ylab, groups = [], [], []
    y = 0.0
    for model in order:
        start = y
        for track, tlabel in TRACKS:
            ypos.append(y)
            ylab.append(tlabel)
            y += 1.0
        groups.append((model, start, y - 1.0))
        y += 1.6  # gap between model blocks, room for the block header

    fig, ax = plt.subplots(figsize=(9.2, 8.8), constrained_layout=True)
    i = 0
    for model in order:
        for track, _ in TRACKS:
            r = sel[(sel.model == model) & (sel.track == track)].iloc[0]
            brazil_row(ax, ypos[i], r)
            ax.text(r.exp_upperlimit_plus2 + 1.2, ypos[i],
                    f"{r.exp_upperlimit:.1f}", va="center", ha="left",
                    fontsize=9, color=INK)
            i += 1

    ax.set_yticks(ypos)
    ax.set_yticklabels(ylab, fontsize=9, color=INK_SOFT)
    ax.invert_yaxis()
    style_axes(ax, "Expected 95% CL upper limit on $\\mu_{HH}$")
    ax.set_xlim(0, sel.exp_upperlimit_plus2.max() * 1.16)

    # model name once per block, in that model's hue, as a section header in
    # the gap above the block (x in axes fraction, y in data coordinates)
    for model, y0, _ in groups:
        ax.text(0.0, y0 - 0.78, model, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=12, fontweight="bold",
                color=MODEL_COLOR[model])

    ax.legend(handles=[
        Line2D([], [], color=INK, lw=2, ls=(0, (4, 2)), label="Expected limit"),
        Patch(facecolor=BAND_1SIG, label="$\\pm 1\\sigma$"),
        Patch(facecolor=BAND_2SIG, label="$\\pm 2\\sigma$"),
    ], loc="lower right", frameon=False, fontsize=9, labelcolor=INK_SOFT)

    ax.set_title(
        f"{CHANNEL_LABEL[channel]}  —  expected limit per model and per run\n",
        fontsize=13, color=INK, loc="left")
    ax.text(0, 1.005,
            "Asimov SR, blind (no observed limit). Statistical uncertainties "
            "only: no systematics, MC-stat gammas float.",
            transform=ax.transAxes, fontsize=8.5, color=INK_SOFT, va="bottom")

    save(fig, f"LimitComparison_{channel}")


def fig_model_comparison(df: pd.DataFrame):
    """Headline: which model gives the best combined expected limit."""
    sel = df[df.track == "statcomb"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    for ax, channel in zip(axes, CHANNELS):
        s = sel[sel.channel == channel].sort_values("exp_upperlimit")
        ypos = np.arange(len(s))
        for y, (_, r) in zip(ypos, s.iterrows()):
            brazil_row(ax, y, r, height=0.24)
            ax.plot([r.exp_upperlimit], [y], "o", ms=8,
                    color=MODEL_COLOR[r.model],
                    markeredgecolor="white", markeredgewidth=1.6, zorder=5)
            ax.text(r.exp_upperlimit_plus2 + 1.0, y, f"{r.exp_upperlimit:.1f}",
                    va="center", ha="left", fontsize=9.5, color=INK)
        ax.set_yticks(ypos)
        ax.set_yticklabels(s.model, fontsize=11, color=INK)
        ax.invert_yaxis()
        style_axes(ax, "Expected 95% CL upper limit on $\\mu_{HH}$")
        ax.set_xlim(0, sel.exp_upperlimit_plus2.max() * 1.14)
        ax.set_title(CHANNEL_LABEL[channel], fontsize=12, color=INK, loc="left")

    fig.suptitle("Run 2 + Run 3 statistical combination — expected limit by model",
                 fontsize=13, color=INK, ha="left", x=0.008)
    save(fig, "CombinedLimit_ModelComparison")


def fig_error_breakdown(df: pd.DataFrame):
    """sigma(mu) of the combination, split into its stat and MC-stat impacts."""
    sel = df[df.track == "statcomb"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    for ax, channel in zip(axes, CHANNELS):
        s = sel[sel.channel == channel].sort_values("mu_up")
        ypos = np.arange(len(s))
        h = 0.26
        ax.barh(ypos - h / 1.8, s.err_stat, height=h, color="#2a78d6",
                label="Data stat.", zorder=3)
        ax.barh(ypos + h / 1.8, s.err_mcstat, height=h, color="#eb6834",
                label="MC stat.", zorder=3)
        for y, (_, r) in zip(ypos, s.iterrows()):
            ax.text(r.err_stat + 0.12, y - h / 1.8, f"{r.err_stat:.2f}",
                    va="center", fontsize=8.5, color=INK)
            ax.text(r.err_mcstat + 0.12, y + h / 1.8, f"{r.err_mcstat:.2f}",
                    va="center", fontsize=8.5, color=INK)
            ax.plot([r.mu_up], [y], "|", ms=22, mew=2, color=INK, zorder=4)
        ax.set_yticks(ypos)
        ax.set_yticklabels(s.model, fontsize=11, color=INK)
        ax.invert_yaxis()
        style_axes(ax, "Impact on $\\mu_{HH}$")
        ax.set_xlim(0, sel[["err_stat", "mu_up"]].max().max() * 1.18)
        ax.set_title(CHANNEL_LABEL[channel], fontsize=12, color=INK, loc="left")

    fig.legend(handles=[
        Patch(facecolor="#2a78d6", label="Data stat."),
        Patch(facecolor="#eb6834", label="MC stat."),
        Line2D([], [], ls="", marker="|", ms=14, mew=2, color=INK,
               label="Total $+\\sigma(\\mu)$"),
    ], loc="outside upper right", ncol=3, frameon=False, fontsize=9,
        labelcolor=INK_SOFT)
    fig.suptitle("Uncertainty breakdown of the Run 2 + Run 3 combined fit\n"
                 "impacts add in quadrature, not linearly",
                 fontsize=12, color=INK, ha="left", x=0.008)
    save(fig, "ErrorBreakdown_Combination")


def fig_two_combineds(df: pd.DataFrame):
    """The two different "combined" things, side by side.

    statcomb  = one likelihood over the run2 fit and the run3 fit, each run
                scored by its own run-trained model (MultiFit).
    combined  = a single fit of the combined-TRAINED model (score_*_comb)
                on the pooled run2+run3 events.
    Same events either way, so this isolates "does training on both runs beat
    combining two run-specific models".

    CAVEAT on the `combined` side: it pools 13 TeV and 13.6 TeV events into
    one region under a single mu and one set of MC-stat gammas. Per the
    supervisor a run2+run3 combination must go through a MultiFit precisely
    because the two runs have different signal cross sections, so this track's
    mu and limit are NOT quotable as a combination result - the comparison
    below is indicative of the classifier, not of the statistics. Doing it
    properly means scoring each run with score_*_comb and MultiFit-ing those
    two fits; see the package README.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    for ax, channel in zip(axes, CHANNELS):
        sc = df[(df.channel == channel) & (df.track == "statcomb")]
        sc = sc.sort_values("exp_upperlimit")
        ct = df[(df.channel == channel) & (df.track == "combined")]
        ypos = np.arange(len(sc))
        for y, (_, r) in zip(ypos, sc.iterrows()):
            c = ct[ct.model == r.model].iloc[0]
            ax.plot([r.exp_upperlimit, c.exp_upperlimit], [y, y],
                    color=GRID, lw=2, zorder=1)
            ax.plot([r.exp_upperlimit], [y], "o", ms=9, color="#2a78d6",
                    markeredgecolor="white", markeredgewidth=1.5, zorder=3)
            ax.plot([c.exp_upperlimit], [y], "D", ms=8, color="#eb6834",
                    markeredgecolor="white", markeredgewidth=1.5, zorder=3)
            better = "stat. comb." if r.exp_upperlimit < c.exp_upperlimit else "comb. training"
            ax.text(max(r.exp_upperlimit, c.exp_upperlimit) + 0.8, y,
                    f"$\\Delta$ = {abs(r.exp_upperlimit - c.exp_upperlimit):.1f}"
                    f"  ({better} tighter)",
                    va="center", fontsize=8.5, color=INK_SOFT)
        ax.set_yticks(ypos)
        ax.set_yticklabels(sc.model, fontsize=11, color=INK)
        ax.invert_yaxis()
        style_axes(ax, "Expected 95% CL upper limit on $\\mu_{HH}$")
        lo = min(sc.exp_upperlimit.min(), ct.exp_upperlimit.min())
        hi = max(sc.exp_upperlimit.max(), ct.exp_upperlimit.max())
        ax.set_xlim(lo - 2, hi + 14)
        ax.set_title(CHANNEL_LABEL[channel], fontsize=12, color=INK, loc="left")

    fig.legend(handles=[
        Line2D([], [], ls="", marker="o", ms=9, color="#2a78d6",
               label="Stat. combination of the two run-trained models"),
        Line2D([], [], ls="", marker="D", ms=8, color="#eb6834",
               label="Single fit of the combined-trained model"),
    ], loc="outside upper right", ncol=2, frameon=False, fontsize=9,
        labelcolor=INK_SOFT)
    fig.suptitle("Two different \"combined\" results\ndiamonds pool 13 and 13.6 TeV "
                 "in one region — indicative of the classifier, not quotable "
                 "as a combination",
                 fontsize=12.5, color=INK, ha="left", x=0.008)
    save(fig, "StatComb_vs_CombinedTraining")


# --------------------------------------------------------------------------
# SR / MVA-score distribution with the combined fit's uncertainty band
# --------------------------------------------------------------------------

# Stack order, bottom to top, using categorical slots 1-5 plus a neutral for
# the three processes that are individually invisible (< 1% of the SR).
STACK = (
    ("$t\\bar{t}$", ("t#bar{t}",), "#2a78d6"),
    ("VV", ("VV",), "#eb6834"),
    ("$W$+jets", ("W+jets",), "#1baf7a"),
    ("$Z$+jets", ("Z+jets",), "#eda100"),
    ("Single-$H$", ("Single-H",), "#e87ba4"),
    ("Other", ("Other top", "VVV", "V#gamma"), "#9a9a94"),
)
SIGNAL_SAMPLE = "HH"


def read_region_yaml(path: Path) -> dict:
    """Per-sample yields, total, band and bin edges from a TRExFitter plot yaml.

    Hand-parsed for the same reason as the error decomposition: the file is a
    fixed, flat shape and the fitting side of the repo carries no yaml dep.
    """
    txt = path.read_text()
    arr = lambda s: np.array([float(x) for x in s.split(",")])
    samples = {name: arr(vals) for name, vals in
               re.findall(r"- Name: (.+?)\s*\n\s*Yield: \[([^\]]+)\]", txt)}
    total = arr(re.search(r"Total:\s*\n\s*- Yield: \[([^\]]+)\]", txt).group(1))
    band = arr(re.search(r"UncertaintyUp: \[([^\]]+)\]", txt).group(1))
    edges = arr(re.search(r"BinEdges: \[([^\]]+)\]", txt).group(1))
    xlabel = re.search(r"XaxisLabel: (.+)", txt).group(1).strip()
    known = {s for _, srcs, _ in STACK for s in srcs} | {SIGNAL_SAMPLE}
    unknown = set(samples) - known
    assert not unknown, f"{path.name}: unmapped samples {unknown} would be dropped"
    return {"samples": samples, "total": total, "band": band,
            "edges": edges, "xlabel": xlabel}


def read_fit_result(path: Path):
    """NP names, symmetrised errors and the correlation matrix of a fit.

    TRExFitter writes CORRELATION_MATRIX with its COLUMNS reversed (it is
    dumped in drawing order, y-axis inverted). Both asserts below fail loudly
    if that ever changes: the diagonal would not be 1 and the matrix would not
    be symmetric.
    """
    names, up, dn, mat, mode = [], [], [], [], None
    for line in path.read_text().splitlines():
        s = line.strip()
        if s == "NUISANCE_PARAMETERS":
            mode = "np"
            continue
        if s == "CORRELATION_MATRIX":
            mode = "cm"
            continue
        p = s.split()
        if mode == "np" and len(p) == 4:
            names.append(p[0])
            up.append(float(p[2]))
            dn.append(abs(float(p[3])))
        elif mode == "cm" and len(p) > 2:
            mat.append([float(x) for x in p])
    corr = np.array(mat)[:, ::-1]
    assert np.allclose(np.diag(corr), 1, atol=1e-6), f"{path.name}: bad diagonal"
    assert np.allclose(corr, corr.T, atol=1e-6), f"{path.name}: not symmetric"
    return names, (np.array(up) + np.array(dn)) / 2, corr


def band_from_fit(fit_dir: Path, region: str, total, signal):
    """Post-fit uncertainty on the total per bin, from that fit's covariance.

    Only the per-bin MC-stat gammas and mu are free, so the propagation is
    exact up to the HESSE symmetrisation - see the module docstring for the
    0.5% closure test against TRExFitter's own band.
    """
    names, err, corr = read_fit_result(fit_dir / "Fits" / f"{fit_dir.name}.txt")
    gi = [names.index(f"gamma_stat_{region}_bin_{i}") for i in range(len(total))]
    mi = names.index("mu_XS_hh")
    sg, sm = err[gi], err[mi]
    rho = np.array([corr[g, mi] for g in gi])
    return np.sqrt((total * sg) ** 2 + (signal * sm) ** 2
                   + 2 * total * signal * rho * sg * sm)


def fig_sr_distribution(channel: str, model: str):
    """SR score distribution for both runs, with the COMBINED fit's band."""
    comb_dir = job_dir(channel, "statcomb", model)
    fig, axes = plt.subplots(
        2, 2, figsize=(11.0, 5.6), sharex="col", constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1]})

    for col, run in enumerate(("run2", "run3")):
        d = job_dir(channel, run, model)
        region = f"SR__{model}_{run.upper()}"
        y = read_region_yaml(d / "Plots" / f"{region}_postfit.yaml")
        edges, total = y["edges"], y["total"]
        sig = y["samples"][SIGNAL_SAMPLE]

        # yields are the SAME pre- and post-fit (all gammas fit to 1 on
        # Asimov); only these two bands differ
        band_run = band_from_fit(d, region, total, sig)
        band_comb = band_from_fit(comb_dir, region, total, sig)

        ax, rax = axes[0, col], axes[1, col]
        bottom = np.zeros_like(total)
        for label, srcs, colour in STACK:
            v = sum((y["samples"][s] for s in srcs if s in y["samples"]),
                    np.zeros_like(total))
            ax.stairs(bottom + v, edges, baseline=bottom, fill=True,
                      color=colour, edgecolor="white", lw=0.8,
                      label=label, zorder=2)
            bottom = bottom + v

        # keep the signal overlay legible but inside the frame: aim for its
        # peak at ~40% of the tallest stacked bin, rounded to a power of ten
        scale = 10 ** int(np.floor(np.log10(0.4 * total.max() / sig.max())))
        ax.stairs(sig * scale, edges, color="#e34948", lw=2, ls="--",
                  label=f"$HH$ $\\times$ {scale:g}", zorder=4)
        ax.stairs(total + band_comb, edges, baseline=total - band_comb,
                  fill=True, facecolor="none", hatch="////",
                  edgecolor=INK_SOFT, lw=0, zorder=3,
                  label="Comb. fit unc.")

        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0, (total + band_comb).max() * 1.38)
        ax.set_title(f"{CHANNEL_LABEL[channel]}  {region}",
                     fontsize=11, color=INK, loc="left")
        style_axes(ax)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=GRID, lw=0.7)
        if col == 0:
            ax.set_ylabel("Events", color=INK_SOFT, fontsize=10)

        # stairs, not step(centres): the Trafo60 bins are irregular, so a
        # mid-referenced step would sit half a bin off the stack above
        # generic label: only the right panel carries the legend, but it has
        # to describe the left panel's run2 curve too
        rax.stairs(100 * band_run / total, edges, lw=2.5, color="#9a9a94",
                   label="that run's fit alone")
        rax.stairs(100 * band_comb / total, edges, lw=2, color="#2a78d6",
                   label="Run 2 + Run 3 fit")
        rax.set_xlim(edges[0], edges[-1])
        rax.set_ylim(0, max(100 * band_run / total) * 1.35)
        style_axes(rax, y["xlabel"])
        rax.grid(axis="y", color=GRID, lw=0.7)
        if col == 0:
            rax.set_ylabel("Unc. [%]", color=INK_SOFT, fontsize=9)

    # axes-level legends: an outside/figure legend lands on the same band as
    # the two-line suptitle and collides with it
    axes[0, 1].legend(ncol=2, frameon=False, fontsize=8.5,
                      labelcolor=INK_SOFT, loc="upper right")
    axes[1, 1].legend(ncol=2, frameon=False, fontsize=8,
                      labelcolor=INK_SOFT, loc="upper left")
    fig.suptitle(
        f"{model} — signal-region score distribution entering the Run 2 + Run 3 "
        f"combination\nAsimov SR, so post-fit yields = pre-fit yields "
        f"(every $\\gamma$ = 1); only the band moves",
        fontsize=11.5, color=INK, ha="left", x=0.008)
    save(fig, f"SRDistribution_{channel}_{model}")


def save(fig, stem: str):
    PLOTS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"{stem}.{ext}", dpi=200,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote plots/{stem}.png / .pdf")


def main():
    df = collect()
    PLOTS.mkdir(parents=True, exist_ok=True)
    cols = ["channel", "track", "model", "job", "exp_upperlimit",
            "exp_upperlimit_minus2", "exp_upperlimit_minus1",
            "exp_upperlimit_plus1", "exp_upperlimit_plus2",
            "mu", "mu_up", "mu_down", "err_stat", "err_mcstat"]
    df[cols].to_csv(PLOTS / "trexfitter_limit_summary.csv", index=False)
    print(f"  wrote plots/trexfitter_limit_summary.csv ({len(df)} jobs)")

    for channel in CHANNELS:
        fig_limit_comparison(df, channel)
    fig_model_comparison(df)
    fig_error_breakdown(df)
    fig_two_combineds(df)
    for channel in CHANNELS:
        for model in MODELS:
            fig_sr_distribution(channel, model)

    print("\nStat-combination expected limits:")
    piv = (df[df.track == "statcomb"]
           .pivot(index="model", columns="channel", values="exp_upperlimit"))
    print(piv.round(2).to_string())


if __name__ == "__main__":
    main()
