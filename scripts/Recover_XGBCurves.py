#!/usr/bin/env python
"""Rebuild XGBoost's per-round log-loss curves from the SAVED boosters. No refit.

The two master notebooks are the only ones with no recoverable training curve:
they print no eval history, and their `plot_feature_set_comparison` cells were
added after the 2026-07-26 run and never executed (zero embedded output). A
re-run would not fix it either -- `train_xgb_fold` passes `eval_set=[(val)]`
only, so `evals_result()` records no TRAIN curve at all.

The saved fold models do contain everything needed. Boosting is additive, so
for any round b the model's output is fully determined by trees[0:b]; replaying
`predict(iteration_range=(0, b))` over a grid of b reconstructs the exact curve
the fit would have produced, for train AND val, under either weighting.

Correctness is not assumed. Two checks run every time:
  1. The reconstructed cumulative margin is compared against a direct
     `iteration_range=(0, n)` prediction (the additive shortcut below).
  2. The fold model's predictions on its held-out test fold are compared
     against the stored `oof_scores_{track}.csv` -- the actual deliverable
     produced by the original run. If the data pipeline were rebuilt even
     slightly wrong, this mismatches and the script aborts.

Preprocessing is not re-derived by hand: the master notebook's own setup cells
(constants + helper defs, everything before the first `load_run_data` call) are
exec'd, and `load_run_data`/`clean_data`/`assign_folds`/`prepare_fold_data` are
called exactly as the notebook calls them.

Fold 0 only, to match the NN/GNN figures, which are all fold-0 preview fits.

Usage:  .venv/bin/python Recover_XGBCurves.py [--tracks run2,run3,combined]
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

NB_DIR = Path("notebooks")
# main() chdir's into notebooks, because the notebooks' BASE_DIR_* constants
# are relative to it; both are rebound there.
ART = NB_DIR / "PPSSP_2026"

MASTERS = {
    "1l2tau": "1L2Tau_Master_Pipeline",
    "2l2tau": "2L2Tau_Master_Pipeline",
}

# track -> (artifact subdir, filename prefix, cell_cols passed to prepare_fold_data)
# cell_cols=("run",) for Combined is the notebook's own convention -- the
# fit-weight balancing is done per run there, not globally.
TRACKS = {
    "run2": ("run2", "Run2", ()),
    "run3": ("run3", "Run3", ()),
    "combined": ("combined", "Combined", ("run",)),
}

N_GRID = 60          # points on the curve
EPS = 1e-7           # probability clip for the log


def setup_namespace(stem: str) -> dict:
    """Exec the notebook's constants + helper defs, stopping before data loading."""
    with open(NB_DIR / f"{stem}.ipynb", encoding="utf-8") as fh:
        nb = json.load(fh)

    code_cells = [
        (i, "".join(c["source"]))
        for i, c in enumerate(nb["cells"])
        if c["cell_type"] == "code"
    ]

    # First cell that CALLS load_run_data (rather than defining it) marks the
    # end of setup. Detected by AST so a mention in a comment/docstring cannot
    # move the boundary.
    boundary = None
    for i, src in code_cells:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        called = {
            n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        if "load_run_data" in called and "load_run_data" not in defined:
            boundary = i
            break
    assert boundary is not None, f"{stem}: no load_run_data call site found"

    ns = {"__name__": "__nb__"}
    for i, src in code_cells:
        if i >= boundary:
            break
        exec(compile(src, f"<{stem}:c{i}>", "exec"), ns)  # noqa: S102
    for need in ("load_run_data", "clean_data", "assign_folds", "prepare_fold_data",
                 "discover_common_features", "compute_process_yield_targets", "to_device"):
        assert need in ns, f"{stem}: setup did not define {need}"
    return ns


def build_track_data(ns: dict, track: str):
    """Reproduce the notebook's load -> clean -> assign_folds for one track."""
    load, clean = ns["load_run_data"], ns["clean_data"]
    discover, assign = ns["discover_common_features"], ns["assign_folds"]
    r2, r3 = ns["BASE_DIR_R2"], ns["BASE_DIR_R3"]

    if track == "run2":
        feats = discover([r2])
        data = load(r2, run_label=2, features=feats, verbose=False)
    elif track == "run3":
        feats = discover([r3])
        data = load(r3, run_label=3, features=feats, verbose=False)
    else:
        feats = discover([r2, r3])
        data = pd.concat(
            [load(r2, run_label=2, features=feats, verbose=False),
             load(r3, run_label=3, features=feats, verbose=False)],
            ignore_index=True,
        )
    data, feats = clean(data, feats, verbose=False)
    return assign(data, n_folds=ns["N_FOLDS"]), feats


def _np(arr) -> np.ndarray:
    """cupy -> numpy. inplace_predict returns device arrays for device input."""
    getter = getattr(arr, "get", None)
    return np.asarray(getter() if callable(getter) else arr, dtype=np.float64)


def cumulative_margins(booster, X, grid):
    """Margin at each round in `grid`, at the cost of ONE full prediction pass.

    Trees are additive: predict(0, b) = base + sum(trees[0:b]). So summing the
    increments predict(a, b) - base over consecutive slices reconstructs every
    cumulative margin while evaluating each tree exactly once, instead of
    re-walking the first b trees for every grid point.
    """
    def margin(a, b):
        return _np(booster.inplace_predict(X, iteration_range=(int(a), int(b)),
                                           predict_type="margin"))

    # Every slice carries the intercept once: g(a,b) = base + sum(trees[a:b]).
    # NOTE iteration_range=(0, 0) does NOT mean "no trees" - 0 is a sentinel for
    # "to the end", so it returns the FULL prediction. Recover the intercept
    # algebraically instead: g(0,b1) + g(b1,b2) - g(0,b2) == base.
    b1, b2 = int(grid[0]), int(grid[1])
    g01 = margin(0, b1)
    base = g01 + margin(b1, b2) - margin(0, b2)
    assert np.ptp(base) < 1e-4, (
        f"intercept is not constant across events (ptp={np.ptp(base):.2e}) - "
        f"the additive decomposition does not hold for this model"
    )

    out, prev, cum = [g01.copy()], b1, g01.copy()
    for b in grid[1:]:
        cum = cum + (margin(prev, b) - base)
        out.append(cum.copy())
        prev = int(b)

    direct = _np(booster.inplace_predict(X, iteration_range=(0, int(grid[-1])),
                                         predict_type="margin"))
    assert np.allclose(out[-1], direct, atol=1e-4), (
        "additive-margin reconstruction disagrees with a direct prediction - "
        "refusing to emit a curve built on a false assumption"
    )
    return out


def weighted_logloss(y, margin, w):
    p = np.clip(1.0 / (1.0 + np.exp(-margin)), EPS, 1.0 - EPS)
    ll = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(np.sum(ll * w) / np.sum(w))


def curve_for_track(ns: dict, channel: str, track: str) -> dict:
    sub, prefix, cell_cols = TRACKS[track]
    art = ART / channel / sub
    with open(art / f"features_{track}.json", encoding="utf-8") as fh:
        final_features = json.load(fh)

    print(f"  [{channel}/{track}] loading data ...", flush=True)
    data, _ = build_track_data(ns, track)

    target_yields = ns["compute_process_yield_targets"](data)
    fd = ns["prepare_fold_data"](data, final_features, target_yields,
                                 cell_cols=cell_cols, n_folds=ns["N_FOLDS"], k=0)

    model = xgb.XGBClassifier()
    model.load_model(art / f"model_{track}_fold0.json")
    booster = model.get_booster()
    n_rounds = booster.num_boosted_rounds()

    to_device, use_gpu = ns["to_device"], ns["USE_GPU"]
    # A booster loaded from JSON carries no device; without this it silently
    # falls back to a CPU DMatrix copy for every cupy input (slow, and warns).
    if use_gpu:
        booster.set_param({"device": "cuda"})

    # --- verification against the stored deliverable -------------------------
    oof = pd.read_csv(art / f"oof_scores_{track}.csv")
    assert len(oof) == len(data), (
        f"{channel}/{track}: OOF has {len(oof)} rows, rebuilt data has {len(data)} - "
        f"the data pipeline did not reproduce the original run"
    )
    test_mask = (data["fold"] == 0).to_numpy()
    pred_test = model.predict_proba(to_device(fd["test_df"][final_features], use_gpu))[:, 1]
    stored = oof.loc[test_mask, "score"].to_numpy()
    max_dev = float(np.max(np.abs(np.asarray(pred_test) - stored)))
    assert max_dev < 1e-5, (
        f"{channel}/{track}: fold-0 predictions deviate from stored OOF by "
        f"{max_dev:.2e} - reconstruction is NOT faithful, refusing to plot"
    )
    print(f"    OOF check passed (max deviation {max_dev:.2e} over "
          f"{test_mask.sum():,} events, {n_rounds} rounds)", flush=True)

    grid = np.unique(np.linspace(1, n_rounds, N_GRID).astype(int))
    series = {}
    for part in ("train", "val"):
        df = fd[f"{part}_df"]
        X = to_device(df[final_features], use_gpu)
        y = df["label"].to_numpy().astype(np.float64)
        w_phys = np.abs(df["w_phys"].to_numpy())
        w_fit = fd[f"w_{part}_fit"]
        margins = cumulative_margins(booster, X, grid)
        series[f"{part}_physical"] = [weighted_logloss(y, m, w_phys) for m in margins]
        series[f"{part}_fit"] = [weighted_logloss(y, m, w_fit) for m in margins]

    return {
        "grid": grid.tolist(),
        "n_rounds": n_rounds,
        "best_iteration": int(booster.attributes().get("best_iteration", n_rounds - 1)),
        "oof_max_deviation": max_dev,
        **series,
    }


def plot_track(channel: str, track: str, cur: dict) -> str:
    sub, prefix, _ = TRACKS[track]
    out_dir = ART / channel / sub / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}TrainingCurves.png"

    g = cur["grid"]
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    ax.plot(g, cur["train_physical"], color="tab:blue", alpha=0.3,
            label="train (physical weights)")
    ax.plot(g, cur["val_physical"], color="tab:orange", alpha=0.3,
            label="val (physical weights, low N_eff — noisy)")
    ax.plot(g, cur["train_fit"], color="tab:blue", lw=2,
            label="train (balanced fit weights)")
    ax.plot(g, cur["val_fit"], color="tab:orange", lw=2,
            label="val (balanced fit weights)")
    ax.axvline(cur["best_iteration"], ls=":", color="0.5", lw=1.2)
    ax.set_xlabel("boosting round")
    ax.set_ylabel("weighted BCE loss")
    ax.set_title(f"Loss (XGBoost, {channel} {prefix}, fold 0)\n"
                 f"dotted = best_iteration ({cur['best_iteration']})", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="run2,run3,combined")
    ap.add_argument("--channels", default="1l2tau,2l2tau")
    args = ap.parse_args()

    # The notebooks use paths relative to notebooks/
    os.chdir(NB_DIR)
    sys.path.insert(0, ".")
    globals()["NB_DIR"] = Path(".")
    globals()["ART"] = Path("PPSSP_2026")

    results = {}
    for channel in args.channels.split(","):
        print(f"\n=== {channel} ({MASTERS[channel]}) ===", flush=True)
        ns = setup_namespace(MASTERS[channel])
        for track in args.tracks.split(","):
            cur = curve_for_track(ns, channel, track)
            path = plot_track(channel, track, cur)
            results[f"{channel}|{track}"] = cur
            print(f"    -> {path}", flush=True)

    out = Path("PPSSP_2026/logloss_curves/xgb_reconstructed_curves.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
