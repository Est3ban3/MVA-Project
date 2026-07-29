#!/usr/bin/env python
"""Recover the per-track training-curve figures from the 2026-07-26 run.

Every NN/GNN notebook draws the same 2-panel figure after each track's fold-0
preview fit -- left "Loss" (train/val, physical *and* balanced-fit weights),
right "AUC" (train dropout-on, train dropout-off, val). Only ``PNN.ipynb``
calls ``savefig`` on it, so only ``ChannelCombinedTrainingCurves_PNN.png`` ever
reached disk. The other 18 are still sitting in the notebooks' saved cell
outputs as embedded PNGs.

This script extracts them, so the full model x era x channel set exists on disk
without retraining anything. It also dumps the per-epoch numbers that were
printed, for anyone who wants to re-plot or overlay.

IMPORTANT -- the printed numbers are NOT the full figure. ``train_model``
records 8 history series but prints only 5:

    printed      train_loss      (dropout ON, balanced fit weights)
                 val_loss        (physical weights -- the noisy light-orange curve)
                 train_auc, train_auc_eval, val_auc   (all 3 AUC series)

    NOT printed  train_loss_eval      (dropout off, physical)   } the three
                 train_loss_eval_fit  (dropout off, fit)        } emphasised
                 val_loss_fit         (balanced fit weights)    } loss curves

So the AUC panel is fully reconstructible from the logs but the loss panel is
not -- which is exactly why the embedded figures are the thing worth keeping.

The XGBoost masters print no eval history and embed no such figure; their
curves would have to be rebuilt from the saved boosters instead.

Usage:  .venv/bin/python Recover_LossCurves.py
"""
from __future__ import annotations

import base64
import json
import os
import re
from collections import OrderedDict

NB_DIR = "notebooks"
ART = os.path.join(NB_DIR, "PPSSP_2026")

THREE = ["Run 2", "Run 3", "Combined"]

# stem, family suffix, channel dir, [tracks in cell order], extract_figures?
#
# PNN is extract=False on purpose: its notebook already savefig's this figure at
# dpi=150 (1650x600), whereas the embedded copy is the inline render at matplotlib's
# default dpi=100 (1089x390). Writing the extracted one would silently DOWNGRADE a
# genuine artifact. The other six never call savefig, so the inline render is the
# only copy that exists and 1089x390 is the ceiling without retraining.
NOTEBOOKS = [
    ("DNN", "DNN", "1l2tau", THREE, True),
    ("DNN_2l2tau", "DNN", "2l2tau", THREE, True),
    ("MLP", "MLP", "1l2tau", THREE, True),
    ("MLP_2l2tau", "MLP", "2l2tau", THREE, True),
    ("GNN_Evie_final_1l2t", "GNN", "1l2tau", THREE, True),
    ("GNN_2l2tau", "GNN", "2l2tau", THREE, True),
    ("PNN", "PNN", "pnn_channel_combined", ["Channel-Combined"], False),
]

# track label -> (artifact subdir, filename prefix), matching the repo's
# existing convention (Run2WeightBalance.png, ChannelCombinedTrainingCurves_PNN.png)
TRACK_DIR = {
    "Run 2": ("run2", "Run2"),
    "Run 3": ("run3", "Run3"),
    "Combined": ("combined", "Combined"),
    "Channel-Combined": ("combined", "ChannelCombined"),
}

EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s*\|\s*train_loss=([\d.]+)\s+val_loss=([\d.]+)\s*\|\s*"
    r"train_auc=([\d.]+)\s+train_auc_eval=([\d.]+)\s+val_auc=([\d.]+)"
)

# The training-curve cell is identified by the two legend labels that only it
# uses, rather than by index, so a cell insertion does not silently shift it.
CURVE_CELL_MARKERS = ("physical weights", "balanced fit")


def cell_text(cell: dict) -> str:
    parts = []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            parts.append("".join(out.get("text", [])))
        elif out.get("output_type") in ("execute_result", "display_data"):
            parts.append("".join(out.get("data", {}).get("text/plain", [])))
    return "".join(parts)


def cell_images(cell: dict) -> list:
    return [o["data"]["image/png"] for o in cell.get("outputs", [])
            if "image/png" in o.get("data", {})]


def check_marker(stem: str, track: str, source: str) -> None:
    """Cross-check declared track against the cell's variable suffixes.

    Only asserts when a marker is present -- Run 2 sections and PNN use bare
    names, so their silence is not evidence either way.
    """
    run3 = re.search(r"_run3\b", source) is not None
    comb = re.search(r"_comb\b", source) is not None
    if run3 and not comb:
        assert track == "Run 3", f"{stem}: declared {track!r} but cell is _run3"
    elif comb and not run3:
        assert track == "Combined", f"{stem}: declared {track!r} but cell is _comb"


def extract_figures(stem: str, suffix: str, channel: str, tracks: list) -> list:
    """Write each track's embedded training-curve PNG to the artifact tree."""
    with open(os.path.join(NB_DIR, f"{stem}.ipynb")) as fh:
        nb = json.load(fh)

    hits = []
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell["source"])
        if not all(m in source for m in CURVE_CELL_MARKERS):
            continue
        imgs = cell_images(cell)
        if not imgs:          # the train_model definition cell also matches the
            continue          # markers (they live in its docstring) but has no figure
        # These cells draw the curves figure and then a second ROC figure.
        # Outputs are stored in execution order, so the curves figure is
        # imgs[0] -- but only if its plotting really does precede the first
        # plt.show(). Check that rather than trusting the ordering blindly.
        first_show = source.find("plt.show()")
        marker_at = min(source.find(m) for m in CURVE_CELL_MARKERS)
        assert 0 <= marker_at < first_show, (
            f"{stem}: curve plotting does not precede the first plt.show() - "
            f"cannot assume imgs[0] is the training-curve figure"
        )
        hits.append((imgs[0], source))

    assert len(hits) == len(tracks), (
        f"{stem}: found {len(hits)} training-curve figures but {len(tracks)} "
        f"tracks declared - cell structure changed, refusing to guess"
    )

    written = []
    for track, (b64, source) in zip(tracks, hits):
        check_marker(stem, track, source)
        sub, prefix = TRACK_DIR[track]
        out_dir = os.path.join(ART, channel, sub, "plots")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{prefix}TrainingCurves_{suffix}.png")
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(b64))
        written.append((track, path))
    return written


def extract_numbers(stem: str, tracks: list) -> "OrderedDict[str, list]":
    """Per-epoch values that were actually printed (see module docstring)."""
    with open(os.path.join(NB_DIR, f"{stem}.ipynb")) as fh:
        nb = json.load(fh)

    blocks = []
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        rows = EPOCH_RE.findall(cell_text(cell))
        if not rows:
            continue
        curve = [
            {"epoch": int(e), "train_loss_dropout_on_fitw": float(tl),
             "val_loss_physical": float(vl), "train_auc_dropout_on": float(ta),
             "train_auc_dropout_off": float(tae), "val_auc": float(va)}
            for e, tl, vl, ta, tae, va in rows
        ]
        eps = [r["epoch"] for r in curve]
        assert eps == sorted(eps) and len(set(eps)) == len(eps), (
            f"{stem}: interleaved epoch blocks in one cell - refusing to guess"
        )
        blocks.append((curve, "".join(cell["source"])))

    assert len(blocks) == len(tracks), (
        f"{stem}: {len(blocks)} printed blocks vs {len(tracks)} tracks declared"
    )
    found = OrderedDict()
    for track, (curve, source) in zip(tracks, blocks):
        check_marker(stem, track, source)
        found[track] = curve
    return found


def main() -> None:
    print("Extracting embedded training-curve figures (Loss + AUC panels):\n")
    total = 0
    numbers = {}
    for stem, suffix, channel, tracks, do_extract in NOTEBOOKS:
        if do_extract:
            written = extract_figures(stem, suffix, channel, tracks)
            for track, path in written:
                print(f"  {stem:22s} {track:16s} -> {path}")
            total += len(written)
        else:
            print(f"  {stem:22s} {'(all)':16s} -> skipped, notebook savefig's a "
                  f"higher-res copy itself")
        for track, curve in extract_numbers(stem, tracks).items():
            numbers[f"{suffix}|{channel}|{track}"] = curve

    out_json = os.path.join(ART, "logloss_curves", "printed_epoch_metrics.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(numbers, fh, indent=1)

    n_ep = sum(len(v) for v in numbers.values())
    print(f"\n  {total} figures written")
    print(f"  {out_json}  ({len(numbers)} curves, {n_ep} epochs)")
    print("\n  Note: the printed numbers cover the AUC panel in full but only "
          "2 of the 4\n  loss series - val_loss_fit and both eval-mode train "
          "losses were never\n  printed. Use the extracted figures for the loss panel.")


if __name__ == "__main__":
    main()
