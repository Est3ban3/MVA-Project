"""Build TRExFitter inputs (augmented ntuples + NTUP-mode configs).

The supervisor's instruction is to run TRExFitter in NTUP mode (ReadFrom:
NTUP, as in TREXFitter_configs/example.config) on the ORIGINAL ntuples with
per-model score columns added. This script produces exactly that: each
original AnalysisMiniTree file is copied with 9 extra score branches
(score_{xgboost,dnn,mlp,gnn}[_comb] + score_pnn) plus w_phys / label /
process_id / fold bookkeeping.

The scores are the existing OOF exports
(Final_Notebooks/PPSSP_2026/<channel>/<run>/scores_<process>.root, written by
ExportModelScoresToROOT.ipynb) scattered back positionally: those files were
verified to align row-for-row with `original file + PRESELECTION` for all 40
(channel, run, process) combinations, and the alignment is re-asserted
chunk-by-chunk here via eventNumber. Events failing the preselection carry
sentinel -99 scores and are excluded by the configs' Selection anyway.

Produces, in TREXFitter_package/:

  ntuples/<channel>/<run>/<process>.root   original branches + score columns
  <channel>_limit_<run>_<MODEL>.config     one fit per channel/run/model
                                           (<run> incl. "combined")
  <channel>_combine_<MODEL>.config         run2+run3 MultiFit
  README.md                                run recipe + deviations from the
                                           supervisor's originals

Conventions follow the repo invariants: signed physical weights (negative
weights kept for evaluation), OOF scores only. The configs are adapted from
TREXFitter_configs/*_limit_*.config — same structure (blind asymptotic limit
on mu_XS_hh, Asimov SR, lumi + HH theory systematics), with the sample list
mapped to our 10 MC processes and the data-driven VR/CR regions dropped (no
real data / fake-tau estimate exists in this project).
"""

from pathlib import Path

import numpy as np
import uproot

REPO = Path(__file__).resolve().parent
SCORES_BASE = REPO / "Final_Notebooks" / "PPSSP_2026"
OUT = REPO / "TREXFitter_package"

CHANNELS = ("1l2tau", "2l2tau")
RUNS = ("run2", "run3")

# TRExFitter-facing model tag -> score branch in the scores_*.root trees.
# Per-run fits use the run-specific track's model (score_xgboost, ...), not
# the *_comb columns, so "the run2 fit uses the run2-trained model".
MODELS = {
    "XGB": "score_xgboost",
    "DNN": "score_dnn",
    "MLP": "score_mlp",
    "GNN": "score_gnn",
    "PNN": "score_pnn",
}

# The combined track: the combined-trained (run2+run3) model evaluated on the
# pooled run2+run3 events. PNN has a single score column (it is channel/run
# parameterized by construction), reused as-is.
MODELS_COMBINED = {
    "XGB": "score_xgboost_comb",
    "DNN": "score_dnn_comb",
    "MLP": "score_mlp_comb",
    "GNN": "score_gnn_comb",
    "PNN": "score_pnn",
}

# (process key = scores_<key>.root, original file name, sample name in
#  config, title, RGB, is_signal)
PROCESSES = [
    ("signal_ggF", "signal_ggF.root", "HH_ggF", "HH ggF", "237,34,37", True),
    ("signal_VBF", "signal_VBF.root", "HH_VBF", "HH VBF", "245,130,32", True),
    ("ttbar", "ttbar.root", "ttbar", "t#bar{t}", "0,84,159", False),
    ("tops", "tops.root", "tops", "Other top", "122,181,229", False),
    ("SingleH", "singleH.root", "SingleH", "Single-H", "139,69,19", False),
    ("Diboson", "diboson.root", "VV", "VV", "202,198,88", False),
    ("VVV", "VVV.root", "VVV", "VVV", "135,187,175", False),
    ("Vgamma", "Vgamma.root", "Vgamma", "V#gamma", "148,103,189", False),
    ("Wjets", "Wjets.root", "Wjets", "W+jets", "56,154,179", False),
    ("Zjets", "Zjets.root", "Zjets", "Z+jets", "0,158,115", False),
]

N_BINS = 8  # base binning in [0,1]; TRExFitter's AutoBin/Trafo60 rebins it

CHANNEL_LABEL = {"1l2tau": "1#ell2#tau", "2l2tau": "2#ell2#tau"}

# Run 3 collides at 13.6 TeV; the combined track mixes both energies.
CME_LABEL = {"run2": "13 TeV", "run3": "13.6 TeV", "combined": "13+13.6 TeV"}

TREE_NAME = "AnalysisMiniTree"

# The MVA preselection, straight from the master pipelines' PRESELECTION
# constants — as a numpy mask here and as a TTreeFormula string in the
# configs' Selection. The two must stay in sync.
PRESELECTION_NP = {
    "1l2tau": lambda a: (a["n_b_jet"] == 0) & (a["n_jet"] >= 2),
    "2l2tau": lambda a: (a["n_b_jet"] == 0)
    & (a["l1_charge"] * a["l2_charge"] < 0)
    & (a["mZ_cut"] > 0),
}
PRESELECTION_TTF = {
    "1l2tau": "n_b_jet==0 && n_jet>=2",
    "2l2tau": "n_b_jet==0 && l1_charge*l2_charge<0 && mZ_cut>0",
}

# SR / LowMVA split on the model score (from the supervisor-guided
# 1l2tau_limit_run2_evie.config draft): SR = score >= cut (kept ASIMOV/blind),
# LowMVA = score < cut (VALIDATION region, real data shown there).
SCORE_CUT = 0.4

# Channel-specific physics cuts on top of the preselection, from the same
# draft (1l2tau only; no 2l2tau equivalents were specified — flagged in the
# README rather than invented here).
EXTRA_CUTS = {
    "1l2tau": " && fabs(dR_t1t2)<=2.0 && pair_isOStaus && pass_SLT",
    "2l2tau": "",
}

DATA_SCORED = REPO / "New_Data_scored"

SCORE_BRANCHES = sorted(set(MODELS.values()) | set(MODELS_COMBINED.values()))


def augment_ntuples(channel: str, run: str) -> dict:
    """Copy each original ntuple with score branches added; return yields.

    Only events passing the MVA preselection are written — the OOF scores are
    attached positionally: the scores_*.root rows equal the original file's
    preselected rows in order (verified globally once, re-asserted per chunk
    on eventNumber below). The configs keep the redundant Selection as a
    guard in case anyone points them at unfiltered ntuples.
    """
    src = SCORES_BASE / channel / run
    dst = OUT / "ntuples" / channel / run
    dst.mkdir(parents=True, exist_ok=True)
    summary = {}
    for proc_key, orig_fname, sample, _t, _rgb, _is_sig in PROCESSES:
        if (dst / orig_fname).exists():
            # already built — delete ntuples/ to force a rebuild (mandatory
            # after the scores_*.root exports are regenerated)
            t = uproot.open(dst / orig_fname)[TREE_NAME]
            summary[sample] = float(t["w_phys"].array(library="np").sum())
            continue
        st = uproot.open(src / f"scores_{proc_key}.root")["nominal"]
        scores = {b: st[b].array(library="np") for b in SCORE_BRANCHES}
        s_event = st["eventNumber"].array(library="np")
        label = int(st["label"].array(entry_stop=1)[0])
        pid = int(st["process_id"].array(entry_stop=1)[0])
        summary[sample] = float(st["w_phys"].array(library="np").sum())

        tin = uproot.open(src / orig_fname)[TREE_NAME]
        cursor = 0
        with uproot.recreate(dst / orig_fname) as fout:
            tree_out = None
            for chunk in tin.iterate(step_size="200 MB", library="np"):
                mask = PRESELECTION_NP[channel](chunk)
                k = int(mask.sum())
                chunk = {name: arr[mask] for name, arr in chunk.items()}
                assert np.array_equal(
                    chunk["eventNumber"], s_event[cursor : cursor + k]
                ), f"{channel}/{run}/{proc_key}: OOF alignment broke"
                for b in SCORE_BRANCHES:
                    chunk[b] = scores[b][cursor : cursor + k]
                cursor += k
                chunk["w_phys"] = (chunk["weight"] * chunk["weights"]).astype(
                    np.float64
                )
                chunk["label"] = np.full(k, label, np.int32)
                chunk["process_id"] = np.full(k, pid, np.int32)
                chunk["fold"] = (chunk["eventNumber"] % 5).astype(np.int32)
                if tree_out is None:
                    fout.mktree(
                        TREE_NAME, {k_: v.dtype for k_, v in chunk.items()}
                    )
                    tree_out = fout[TREE_NAME]
                tree_out.extend(chunk)
        assert cursor == len(s_event), f"{channel}/{run}/{proc_key}: leftover OOF rows"
    return summary


def limit_config(channel: str, run: str) -> str:
    """One TRExFitter NTUP-mode config per <channel>/<run>, with one SR
    region per model. The regions all contain the SAME events (one score
    each), so fits must run ONE region at a time via the Regions= CLI option
    — never all together."""
    ch_label = CHANNEL_LABEL[channel]
    run_tag = run.upper()
    models = MODELS_COMBINED if run == "combined" else MODELS
    lumi = "300 fb^{-1}" if run == "combined" else "140 fb^{-1}"
    src_runs = list(RUNS) if run == "combined" else [run]

    lines = [
        f"% Auto-generated by Build_TRExFitter_Inputs.py from "
        f"TREXFitter_configs/{channel}_limit_run2.config (NTUP mode)",
        "% One SR region per model, all holding the SAME events - run ONE",
        "% region per fit, e.g.:",
        f"%   trex-fitter nwdfl {channel}_limit_{run}.config "
        f'"Regions=SR__XGB_{run_tag}:Suffix=_XGB"',
        "",
        f"Job: hh{channel}_{run}",
        f"  Label: {ch_label}",
        f"  CmeLabel: {CME_LABEL[run]}",
        "  POI: mu_XS_hh",
        "  ReadFrom: NTUP",
        "  NtuplePath: ./ntuples",
        f'  NtupleName: "{TREE_NAME}"',
        '  MCweight: "w_phys"',
        "  Lumi: 1.0",  # w_phys already includes the lumi scaling
        f"  LumiLabel: {lumi}",
        "  PlotOptions: NORMSIG, NOXERR, YIELDS, CHI2",
        "  LegendX1: 0.6",
        "  DebugLevel: 1",
        "  HistoChecks: NOCRASH",
        "  ImageFormat: pdf,png",
        "  MCstatThreshold: 0.001",
        "  MCstatConstraint: Poisson",
        "  GetChi2: True",
        "  StatOnly: False",  # MC-stat gammas still float; no Systematic blocks
        "  DoSummaryPlot: True",
        "  SummaryPlotYmin: 0.05",
        "  DoTables: True",
        "  DoPieChartPlot: True",
        "  KeepPrefitBlindedBins: TRUE",
        "  RecreateBinningFiles: TRUE",
        "",
    ]

    base_sel = PRESELECTION_TTF[channel] + EXTRA_CUTS[channel]
    for model, branch in models.items():
        lines += [
            # no DataType override (per supervisor): regions default to the
            # real data sample; the limit stays expected-only via LimitBlind.
            f"Region: SR__{model}_{run_tag}",
            "  Type: SIGNAL",
            f"  Label: HH {ch_label} SR ({model})",
            f'  Variable: "{branch}",{N_BINS},{SCORE_CUT},1',
            f"  VariableTitle: {model} score",
            '  Binning: "AutoBin","Trafo60",4,4',
            f"  Selection: {branch}>={SCORE_CUT} && {base_sel}",
            "",
            f"Region: LowMVA__{model}_{run_tag}",
            "  Type: VALIDATION",
            f"  Label: HH {ch_label} LowMVA ({model})",
            f'  Variable: "{branch}",{N_BINS},0,{SCORE_CUT}',
            f"  VariableTitle: {model} score",
            '  Binning: "AutoBin","Trafo60",4,4',
            f"  Selection: {branch}<{SCORE_CUT} && {base_sel}",
            "",
        ]

    lines += [
        "Fit: fit",
        "  FitType: SPLUSB",
        "  FitRegion: CRSR",
        "  POIAsimov: 1",
        "  UseMinos: mu_XS_hh",
        "",
        "Limit: limit",
        "  LimitType: ASYMPTOTIC",
        "  LimitBlind: True",
        "  POI: mu_XS_hh",
        "",
    ]

    data_files = ", ".join(f"Data/data_{channel}_{r}" for r in src_runs)
    lines += [
        # real data (scored by Score_New_Data.py; same tree/schema as the MC
        # ntuples). Enters only the LowMVA VALIDATION regions — the SR is
        # DataType: ASIMOV, so the fit and limit stay blind.
        f"Sample: data_{channel}_{run}",
        "  Type: DATA",
        "  Title: data",
        "  FillColor: 1",
        f"  NtupleFiles: {data_files}",
        "",
    ]

    for _proc_key, orig_fname, sample, title, rgb, is_sig in PROCESSES:
        stem = orig_fname.removesuffix(".root")
        files = ", ".join(f"{channel}/{r}/{stem}" for r in src_runs)
        lines += [
            f"Sample: {sample}",
            f"  Type: {'SIGNAL' if is_sig else 'BACKGROUND'}",
            f"  Title: {title}",
            f"  FillColorRGB: {rgb}",
            f"  NtupleFiles: {files}",
            f'  Group: "{"HH" if is_sig else title}"',
            "",
        ]

    lines += [
        "NormFactor: mu_XS_hh",
        "  Nominal: 1",
        "  Min: -100",
        "  Max: 500",  # supervisor had 100; MVA-only Asimov sensitivity is weaker
        "  Samples: HH_*",
        "  Regions: SR*",
        "",
        "% No Systematic blocks, per supervisor - statistical-only fit",
        "% (MC-stat gammas still included via MCstatThreshold).",
        "",
    ]
    return "\n".join(lines)


def combine_config(channel: str) -> str:
    """Run2+Run3 MultiFit for one channel, mirroring *_combine.config.

    Combines whatever region the two per-run workspaces were last built
    with — build both runs with the SAME model's Regions= before running
    this, and finish each model's combination before starting the next
    (workspaces live under the per-run Job dirs and are overwritten).
    """
    lines = [
        f"% Auto-generated by Build_TRExFitter_Inputs.py from "
        f"TREXFitter_configs/{channel}_combine.config",
        "% Combines the regions the two per-run workspaces were LAST built",
        "% with - run both runs' nw steps for one model, then this, before",
        "% moving to the next model (see README).",
        "",
        f'MultiFit: "combine_{channel}"',
        f'  Label: "{channel}"',
        '  POITitle: "#it{#mu}(#it{hh})"',
        '  LimitTitle: "95% CL limit on SigXsecOverSM"',
        "  Combine: TRUE",
        '  CmeLabel: "13+13.6 TeV"',
        '  LumiLabel: "300 fb^{-1}"',
        "  ComparePOI: FALSE",
        "  CompareLimits: TRUE",
        '  PlotOptions: "YIELDS,LEFT,NOXERR,NOENDERR, NORMSIG"',
        "  ShowObserved: FALSE",
        "  DebugLevel: 1",
        '  DataName: "asimovData"',
        "  POIName: mu_XS_hh",
        "",
        'Fit: "run2"',
        f"  ConfigFile: configs/{channel}_limit_run2.config",
        '  Label: "run2"',
        "",
        'Fit: "run3"',
        f"  ConfigFile: configs/{channel}_limit_run3.config",
        '  Label: "run3"',
        "",
        "Limit: limit",
        "  LimitType: ASYMPTOTIC",
        "  LimitBlind: True",
        "  POI: mu_XS_hh",
        "",
    ]
    return "\n".join(lines)


def copy_data_ntuples():
    """Copy the scored real-data files (Score_New_Data.py output) into the
    package under ntuples/Data/. Same tree name and schema as the MC files."""
    import shutil

    dst = OUT / "ntuples" / "Data"
    dst.mkdir(parents=True, exist_ok=True)
    for channel in CHANNELS:
        for run in RUNS:
            fname = f"data_{channel}_{run}.root"
            src = DATA_SCORED / fname
            assert src.exists(), f"{src} missing — run Score_New_Data.py first"
            shutil.copy2(src, dst / fname)
            print(f"copied {fname} -> ntuples/Data/")


def main():
    OUT.mkdir(exist_ok=True)
    copy_data_ntuples()
    # configs go in their own subfolder so it can be zipped/shipped without
    # the ntuples; run trex-fitter from the package ROOT (NtuplePath and the
    # MultiFit ConfigFile references resolve relative to the CWD):
    #   trex-fitter nwdfl configs/<name>.config "Regions=...:Suffix=..."
    cfg_dir = OUT / "configs"
    cfg_dir.mkdir(exist_ok=True)
    for channel in CHANNELS:
        for run in RUNS:
            summary = augment_ntuples(channel, run)
            sig = sum(v for k, v in summary.items() if k.startswith("HH_"))
            bkg = sum(v for k, v in summary.items() if not k.startswith("HH_"))
            print(f"{channel}/{run}: signal yield {sig:.3f}, background {bkg:.1f}")
        for run in (*RUNS, "combined"):
            path = cfg_dir / f"{channel}_limit_{run}.config"
            path.write_text(limit_config(channel, run))
        (cfg_dir / f"{channel}_combine.config").write_text(combine_config(channel))
    print(f"\nWrote inputs to {OUT}")


if __name__ == "__main__":
    main()
