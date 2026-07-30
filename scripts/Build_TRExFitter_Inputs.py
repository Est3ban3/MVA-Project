"""Build TRExFitter inputs (augmented ntuples + NTUP-mode configs).

The supervisor's instruction is to run TRExFitter in NTUP mode (ReadFrom:
NTUP, as in trexfitter/configs_original/example.config) on the ORIGINAL ntuples with
per-model score columns added. This script produces exactly that: each
original AnalysisMiniTree file is copied with 9 extra score branches
(score_{xgboost,dnn,mlp,gnn}[_comb] + score_pnn) plus w_phys / label /
process_id / fold bookkeeping.

The scores are the existing OOF exports
(notebooks/PPSSP_2026/<channel>/<run>/scores_<process>.root, written by
ExportModelScoresToROOT.ipynb) scattered back positionally: those files were
verified to align row-for-row with `original file + PRESELECTION` for all 40
(channel, run, process) combinations, and the alignment is re-asserted
chunk-by-chunk here via eventNumber. Events failing the preselection carry
sentinel -99 scores and are excluded by the configs' Selection anyway.

Produces, in trexfitter/package/:

  ntuples/<channel>/<run>/<process>.root   original branches + score columns
  <channel>_limit_<run>_<MODEL>.config     one fit per channel/run/model
                                           (<run> incl. "combined")
  <channel>_combine_<MODEL>.config         run2+run3 MultiFit
  <channel>_limit_<run>.config             all-models convenience copy (only
                                           valid with Regions= one model at
                                           a time — see its header)
  configs/run_all.sh                       every fit + combine command, in
                                           dependency order (nwd, fp, l)
  README.md                                run recipe + deviations from the
                                           supervisor's originals

Conventions follow the repo invariants: signed physical weights (negative
weights kept for evaluation), OOF scores only. The configs are adapted from
trexfitter/configs_original/*_limit_*.config — same structure (blind asymptotic limit
on mu_XS_hh, Asimov SR, lumi + HH theory systematics), with the sample list
mapped to our 10 MC processes and the data-driven VR/CR regions dropped (no
real data / fake-tau estimate exists in this project).
"""

from pathlib import Path

import numpy as np
import uproot

REPO = Path(__file__).resolve().parent.parent
SCORES_BASE = REPO / "notebooks" / "PPSSP_2026"
OUT = REPO / "trexfitter" / "package"

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

# Channel-specific physics cuts on top of the preselection, as specified by
# the supervisor. `pair_isOStaus` = the two taus have opposite charges.
#
# 2l2tau's was supplied later (2026-07-30) as
# `fabs(dR_t1t2)<=2.0 && pair_isOStaus && low_mass_cut && passTriggers`.
# There is no `passTriggers` branch — the only trigger flags in the 2l2tau
# ntuples are `pass_SLT` and `pass_DLT` — so it is expanded here to their OR
# (user-confirmed), matching the fact that 1l2tau, which carries five trigger
# flags, names `pass_SLT` explicitly instead of asking for "the triggers".
#
# `low_mass_cut` is ~a no-op on the samples we have (identically 1 in every
# process except a handful of diboson/Zjets events) but is kept because it is
# part of the stated selection and costs nothing.
EXTRA_CUTS = {
    "1l2tau": " && fabs(dR_t1t2)<=2.0 && pair_isOStaus && pass_SLT",
    "2l2tau": " && fabs(dR_t1t2)<=2.0 && pair_isOStaus && low_mass_cut"
    " && (pass_SLT || pass_DLT)",
}

DATA_SCORED = REPO / "new_data_scored"

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


def limit_config(
    channel: str, run: str, model: str | None = None, branch: str | None = None
) -> str:
    """One TRExFitter NTUP-mode config per <channel>/<run>/<model>: that
    model's SR + LowMVA pair only, under its own Job name. The five models
    score the SAME events, so they must never share a likelihood — one model
    per config makes the invalid all-models fit unexpressible instead of
    relying on a Regions= CLI option being remembered (the HH_RUNS 2026-07
    fits forgot it and double-counted the data five times).

    model=None emits the all-models-in-one convenience layout instead
    (6 configs, user-requested): every model's region pair in one file,
    valid ONLY when every trex-fitter step is run with Regions= restricted
    to a single model's pair (exact commands in the file header). The fits
    are identical to the per-model configs' — prefer those / run_all.sh.
    """
    ch_label = CHANNEL_LABEL[channel]
    run_tag = run.upper()
    lumi = "300 fb^{-1}" if run == "combined" else "140 fb^{-1}"
    src_runs = list(RUNS) if run == "combined" else [run]
    if model is not None:
        models = {model: branch}
        header = [
            f"% {model} only. The five models score the SAME events, so each",
            "% model gets its own config/Job - never merge them into one fit.",
        ]
        job = f"hh{channel}_{run}_{model}"
    else:
        models = MODELS_COMBINED if run == "combined" else MODELS
        cfg = f"configs/{channel}_limit_{run}.config"
        opts = f'"Regions=SR__XGB_{run_tag},LowMVA__XGB_{run_tag}:Suffix=_XGB"'
        header = [
            "% ALL-MODELS convenience config - same fits as the per-model",
            "% configs (prefer those / run_all.sh). The five models' regions",
            "% hold the SAME events, so NEVER run this file bare: pass",
            "% Regions= with ONE model's pair (+ Suffix) to EVERY step:",
            f"%   trex-fitter nwd {cfg} {opts}",
            f"%   trex-fitter fp  {cfg} {opts}",
            f"%   trex-fitter l   {cfg} {opts}",
            "% then the same three lines for DNN, MLP, GNN, PNN.",
        ]
        job = f"hh{channel}_{run}"

    lines = [
        f"% Auto-generated by Build_TRExFitter_Inputs.py from "
        f"trexfitter/configs_original/{channel}_limit_run2.config (NTUP mode)",
        *header,
        "",
        f"Job: {job}",
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
    for m, b in models.items():
        lines += [
            f"Region: SR__{m}_{run_tag}",
            "  Type: SIGNAL",
            # SR stays blind, as in the supervisor's HIST-mode template: the
            # fit sees the mu=1 Asimov dataset here (POIAsimov in the Fit
            # block), never real data. Real data enters only the LowMVA
            # VALIDATION region below.
            "  DataType: ASIMOV",
            f"  Label: HH {ch_label} SR ({m})",
            f'  Variable: "{b}",{N_BINS},{SCORE_CUT},1',
            f"  VariableTitle: {m} score",
            '  Binning: "AutoBin","Trafo60",4,4',
            f"  Selection: {b}>={SCORE_CUT} && {base_sel}",
            "",
            f"Region: LowMVA__{m}_{run_tag}",
            "  Type: VALIDATION",
            f"  Label: HH {ch_label} LowMVA ({m})",
            f'  Variable: "{b}",{N_BINS},0,{SCORE_CUT}',
            f"  VariableTitle: {m} score",
            '  Binning: "AutoBin","Trafo60",4,4',
            f"  Selection: {b}<{SCORE_CUT} && {base_sel}",
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
        # ntuples). Enters only the LowMVA VALIDATION region — the SR is
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


def combine_config(channel: str, model: str) -> str:
    """Run2+Run3 MultiFit for one channel/model, mirroring *_combine.config.

    References the per-model run2/run3 configs; their Job names (and thus
    workspace paths) are unique per model, so the ten combinations are
    independent — no overwrite hazard, any order works once the per-run
    workspaces exist.

    MultiFit combines WORKSPACES, not events: it never reads an ntuple, so it
    writes no Histograms/, Plots/ or Tables/ however many of `n`/`d`/`p` are
    passed — those letters are no-ops in `m` mode. Its only figures are the
    Compare* ones, which the first revision of this generator left off (it
    mirrored the supervisor's original, where `%Compare: TRUE` is commented
    out), so the 2026-07-28 combinations came back with a limit but no limit
    plot. They are on now.

    `ComparePulls` stays off deliberately: with no systematics the only NPs
    are per-run MC-stat gammas (gamma_stat_SR__<MODEL>_RUN2/RUN3), so the two
    fits share not a single nuisance parameter and the pull comparison would
    be empty."""
    lines = [
        f"% Auto-generated by Build_TRExFitter_Inputs.py from "
        f"trexfitter/configs_original/{channel}_combine.config",
        f"% Statistical run2+run3 combination for {model} (each run keeps",
        "% its own run-trained model).",
        "",
        f'MultiFit: "combine_{channel}_{model}"',
        f'  Label: "{channel} {model}"',
        '  POITitle: "#it{#mu}(#it{hh})"',
        '  LimitTitle: "95% CL limit on SigXsecOverSM"',
        "  Combine: TRUE",
        '  CmeLabel: "13+13.6 TeV"',
        '  LumiLabel: "300 fb^{-1}"',
        # Compare: master switch for the comparison figures. Without it the
        # MultiFit produces no plot at all (see the docstring).
        "  Compare: TRUE",
        "  ComparePOI: TRUE",
        "  CompareLimits: TRUE",
        "  PlotCombCorrMatrix: TRUE",
        '  PlotOptions: "YIELDS,LEFT,NOXERR,NOENDERR, NORMSIG"',
        "  ShowObserved: FALSE",
        "  DebugLevel: 1",
        '  DataName: "asimovData"',
        "  POIName: mu_XS_hh",
        "",
        'Fit: "run2"',
        f"  ConfigFile: configs/{channel}_limit_run2_{model}.config",
        '  Label: "run2"',
        "",
        'Fit: "run3"',
        f"  ConfigFile: configs/{channel}_limit_run3_{model}.config",
        '  Label: "run3"',
        "",
        "Limit: limit",
        "  LimitType: ASYMPTOTIC",
        "  LimitBlind: True",
        "  POI: mu_XS_hh",
        "",
    ]
    return "\n".join(lines)


def postfit_from_combination_config(channel: str, run: str, model: str) -> str:
    """Per-run config that redraws its SR/LowMVA plots using the MULTIFIT result.

    The combination itself stays a MultiFit — per the supervisor, run2 and run3
    must not be merged into one likelihood by hand, because the two runs are at
    different centre-of-mass energies (13 vs 13.6 TeV) and therefore different
    signal cross sections. Only the plots are redrawn here; no fit is repeated
    and no number is re-derived. The mu, its uncertainty and the limit all
    remain the MultiFit's.

    TRExFitter has no `mp` step: a MultiFit combines workspaces and builds no
    histograms, so it cannot draw a region plot. A normal job can — it owns the
    histograms — but has no option to read a fit result from another directory
    (`FitResultsFile` exists only in the MultiFit `Fit` block, checked against
    the settings list). What an ordinary job has is `Suffix`, "added to file
    names of plots, workspace, fit results etc.": the `p` step then reads
    `<Job>/Fits/<Job><Suffix>.txt` and writes `Plots/<region>_postFit<Suffix>`.
    So the recipe is to stage the MultiFit's fit result into this job's `Fits/`
    under the suffixed name and re-run `p` — what run_postfit_combined.sh does.

    Why one config per (channel, model, RUN) and not per (channel, model): the
    combined likelihood spans two regions, `SR__<MODEL>_RUN2` and
    `SR__<MODEL>_RUN3`, and each lives in a different job directory with its
    own histograms. A Job draws only its own regions, so covering both runs of
    all ten combinations takes 10 x 2 = 20 invocations.

    `Suffix` does not touch histogram file names (that is `SaveSuffix`), so
    `trex-fitter hp` reads the histograms the original `n` step already wrote:
    no ntuples needed, nothing refitted, and the existing unsuffixed plots are
    not overwritten.

    The MultiFit result carries both runs' gammas; the ones belonging to the
    other run simply do not match any parameter here and are ignored.
    """
    base = limit_config(channel, run, model, MODELS[model])
    job = f"hh{channel}_{run}_{model}"
    comb = f"combine_{channel}_{model}"
    header = [
        "% Auto-generated by Build_TRExFitter_Inputs.py.",
        f"% Redraws {job}'s regions using the MultiFit result of {comb}.",
        "% The combination stays a MultiFit (supervisor: run2 and run3 are at",
        "% different energies / cross sections, so they must not be merged",
        "% into one hand-written likelihood). Nothing is refitted here - only",
        "% the region plots are redrawn with the combined fit's parameters.",
        "%",
        "% Identical to the fit config except for the Suffix below - the",
        "% regions, samples and binning MUST stay byte-identical, since the",
        "% `h` step reads the histograms the original job already wrote.",
        "%",
        "% PREREQUISITE (run_postfit_combined.sh does this for you):",
        f"%   cp {comb}/Fits/{comb}.txt \\",
        f"%      {job}/Fits/{job}_combFit.txt",
        "% then:",
        f"%   trex-fitter hp configs/{channel}_postfit_{run}_{model}_combFit.config",
        "%",
        "% Writes Plots/<region>_postFit_combFit.{png,pdf}, leaving the",
        "% original single-run post-fit plots untouched.",
        "%",
        "% NB the stack will be the same as the single-run post-fit plot: the",
        "% SR is DataType: ASIMOV at mu=1, so every gamma and mu fit to 1.000",
        "% in BOTH fits (largest deviation across all 40 jobs: 4e-6). What",
        "% changes is the uncertainty band, which is what this is for.",
    ]
    # keep everything from `Job:` on verbatim, swap the generated preamble
    body = base[base.index("Job: "):]
    body = body.replace(f"Job: {job}\n", f"Job: {job}\n  Suffix: _combFit\n", 1)
    return "\n".join(header) + "\n\n" + body


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
    #   bash configs/run_all.sh
    cfg_dir = OUT / "configs"
    cfg_dir.mkdir(exist_ok=True)
    # superseded layouts (e.g. the all-models-in-one-Job configs) must not
    # linger as runnable files — they are exactly the double-count hazard
    for stale in cfg_dir.glob("*.config"):
        stale.unlink()
    fit_cmds, simple_cmds, combine_cmds, simple_combine_cmds, inputs = (
        [], [], [], [], []
    )
    postfit_jobs = []  # (channel, run, model, config) for the redraw pass
    for channel in CHANNELS:
        for run in RUNS:
            inputs.append(f"Data/data_{channel}_{run}.root")
            inputs += [f"{channel}/{run}/{p[1]}" for p in PROCESSES]
            summary = augment_ntuples(channel, run)
            sig = sum(v for k, v in summary.items() if k.startswith("HH_"))
            bkg = sum(v for k, v in summary.items() if not k.startswith("HH_"))
            print(f"{channel}/{run}: signal yield {sig:.3f}, background {bkg:.1f}")
        for run in (*RUNS, "combined"):
            models = MODELS_COMBINED if run == "combined" else MODELS
            for model, branch in models.items():
                name = f"{channel}_limit_{run}_{model}.config"
                (cfg_dir / name).write_text(limit_config(channel, run, model, branch))
                # supervisor's steps (n w d f p l) in ONE invocation. Split
                # across processes (nwd / fp / l) the fit silently produces
                # no Fits/*.txt in NTUP mode: only `n` builds the histograms,
                # and a fresh process has none loaded. Keep them together —
                # or prefix each later call with `h` to re-read the cached
                # histograms (trex-fitter hfp / hl).
                fit_cmds.append(
                    f"run_job configs/{name} hh{channel}_{run}_{model}"
                )
                # supervisor's step grouping (user-verified on lxplus): the
                # merged nwdfpl run dies with free(): invalid pointer,
                # while nwd / fp / l as separate processes complete.
                simple_cmds += [
                    f"trex-fitter {steps} configs/{name}"
                    for steps in ("nwd", "fp", "l")
                ] + [""]
            # all-models-in-one convenience copy (user preference): only
            # valid with Regions= one model at a time — see its header.
            # NOT in run_all.sh; the per-model configs above cover the fits.
            (cfg_dir / f"{channel}_limit_{run}.config").write_text(
                limit_config(channel, run)
            )
        for model in MODELS:
            name = f"{channel}_combine_{model}.config"
            (cfg_dir / name).write_text(combine_config(channel, model))
            # `w` combines the two per-run workspaces, `f` fits it, `l` puts
            # the limit on it. `n`/`d`/`p` are NOT included: MultiFit reads
            # workspaces, not ntuples, so they are no-ops (the 2026-07-28 run
            # used `mnwdfpl` and got no Histograms/, Plots/ or Tables/).
            # The trailing bare `m` is the comparison pass — it needs every
            # referenced fit AND limit to exist, so it must come last.
            combine_cmds += [
                f"trex-fitter mwfl configs/{name}",
                f"trex-fitter m    configs/{name}",
            ]
            # redraw both of the combination's regions with the MultiFit
            # result - one per run, since a Job draws only its own regions
            for run in RUNS:
                pf = f"{channel}_postfit_{run}_{model}_combFit.config"
                (cfg_dir / pf).write_text(
                    postfit_from_combination_config(channel, run, model)
                )
                postfit_jobs.append((channel, run, model, pf))
            # separated variant for run_fits.sh, at the supervisor's own
            # granularity (mw / mf / ml), plus the same comparison pass
            simple_combine_cmds += [
                f"trex-fitter {steps} configs/{name}"
                for steps in ("mwf", "ml", "m")
            ] + [""]
    # ---- optional pass: region plots drawn with the MultiFit result -------
    postfit = ["#!/usr/bin/env bash",
               "# Auto-generated by Build_TRExFitter_Inputs.py.",
               "#",
               "# Redraw every SR/LowMVA plot using the Run2+Run3 MULTIFIT",
               "# result instead of the single-run fit. Run AFTER run_all.sh.",
               "#",
               "# The combination itself stays a MultiFit - run2 and run3 are at",
               "# different energies (13 vs 13.6 TeV) and so different signal",
               "# cross sections, and per the supervisor must not be merged into",
               "# one hand-written likelihood. Nothing here refits anything: the",
               "# mu, its uncertainty and the limit stay the MultiFit's, and only",
               "# the region plots are redrawn with its parameter values.",
               "#",
               "# TRExFitter has no `mp` step: a MultiFit combines workspaces and",
               "# builds no histograms, so it cannot draw a region plot. A normal",
               "# job can - it owns the histograms - but has no option to read a",
               "# fit result from another directory (`FitResultsFile` is MultiFit",
               "# only). `Suffix` is the hinge: the `p` step reads",
               "# <Job>/Fits/<Job><Suffix>.txt, so staging the MultiFit result",
               "# under that name is what makes this work.",
               "#",
               "# 20 invocations = 10 combinations x the 2 regions each spans,",
               "# which live in 2 different job directories.",
               "#",
               "# `h` re-reads the cached histograms, so no ntuples are touched and",
               "# nothing is refitted. Output: Plots/<region>_postFit_combFit.*,",
               "# alongside (not replacing) the single-run post-fit plots.",
               "set -euo pipefail",
               'ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." '
               '&& pwd)"',
               'cd "$ROOT"',
               "",
               "stage() {  # <multifit job> <fit job>",
               '  src="$1/Fits/$1.txt"',
               '  dst="$2/Fits/$2_combFit.txt"',
               '  if [ ! -s "$src" ]; then',
               '    echo "ERROR: $src missing - run the MultiFit first." >&2',
               "    exit 1",
               "  fi",
               '  cp "$src" "$dst"',
               "  # the .txt is what FitResults::ReadFromTXT reads; copy the",
               "  # RooFitResult too when it exists, so nothing downstream can",
               "  # fall back to the single-run one under the suffixed name",
               '  [ -s "$1/Fits/$1.root" ] && cp "$1/Fits/$1.root" '
               '"$2/Fits/$2_combFit.root"',
               "}",
               ""]
    for channel, run, model, name in postfit_jobs:
        job = f"hh{channel}_{run}_{model}"
        postfit += [f"stage combine_{channel}_{model} {job}",
                    f"trex-fitter hp configs/{name}",
                    ""]
    postfit += ['echo "MultiFit region plots:"',
                "ls hh*/Plots/*_postFit_combFit.png"]
    (OUT / "run_postfit_combined.sh").write_text("\n".join(postfit) + "\n")

    n_fits = len(fit_cmds)
    script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "# Auto-generated by Build_TRExFitter_Inputs.py.",
            "# Launch from anywhere - it cd's to the package root itself:",
            "#   bash configs/run_all.sh   |   cd configs && bash run_all.sh",
            "# One job per (channel, run, model): each config holds a single",
            "# model, so no Regions=/Suffix= CLI options are needed.",
            "# Steps per job are the supervisor's (n w d f p l) in ONE",
            "# invocation: run as separate processes the fit writes no",
            "# Fits/*.txt, because only `n` builds the NTUP histograms.",
            "set -euo pipefail",
            "",
            "# every path below (and NtuplePath / the MultiFit ConfigFile",
            "# references inside the configs) resolves against the CWD, which",
            "# must be the package root - the dir holding ntuples/ and configs/",
            'ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." '
            '&& pwd)"',
            'cd "$ROOT"',
            "",
            "# ---- trex-fitter environment --------------------------------",
            "# trex-fitter is NOT in PATH in a fresh shell. On lxplus it ships",
            "# inside the StatAnalysis release; set that up here so the script",
            "# works from a clean login. Pin a release with e.g.",
            "#   STATANALYSIS=StatAnalysis,0.5.3 bash configs/run_all.sh",
            "if ! command -v trex-fitter >/dev/null 2>&1; then",
            "  ALRB=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase",
            '  if [ -e "$ALRB/user/atlasLocalSetup.sh" ]; then',
            '    echo "note: trex-fitter not in PATH - running asetup '
            '${STATANALYSIS:-StatAnalysis,0.6.3}"',
            "    set +eu  # ALRB/asetup are not set -eu clean",
            '    export ATLAS_LOCAL_ROOT_BASE="$ALRB"',
            '    source "$ALRB/user/atlasLocalSetup.sh" --quiet',
            '    asetup "${STATANALYSIS:-StatAnalysis,0.6.3}"',
            "    set -eu",
            "  fi",
            "fi",
            "if ! command -v trex-fitter >/dev/null 2>&1; then",
            '  echo "ERROR: trex-fitter not found in PATH (and no cvmfs to'
            ' set it up from)." >&2',
            '  echo "  This script needs a machine with TRExFitter - i.e.'
            ' lxplus, not a" >&2',
            '  echo "  local box. There: setupATLAS && asetup'
            ' StatAnalysis,0.6.3" >&2',
            '  echo "  (or pin: STATANALYSIS=StatAnalysis,<version> bash'
            ' configs/run_all.sh)" >&2',
            "  exit 1",
            "fi",
            'echo "using trex-fitter: $(command -v trex-fitter)"',
            "",
            "# ---- locate the ntuples -------------------------------------",
            "# NTUPLE_PATH=/some/where/ntuples bash configs/run_all.sh",
            "# overrides the search below.",
            'NTUPLES="${NTUPLE_PATH:-}"',
            'if [ -z "$NTUPLES" ]; then',
            "  # single line on purpose: a backslash continuation here got"
            " corrupted",
            "  # in a clipboard transfer once and broke the whole script",
            '  for cand in "$ROOT/ntuples" "$ROOT/configs/ntuples"'
            ' "$ROOT/../ntuples" "$ROOT/ntuples/ntuples"; do',
            '    if [ -d "$cand" ]; then NTUPLES="$cand"; break; fi',
            "  done",
            "fi",
            'NTUPLES="$(readlink -f "${NTUPLES:-/nonexistent}")"',
            'if [ ! -d "$NTUPLES" ]; then',
            "  echo \"ERROR: no ntuples directory found.\" >&2",
            '  echo "  searched: \\$ROOT/ntuples, \\$ROOT/configs/ntuples,'
            ' \\$ROOT/../ntuples" >&2',
            '  echo "  ROOT=$ROOT" >&2',
            '  echo "  Copy ntuples/ next to configs/ (702 MB, 44 .root'
            ' files), or:" >&2',
            '  echo "    NTUPLE_PATH=/path/to/ntuples bash configs/run_all.sh"'
            " >&2",
            "  exit 1",
            "fi",
            "",
            "# the configs say 'NtuplePath: ./ntuples', i.e. relative to this",
            "# CWD - link whatever we found into place so that resolves",
            'if [ "$NTUPLES" != "$ROOT/ntuples" ]; then',
            '  ln -sfn "$NTUPLES" "$ROOT/ntuples"',
            '  echo "note: linked $ROOT/ntuples -> $NTUPLES"',
            "fi",
            "",
            "# ---- preflight ----------------------------------------------",
            "# TRExFitter only WARNS on a missing input and then fits empty",
            "# histograms (single-bin auto-binning, no Fits/*.txt), so check",
            "# every file the configs will open BEFORE burning 40 jobs.",
            "INPUTS=(",
            *[f'  "{f}"' for f in inputs],
            ")",
            "missing=0",
            'for f in "${INPUTS[@]}"; do',
            '  if [ ! -r "$ROOT/ntuples/$f" ]; then',
            '    echo "MISSING: $ROOT/ntuples/$f" >&2',
            "    missing=$((missing + 1))",
            "  fi",
            "done",
            'if [ "$missing" -ne 0 ]; then',
            '  echo "ERROR: $missing of ${#INPUTS[@]} inputs missing under'
            ' $NTUPLES" >&2',
            '  echo "  (expected layout: ntuples/{1l2tau,2l2tau}/{run2,run3}/'
            '*.root + ntuples/Data/*.root)" >&2',
            "  exit 1",
            "fi",
            'echo "preflight OK: ${#INPUTS[@]} inputs readable under $NTUPLES"',
            "",
            "# ---- run ----------------------------------------------------",
            "# trex-fitter exits 0 even when a job produces nothing, so check",
            "# the fit result exists before moving on.",
            "run_job() {  # <config> <job dir>",
            '  echo "=== $1"',
            '  trex-fitter nwdfpl "$1"',
            '  if [ ! -s "$2/Fits/$2.txt" ]; then',
            '    echo "ERROR: $1 produced no $2/Fits/$2.txt - stopping." >&2',
            "    exit 1",
            "  fi",
            "}",
            "",
            *fit_cmds,
            "# run2+run3 statistical combinations (need the per-run workspaces above).",
            "# MultiFit combines workspaces, not ntuples: n/d/p are no-ops in m mode,",
            "# so there are no combined Histograms/Plots/Tables to ask for. The bare",
            "# `m` pass draws the Compare* figures and needs the fit AND limit of both",
            "# referenced jobs to exist already.",
            *combine_cmds,
            "",
        ]
    )
    script_path = cfg_dir / "run_all.sh"
    script_path.write_text(script)
    script_path.chmod(0o755)
    # the SIMPLE runner (user request): lives at the package ROOT next to
    # ntuples/ and configs/, no guards, no path magic — a bare list of the
    # 40 commands so single lines can be copy-pasted. Run it FROM the
    # package root with trex-fitter already set up (on lxplus:
    # setupATLAS && asetup StatAnalysis,0.6.3). run_all.sh stays the
    # guarded variant.
    simple = "\n".join(
        [
            "#!/usr/bin/env bash",
            "",
            *simple_cmds,
            "# run2+run3 statistical combinations (need the fits above).",
            "# mwf = combine the two workspaces + fit, ml = limit on it,",
            "# m = the Compare* figures (must come last: it reads both",
            "# referenced jobs' fit and limit results off disk).",
            *simple_combine_cmds,
            'echo "all jobs done. fit results:"',
            "ls hh*/Fits/*.txt",
            "",
        ]
    )
    simple_path = OUT / "run_fits.sh"
    simple_path.write_text(simple)
    simple_path.chmod(0o755)
    print(
        f"\nWrote inputs to {OUT} "
        f"({n_fits} fits + {len(combine_cmds)} combines in configs/run_all.sh "
        f"and run_fits.sh, + 6 all-models convenience configs)"
    )


if __name__ == "__main__":
    main()
