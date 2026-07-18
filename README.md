# MVA-Project

This repository trains XGBoost BDTs to separate Higgs signal (`ggF` + `VBF`)
from Standard Model backgrounds, using ATLAS-style ROOT ntuples, for two
channels (`1l2tau`, `2l2tau`) and two data-taking campaigns (Run 2, Run 3).

The two canonical, ready-to-run pipelines are:

- [`Final_Notebooks/1L2Tau_Master_Pipeline.ipynb`](Final_Notebooks/1L2Tau_Master_Pipeline.ipynb) — 1 lepton + 2 tau channel
- [`Final_Notebooks/2L2Tau_Master_Pipeline.ipynb`](Final_Notebooks/2L2Tau_Master_Pipeline.ipynb) — 2 lepton + 2 tau channel

They replace the earlier, duplicated exploration notebooks under `Esteban/`
and `Evie/` (`First_Training.ipynb`, `Combined_Runs.ipynb`, `Comparison.ipynb`,
`Yields.ipynb`, `1L2TRun2_combo.ipynb`, `HISTOALL2.ipynb`, ...), which are kept
around for reference/history but are no longer the recommended entry point.
Both master pipelines share the exact same architecture and section layout —
see [Master pipelines](#master-pipelines) below for full documentation.


## ROOT File Meaning

| Sample | Meaning |
| --- | --- |
| Diboson | Diboson background processes, for example WW/WZ/ZZ. |
| signal_ggF | Signal sample produced via gluon-gluon fusion (ggF). |
| signal_VBF | Signal sample produced via vector boson fusion (VBF). |
| SingleH | Single-Higgs background sample. |
| tops | Single-top and related top backgrounds. |
| tt_bar / ttbar | Top pair background sample. |
| Vgamma | Vector boson plus photon background. |
| VVV | Triboson background processes. |
| Wjets | W plus jets background. |
| Zjets | Z plus jets background. |


## Cuts Needed

### 1 Lepton 2 Taus

- n_b_jet == 0
- n_jet >= 2

### 2 Lepton 2 Taus

- n_b_jet == 0
- 2l_charge * 1l_charge < 0 or 2l_charge != 1l_charge
- mZ_cut > 0

## Master Pipelines

Both `Final_Notebooks/1L2Tau_Master_Pipeline.ipynb` and
`Final_Notebooks/2L2Tau_Master_Pipeline.ipynb` are self-contained, documented
notebooks built around the same idea: the loading / cleaning / training /
pruning / tuning logic is written **once** as a small library of helper
functions, and every run (Run 2, Run 3, Combined, domain-shift check) just
calls those functions instead of copy-pasting the logic. This keeps every
track consistent by construction instead of by copy-paste discipline. The
2l2tau notebook is a channel-specific mirror of the 1l2tau one — same section
layout, same helper functions, only the preselection, blocklist and paths
differ where the channel requires it.

Each notebook is written to be run top-to-bottom against its own
`PPSSP_2026/<channel>/{run2,run3}/*.root` files (tree `AnalysisMiniTree`).

### Common setup

| | 1L2Tau | 2L2Tau |
| --- | --- | --- |
| Processes (label = 1) | `signal_ggF`, `signal_VBF` | `signal_ggF`, `signal_VBF` |
| Processes (label = 0) | `Diboson`, `Zjets`, `Wjets`, `ttbar`, `tops`, `SingleH`, `Vgamma`, `VVV` | same 8 backgrounds |
| Preselection | `n_b_jet == 0` & `n_jet >= 2` | `n_b_jet == 0` & `l1_charge * l2_charge < 0` & `mZ_cut > 0` |
| Event weight | `w_phys = weight * weights` (`weight` = normalization, `weights` = Sherpa NLO generator weight, can be negative); `\|w_phys\|` is used for training/AUC, the sign only matters for yields | same convention |

**Leakage-free feature policy** (shared by both notebooks): candidate features
are discovered from branches common to every input file, then filtered by a
blocklist so the model can't cheat using truth/weight/efficiency information —
blocked substrings include `weight`, `effsf`, `_ff`, `truth`, `istrue`, `fake`,
`anti`, `dsid`, `eventnumber`, `_RNNTight`, `_isOS`, `_d0sig`, plus an explicit
blocklist of exact branch names (e.g. `n_b_jet`, the channel's `pass1l2tau` /
`pass2l2tau` flag, `hhml_subchannelflavor`, RNN tau-ID scores, OS-pair flags).
Constant/empty columns are auto-dropped. Sentinel values below `-100`
(ATLAS convention, e.g. `-999`) are masked to `NaN`, which XGBoost handles
natively via its learned per-split default direction.

### Notebook sections

Both notebooks follow the same seven sections:

1. **Setup & global configuration** — paths, tree name, preselection cut and
   the leakage-free feature policy, defined once and shared by every section
   below.
2. **Shared helper function library** — the only place the logic is written:
   - `to_device` — moves data onto the GPU so XGBoost never falls back to a
     CPU↔GPU copy during training/eval/predict
   - `discover_common_features`, `load_run_data`, `clean_data` — I/O + cleaning
   - `compute_yields` — per-process yield / S over B table
   - `make_train_val_split`, `make_fit_weights` — splitting & training weights
   - `train_xgb_baseline`, `get_importance`, `plot_importance_bar` — baseline model
   - `top_pairs`, `grouped_correlations`, `plot_group_correlations`,
     `prune_correlated` — correlation-based feature pruning (drop a feature if
     `\|corr\| > 0.75` with an already-kept, more important feature)
   - `run_optuna_search`, `train_final_tuned_model`, `plot_optuna_diagnostics` —
     GPU-aware hyperparameter tuning (Optuna TPE sampler, `MedianPruner`,
     objective = mean weighted AUC over stratified k-fold CV)
3. **Run 2 — solo track** — load → yields → baseline model → correlation
   pruning → Optuna tuning, using only Run 2 files.
4. **Run 3 — solo track** — identical procedure to Section 3, independently on
   Run 3 only (its own feature-discovery pass, in case branches differ
   slightly between campaigns).
5. **Run 2 + Run 3 — combined track** — concatenates both signals and all
   backgrounds from both campaigns into one training sample and reruns the
   same pipeline (feature list rebuilt from branches common to *all* files
   across both runs; a bookkeeping `run` column is kept, never used as a
   feature, so train/val can be stratified on `label` and `run` jointly).
   Includes a top-15-features cut and a comparison of how each track's
   feature ranking shifts between Run 2-only / Run 3-only / Combined.
6. **Run 2 vs Run 3 domain-shift check** — a separate, lightweight classifier
   whose only job is telling Run 2 signal events apart from Run 3 signal
   events, reusing the Combined track's tuned feature list (not its model).
   `AUC ≈ 0.5` means the two campaigns are indistinguishable in the features
   that matter; `AUC - 0.5` is roughly the effect size of any real shift. Also
   repeats the same two-sample test per-process to localize any shift to
   specific backgrounds.
7. **Summary** — consolidated AUC comparison across tracks and a table of
   every artifact written to disk.

### Outputs

All paths below are relative to `Final_Notebooks/`, with `<ch>` = `1l2tau` or
`2l2tau` and the 2l2tau top-level domain-shift outputs suffixed `_2l2tau`:

| Path | Contents |
| --- | --- |
| `PPSSP_2026/<ch>/run2/final_model_run2.json` | Run-2 tuned XGBoost model |
| `PPSSP_2026/<ch>/run2/optuna_features_run2.json` | Run-2 tuned feature list |
| `PPSSP_2026/<ch>/run2/splits/{train,val}.root` | Run-2 train/val split |
| `PPSSP_2026/<ch>/run2/plots/*.png` | Run-2 diagnostic plots (importance, correlation, Optuna) |
| `PPSSP_2026/<ch>/run3/final_model_run3.json` | Run-3 tuned XGBoost model |
| `PPSSP_2026/<ch>/run3/optuna_features_run3.json` | Run-3 tuned feature list |
| `PPSSP_2026/<ch>/run3/splits/{train,val}.root` | Run-3 train/val split |
| `PPSSP_2026/<ch>/run3/plots/*.png` | Run-3 diagnostic plots (importance, correlation, Optuna) |
| `PPSSP_2026/<ch>/combined/final_model_combined.json` | Combined tuned XGBoost model |
| `PPSSP_2026/<ch>/combined/optuna_features_combined.json` | Combined tuned feature list |
| `PPSSP_2026/<ch>/combined/plots/*.png` | Combined diagnostic plots (importance, correlation, Optuna, model comparison) |
| `run2_vs_run3_auc_summary(_2l2tau).csv` | Per-process Run2/Run3 domain-shift AUC |
| `run2_vs_run3_plots(_2l2tau)/*.png` | Signal domain-shift importance/SHAP plots + per-process feature-importance plots |
| `PPSSP_2026/<ch>/summary_plots/AUCSummaryByTrack.png` | Consolidated AUC comparison across tracks |

