"""Score the supervisor's new_data/*.root files with every trained model.

Loads the saved fold models of all 5 families (XGBoost, DNN, MLP, GNN, PNN)
for all three tracks and writes, per data file, a preselected copy with the
same 13 extra branches the MC ntuples in trexfitter/package/ carry:
score_{xgboost,dnn,mlp,gnn}[_comb], score_pnn, w_phys, label, process_id,
fold. Output goes to new_data_scored/ (kept separate from the package on
purpose).

Conventions reproduced from the notebooks (verified — see --verify):
  - preselection per channel, sentinel (< -100) -> NaN on feature columns
    (clean_data), fold = eventNumber % 5;
  - every event is scored by the fold model whose TEST fold it belongs to —
    the same rule that produced the MC OOF scores, applied unchanged to data
    (data events were never trained on, so any fold model is valid; using
    the fold rule keeps data and MC scoring conventions identical);
  - DNN/MLP: median-impute (per-fold train medians) -> StandardScaler ->
    append __isnan flags -> sigmoid(SimpleMLP logits);
  - GNN: median-impute -> scale continuous node cols -> 6/7-node fixed graph
    -> ObjectGNN (2x GATv2Conv, mean+max pool);
  - PNN: like DNN but input = [15 scaled physics, channel_id, 3 flags].
    Its preprocess pickle is missing on disk, so the per-fold scalers are
    reconstructed from the MC (deterministic: train split = folds not in
    {k, (k+1)%5}, negative weights dropped, medians+scaler on that split) —
    correctness proven by --verify reproducing the stored MC OOF scores.

--verify re-scores the MC ntuples in trexfitter/package/ntuples/ and compares
against their stored score branches for every (channel, run, model, track);
run it after any artifact regeneration. Scoring data refuses to run unless
verification has passed in the same invocation (default) or --skip-verify is
given explicitly.
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import uproot
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool

REPO = Path(__file__).resolve().parent.parent
MC_BASE = REPO / "notebooks" / "PPSSP_2026"
PKG_NTUPLES = REPO / "trexfitter" / "package" / "ntuples"
DATA_DIR = REPO / "new_data"
OUT_DIR = REPO / "new_data_scored"

TREE_NAME = "AnalysisMiniTree"
N_FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHANNELS = ("1l2tau", "2l2tau")
RUNS = ("run2", "run3")

PRESELECTION_NP = {
    "1l2tau": lambda a: (a["n_b_jet"] == 0) & (a["n_jet"] >= 2),
    "2l2tau": lambda a: (a["n_b_jet"] == 0)
    & (a["l1_charge"] * a["l2_charge"] < 0)
    & (a["mZ_cut"] > 0),
}

# score branch -> (family, track): per-run columns use the run's own track,
# _comb columns the combined track, PNN its single channel-combined model.
SCORE_COLUMNS = {
    "score_xgboost": ("xgb", "run"),
    "score_dnn": ("dnn", "run"),
    "score_mlp": ("mlp", "run"),
    "score_gnn": ("gnn", "run"),
    "score_xgboost_comb": ("xgb", "combined"),
    "score_dnn_comb": ("dnn", "combined"),
    "score_mlp_comb": ("mlp", "combined"),
    "score_gnn_comb": ("gnn", "combined"),
    "score_pnn": ("pnn", "pnn"),
}

# ---------------------------------------------------------------------------
# Model classes — verbatim architectures from the notebooks
# ---------------------------------------------------------------------------


class SimpleMLP(nn.Module):
    """DNN.ipynb / MLP.ipynb / PNN.ipynb classifier (identical in all three)."""

    def __init__(self, n_features, hidden_sizes, dropout):
        super().__init__()
        layers = []
        in_size = n_features
        for hidden_size in hidden_sizes:
            layers += [nn.Linear(in_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
            in_size = hidden_size
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ObjectGNN(nn.Module):
    """GNN_Evie_final_1l2t.ipynb / GNN_2l2tau.ipynb classifier."""

    def __init__(self, in_channels, hidden_channels, dropout):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels)
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.head(x).squeeze(-1)


# ---------------------------------------------------------------------------
# GNN node schemas — verbatim from the two GNN notebooks
# ---------------------------------------------------------------------------

GNN_SCHEMA = {
    "1l2tau": {
        "OBJECT_COLUMNS": {
            "lepton": {"pt": "l1_pt", "eta": "l1_eta", "phi": "l1_phi", "e": "l1_e", "charge": "l1_charge", "pdg": "l1_pdg"},
            "tau1": {"pt": "tau1_pt", "eta": "tau1_eta", "phi": "tau1_phi"},
            "tau2": {"pt": "tau2_pt", "eta": "tau2_eta", "phi": "tau2_phi"},
            "jet1": {"pt": "j1_pt", "eta": "j1_eta", "phi": "j1_phi", "e": "j1_e"},
            "jet2": {"pt": "j2_pt", "eta": "j2_eta", "phi": "j2_phi", "e": "j2_e"},
            "met": {"pt": "met_met", "phi": "met_phi", "e": "met_sumet"},
        },
        "NODE_ORDER": ["lepton", "tau1", "tau2", "jet1", "jet2", "met"],
        "NODE_TYPE": {"lepton": "lepton", "tau1": "tau", "tau2": "tau", "jet1": "jet", "jet2": "jet", "met": "met"},
    },
    "2l2tau": {
        "OBJECT_COLUMNS": {
            "lepton1": {"pt": "l1_pt", "eta": "l1_eta", "phi": "l1_phi", "e": "l1_e", "charge": "l1_charge", "pdg": "l1_pdg"},
            "lepton2": {"pt": "l2_pt", "eta": "l2_eta", "phi": "l2_phi", "e": "l2_e", "charge": "l2_charge", "pdg": "l2_pdg"},
            "tau1": {"pt": "tau1_pt", "eta": "tau1_eta", "phi": "tau1_phi"},
            "tau2": {"pt": "tau2_pt", "eta": "tau2_eta", "phi": "tau2_phi"},
            "jet1": {"pt": "j1_pt", "eta": "j1_eta", "phi": "j1_phi", "e": "j1_e"},
            "jet2": {"pt": "j2_pt", "eta": "j2_eta", "phi": "j2_phi", "e": "j2_e"},
            "met": {"pt": "met_met", "phi": "met_phi", "e": "met_sumet"},
        },
        "NODE_ORDER": ["lepton1", "lepton2", "tau1", "tau2", "jet1", "jet2", "met"],
        "NODE_TYPE": {"lepton1": "lepton", "lepton2": "lepton", "tau1": "tau", "tau2": "tau", "jet1": "jet", "jet2": "jet", "met": "met"},
    },
}
TYPE_LIST = ["lepton", "tau", "jet", "met"]
N_NODE_FEATURES = 7 + len(TYPE_LIST)
GNN_BATCH = 8192


def mask_sentinels(X):
    """clean_data's sentinel handling: values < -100 (e.g. -999) -> NaN."""
    return X.where(~(X < -100))


def fold_models_dir(channel, track):
    return MC_BASE / channel / track


# ---------------------------------------------------------------------------
# Per-family scorers. All take df (sentinel handling done inside on the
# feature columns they use) with a `fold` column, and return per-event scores
# in [0, 1] via the fold-model rule: model k scores rows with fold == k.
# ---------------------------------------------------------------------------


def score_xgb(df, channel, track):
    d = fold_models_dir(channel, track)
    features = json.load(open(d / f"features_{track}.json"))
    X = mask_sentinels(df[features])
    out = np.full(len(df), np.nan)
    for k in range(N_FOLDS):
        rows = (df["fold"] == k).to_numpy()
        if not rows.any():
            continue
        model = xgb.XGBClassifier()
        model.load_model(d / f"model_{track}_fold{k}.json")
        model.set_params(device="cpu")
        out[rows] = model.predict_proba(X.loc[rows])[:, 1]
    return out


def _torch_scores(model, X_np, batch=65536):
    model.eval()
    parts = []
    with torch.no_grad():
        for s in range(0, len(X_np), batch):
            t = torch.tensor(X_np[s : s + batch], dtype=torch.float32, device=DEVICE)
            parts.append(torch.sigmoid(model(t)).cpu().numpy())
    return np.concatenate(parts) if parts else np.empty(0)


def score_nn(df, channel, track, suffix):
    """DNN and MLP (suffix 'dnn'/'mlp')."""
    d = fold_models_dir(channel, track)
    meta = json.load(open(d / f"features_{track}_{suffix}.json"))
    features, flag_cols = meta["features"], meta["flag_cols"]
    hp = json.load(open(d / f"hyperparams_{track}_{suffix}.json"))
    with open(d / f"preprocess_{track}_{suffix}.pkl", "rb") as f:
        pp = pickle.load(f)
    X_raw = mask_sentinels(df[features])
    n_in = len(features) + len(flag_cols)
    out = np.full(len(df), np.nan)
    for k in range(N_FOLDS):
        rows = (df["fold"] == k).to_numpy()
        if not rows.any():
            continue
        Xr = X_raw.loc[rows]
        X_imp = Xr.fillna(pp["medians"][k])
        X_scaled = pp["scalers"][k].transform(X_imp)
        flags = Xr[flag_cols].isna().astype(np.float32).to_numpy() if flag_cols else np.empty((rows.sum(), 0), np.float32)
        model = SimpleMLP(n_in, hp["hidden_sizes"], hp["dropout"]).to(DEVICE)
        model.load_state_dict(torch.load(d / f"model_{track}_fold{k}_{suffix}.pt", map_location=DEVICE))
        out[rows] = _torch_scores(model, np.hstack([X_scaled, flags]))
    return out


def _stack_node_features(df_scaled, df_imp, schema):
    n = len(df_scaled)
    node_arrays = []
    for name in schema["NODE_ORDER"]:
        cols = schema["OBJECT_COLUMNS"][name]
        pt = df_scaled[cols["pt"]].to_numpy(dtype=np.float32)
        eta = df_scaled[cols["eta"]].to_numpy(dtype=np.float32) if "eta" in cols else np.zeros(n, dtype=np.float32)
        phi = df_imp[cols["phi"]].to_numpy(dtype=np.float32)
        e = df_scaled[cols["e"]].to_numpy(dtype=np.float32) if "e" in cols else np.zeros(n, dtype=np.float32)
        charge = df_imp[cols["charge"]].to_numpy(dtype=np.float32) if "charge" in cols else np.zeros(n, dtype=np.float32)
        is_electron = (
            (np.abs(df_imp[cols["pdg"]].to_numpy()) == 11).astype(np.float32)
            if "pdg" in cols else np.zeros(n, dtype=np.float32)
        )
        type_onehot = np.tile(
            np.array([float(schema["NODE_TYPE"][name] == t) for t in TYPE_LIST], dtype=np.float32), (n, 1)
        )
        node_arrays.append(np.column_stack([pt, eta, np.sin(phi), np.cos(phi), e, charge, is_electron, type_onehot]))
    return np.stack(node_arrays, axis=1)


def score_gnn(df, channel, track):
    d = fold_models_dir(channel, track)
    schema = GNN_SCHEMA[channel]
    features = json.load(open(d / f"features_{track}_gnn.json"))["features"]
    cont_cols = sorted({
        c for cols in schema["OBJECT_COLUMNS"].values() for key, c in cols.items() if key in ("pt", "eta", "e")
    })
    hp = json.load(open(d / f"hyperparams_{track}_gnn.json"))
    with open(d / f"preprocess_{track}_gnn.pkl", "rb") as f:
        pp = pickle.load(f)
    n_nodes = len(schema["NODE_ORDER"])
    edge_index_1 = torch.tensor(
        [[i, j] for i in range(n_nodes) for j in range(n_nodes) if i != j], dtype=torch.long
    ).t().contiguous().to(DEVICE)

    X_raw = mask_sentinels(df[features])
    out = np.full(len(df), np.nan)
    for k in range(N_FOLDS):
        rows = (df["fold"] == k).to_numpy()
        if not rows.any():
            continue
        imp = X_raw.loc[rows].fillna(pp["medians"][k])
        scaled = imp.copy()
        scaled[cont_cols] = pp["scalers"][k].transform(imp[cont_cols])
        assert np.isfinite(scaled.to_numpy()).all(), "NaN/inf reached the GNN input"
        x_all = torch.from_numpy(_stack_node_features(scaled, imp, schema)).to(DEVICE)
        model = ObjectGNN(N_NODE_FEATURES, hp["hidden_channels"], hp["dropout"]).to(DEVICE)
        model.load_state_dict(torch.load(d / f"model_{track}_fold{k}_gnn.pt", map_location=DEVICE))
        model.eval()
        parts = []
        with torch.no_grad():
            for s in range(0, x_all.shape[0], GNN_BATCH):
                bx = x_all[s : s + GNN_BATCH]
                b = bx.shape[0]
                offs = torch.arange(b, device=DEVICE).repeat_interleave(edge_index_1.shape[1]) * n_nodes
                eidx = edge_index_1.repeat(1, b) + offs
                bvec = torch.arange(b, device=DEVICE).repeat_interleave(n_nodes)
                logits = model(bx.reshape(b * n_nodes, N_NODE_FEATURES), eidx, bvec)
                parts.append(torch.sigmoid(logits).cpu().numpy())
        out[rows] = np.concatenate(parts)
    return out


# ---------------------------------------------------------------------------
# PNN — preprocess pickle missing on disk; reconstruct it from the MC
# ---------------------------------------------------------------------------

PNN_DIR = MC_BASE / "pnn_channel_combined" / "combined"
_pnn_cache = None


def pnn_meta():
    meta = json.load(open(PNN_DIR / "features_combined.json"))
    physics = [f for f in meta["features"] if f != "channel_id"]
    return physics, meta["flag_cols"], json.load(open(PNN_DIR / "hyperparams_combined.json"))


def reconstruct_pnn_preprocess():
    """Recompute PNN's per-fold medians + scalers from the MC ntuples.

    Deterministic reproduction of prepare_fold_data + prepare_fold_tensors_pnn:
    train split of fold k = events with fold not in {k, (k+1)%5} and
    w_phys >= 0, over BOTH channels and BOTH runs pooled (the PNN's training
    dataset); medians over the 15 physics features of that split, scaler fit
    on the median-imputed split. Verified against the stored OOF scores.
    """
    global _pnn_cache
    if _pnn_cache is not None:
        return _pnn_cache
    physics, _flags, _hp = pnn_meta()
    frames = []
    for channel in CHANNELS:
        for run in RUNS:
            for f in sorted((PKG_NTUPLES / channel / run).glob("*.root")):
                t = uproot.open(f)[TREE_NAME]
                arr = t.arrays([*physics, "w_phys", "fold"], library="pd")
                arr["channel_id"] = 0.0 if channel == "1l2tau" else 1.0
                frames.append(arr)
    data = pd.concat(frames, ignore_index=True)
    data[physics] = mask_sentinels(data[physics])
    scalers, medians = [], []
    for k in range(N_FOLDS):
        val_fold = (k + 1) % N_FOLDS
        train = data.loc[~data["fold"].isin([k, val_fold]) & (data["w_phys"] >= 0)]
        med = train[physics].median()
        scaler = StandardScaler().fit(train[physics].fillna(med))
        medians.append(med)
        scalers.append(scaler)
    _pnn_cache = {"scalers": scalers, "medians": medians}
    return _pnn_cache


def score_pnn(df, channel):
    physics, flag_cols, hp = pnn_meta()
    pp = reconstruct_pnn_preprocess()
    X_raw = mask_sentinels(df[physics])
    channel_id = np.full((len(df), 1), 0.0 if channel == "1l2tau" else 1.0, np.float32)
    n_in = len(physics) + 1 + len(flag_cols)
    out = np.full(len(df), np.nan)
    for k in range(N_FOLDS):
        rows = (df["fold"] == k).to_numpy()
        if not rows.any():
            continue
        Xr = X_raw.loc[rows]
        X_scaled = pp["scalers"][k].transform(Xr.fillna(pp["medians"][k]))
        flags = Xr[flag_cols].isna().astype(np.float32).to_numpy()
        model = SimpleMLP(n_in, hp["hidden_sizes"], hp["dropout"]).to(DEVICE)
        model.load_state_dict(torch.load(PNN_DIR / f"model_combined_fold{k}.pt", map_location=DEVICE))
        out[rows] = _torch_scores(model, np.hstack([X_scaled, channel_id[rows], flags]))
    return out


def score_all(df, channel, run):
    """All 9 score columns for one (channel, run) DataFrame."""
    scores = {}
    for col, (family, track_kind) in SCORE_COLUMNS.items():
        track = {"run": run, "combined": "combined", "pnn": None}[track_kind]
        if family == "xgb":
            scores[col] = score_xgb(df, channel, track)
        elif family in ("dnn", "mlp"):
            scores[col] = score_nn(df, channel, track, family)
        elif family == "gnn":
            scores[col] = score_gnn(df, channel, track)
        else:
            scores[col] = score_pnn(df, channel)
        assert np.isfinite(scores[col]).all(), f"non-finite {col}"
    return scores


# ---------------------------------------------------------------------------
# Verification: re-score the MC and compare with the stored OOF branches
# ---------------------------------------------------------------------------

TOLERANCE = {"xgb": 5e-6, "dnn": 5e-5, "mlp": 5e-5, "gnn": 5e-4, "pnn": 5e-5}


def verify():
    print(f"== VERIFICATION (device: {DEVICE}) ==")
    all_ok = True
    for channel in CHANNELS:
        for run in RUNS:
            files = sorted((PKG_NTUPLES / channel / run).glob("*.root"))
            branches = None
            frames = []
            for f in files:
                t = uproot.open(f)[TREE_NAME]
                if branches is None:
                    branches = [b for b in t.keys()]
                frames.append(t.arrays(branches, library="pd"))
            df = pd.concat(frames, ignore_index=True)
            new = score_all(df, channel, run)
            for col, arr in new.items():
                family = SCORE_COLUMNS[col][0]
                stored = df[col].to_numpy()
                dmax = np.abs(arr - stored).max()
                ok = dmax <= TOLERANCE[family]
                all_ok &= ok
                print(f"  {channel}/{run} {col:20s} max|new-stored| = {dmax:.2e} "
                      f"{'OK' if ok else f'FAIL (tol {TOLERANCE[family]:.0e})'}")
    print("== VERIFICATION", "PASSED ==" if all_ok else "FAILED ==")
    return all_ok


# ---------------------------------------------------------------------------
# Data scoring
# ---------------------------------------------------------------------------


def score_data_file(channel, run):
    src = DATA_DIR / f"data_{channel}_{run}.root"
    t = uproot.open(src)[TREE_NAME]
    df = t.arrays(t.keys(), library="pd")
    n_all = len(df)
    df = df.loc[PRESELECTION_NP[channel](df)].reset_index(drop=True)
    df["fold"] = (df["eventNumber"] % N_FOLDS).astype(np.int32)
    scores = score_all(df, channel, run)

    out = {name: df[name].to_numpy() for name in t.keys()}
    for col in SCORE_COLUMNS:
        out[col] = scores[col].astype(np.float64)
    out["w_phys"] = (df["weight"] * df["weights"]).to_numpy(np.float64)  # data: 1.0
    out["label"] = np.full(len(df), -1, np.int32)
    out["process_id"] = np.full(len(df), -1, np.int32)
    out["fold"] = df["fold"].to_numpy(np.int32)

    OUT_DIR.mkdir(exist_ok=True)
    dst = OUT_DIR / f"data_{channel}_{run}.root"
    with uproot.recreate(dst) as fout:
        fout.mktree(TREE_NAME, {k: v.dtype for k, v in out.items()})
        fout[TREE_NAME].extend(out)
    print(f"{dst.name}: {n_all} -> {len(df)} preselected events, "
          f"{len(SCORE_COLUMNS)} score branches written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true", help="only run the MC verification")
    ap.add_argument("--skip-verify", action="store_true", help="score data without the MC verification")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message=".*mismatched devices.*")

    if not args.skip_verify:
        if not verify():
            raise SystemExit("MC verification FAILED - not scoring data. Investigate first.")
    if args.verify_only:
        return
    print("\n== SCORING DATA ==")
    for channel in CHANNELS:
        for run in RUNS:
            score_data_file(channel, run)


if __name__ == "__main__":
    main()
