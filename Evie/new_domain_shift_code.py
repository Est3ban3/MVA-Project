# ---- Domain-shift plot: can Run 2 and Run 3 be pooled? ----------------------
# Reframed around the decision actually being made. The question is not "is
# there a detectable shift" - at these sample sizes there always is, and the
# z-scores just measure how much simulation we generated. The question is
# whether the shift is SMALL ENOUGH to train on the combined sample, which is
# a question about the size of the AUC. So AUC is what the plot encodes:
# position against labelled tolerance bands, value printed on every row.
# z-scores are dropped entirely - they scaled with statistics, not with shift,
# and flagged our best-measured processes (Diboson) while clearing our worst-
# measured ones (Vgamma), which is backwards for a pooling decision.

AUC_NEGLIGIBLE = 0.55   # below: periods effectively interchangeable
AUC_TOLERABLE  = 0.60   # above: don't pool without reweighting / domain adaptation

is_signal_process = results_df["process"].map(lambda p: FILES.get(p, (None, 0))[1] == 1)
bkg_df = results_df.loc[~is_signal_process].dropna(subset=["ci_lo", "ci_hi"]).copy()

signal_row = pd.DataFrame([{
    "process": "Signal (ggF+VBF)", "mean_auc": fold_aucs.mean(),
    "ci_lo": ci_lo_signal, "ci_hi": ci_hi_signal,
}])

# Ascending sort + upward y-axis puts the largest shift at the top, where the
# eye lands first - those are the rows that decide the answer.
plot_df = pd.concat(
    [bkg_df[["process", "mean_auc", "ci_lo", "ci_hi"]], signal_row], ignore_index=True
).sort_values("mean_auc").reset_index(drop=True)

y_pos = np.arange(len(plot_df))
mean_auc = plot_df["mean_auc"].to_numpy()
ci_lo = plot_df["ci_lo"].to_numpy()
ci_hi = plot_df["ci_hi"].to_numpy()
is_signal = (plot_df["process"] == "Signal (ggF+VBF)").to_numpy()
is_watch = mean_auc >= AUC_NEGLIGIBLE

INK, MUTED = "#22201d", "#6b6862"
C_LOW, C_WATCH = "#6BA3C7", "#1F5C8B"   # one hue, two steps - redundant with position

fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.46 * len(plot_df))))

# Tolerance bands carry the verdict; alpha kept low so the marks stay dominant.
ax.axvspan(0.50, AUC_NEGLIGIBLE, color="#4C9A6A", alpha=0.10, zorder=0)
ax.axvspan(AUC_NEGLIGIBLE, AUC_TOLERABLE, color="#BA7517", alpha=0.10, zorder=0)
ax.axvspan(AUC_TOLERABLE, 0.70, color="#A62B1F", alpha=0.10, zorder=0)
for x, lab in [(AUC_NEGLIGIBLE, f"{AUC_NEGLIGIBLE:g}"), (AUC_TOLERABLE, f"{AUC_TOLERABLE:g}")]:
    ax.axvline(x, color=MUTED, lw=1, ls=":", zorder=1)
ax.axvline(0.5, color=INK, lw=1.2, ls="--", zorder=1)

for label, colour, mask, marker, size in [
    ("background, negligible (< 0.55)", C_LOW,   ~is_watch & ~is_signal, "o", 55),
    ("background, mild (0.55 - 0.60)",  C_WATCH,  is_watch & ~is_signal, "o", 55),
    ("signal (ggF+VBF, combined)",      C_WATCH if is_watch[is_signal][0] else C_LOW,
                                                              is_signal, "D", 95),
]:
    if not mask.any():
        continue
    ax.hlines(y_pos[mask], 0.5, mean_auc[mask], color=colour, lw=1.5, zorder=2)
    ax.errorbar(mean_auc[mask], y_pos[mask],
                xerr=[mean_auc[mask] - ci_lo[mask], ci_hi[mask] - mean_auc[mask]],
                fmt="none", ecolor=colour, capsize=3, lw=1.2, zorder=2)
    ax.scatter(mean_auc[mask], y_pos[mask], color=colour, s=size, marker=marker,
               zorder=3, label=label, edgecolor="white", linewidth=0.8)

# Direct-label every row with the number being read - no legend lookup needed.
for yi in y_pos:
    ax.text(ci_hi[yi] + 0.0035, yi, f"{mean_auc[yi]:.3f}", va="center",
            fontsize=9, color=INK)

x_max = max(AUC_TOLERABLE + 0.012, ci_hi.max() + 0.022)
ax.set(yticks=y_pos, yticklabels=plot_df["process"], xlim=(0.492, x_max),
    ylim=(-0.8, len(plot_df) - 0.2),
    xlabel="Run 2 vs Run 3 domain AUC  (0.5 = indistinguishable; bootstrap 95% CI)")

ax.text(0.5015, len(plot_df) - 0.55, "negligible", fontsize=8, color=MUTED, style="italic")
ax.text(AUC_NEGLIGIBLE + 0.0015, len(plot_df) - 0.55, "mild", fontsize=8, color=MUTED, style="italic")
if x_max > AUC_TOLERABLE:
    ax.text(AUC_TOLERABLE + 0.0015, len(plot_df) - 0.55, "don't pool", fontsize=8,
            color=MUTED, style="italic")

worst = plot_df.iloc[-1]
verdict = ("supports pooling" if mean_auc.max() < AUC_TOLERABLE else "pooling NOT supported")
ax.set_title(f"Run 2 / Run 3 compatibility by process — {verdict}\n"
            f"largest shift: {worst['process']} at AUC {worst['mean_auc']:.3f} "
            f"(all < {AUC_TOLERABLE:g})", fontsize=11, loc="left")

ax.grid(axis="x", alpha=0.25, zorder=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.95)
plt.tight_layout()

plot_path = PLOTS_DIR_SUMMARY / "DomainShiftPoolingCheck.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Saved plot -> {plot_path}")
plt.show()

watch = plot_df.loc[plot_df["mean_auc"] >= AUC_NEGLIGIBLE, ["process", "mean_auc"]]
print(f"\nMax domain AUC = {mean_auc.max():.3f}  (tolerance {AUC_TOLERABLE:g})")
print(f"Verdict: {verdict}")
print(f"\nMild shift, worth monitoring (AUC >= {AUC_NEGLIGIBLE:g}):")
print(watch.to_string(index=False) if len(watch) else "  none")