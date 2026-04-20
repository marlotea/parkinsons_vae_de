"""
Diagnostic comparison figures following Murphy et al. (2023) eLife framework.

Produces five publication-quality figures:
  1. DEG count vs cell count (Murphy diagnostic) with permutation controls
  2. LFC correlation between Wilcoxon and scVI
  3. QC impact: cells retained under each strategy
  4. DEG overlap (UpSet / Venn)
  5. scVI latent space batch mixing vs PCA
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
import anndata as ad

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

PD_GWAS_GENES = ["SNCA", "LRRK2", "MAPT", "GBA", "PINK1", "PARK7"]
N_PERMUTATIONS = 100
RANDOM_SEED = 42


def load_results() -> dict:
    """
    Load all available DE result tables and AnnData objects.

    Returns:
        Dict with keys: wilcoxon_df, pseudobulk_df, scvi_df,
        adata_raw, adata_fern, adata_bp, adata_clustered, adata_scvi.
    """
    results = {}

    def safe_load_csv(path, index_col=None):
        return pd.read_csv(path, index_col=index_col) if path.exists() else None

    def safe_load_h5ad(path):
        return sc.read_h5ad(path) if path.exists() else None

    results["wilcoxon_df"] = safe_load_csv(RESULTS_DIR / "degs_wilcoxon_per_cluster.csv")
    results["pseudobulk_df"] = safe_load_csv(RESULTS_DIR / "degs_pseudobulk_edger.csv")
    results["scvi_df"] = safe_load_csv(RESULTS_DIR / "degs_scvi.csv", index_col=0)
    results["adata_raw"] = safe_load_h5ad(PROCESSED_DIR / "adata_raw.h5ad")
    results["adata_fern"] = safe_load_h5ad(PROCESSED_DIR / "adata_qc_fernandes.h5ad")
    results["adata_bp"] = safe_load_h5ad(PROCESSED_DIR / "adata_qc_bestpractice.h5ad")
    results["adata_clustered"] = safe_load_h5ad(PROCESSED_DIR / "adata_fernandes_clustered.h5ad")
    results["adata_scvi"] = safe_load_h5ad(PROCESSED_DIR / "adata_scvi.h5ad")

    for k, v in results.items():
        status = "loaded" if v is not None else "NOT FOUND"
        log.info("  %-25s %s", k, status)

    return results


def run_wilcoxon_permutation(adata_cluster: ad.AnnData, n_perm: int = N_PERMUTATIONS) -> list[int]:
    """
    Run Wilcoxon DE with permuted genotype labels to generate a null distribution.

    Args:
        adata_cluster: AnnData for one cluster (untreated cells only).
        n_perm: Number of permutations.

    Returns:
        List of DEG counts per permutation.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    perm_deg_counts = []
    genotypes = adata_cluster.obs["genotype"].values.copy()

    for _ in range(n_perm):
        shuffled = rng.permutation(genotypes)
        adata_perm = adata_cluster.copy()
        adata_perm.obs["genotype"] = shuffled

        if (adata_perm.obs["genotype"] == "WT").sum() < 5 or \
           (adata_perm.obs["genotype"] == "SNCA_A53T").sum() < 5:
            perm_deg_counts.append(0)
            continue

        sc.tl.rank_genes_groups(
            adata_perm, groupby="genotype", groups=["SNCA_A53T"], reference="WT",
            method="wilcoxon", key_added="rank_perm", use_raw=False,
        )
        res = sc.get.rank_genes_groups_df(
            adata_perm, group="SNCA_A53T", key="rank_perm", pval_cutoff=1.0
        )
        n_de = ((res["pvals_adj"] < 0.05) & (res["logfoldchanges"].abs() > 0.25)).sum()
        perm_deg_counts.append(n_de)

    return perm_deg_counts


def figure1_deg_vs_cells(results: dict, out_path: Path):
    """
    Three-panel Murphy et al. diagnostic figure.

    Panel A: Wilcoxon DEG count vs cluster cell count with permutation null —
             the actual Murphy diagnostic showing spurious inflation.
    Panel B: Biological replicate count per genotype × cell type — explains
             why pseudobulk edgeR cannot run (n=1 per group).
    Panel C: scVI volcano — Bayes factor vs log2(mean_A53T / mean_WT), colored
             by significance. Replaces a per-cluster scatter that is impossible
             with a single aggregate comparison.

    Args:
        results: Dict from load_results().
        out_path: Output PDF path.
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        "Figure 1 — Murphy et al. Diagnostic & Method Comparison",
        fontsize=12, fontweight="bold",
    )

    # ── Panel A: Wilcoxon DEG count vs cell count ──────────────────────────
    ax = axes[0]
    ax.set_title("A  Wilcoxon: DEG count vs cluster size\n(pseudoreplication diagnostic)",
                 fontsize=9)
    ax.set_xlabel("Cells in cluster (untreated)")
    ax.set_ylabel("DEGs (FDR < 0.05, |LFC| > 0.25)")

    wil_df = results["wilcoxon_df"]
    if wil_df is not None and "n_cells_total" in wil_df.columns:
        summary = (
            wil_df.groupby("cluster")
            .agg(
                n_cells=("n_cells_total", "first"),
                n_degs=("pvals_adj", lambda x: (
                    (x < 0.05) & (wil_df.loc[x.index, "logfoldchanges"].abs() > 0.25)
                ).sum()),
            )
            .reset_index()
        )
        x = summary["n_cells"].values
        y = summary["n_degs"].values
        ax.scatter(x, y, c="coral", s=60, zorder=3, label="Observed")

        if len(x) > 1:
            r, p = stats.pearsonr(x, y)
            m, b = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, m * x_line + b, "--", color="black", lw=1, alpha=0.6)
            ax.text(0.05, 0.95, f"r = {r:.3f},  p = {p:.2e}",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        # Permutation null — use adata_clustered (scaled HVG, no .raw needed)
        adata_c = results["adata_clustered"]
        if adata_c is not None:
            log.info("Running %d permutations for Wilcoxon null...", N_PERMUTATIONS)
            untreated = adata_c.obs["condition"] == "untreated"
            perm_ys = []
            for cluster_id in summary["cluster"]:
                mask = untreated & (adata_c.obs["louvain"] == cluster_id)
                adata_cl = adata_c[mask].copy()
                if adata_cl.n_obs < 20:
                    perm_ys.append([0] * N_PERMUTATIONS)
                    continue
                perms = run_wilcoxon_permutation(adata_cl, n_perm=N_PERMUTATIONS)
                perm_ys.append(perms)

            for xi, perms in zip(x, perm_ys):
                jitter = np.random.default_rng(RANDOM_SEED).normal(0, max(xi * 0.02, 1), len(perms))
                ax.scatter(xi + jitter, perms, c="gray", s=4, alpha=0.15, zorder=1)

            ax.legend(handles=[
                mpatches.Patch(color="coral", label="Observed"),
                mpatches.Patch(color="gray", alpha=0.4, label="Permuted labels"),
            ], fontsize=7)
    else:
        ax.text(0.5, 0.5, "No Wilcoxon data.\nRun step 03 first.",
                ha="center", va="center", transform=ax.transAxes, color="gray")

    # ── Panel B: Biological replicate counts ───────────────────────────────
    ax = axes[1]
    ax.set_title("B  Biological replicates per group\n(pseudobulk edgeR requirement)",
                 fontsize=9)

    pb_meta_path = RESULTS_DIR / "pseudobulk_metadata.csv"
    if pb_meta_path.exists():
        meta = pd.read_csv(pb_meta_path)
        untreated_meta = meta[meta["condition"] == "untreated"]
        rep_counts = (
            untreated_meta.groupby(["cell_type", "genotype"])
            .size()
            .reset_index(name="n_replicates")
        )

        cell_types = rep_counts["cell_type"].unique()
        genotypes = ["WT", "SNCA_A53T"]
        colors_geno = {"WT": "steelblue", "SNCA_A53T": "coral"}
        x_pos = np.arange(len(cell_types))
        w = 0.35

        for i, geno in enumerate(genotypes):
            sub = rep_counts[rep_counts["genotype"] == geno].set_index("cell_type")
            vals = [sub.loc[ct, "n_replicates"] if ct in sub.index else 0 for ct in cell_types]
            ax.bar(x_pos + (i - 0.5) * w, vals, w,
                   label=geno, color=colors_geno[geno], alpha=0.8)

        ax.axhline(3, color="black", ls="--", lw=1.2, label="Min. replicates needed (n=3)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(cell_types, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("iPSC line replicates per group")
        ax.legend(fontsize=7)
        ax.set_ylim(0, max(4, rep_counts["n_replicates"].max() + 1))
        ax.text(0.5, 0.92,
                "n=1 per genotype → edgeR underpowered",
                transform=ax.transAxes, ha="center", fontsize=8, color="darkred",
                bbox=dict(boxstyle="round", facecolor="mistyrose", alpha=0.8))
    else:
        ax.text(0.5, 0.5, "pseudobulk_metadata.csv not found.\nRun step 04 first.",
                ha="center", va="center", transform=ax.transAxes, color="gray")

    # ── Panel C: scVI volcano ──────────────────────────────────────────────
    ax = axes[2]
    ax.set_title("C  scVI: Bayes factor vs log₂FC (DAn1)\n(generative model DE)",
                 fontsize=9)
    ax.set_xlabel("log₂FC  (A53T / WT,  from normalized means)")
    ax.set_ylabel("Bayes factor")

    scvi_df = results["scvi_df"]
    if scvi_df is not None and "bayes_factor" in scvi_df.columns:
        lfc = np.log2(
            (scvi_df["raw_normalized_mean1"] + 1e-6) /
            (scvi_df["raw_normalized_mean2"] + 1e-6)
        )
        bf = scvi_df["bayes_factor"]
        sig_col = next((c for c in ["is_de_fdr_0.05", "is_de"] if c in scvi_df.columns), None)
        sig = scvi_df[sig_col].astype(bool) if sig_col else pd.Series(False, index=scvi_df.index)

        point_colors = np.where(sig, "mediumseagreen", "lightgray")
        ax.scatter(lfc, bf, c=point_colors, s=6, alpha=0.6, linewidths=0)

        # Threshold lines
        ax.axhline(0.5, color="black", ls="--", lw=0.8, alpha=0.6, label="BF = 0.5")
        ax.axvline(0, color="black", lw=0.5, alpha=0.4)

        # Annotate PD GWAS genes
        for gene in PD_GWAS_GENES:
            if gene in scvi_df.index:
                gx = lfc.loc[gene]
                gy = bf.loc[gene]
                ax.annotate(gene, (gx, gy), fontsize=7, color="darkred",
                            xytext=(4, 4), textcoords="offset points")
                ax.scatter([gx], [gy], c="darkred", s=25, zorder=5)

        n_sig = sig.sum()
        ax.legend(handles=[
            mpatches.Patch(color="mediumseagreen", label=f"Significant (n={n_sig})"),
            mpatches.Patch(color="lightgray", label="Not significant"),
        ], fontsize=7)
    else:
        ax.text(0.5, 0.5, "No scVI DE data.\nRun step 05 first.",
                ha="center", va="center", transform=ax.transAxes, color="gray")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure 1 saved to %s", out_path)


def figure2_lfc_comparison(results: dict, out_path: Path):
    """
    Scatter plot comparing log2FC from Wilcoxon and scVI for DAn1.

    Args:
        results: Dict from load_results().
        out_path: Output PDF path.
    """
    wil_df = results["wilcoxon_df"]
    scvi_df = results["scvi_df"]

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("Figure 2 — LFC Comparison: Wilcoxon vs scVI (DAn1 WT vs A53T)",
                 fontsize=11, fontweight="bold")

    if wil_df is None or scvi_df is None:
        ax.text(0.5, 0.5, "DE results not available.\nRun steps 03 and 05 first.",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    # Filter Wilcoxon to DAn1
    dan1_mask = wil_df.get("cell_type", wil_df.get("cluster", "")) == "DAn1"
    if isinstance(dan1_mask, str):
        wil_dan1 = wil_df.copy()
    else:
        wil_dan1 = wil_df[dan1_mask].copy() if dan1_mask.any() else wil_df.copy()

    wil_dan1 = wil_dan1.rename(columns={"names": "gene", "logfoldchanges": "lfc_wilcoxon",
                                         "pvals_adj": "fdr_wilcoxon"})

    # column names vary by scvi-tools version
    lfc_col = next((c for c in ["lfc_mean", "lfc_median"] if c in scvi_df.columns), None)
    sig_col = next((c for c in ["is_de_fdr_0.05", "is_de"] if c in scvi_df.columns), None)
    scvi_rename = {}
    if lfc_col:
        scvi_rename[lfc_col] = "lfc_scvi"
    if sig_col:
        scvi_rename[sig_col] = "sig_scvi"
    scvi_sub = scvi_df.rename(columns=scvi_rename)
    # if no lfc column, derive from log-ratio of normalized means
    if "lfc_scvi" not in scvi_sub.columns:
        if "raw_normalized_mean1" in scvi_sub.columns and "raw_normalized_mean2" in scvi_sub.columns:
            scvi_sub["lfc_scvi"] = np.log2(
                (scvi_sub["raw_normalized_mean1"] + 1e-6) / (scvi_sub["raw_normalized_mean2"] + 1e-6)
            )
        else:
            scvi_sub["lfc_scvi"] = np.nan
    if "sig_scvi" not in scvi_sub.columns:
        scvi_sub["sig_scvi"] = False
    if "gene" not in scvi_sub.columns:
        scvi_sub.index.name = "gene"
        scvi_sub = scvi_sub.reset_index()

    merged = wil_dan1.merge(scvi_sub[["gene", "lfc_scvi", "sig_scvi"]],
                             on="gene", how="inner")

    if merged.empty:
        ax.text(0.5, 0.5, "No matching genes between methods.",
                ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    merged["sig_wilcoxon"] = (
        (merged["fdr_wilcoxon"] < 0.05) & (merged["lfc_wilcoxon"].abs() > 0.25)
    ) if "fdr_wilcoxon" in merged.columns else False

    # Color by significance category
    colors = np.where(
        merged["sig_wilcoxon"] & merged["sig_scvi"], "purple",
        np.where(merged["sig_wilcoxon"], "coral",
                 np.where(merged["sig_scvi"], "steelblue", "lightgray"))
    )

    ax.scatter(merged["lfc_wilcoxon"], merged["lfc_scvi"], c=colors, s=8, alpha=0.6)

    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.axhline(-0.5, color="gray", ls="--", lw=0.8)
    ax.axvline(0.5, color="gray", ls="--", lw=0.8)
    ax.axvline(-0.5, color="gray", ls="--", lw=0.8)
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)

    r, p = stats.pearsonr(merged["lfc_wilcoxon"], merged["lfc_scvi"])
    ax.text(0.05, 0.95, f"Pearson r = {r:.3f}\np = {p:.2e}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Label PD GWAS genes
    for _, row in merged[merged["gene"].isin(PD_GWAS_GENES)].iterrows():
        ax.annotate(row["gene"], (row["lfc_wilcoxon"], row["lfc_scvi"]),
                    fontsize=7, color="darkred",
                    arrowprops=dict(arrowstyle="-", lw=0.5))

    legend_handles = [
        mpatches.Patch(color="purple", label="Both significant"),
        mpatches.Patch(color="coral", label="Wilcoxon only"),
        mpatches.Patch(color="steelblue", label="scVI only"),
        mpatches.Patch(color="lightgray", label="Neither"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")
    ax.set_xlabel("log2FC (Wilcoxon)", fontsize=11)
    ax.set_ylabel("log2FC (scVI)", fontsize=11)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure 2 saved to %s", out_path)


def figure3_qc_impact(results: dict, out_path: Path):
    """
    Bar chart and violin of cells retained under each QC strategy.

    Args:
        results: Dict from load_results().
        out_path: Output PDF path.
    """
    adata_raw = results["adata_raw"]
    adata_fern = results["adata_fern"]
    adata_bp = results["adata_bp"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Figure 3 — QC Strategy Impact", fontsize=12, fontweight="bold")

    # Panel A: cells per sample under each strategy
    ax = axes[0]
    if adata_raw and adata_fern and adata_bp:
        samples = adata_raw.obs["sample_id"].unique()
        raw_counts = adata_raw.obs.groupby("sample_id", observed=True).size().reindex(samples).fillna(0)
        fern_counts = adata_fern.obs.groupby("sample_id", observed=True).size().reindex(samples).fillna(0)
        bp_counts = adata_bp.obs.groupby("sample_id", observed=True).size().reindex(samples).fillna(0)

        x = np.arange(len(samples))
        w = 0.25
        ax.bar(x - w, raw_counts, w, label="Raw", color="lightgray")
        ax.bar(x, fern_counts, w, label="Fernandes QC", color="coral")
        ax.bar(x + w, bp_counts, w, label="Best-practice QC", color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels(samples, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Number of cells")
        ax.set_title("Cells per sample")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Data not available", ha="center", va="center",
                transform=ax.transAxes)

    # Panel B: MT% for cells retained by Fernandes but excluded by best-practice
    ax = axes[1]
    if adata_raw and adata_fern and adata_bp:
        fern_only = set(adata_fern.obs_names) - set(adata_bp.obs_names)
        if fern_only:
            mt_fern_only = adata_fern.obs.loc[
                adata_fern.obs_names.isin(fern_only), "pct_counts_mt"
            ]
            mt_bp = adata_bp.obs["pct_counts_mt"]
            ax.violinplot([mt_bp.values, mt_fern_only.values], positions=[0, 1],
                          showmedians=True)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Best-practice kept\n(n={:,})".format(len(mt_bp)),
                                 "Fern kept/BP removed\n(n={:,})".format(len(mt_fern_only))])
            ax.axhline(10, color="steelblue", ls="--", lw=1, label="BP 10% MT cutoff")
            ax.axhline(40, color="coral", ls="--", lw=1, label="Fernandes 40% MT cutoff")
            ax.set_ylabel("% Mitochondrial reads")
            ax.set_title("MT% distribution:\nFernandes-retained vs BP-removed cells")
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "No cells exclusively retained\nby Fernandes", ha="center",
                    va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "Data not available", ha="center", va="center",
                transform=ax.transAxes)

    # Panel C: summary bar
    ax = axes[2]
    if adata_raw and adata_fern and adata_bp:
        totals = [adata_raw.n_obs, adata_fern.n_obs, adata_bp.n_obs]
        labels = ["Raw", "Fernandes QC", "Best-practice QC"]
        bars = ax.bar(labels, totals, color=["lightgray", "coral", "steelblue"])
        for bar, total in zip(bars, totals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                    f"{total:,}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Total cells")
        ax.set_title("Total cells retained")
    else:
        ax.text(0.5, 0.5, "Data not available", ha="center", va="center",
                transform=ax.transAxes)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure 3 saved to %s", out_path)


def figure4_deg_overlap(results: dict, out_path: Path):
    """
    UpSet-style or Venn DEG overlap between methods.

    Args:
        results: Dict from load_results().
        out_path: Output PDF path.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle("Figure 4 — DEG Overlap Between Methods (DAn1 WT vs A53T)",
                 fontsize=11, fontweight="bold")

    # Collect significant gene sets per method
    gene_sets = {}

    wil_df = results["wilcoxon_df"]
    if wil_df is not None and not wil_df.empty:
        dan1 = wil_df[wil_df.get("cell_type", wil_df.get("cluster", pd.Series())) == "DAn1"] \
               if "cell_type" in wil_df.columns else wil_df
        sig = dan1[
            (dan1["pvals_adj"] < 0.05) & (dan1["logfoldchanges"].abs() > 0.25)
        ] if "pvals_adj" in dan1.columns else dan1
        gene_col = "names" if "names" in sig.columns else "gene"
        gene_sets["Wilcoxon"] = set(sig[gene_col].dropna())

    pb_df = results["pseudobulk_df"]
    if pb_df is not None and "FDR" in pb_df.columns:
        sig = pb_df[pb_df["FDR"] < 0.05]
        gene_sets["Pseudobulk"] = set(sig["gene"].dropna()) if "gene" in sig.columns else set()

    scvi_df = results["scvi_df"]
    if scvi_df is not None:
        _scvi_sig_col = next((c for c in ["is_de_fdr_0.05", "is_de"] if c in scvi_df.columns), None)
        if _scvi_sig_col:
            sig = scvi_df[scvi_df[_scvi_sig_col].astype(bool)]
            gene_col = "gene" if "gene" in sig.columns else sig.index.name or "index"
            gene_sets["scVI"] = set(sig.index if gene_col == "index" else sig[gene_col].dropna())

    if not gene_sets:
        ax.text(0.5, 0.5, "No DE results available.\nRun steps 03–05 first.",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    methods = list(gene_sets.keys())
    n = len(methods)

    # Simple bar chart if only one method
    if n == 1:
        name = methods[0]
        genes = list(gene_sets[name])
        pd_hits = [g for g in genes if g in PD_GWAS_GENES]
        ax.bar([name], [len(genes)], color="steelblue")
        ax.set_ylabel("Number of DEGs")
        ax.set_title(f"{name}: {len(genes)} DEGs\nPD GWAS hits: {', '.join(pd_hits) or 'none'}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    # Venn-style sets
    try:
        from matplotlib_venn import venn2, venn3
        if n == 2:
            m1, m2 = methods
            venn2([gene_sets[m1], gene_sets[m2]], set_labels=(m1, m2), ax=ax)
        elif n >= 3:
            m1, m2, m3 = methods[:3]
            venn3([gene_sets[m1], gene_sets[m2], gene_sets[m3]],
                  set_labels=(m1, m2, m3), ax=ax)
    except ImportError:
        # Fallback: UpSet-style bar chart
        from itertools import combinations
        all_combos = {}
        all_genes = set().union(*gene_sets.values())
        for size in range(1, n + 1):
            for combo in combinations(methods, size):
                in_all = set.intersection(*[gene_sets[m] for m in combo])
                if size < n:
                    exclude = set.union(*[gene_sets[m] for m in methods if m not in combo])
                    in_all = in_all - exclude
                if in_all:
                    all_combos[" ∩ ".join(combo)] = len(in_all)
        ax.barh(list(all_combos.keys()), list(all_combos.values()), color="steelblue")
        ax.set_xlabel("Number of genes")

    # Annotate PD GWAS genes
    shared_genes = set.intersection(*gene_sets.values()) if gene_sets else set()
    pd_shared = shared_genes & set(PD_GWAS_GENES)
    ax.text(0.5, 0.02,
            f"PD GWAS genes in all sets: {', '.join(pd_shared) or 'none'}",
            ha="center", va="bottom", transform=ax.transAxes, fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure 4 saved to %s", out_path)


def figure5_batch_mixing(results: dict, out_path: Path):
    """
    Side-by-side UMAP coloured by sample_id to visualize batch mixing.

    Computes iLISI (integration LISI) score as a quantitative batch-mixing metric.

    Args:
        results: Dict from load_results().
        out_path: Output PDF path.
    """
    adata_clustered = results["adata_clustered"]
    adata_scvi = results["adata_scvi"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Figure 5 — Batch Mixing: PCA UMAP vs scVI UMAP",
                 fontsize=12, fontweight="bold")

    color_key = "sample_id"

    def plot_umap(ax, adata, umap_key, title):
        if adata is None or umap_key not in adata.obsm:
            ax.text(0.5, 0.5, f"No {umap_key}", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title)
            return
        coords = adata.obsm[umap_key]
        if color_key not in adata.obs.columns:
            ax.scatter(coords[:, 0], coords[:, 1], s=1, alpha=0.3, color="gray")
            ax.set_title(title)
            return
        cats = adata.obs[color_key].astype("category")
        cmap_colors = plt.cm.tab20(np.linspace(0, 1, len(cats.cat.categories)))
        for i, cat in enumerate(cats.cat.categories):
            mask = cats == cat
            ax.scatter(coords[mask, 0], coords[mask, 1], s=1, alpha=0.4,
                       color=cmap_colors[i], label=str(cat))
        ax.legend(fontsize=5, markerscale=4, loc="best", ncol=2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("UMAP1", fontsize=8)
        ax.set_ylabel("UMAP2", fontsize=8)

    plot_umap(axes[0], adata_clustered, "X_umap", "PCA-based UMAP\n(coloured by sample_id)")
    plot_umap(axes[1], adata_scvi, "X_umap_scvi", "scVI UMAP\n(coloured by sample_id)")

    # Attempt LISI score
    try:
        from harmonypy import compute_lisi
        lisi_scores = {}
        for name, adata, key in [
            ("PCA", adata_clustered, "X_pca"),
            ("scVI", adata_scvi, "X_scVI"),
        ]:
            if adata is not None and key in adata.obsm and color_key in adata.obs.columns:
                lisi = compute_lisi(
                    adata.obsm[key][:, :30] if adata.obsm[key].shape[1] > 30 else adata.obsm[key],
                    adata.obs[[color_key]],
                    [color_key],
                )
                lisi_scores[name] = float(np.median(lisi))
        if lisi_scores:
            txt = "  ".join(f"LISI_{k}={v:.2f}" for k, v in lisi_scores.items())
            fig.text(0.5, 0.01, f"Median iLISI (higher = better mixing): {txt}",
                     ha="center", fontsize=9)
    except ImportError:
        log.info("harmonypy not installed; skipping LISI score computation.")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure 5 saved to %s", out_path)


def main():
    np.random.seed(RANDOM_SEED)

    log.info("Loading all results...")
    results = load_results()

    log.info("Generating Figure 1: DEG count vs cell count...")
    figure1_deg_vs_cells(results, FIGURES_DIR / "fig1_deg_vs_cells.pdf")

    log.info("Generating Figure 2: LFC comparison...")
    figure2_lfc_comparison(results, FIGURES_DIR / "fig2_lfc_comparison.pdf")

    log.info("Generating Figure 3: QC impact...")
    figure3_qc_impact(results, FIGURES_DIR / "fig3_qc_impact.pdf")

    log.info("Generating Figure 4: DEG overlap...")
    figure4_deg_overlap(results, FIGURES_DIR / "fig4_deg_overlap.pdf")

    log.info("Generating Figure 5: Batch mixing...")
    figure5_batch_mixing(results, FIGURES_DIR / "fig5_batch_mixing.pdf")

    log.info("All figures saved to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
