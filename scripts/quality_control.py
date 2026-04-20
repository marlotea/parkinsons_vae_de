"""
Quality control filtering of raw scRNA-seq data.

Applies two QC strategies in parallel:
  1. Fernandes et al. (2020) thresholds: min 200 genes, max 8000 genes, <40% MT
  2. Best-practice adaptive thresholds: MAD-based upper gene cutoff, <10% MT,
     plus Scrublet doublet detection per sample.

Produces figures/qc_comparison.pdf and saves two filtered AnnData objects.
"""

import logging
import sys
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scrublet as scr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Fernandes et al. thresholds
FERN_MIN_GENES = 200
FERN_MAX_GENES = 8000
FERN_MAX_MT = 40.0

# Best-practice fixed lower bound; upper bound is MAD-adaptive
BP_MIN_GENES = 200
BP_MAX_MT = 10.0
BP_MAD_MULTIPLIER = 4.0


def mad(x: np.ndarray) -> float:
    """Return the median absolute deviation of array x."""
    return float(np.median(np.abs(x - np.median(x))))


def compute_adaptive_max_genes(log_genes: np.ndarray, multiplier: float = BP_MAD_MULTIPLIER) -> float:
    """
    Compute MAD-based adaptive upper threshold for log(n_genes).

    Args:
        log_genes: Array of log-transformed gene counts per cell.
        multiplier: Number of MADs above median to set threshold.

    Returns:
        Upper threshold on the original (non-log) scale.
    """
    median = float(np.median(log_genes))
    m = mad(log_genes)
    threshold_log = median + multiplier * m
    return float(np.exp(threshold_log))


def run_scrublet(adata_sample: ad.AnnData, sample_id: str) -> np.ndarray:
    """
    Run Scrublet doublet detection on a single-sample AnnData.

    Args:
        adata_sample: AnnData for one sample (raw counts).
        sample_id: Used for logging.

    Returns:
        Boolean array of length n_obs; True = predicted doublet.
    """
    counts = adata_sample.X
    if not isinstance(counts, np.ndarray):
        counts = counts.toarray()

    scrub = scr.Scrublet(counts, random_state=42)
    doublet_scores, predicted_doublets = scrub.scrub_doublets(
        min_counts=2, min_cells=3, min_gene_variability_pctl=85, n_prin_comps=30
    )
    n_doublets = predicted_doublets.sum()
    log.info(
        "  Scrublet [%s]: %.1f%% doublets (%d / %d cells)",
        sample_id,
        100 * n_doublets / len(predicted_doublets),
        n_doublets,
        len(predicted_doublets),
    )
    return predicted_doublets


def compute_qc_metrics(adata: ad.AnnData) -> ad.AnnData:
    """
    Compute per-cell QC metrics in-place and return the AnnData.

    Adds obs columns: n_genes_by_counts, total_counts, pct_counts_mt.

    Args:
        adata: AnnData with raw count matrix.

    Returns:
        Same AnnData with QC metrics added to .obs.
    """
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    return adata


def apply_fernandes_thresholds(adata: ad.AnnData) -> ad.AnnData:
    """
    Filter cells using Fernandes et al. (2020) QC thresholds.

    Args:
        adata: AnnData with QC metrics computed.

    Returns:
        Filtered AnnData.
    """
    mask = (
        (adata.obs["n_genes_by_counts"] >= FERN_MIN_GENES)
        & (adata.obs["n_genes_by_counts"] <= FERN_MAX_GENES)
        & (adata.obs["pct_counts_mt"] < FERN_MAX_MT)
    )
    log.info(
        "Fernandes thresholds: %d / %d cells pass (removed %d)",
        mask.sum(), len(mask), (~mask).sum(),
    )
    return adata[mask].copy()


def apply_bestpractice_thresholds(adata: ad.AnnData) -> ad.AnnData:
    """
    Filter cells using MAD-adaptive thresholds + Scrublet doublet removal.

    Scrublet is run per sample to avoid batch effects influencing doublet scores.

    Args:
        adata: AnnData with QC metrics computed.

    Returns:
        Filtered AnnData with doublet column added to obs.
    """
    # MAD-based upper gene threshold computed globally across all samples
    log_genes = np.log(adata.obs["n_genes_by_counts"].values + 1)
    adaptive_max = compute_adaptive_max_genes(log_genes)
    log.info(
        "Adaptive max genes threshold: %.0f (median + %g * MAD on log scale)",
        adaptive_max, BP_MAD_MULTIPLIER,
    )

    # Per-sample doublet detection
    doublet_flags = np.zeros(adata.n_obs, dtype=bool)
    for sample_id in adata.obs["sample_id"].unique():
        idx = np.where(adata.obs["sample_id"] == sample_id)[0]
        sample_adata = adata[idx]
        flags = run_scrublet(sample_adata, sample_id)
        doublet_flags[idx] = flags

    adata.obs["predicted_doublet"] = doublet_flags

    mask = (
        (adata.obs["n_genes_by_counts"] >= BP_MIN_GENES)
        & (adata.obs["n_genes_by_counts"] <= adaptive_max)
        & (adata.obs["pct_counts_mt"] < BP_MAX_MT)
        & (~adata.obs["predicted_doublet"])
    )
    log.info(
        "Best-practice thresholds: %d / %d cells pass (removed %d)",
        mask.sum(), len(mask), (~mask).sum(),
    )
    return adata[mask].copy()


def plot_qc_comparison(adata_raw: ad.AnnData, adata_fern: ad.AnnData,
                       adata_bp: ad.AnnData, out_path: Path):
    """
    Produce a QC comparison figure with violin plots before/after filtering.

    Args:
        adata_raw: Unfiltered AnnData.
        adata_fern: AnnData after Fernandes thresholds.
        adata_bp: AnnData after best-practice thresholds.
        out_path: Path to save the PDF figure.
    """
    metrics = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
    labels = ["n_genes", "total_counts", "% MT"]
    datasets = [("Raw", adata_raw), ("Fernandes QC", adata_fern), ("Best-practice QC", adata_bp)]

    fig, axes = plt.subplots(len(metrics), 3, figsize=(14, 12))
    fig.suptitle("QC Metrics Before and After Filtering", fontsize=14, fontweight="bold")

    for row, (metric, label) in enumerate(zip(metrics, labels)):
        for col, (ds_label, adata) in enumerate(datasets):
            ax = axes[row, col]
            vals = adata.obs[metric].values
            ax.violinplot(vals, positions=[0], showmedians=True)
            ax.set_title(f"{ds_label}\nn={adata.n_obs:,}", fontsize=9)
            ax.set_ylabel(label if col == 0 else "")
            ax.set_xticks([])

            # Add Fernandes threshold lines for reference
            if metric == "n_genes_by_counts":
                ax.axhline(FERN_MIN_GENES, color="orange", ls="--", lw=0.8, label="Fern min")
                ax.axhline(FERN_MAX_GENES, color="orange", ls="-.", lw=0.8, label="Fern max")
                log_v = np.log(vals + 1)
                adaptive = compute_adaptive_max_genes(log_v)
                ax.axhline(adaptive, color="blue", ls="--", lw=0.8, label="BP adaptive max")
                if col == 0:
                    ax.legend(fontsize=6)
            elif metric == "pct_counts_mt":
                ax.axhline(FERN_MAX_MT, color="orange", ls="--", lw=0.8, label="Fern 40%")
                ax.axhline(BP_MAX_MT, color="blue", ls="--", lw=0.8, label="BP 10%")
                if col == 0:
                    ax.legend(fontsize=6)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("QC comparison figure saved to %s", out_path)


def print_qc_table(adata_raw: ad.AnnData, adata_fern: ad.AnnData, adata_bp: ad.AnnData):
    """
    Print a per-sample cell count table for each QC strategy.

    Args:
        adata_raw: Unfiltered AnnData.
        adata_fern: AnnData after Fernandes thresholds.
        adata_bp: AnnData after best-practice thresholds.
    """
    def counts_per_sample(adata):
        return adata.obs.groupby("sample_id", observed=True).size()

    raw_counts = counts_per_sample(adata_raw).rename("raw")
    fern_counts = counts_per_sample(adata_fern).rename("fernandes_qc")
    bp_counts = counts_per_sample(adata_bp).rename("bestpractice_qc")

    table = pd.concat([raw_counts, fern_counts, bp_counts], axis=1).fillna(0).astype(int)
    table["removed_by_fernandes"] = table["raw"] - table["fernandes_qc"]
    table["removed_by_bestpractice"] = table["raw"] - table["bestpractice_qc"]

    log.info("\n=== QC CELL COUNT TABLE ===\n%s\n", table.to_string())
    log.info(
        "TOTAL: raw=%d | Fernandes=%d | best-practice=%d",
        adata_raw.n_obs, adata_fern.n_obs, adata_bp.n_obs,
    )

    # Flag the mito cutoff discrepancy
    extra_mito = (
        (adata_raw.obs["pct_counts_mt"] >= BP_MAX_MT)
        & (adata_raw.obs["pct_counts_mt"] < FERN_MAX_MT)
    ).sum()
    log.info(
        "\n*** KEY FINDING: %d cells (%.1f%% of raw) have >=10%% but <40%% MT reads. "
        "Fernandes et al. retain these potentially dead/dying cells. ***",
        extra_mito, 100 * extra_mito / adata_raw.n_obs,
    )


def main():
    np.random.seed(42)

    raw_path = PROCESSED_DIR / "adata_raw.h5ad"
    log.info("Loading %s", raw_path)
    adata = sc.read_h5ad(raw_path)

    log.info("Computing QC metrics...")
    adata = compute_qc_metrics(adata)

    log.info("Applying Fernandes et al. thresholds...")
    adata_fern = apply_fernandes_thresholds(adata)

    log.info("Applying best-practice thresholds (with Scrublet)...")
    adata_bp = apply_bestpractice_thresholds(adata)

    print_qc_table(adata, adata_fern, adata_bp)

    plot_qc_comparison(adata, adata_fern, adata_bp, FIGURES_DIR / "qc_comparison.pdf")

    fern_out = PROCESSED_DIR / "adata_qc_fernandes.h5ad"
    bp_out = PROCESSED_DIR / "adata_qc_bestpractice.h5ad"
    adata_fern.write_h5ad(fern_out)
    adata_bp.write_h5ad(bp_out)
    log.info("Saved: %s", fern_out)
    log.info("Saved: %s", bp_out)


if __name__ == "__main__":
    main()
