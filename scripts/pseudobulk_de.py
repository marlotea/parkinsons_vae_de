"""
Pseudobulk differential expression using edgeR (via rpy2).

Aggregates raw counts per (cell_type, sample_id) combination, then applies
edgeR's glmQLFTest for WT vs SNCA-A53T comparison. Explicitly checks whether
sufficient biological replicates exist — insufficient replicates is itself a
key finding demonstrating the study is underpowered at the appropriate unit.
"""

import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MIN_CELLS_PER_PSEUDOBULK = 10
MIN_REPLICATES_PER_GROUP = 3


def aggregate_pseudobulk(adata: ad.AnnData, cell_type_key: str = "cell_type",
                          sample_key: str = "sample_id") -> pd.DataFrame:
    """
    Sum raw counts across cells for each (cell_type, sample_id) combination.

    Only retains combinations with >= MIN_CELLS_PER_PSEUDOBULK cells.

    Args:
        adata: AnnData with raw count matrix (.X) and obs annotations.
        cell_type_key: obs column for cell type labels.
        sample_key: obs column for sample/batch identifiers.

    Returns:
        DataFrame of shape (n_genes, n_pseudosamples) with column names
        "CellType__SampleID", and a separate metadata DataFrame.
    """
    log.info("Aggregating pseudobulk counts...")

    groups = adata.obs[[cell_type_key, sample_key]].copy()
    groups["group_key"] = groups[cell_type_key].astype(str) + "__" + groups[sample_key].astype(str)
    unique_groups = groups["group_key"].unique()

    count_cols = {}
    meta_rows = []

    for key in sorted(unique_groups):
        cell_type, sample_id = key.split("__", 1)
        mask = groups["group_key"] == key
        n_cells = mask.sum()

        if n_cells < MIN_CELLS_PER_PSEUDOBULK:
            log.debug("Skipping %s: only %d cells (< %d)", key, n_cells, MIN_CELLS_PER_PSEUDOBULK)
            continue

        subset = adata[mask.values]
        X = subset.X
        if sp.issparse(X):
            X = X.toarray()
        summed = X.sum(axis=0)

        count_cols[key] = summed

        sample_meta = adata.obs[adata.obs[sample_key] == sample_id].iloc[0]
        meta_rows.append({
            "pseudosample": key,
            "cell_type": cell_type,
            "sample_id": sample_id,
            "genotype": sample_meta.get("genotype", "unknown"),
            "condition": sample_meta.get("condition", "unknown"),
            "n_cells": n_cells,
        })

    count_df = pd.DataFrame(count_cols, index=adata.var_names)
    meta_df = pd.DataFrame(meta_rows)

    log.info(
        "Pseudobulk matrix: %d genes x %d pseudo-samples (%d combinations with >= %d cells)",
        count_df.shape[0], count_df.shape[1], len(meta_rows), MIN_CELLS_PER_PSEUDOBULK,
    )
    return count_df, meta_df


def check_replicates(meta_df: pd.DataFrame, cell_type: str, condition: str = "untreated") -> dict:
    """
    Count biological replicates per genotype group for a given cell type and condition.

    Args:
        meta_df: Pseudobulk metadata DataFrame.
        cell_type: Cell type to inspect.
        condition: Condition to filter on.

    Returns:
        Dict mapping genotype → replicate count.
    """
    subset = meta_df[
        (meta_df["cell_type"] == cell_type) & (meta_df["condition"] == condition)
    ]
    return subset.groupby("genotype").size().to_dict()


def run_edger_via_rpy2(count_df: pd.DataFrame, meta_df: pd.DataFrame,
                       cell_type: str, condition: str = "untreated") -> pd.DataFrame:
    """
    Run edgeR glmQLFTest for WT vs SNCA-A53T in a specified cell type.

    Uses rpy2 to call R; requires edgeR to be installed in the R environment.

    Args:
        count_df: Gene x pseudo-sample count matrix (all cell types / conditions).
        meta_df: Metadata for each pseudo-sample column.
        cell_type: Cell type to subset for DE.
        condition: Treatment condition to subset (default: "untreated").

    Returns:
        DataFrame with edgeR DE results (gene, logFC, logCPM, F, PValue, FDR).
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        pandas2ri.activate()
        edger = importr("edgeR")
        base = importr("base")
        stats = importr("stats")
    except ImportError as exc:
        raise RuntimeError(
            "rpy2 or edgeR not available. Install rpy2 and edgeR in R: "
            "BiocManager::install('edgeR')"
        ) from exc

    subset_meta = meta_df[
        (meta_df["cell_type"] == cell_type) & (meta_df["condition"] == condition)
    ].copy()

    if subset_meta.empty:
        raise ValueError(f"No pseudo-samples for cell_type={cell_type}, condition={condition}")

    cols = subset_meta["pseudosample"].tolist()
    counts_subset = count_df[cols].copy()

    # Remove genes with zero counts in all samples
    nonzero = (counts_subset > 0).any(axis=1)
    counts_subset = counts_subset[nonzero]
    log.info(
        "edgeR input: %d genes x %d samples (cell_type=%s, condition=%s)",
        counts_subset.shape[0], counts_subset.shape[1], cell_type, condition,
    )

    # Transfer to R
    r_counts = pandas2ri.py2rpy(counts_subset.astype(int))
    r_genotype = ro.StrVector(subset_meta["genotype"].tolist())
    r_genotype_factor = base.factor(r_genotype, levels=["WT", "SNCA_A53T"])

    # edgeR pipeline
    dge = edger.DGEList(counts=r_counts)
    dge = edger.calcNormFactors(dge, method="TMM")

    design_formula = stats.formula("~ genotype")
    ro.globalenv["genotype"] = r_genotype_factor
    design_mat = stats.model_matrix(design_formula)

    dge = edger.estimateDisp(dge, design=design_mat)
    fit = edger.glmQLFit(dge, design=design_mat)
    qlf = edger.glmQLFTest(fit, coef=2)

    # Extract results
    top_tags = edger.topTags(qlf, n=counts_subset.shape[0], sort_by="PValue")
    result_r = base.as_data_frame(top_tags.rx2("table"))
    result_df = pandas2ri.rpy2py(result_r)
    result_df.index.name = "gene"
    result_df = result_df.reset_index()
    result_df.columns = ["gene", "logFC", "logCPM", "F", "PValue", "FDR"]

    n_de = (result_df["FDR"] < 0.05).sum()
    log.info("edgeR: %d DEGs at FDR < 0.05 (cell_type=%s)", n_de, cell_type)
    return result_df


def main():
    np.random.seed(42)

    log.info("Loading best-practice QC data...")
    adata = sc.read_h5ad(PROCESSED_DIR / "adata_qc_bestpractice.h5ad")

    # Load cluster/cell_type labels from the clustered object if available
    clustered_path = PROCESSED_DIR / "adata_fernandes_clustered.h5ad"
    if clustered_path.exists():
        adata_clustered = sc.read_h5ad(clustered_path)
        # Transfer cell_type only for cells present in both
        common_cells = adata.obs_names.intersection(adata_clustered.obs_names)
        if len(common_cells) > 0:
            adata.obs["cell_type"] = adata_clustered.obs.loc[common_cells, "cell_type"] \
                .reindex(adata.obs_names)
            log.info("Transferred cell_type labels for %d cells", len(common_cells))
        else:
            log.warning("No overlapping cell barcodes between QC sets; using placeholder labels.")
            adata.obs["cell_type"] = "DAn1"
    else:
        log.warning("Clustered AnnData not found; run 03_reproduce_original.py first.")
        adata.obs["cell_type"] = "DAn1"

    count_df, meta_df = aggregate_pseudobulk(adata)

    # Print replicate counts
    log.info("\n=== BIOLOGICAL REPLICATE COUNTS (untreated) ===")
    for ct in meta_df["cell_type"].unique():
        rep_counts = check_replicates(meta_df, ct, condition="untreated")
        log.info("  %s: %s", ct, rep_counts)
        for genotype, n_reps in rep_counts.items():
            if n_reps < MIN_REPLICATES_PER_GROUP:
                log.warning(
                    "\n*** KEY FINDING: cell_type=%s, genotype=%s has only %d biological "
                    "replicate(s) (need >= %d for valid pseudobulk DE). The study is "
                    "statistically underpowered at the appropriate unit of replication. "
                    "This is a direct consequence of the pseudoreplication design. ***",
                    ct, genotype, n_reps, MIN_REPLICATES_PER_GROUP,
                )

    # Attempt edgeR DE for DAn1
    target_cell_type = "DAn1"
    if target_cell_type not in meta_df["cell_type"].values:
        available = meta_df["cell_type"].unique().tolist()
        log.warning("'%s' not found; using '%s' instead.", target_cell_type, available[0])
        target_cell_type = available[0]

    rep_counts = check_replicates(meta_df, target_cell_type, condition="untreated")
    min_reps = min(rep_counts.values()) if rep_counts else 0

    if min_reps < MIN_REPLICATES_PER_GROUP:
        log.error(
            "\n*** PSEUDOBULK DE NOT PERFORMED for %s: insufficient replicates (%s). "
            "This is itself a key finding — the original paper's DE claims lack a "
            "valid biological replicate basis at the iPSC line level. "
            "Saving placeholder results. ***",
            target_cell_type, rep_counts,
        )
        result_df = pd.DataFrame({
            "gene": [], "logFC": [], "logCPM": [], "F": [], "PValue": [], "FDR": [],
            "note": [],
        })
        result_df.loc[0] = ["N/A", np.nan, np.nan, np.nan, np.nan, np.nan,
                            f"Insufficient replicates: {rep_counts}"]
    else:
        log.info("Running edgeR for %s (untreated WT vs A53T)...", target_cell_type)
        try:
            result_df = run_edger_via_rpy2(count_df, meta_df, target_cell_type)
        except RuntimeError as exc:
            log.error("edgeR failed: %s", exc)
            result_df = pd.DataFrame({"error": [str(exc)]})

    out_csv = RESULTS_DIR / "degs_pseudobulk_edger.csv"
    result_df.to_csv(out_csv, index=False)
    log.info("Pseudobulk DE results saved to %s", out_csv)

    # Save pseudobulk matrices for downstream use
    meta_df.to_csv(RESULTS_DIR / "pseudobulk_metadata.csv", index=False)
    count_df.to_csv(RESULTS_DIR / "pseudobulk_counts.csv")
    log.info("Pseudobulk matrices saved to %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
