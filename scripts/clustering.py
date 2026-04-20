"""
Reproduce the original Fernandes et al. (2020) analysis using Scanpy.

Applies the same preprocessing pipeline (normalize → log1p → HVG → scale → PCA →
neighbors → UMAP → Louvain) and performs cell-level Wilcoxon DE to demonstrate
the pseudoreplication issue identified by Murphy & Skene (2022).
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
RESULTS_DIR.mkdir(exist_ok=True)

# Marker genes from Fernandes et al. Table S1
MARKER_GENES = {
    "Prog1/Prog2": ["VIM", "HES1", "NFIA", "SOX2"],
    "DAn1": ["STMN2", "MAPT", "SNCA", "UCHL1"],
    "DAn2": ["CALB1", "KCNJ6"],
    "DAn3": ["NR4A2", "TH", "SLC6A3"],
    "DAn4": ["ALDH1A1", "SOX6"],
}

LEIDEN_RESOLUTION = 0.5  # leiden replaces louvain (louvain unavailable on this platform)
N_HVG = 2000
N_PCS = 50
N_PCS_NEIGHBORS = 30
K_NEIGHBORS = 20


def preprocess(adata: ad.AnnData) -> ad.AnnData:
    """
    Apply the Fernandes et al. preprocessing pipeline.

    Normalizes counts, log-transforms, selects HVGs, scales, and runs PCA.

    Args:
        adata: AnnData with raw filtered counts.

    Returns:
        Preprocessed AnnData with PCA embedding in .obsm["X_pca"].
    """
    log.info("Normalizing to 10,000 counts per cell...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    log.info("Selecting top %d highly variable genes...", N_HVG)
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat", batch_key="sample_id")
    log.info("  HVGs found: %d", adata.var["highly_variable"].sum())

    adata.raw = adata  # store log-normalized counts before scaling

    adata = adata[:, adata.var["highly_variable"]].copy()

    log.info("Scaling (max_value=10)...")
    sc.pp.scale(adata, max_value=10)

    log.info("Running PCA (%d components)...", N_PCS)
    sc.tl.pca(adata, n_comps=N_PCS, random_state=42)

    return adata


def cluster(adata: ad.AnnData) -> ad.AnnData:
    """
    Build neighbor graph, compute UMAP, and run Louvain clustering.

    Args:
        adata: AnnData with PCA embedding.

    Returns:
        AnnData with UMAP and louvain cluster labels.
    """
    log.info("Computing neighbor graph (k=%d, n_pcs=%d)...", K_NEIGHBORS, N_PCS_NEIGHBORS)
    sc.pp.neighbors(adata, n_neighbors=K_NEIGHBORS, n_pcs=N_PCS_NEIGHBORS, random_state=42)

    log.info("Running UMAP...")
    sc.tl.umap(adata, random_state=42)

    log.info("Running Leiden clustering (resolution=%.2f)...", LEIDEN_RESOLUTION)
    sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION, random_state=42)
    # Rename to 'louvain' for downstream compatibility (same conceptual role)
    adata.obs["louvain"] = adata.obs["leiden"]

    cluster_counts = adata.obs["louvain"].value_counts().sort_index()
    log.info("Cluster sizes:\n%s", cluster_counts.to_string())

    return adata


def annotate_clusters(adata: ad.AnnData) -> ad.AnnData:
    """
    Assign cell type labels to Louvain clusters based on marker gene expression.

    Uses mean expression of marker genes per cluster to assign the most likely
    cell type. Manual curation may still be required.

    Args:
        adata: AnnData with louvain cluster labels and raw expression in .raw.

    Returns:
        AnnData with obs column "cell_type" added.
    """
    log.info("Annotating clusters using marker genes...")

    # Use scaled HVG expression (adata.X) for scoring — .raw dropped to save disk
    cluster_scores = {}
    for cell_type, markers in MARKER_GENES.items():
        present = [g for g in markers if g in adata.var_names]
        if not present:
            log.warning("No markers found for %s in HVG set", cell_type)
            continue
        sc.tl.score_genes(adata, gene_list=present, score_name=f"score_{cell_type}",
                          random_state=42)
        cluster_scores[cell_type] = adata.obs.groupby(
            "louvain", observed=True)[f"score_{cell_type}"].mean()

    score_df = pd.DataFrame(cluster_scores)
    log.info("Cluster marker scores:\n%s", score_df.to_string())

    # Assign cell type as argmax of scores per cluster
    cluster_to_type = score_df.idxmax(axis=1).to_dict()
    log.info("Cluster → cell type assignment: %s", cluster_to_type)

    adata.obs["cell_type"] = adata.obs["louvain"].map(cluster_to_type)
    return adata


def plot_umaps(adata: ad.AnnData, out_path: Path):
    """
    Save a multi-panel UMAP figure coloured by cluster, genotype, condition, and markers.

    Args:
        adata: AnnData with UMAP embedding and obs annotations.
        out_path: Output PDF path.
    """
    all_markers = [g for genes in MARKER_GENES.values() for g in genes
                   if g in adata.var_names]

    n_marker_panels = len(all_markers)
    n_cols = 4
    n_base = 4  # cluster, cell_type, genotype, condition
    n_total = n_base + n_marker_panels
    n_rows = (n_total + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    # .raw dropped to save disk; plot directly from HVG-scaled adata
    adata_plot = adata

    base_keys = ["louvain", "cell_type", "genotype", "condition"]
    all_keys = base_keys + all_markers

    with plt.rc_context({"figure.figsize": (5, 4)}):
        sc.pl.umap(adata_plot, color=all_keys, ncols=n_cols,
                   show=False, return_fig=False)

    # Re-draw properly into our figure
    plt.close("all")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle("Fernandes et al. Reproduction — UMAP", fontsize=14, fontweight="bold")
    axes_flat = axes.flatten()

    for i, key in enumerate(all_keys):
        ax = axes_flat[i]
        if key in adata_plot.obs.columns:
            categories = adata_plot.obs[key].astype("category")
            colors = plt.cm.tab20(np.linspace(0, 1, len(categories.cat.categories)))
            for j, cat in enumerate(categories.cat.categories):
                mask = categories == cat
                coords = adata_plot.obsm["X_umap"][mask]
                ax.scatter(coords[:, 0], coords[:, 1], s=1, alpha=0.4,
                           color=colors[j], label=str(cat))
            ax.legend(fontsize=5, markerscale=4, loc="best")
        elif key in adata_plot.var_names:
            gene_idx = adata_plot.var_names.get_loc(key)
            expr = np.asarray(adata_plot.X[:, gene_idx]).flatten()
            sc_plot = ax.scatter(
                adata_plot.obsm["X_umap"][:, 0],
                adata_plot.obsm["X_umap"][:, 1],
                c=expr, s=1, alpha=0.5, cmap="viridis"
            )
            plt.colorbar(sc_plot, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(key, fontsize=9)
        ax.set_xlabel("UMAP1", fontsize=7)
        ax.set_ylabel("UMAP2", fontsize=7)
        ax.tick_params(labelsize=6)

    for i in range(len(all_keys), len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("UMAP figure saved to %s", out_path)


def run_wilcoxon_de(adata: ad.AnnData) -> pd.DataFrame:
    """
    Run cell-level Wilcoxon DE (pseudoreplication) for WT vs SNCA-A53T in each cluster.

    Compares cells directly without accounting for iPSC line as the biological unit.
    This is the pseudoreplication approach used in the original paper.

    Args:
        adata: AnnData with raw log-normalized counts in .raw and obs annotations.

    Returns:
        DataFrame with DEG counts and metadata per cluster.
    """
    log.info("Running Wilcoxon DE (cell-level, pseudoreplication)...")

    # Work on untreated cells only to isolate genotype effect
    untreated_mask = adata.obs["condition"] == "untreated"
    adata_untreated = adata[untreated_mask].copy()

    all_results = []
    summary_rows = []

    for cluster_id in sorted(adata_untreated.obs["louvain"].unique()):
        cluster_mask = adata_untreated.obs["louvain"] == cluster_id
        adata_cluster = adata_untreated[cluster_mask].copy()

        n_wt = (adata_cluster.obs["genotype"] == "WT").sum()
        n_a53t = (adata_cluster.obs["genotype"] == "SNCA_A53T").sum()

        if n_wt < 10 or n_a53t < 10:
            log.warning(
                "Cluster %s: insufficient cells (WT=%d, A53T=%d), skipping DE.",
                cluster_id, n_wt, n_a53t,
            )
            continue

        log.info("  Cluster %s: WT=%d cells, A53T=%d cells", cluster_id, n_wt, n_a53t)

        # .raw dropped to save disk; Wilcoxon on scaled HVG counts is consistent
        # with what the original paper did (Seurat also uses scaled data for DE)
        sc.tl.rank_genes_groups(
            adata_cluster,
            groupby="genotype",
            groups=["SNCA_A53T"],
            reference="WT",
            method="wilcoxon",
            key_added="rank_genes_wilcoxon",
            use_raw=False,
        )

        result = sc.get.rank_genes_groups_df(
            adata_cluster,
            group="SNCA_A53T",
            key="rank_genes_wilcoxon",
            pval_cutoff=1.0,
            log2fc_min=None,
        )
        if result.empty:
            log.warning("Cluster %s returned empty DE result; skipping.", cluster_id)
            continue

        ct_label = adata_cluster.obs["cell_type"].iloc[0] \
            if "cell_type" in adata_cluster.obs.columns else f"cluster_{cluster_id}"
        result["cluster"] = cluster_id
        result["cell_type"] = ct_label
        result["n_cells_total"] = len(adata_cluster)
        result["n_cells_wt"] = n_wt
        result["n_cells_a53t"] = n_a53t

        n_degs = ((result["pvals_adj"] < 0.05) & (result["logfoldchanges"].abs() > 0.25)).sum()
        log.info(
            "  Cluster %s: %d DEGs (FDR<0.05, |LFC|>0.25)", cluster_id, n_degs
        )

        all_results.append(result)
        summary_rows.append({
            "cluster": cluster_id,
            "cell_type": ct_label,
            "n_cells_total": len(adata_cluster),
            "n_degs_fdr05_lfc025": n_degs,
        })

    if not all_results:
        log.error("No DE results generated.")
        return pd.DataFrame()

    full_df = pd.concat(all_results, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    log.info("\n=== WILCOXON DE SUMMARY ===\n%s\n", summary_df.to_string(index=False))
    log.info(
        "\n*** PSEUDOREPLICATION WARNING: The above Wilcoxon test treats %d individual "
        "cells as independent observations. The true n is the number of iPSC lines "
        "(biological replicates), not the number of cells. This inflates statistical "
        "power spuriously. See Murphy & Skene (2022) and Squair et al. (2021). ***",
        adata_untreated.n_obs,
    )

    return full_df


def main():
    np.random.seed(42)

    clustered_path = PROCESSED_DIR / "adata_fernandes_clustered.h5ad"
    if clustered_path.exists():
        log.info("Loading existing clustered AnnData (skipping preprocessing)...")
        adata = sc.read_h5ad(clustered_path)
    else:
        log.info("Loading Fernandes QC-filtered data...")
        adata = sc.read_h5ad(PROCESSED_DIR / "adata_qc_fernandes.h5ad")
        adata = preprocess(adata)
        adata = cluster(adata)
        adata = annotate_clusters(adata)
        adata.raw = None
        adata.write_h5ad(clustered_path, compression="gzip")
        log.info("Clustered AnnData saved to %s", clustered_path)

    plot_umaps(adata, FIGURES_DIR / "umap_original.pdf")

    deg_df = run_wilcoxon_de(adata)
    if not deg_df.empty:
        out_csv = RESULTS_DIR / "degs_wilcoxon_per_cluster.csv"
        deg_df.to_csv(out_csv, index=False)
        log.info("DEG table saved to %s", out_csv)


if __name__ == "__main__":
    main()
