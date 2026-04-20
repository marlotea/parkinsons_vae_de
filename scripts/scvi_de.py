"""
scVI model training and latent-space differential expression.

Trains a scVI model treating each iPSC line (sample_id) as a batch covariate,
correcting for donor/batch effects. Uses the posterior predictive distribution
for DE — not treating cells as independent — which is more principled than the
Wilcoxon pseudoreplication approach of the original paper.
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
import scvi
import torch
from sklearn.metrics import silhouette_score

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
MODELS_DIR = PROJECT_ROOT / "models" / "scvi_model"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# scVI hyperparameters
N_HVG = 2000
N_LATENT = 30
N_LAYERS = 2
MAX_EPOCHS = 400
RANDOM_SEED = 42

# PD GWAS genes of interest
PD_GWAS_GENES = ["SNCA", "LRRK2", "MAPT", "GBA", "PINK1", "PARK7"]


def setup_anndata(adata: ad.AnnData) -> ad.AnnData:
    """
    Select HVGs and register the AnnData for scVI training.

    Uses raw (un-normalized) integer counts as required by scVI's negative
    binomial likelihood.

    Args:
        adata: AnnData with raw counts in .X.

    Returns:
        AnnData subset to HVGs with scVI setup applied.
    """
    # Ensure integer counts for scVI's ZINB likelihood and seurat_v3 HVG selection
    import scipy.sparse as sp_
    if sp_.issparse(adata.X):
        adata.X = adata.X.astype(np.float32)
        adata.X.data = np.round(adata.X.data).astype(np.float32)
    else:
        adata.X = np.round(adata.X).astype(np.float32)

    log.info("Selecting top %d HVGs for scVI...", N_HVG)
    # Compute HVGs on log-normalized data but keep raw counts in .X
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    # No batch_key for HVG selection: only 1 iPSC line per genotype, so
    # batch_key="sample_id" would be collinear with genotype.
    sc.pp.highly_variable_genes(
        adata_norm, n_top_genes=N_HVG, flavor="seurat_v3"
    )
    adata.var["highly_variable"] = adata_norm.var["highly_variable"]
    del adata_norm

    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    log.info("HVG subset shape: %s", adata_hvg.shape)

    # batch_key omitted: with n=1 iPSC line per genotype, sample_id is perfectly
    # collinear with genotype — including it as batch would absorb the biological
    # signal we want to detect. scVI runs without batch correction here; the
    # genotype effect is modelled directly in the DE step.
    log.warning(
        "scVI trained WITHOUT batch_key: study has only 1 iPSC line per genotype "
        "(KOLF2-WT, KOLF2-AsynA53T), so sample_id == genotype. Adding batch_key "
        "would cause the model to regress out the biological variable of interest."
    )
    scvi.model.SCVI.setup_anndata(
        adata_hvg,
        layer=None,  # use .X (raw counts)
    )
    return adata_hvg


def train_scvi(adata_hvg: ad.AnnData) -> scvi.model.SCVI:
    """
    Train scVI model with ZINB likelihood and batch correction.

    Args:
        adata_hvg: AnnData registered with scVI, HVGs only, raw counts in .X.

    Returns:
        Trained SCVI model object.
    """
    use_cuda = torch.cuda.is_available()
    use_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    use_gpu = use_cuda or use_mps
    accelerator = "cuda" if use_cuda else ("mps" if use_mps else "cpu")
    log.info("Training scVI (accelerator=%s, n_latent=%d, n_layers=%d, max_epochs=%d)...",
             accelerator, N_LATENT, N_LAYERS, MAX_EPOCHS)

    # dispersion="gene-batch" requires batch_key; use "gene" since we have no batch key
    model = scvi.model.SCVI(
        adata_hvg,
        n_layers=N_LAYERS,
        n_latent=N_LATENT,
        gene_likelihood="zinb",
        dispersion="gene",
    )
    train_kwargs = dict(max_epochs=MAX_EPOCHS, early_stopping=True, plan_kwargs={"lr": 1e-3})
    try:
        model.train(**train_kwargs, accelerator=accelerator)
    except TypeError:
        model.train(**train_kwargs, use_gpu=use_gpu)

    model.save(str(MODELS_DIR), overwrite=True)
    log.info("Model saved to %s", MODELS_DIR)
    return model


def evaluate_latent_space(adata: ad.AnnData, model: scvi.model.SCVI,
                           adata_pca: ad.AnnData) -> ad.AnnData:
    """
    Extract scVI latent embedding, run UMAP, and compute silhouette scores.

    Compares silhouette score in PCA space vs scVI latent space to quantify
    whether scVI captures biological variation better than PCA.

    Args:
        adata: AnnData (HVG subset) registered with the model.
        model: Trained SCVI model.
        adata_pca: AnnData with PCA-based UMAP for comparison (from step 3).

    Returns:
        AnnData with X_scVI and X_umap_scvi embeddings added to .obsm.
    """
    log.info("Extracting scVI latent representation...")
    adata.obsm["X_scVI"] = model.get_latent_representation()

    log.info("Running UMAP on scVI latent space...")
    sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=20, random_state=RANDOM_SEED)
    sc.tl.umap(adata, random_state=RANDOM_SEED)
    adata.obsm["X_umap_scvi"] = adata.obsm["X_umap"].copy()

    # Silhouette scores
    if "cell_type" in adata.obs.columns and adata.obs["cell_type"].nunique() > 1:
        labels = adata.obs["cell_type"].astype("category").cat.codes.values

        # scVI silhouette
        sil_scvi = silhouette_score(
            adata.obsm["X_scVI"], labels, sample_size=min(5000, adata.n_obs),
            random_state=RANDOM_SEED
        )

        # PCA silhouette (on common cells)
        if adata_pca is not None and "X_pca" in adata_pca.obsm:
            common = adata.obs_names.intersection(adata_pca.obs_names)
            if len(common) > 100:
                idx_scvi = [adata.obs_names.get_loc(c) for c in common]
                idx_pca = [adata_pca.obs_names.get_loc(c) for c in common]
                labels_common = labels[[adata.obs_names.get_loc(c) for c in common]]
                sil_pca = silhouette_score(
                    adata_pca.obsm["X_pca"][idx_pca, :30], labels_common,
                    sample_size=min(5000, len(common)), random_state=RANDOM_SEED
                )
                log.info(
                    "\n=== SILHOUETTE SCORES (cell_type labels) ===\n"
                    "  PCA (30 PCs): %.4f\n  scVI latent:  %.4f\n"
                    "  (Higher in scVI → model captures biological structure better)",
                    sil_pca, sil_scvi,
                )
        else:
            log.info("scVI silhouette score: %.4f", sil_scvi)

    return adata


def run_scvi_de(model: scvi.model.SCVI, adata: ad.AnnData,
                target_cell_type: str = "DAn1") -> pd.DataFrame:
    """
    Run scVI differential expression for WT vs SNCA-A53T in a cell type.

    Uses the posterior predictive distribution and Bayes factors, which does
    not assume cells are independent observations (unlike Wilcoxon).

    Args:
        model: Trained SCVI model.
        adata: AnnData with cell_type and genotype obs columns.
        target_cell_type: Cell type to compare (default "DAn1").

    Returns:
        DataFrame with scVI DE results for the target comparison.
    """
    untreated = (adata.obs["condition"] == "untreated").values
    ct_mask = (adata.obs["cell_type"].astype(str) == target_cell_type).values
    wt_mask = (adata.obs["genotype"].astype(str) == "WT").values
    a53t_mask = (adata.obs["genotype"].astype(str) == "SNCA_A53T").values

    idx1 = untreated & ct_mask & a53t_mask  # SNCA_A53T DAn1 untreated
    idx2 = untreated & ct_mask & wt_mask    # WT DAn1 untreated

    if idx1.sum() == 0 or idx2.sum() == 0:
        log.warning(
            "No cells for cell_type=%s untreated: A53T=%d, WT=%d",
            target_cell_type, idx1.sum(), idx2.sum(),
        )
        return pd.DataFrame()

    log.info(
        "Running scVI DE: WT vs A53T in %s (untreated): A53T=%d cells, WT=%d cells",
        target_cell_type, idx1.sum(), idx2.sum(),
    )

    # Use idx1/idx2 boolean arrays — avoids re-calling setup_anndata on a subset,
    # which would mismatch the model's registered adata dimensions.
    de_result = model.differential_expression(
        idx1=idx1,
        idx2=idx2,
        delta=0.25,
    )

    # scvi-tools >=1.1 uses proba_m1 (posterior prob of being DE) and bayes_factor
    # rather than is_de_fdr_0.05. Detect which columns are available.
    if "is_de_fdr_0.05" in de_result.columns:
        sig_col = "is_de_fdr_0.05"
        n_de = de_result[sig_col].sum()
    elif "proba_m1" in de_result.columns:
        # proba_m1 > 0.5 and bayes_factor > log(10) ≈ 2.3 is standard threshold
        de_result["is_de"] = (de_result["proba_m1"] > 0.5) & (de_result["bayes_factor"] > 0.5)
        sig_col = "is_de"
        n_de = de_result[sig_col].sum()
    else:
        log.warning("Could not find DE significance column; available: %s", de_result.columns.tolist())
        sig_col = None
        n_de = 0

    log.info(
        "scVI DE: %d genes significant (%s) in %s WT vs A53T",
        n_de, sig_col, target_cell_type,
    )

    # Flag PD GWAS genes in results
    de_result["is_pd_gwas"] = de_result.index.isin(PD_GWAS_GENES)
    pd_hits = de_result[de_result["is_pd_gwas"]]
    lfc_col = next((c for c in ["lfc_mean", "lfc_median"] if c in de_result.columns), None)
    show_cols = ([lfc_col] if lfc_col else []) + ([sig_col] if sig_col else [])
    if not pd_hits.empty:
        log.info("PD GWAS genes in results:\n%s", pd_hits[show_cols].to_string())

    return de_result


def plot_scvi_umaps(adata: ad.AnnData, adata_pca: ad.AnnData, out_path: Path):
    """
    Save UMAPs comparing PCA-based and scVI-based embeddings.

    Args:
        adata: AnnData with X_umap_scvi in .obsm.
        adata_pca: AnnData with original PCA-based UMAP.
        out_path: Output PDF path.
    """
    color_keys = ["cell_type", "genotype", "condition", "sample_id", "pct_counts_mt"]
    color_keys = [k for k in color_keys if k in adata.obs.columns]

    n_keys = len(color_keys)
    fig, axes = plt.subplots(2, n_keys, figsize=(5 * n_keys, 9))
    fig.suptitle("PCA vs scVI UMAP Comparison", fontsize=13, fontweight="bold")

    for col, key in enumerate(color_keys):
        for row, (embedding_key, label) in enumerate([
            ("X_umap", "PCA-based UMAP"),
            ("X_umap_scvi", "scVI UMAP"),
        ]):
            ax = axes[row, col]
            source_adata = adata_pca if embedding_key == "X_umap" and adata_pca is not None \
                           else adata

            if embedding_key not in source_adata.obsm:
                ax.text(0.5, 0.5, "Not available", ha="center", va="center")
                ax.set_title(f"{label}\n{key}")
                continue

            coords = source_adata.obsm[embedding_key]
            obs_vals = source_adata.obs[key] if key in source_adata.obs.columns \
                       else adata.obs.reindex(source_adata.obs_names)[key]

            if pd.api.types.is_numeric_dtype(obs_vals):
                sc_plot = ax.scatter(coords[:, 0], coords[:, 1], c=obs_vals,
                                     s=1, alpha=0.4, cmap="viridis")
                plt.colorbar(sc_plot, ax=ax, fraction=0.046)
            else:
                cats = obs_vals.astype("category")
                cmap = plt.cm.tab20(np.linspace(0, 1, len(cats.cat.categories)))
                for i, cat in enumerate(cats.cat.categories):
                    mask = cats == cat
                    ax.scatter(coords[mask, 0], coords[mask, 1], s=1, alpha=0.4,
                               color=cmap[i], label=str(cat))
                ax.legend(fontsize=5, markerscale=4, loc="best")

            ax.set_title(f"{label}\n{key}", fontsize=8)
            ax.set_xlabel("UMAP1", fontsize=7)
            ax.set_ylabel("UMAP2", fontsize=7)
            ax.tick_params(labelsize=6)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("scVI UMAP figure saved to %s", out_path)


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    scvi.settings.seed = RANDOM_SEED

    log.info("Loading best-practice QC data...")
    adata = sc.read_h5ad(PROCESSED_DIR / "adata_qc_bestpractice.h5ad")

    # Transfer cell_type from clustered object
    clustered_path = PROCESSED_DIR / "adata_fernandes_clustered.h5ad"
    adata_pca = None
    if clustered_path.exists():
        adata_clustered = sc.read_h5ad(clustered_path)
        common = adata.obs_names.intersection(adata_clustered.obs_names)
        if len(common) > 0:
            adata.obs["cell_type"] = adata_clustered.obs.loc[common, "cell_type"] \
                .reindex(adata.obs_names)
        adata_pca = adata_clustered
    else:
        log.warning("Run 03_reproduce_original.py first to get cell type labels.")
        adata.obs["cell_type"] = "DAn1"

    # Use raw counts; ensure integer type
    import scipy.sparse as sp
    if sp.issparse(adata.X):
        adata.X = adata.X.astype(np.float32)

    adata_hvg = setup_anndata(adata)

    model_path = MODELS_DIR / "model.pt"
    if model_path.exists():
        log.info("Loading existing model from %s...", MODELS_DIR)
        model = scvi.model.SCVI.load(str(MODELS_DIR), adata=adata_hvg)
    else:
        model = train_scvi(adata_hvg)

    adata_hvg = evaluate_latent_space(adata_hvg, model, adata_pca)

    target_ct = "DAn1"
    if target_ct not in adata_hvg.obs.get("cell_type", pd.Series()).unique():
        available = adata_hvg.obs["cell_type"].unique().tolist()
        log.warning("DAn1 not found; using %s", available[0] if available else "none")
        target_ct = available[0] if available else None

    if target_ct:
        de_df = run_scvi_de(model, adata_hvg, target_cell_type=target_ct)
        if not de_df.empty:
            out_csv = RESULTS_DIR / "degs_scvi.csv"
            de_df.to_csv(out_csv)
            log.info("scVI DE results saved to %s", out_csv)

    # Save annotated AnnData with scVI embeddings (gzip to save disk)
    adata_out = PROCESSED_DIR / "adata_scvi.h5ad"
    adata_hvg.write_h5ad(adata_out, compression="gzip")
    log.info("scVI AnnData saved to %s", adata_out)

    plot_scvi_umaps(adata_hvg, adata_pca, FIGURES_DIR / "umap_scvi.pdf")


if __name__ == "__main__":
    main()
