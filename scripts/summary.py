"""
Results summary table for the Fernandes et al. (2020) reanalysis.

Prints a clean comparison of DE results across methods, QC impact statistics,
and biological replicate counts to consolidate the key findings.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_wilcoxon_stats() -> dict:
    """
    Load Wilcoxon DE results and compute summary statistics for DAn1.

    Returns:
        Dict with keys: n_degs, pearson_r, median_lfc.
    """
    path = RESULTS_DIR / "degs_wilcoxon_per_cluster.csv"
    if not path.exists():
        return {"n_degs": "N/A", "pearson_r": "N/A", "median_lfc": "N/A"}

    df = pd.read_csv(path)

    # Per-cluster DEG counts and cell counts for Pearson r
    if "cluster" in df.columns and "n_cells_total" in df.columns:
        summary = df.groupby("cluster").agg(
            n_cells=("n_cells_total", "first"),
            n_degs=("pvals_adj", lambda x: (
                (x < 0.05) & (df.loc[x.index, "logfoldchanges"].abs() > 0.25)
            ).sum()),
        )
        if len(summary) > 1:
            r, _ = stats.pearsonr(summary["n_cells"], summary["n_degs"])
        else:
            r = float("nan")
    else:
        r = float("nan")

    # DAn1 specific stats
    dan1_mask = df.get("cell_type", df.get("cluster", pd.Series())) == "DAn1"
    if isinstance(dan1_mask, str):
        dan1 = df
    else:
        dan1 = df[dan1_mask] if dan1_mask.any() else df

    sig = dan1[
        (dan1["pvals_adj"] < 0.05) & (dan1["logfoldchanges"].abs() > 0.25)
    ] if "pvals_adj" in dan1.columns else dan1

    n_degs = len(sig)
    median_lfc = float(sig["logfoldchanges"].abs().median()) if not sig.empty else float("nan")

    return {
        "n_degs": n_degs,
        "pearson_r": f"{r:.3f}" if not np.isnan(r) else "N/A (single cluster)",
        "median_lfc": f"{median_lfc:.3f}" if not np.isnan(median_lfc) else "N/A",
    }


def load_pseudobulk_stats() -> dict:
    """
    Load pseudobulk edgeR results and compute summary statistics.

    Returns:
        Dict with keys: n_degs, pearson_r, median_lfc.
    """
    path = RESULTS_DIR / "degs_pseudobulk_edger.csv"
    if not path.exists():
        return {"n_degs": "N/A", "pearson_r": "N/A", "median_lfc": "N/A"}

    df = pd.read_csv(path)

    if "note" in df.columns and "Insufficient" in str(df["note"].iloc[0]):
        return {
            "n_degs": "N/A (n<3 replicates)",
            "pearson_r": "N/A",
            "median_lfc": "N/A",
        }

    if "FDR" not in df.columns:
        return {"n_degs": "error", "pearson_r": "N/A", "median_lfc": "N/A"}

    sig = df[df["FDR"] < 0.05]
    n_degs = len(sig)
    median_lfc = float(sig["logFC"].abs().median()) if not sig.empty else float("nan")

    # Only one comparison — no per-cluster Pearson r
    return {
        "n_degs": n_degs,
        "pearson_r": "N/A (single comparison)",
        "median_lfc": f"{median_lfc:.3f}" if not np.isnan(median_lfc) else "N/A",
    }


def load_scvi_stats() -> dict:
    """
    Load scVI DE results and compute summary statistics for DAn1.

    Returns:
        Dict with keys: n_degs, pearson_r, median_lfc.
    """
    path = RESULTS_DIR / "degs_scvi.csv"
    if not path.exists():
        return {"n_degs": "N/A", "pearson_r": "N/A", "median_lfc": "N/A"}

    df = pd.read_csv(path, index_col=0)

    if "is_de_fdr_0.05" not in df.columns:
        return {"n_degs": "error", "pearson_r": "N/A", "median_lfc": "N/A"}

    sig = df[df["is_de_fdr_0.05"]]
    n_degs = len(sig)
    lfc_col = "lfc_mean" if "lfc_mean" in sig.columns else None
    median_lfc = float(sig[lfc_col].abs().median()) if lfc_col and not sig.empty \
        else float("nan")

    return {
        "n_degs": n_degs,
        "pearson_r": "N/A (single comparison)",
        "median_lfc": f"{median_lfc:.3f}" if not np.isnan(median_lfc) else "N/A",
    }


def load_qc_stats() -> dict:
    """
    Load QC-filtered AnnData objects and compute per-condition cell counts.

    Returns:
        Dict with QC summary statistics.
    """
    stats_dict = {}

    for key, fname in [("raw", "adata_raw.h5ad"),
                        ("fernandes", "adata_qc_fernandes.h5ad"),
                        ("bestpractice", "adata_qc_bestpractice.h5ad")]:
        path = PROCESSED_DIR / fname
        if path.exists():
            adata = sc.read_h5ad(path)
            stats_dict[f"n_cells_{key}"] = adata.n_obs
            stats_dict[f"n_genes_{key}"] = adata.n_vars
        else:
            stats_dict[f"n_cells_{key}"] = "N/A"
            stats_dict[f"n_genes_{key}"] = "N/A"

    return stats_dict


def load_replicate_counts() -> pd.DataFrame:
    """
    Load pseudobulk metadata to count biological replicates per condition.

    Returns:
        DataFrame summarizing replicate counts per cell_type and genotype.
    """
    path = RESULTS_DIR / "pseudobulk_metadata.csv"
    if not path.exists():
        return pd.DataFrame({"note": ["Run step 04 to generate pseudobulk metadata."]})

    meta = pd.read_csv(path)
    untreated = meta[meta["condition"] == "untreated"]
    summary = untreated.groupby(["cell_type", "genotype"]).size().reset_index(name="n_replicates")
    return summary


def main():
    wil = load_wilcoxon_stats()
    pb = load_pseudobulk_stats()
    scvi = load_scvi_stats()
    qc = load_qc_stats()
    rep_df = load_replicate_counts()

    print("\n" + "=" * 75)
    print("  CPSC 445 FINAL PROJECT — RESULTS SUMMARY")
    print("  Reanalysis of Fernandes et al. (2020)")
    print("=" * 75)

    print("\n--- DE METHOD COMPARISON (DAn1: WT vs SNCA-A53T, untreated) ---\n")
    rows = [
        ["Wilcoxon (pseudoreplication)", wil["n_degs"], wil["pearson_r"], wil["median_lfc"]],
        ["Pseudobulk edgeR",             pb["n_degs"],  pb["pearson_r"],  pb["median_lfc"]],
        ["scVI (generative model)",      scvi["n_degs"], scvi["pearson_r"], scvi["median_lfc"]],
    ]
    header = ["Method", "n_DEGs (DAn1)", "Pearson r (DEGs~cells)", "Median |LFC|"]
    col_widths = [max(len(header[i]), max(len(str(r[i])) for r in rows)) for i in range(4)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)

    print(fmt.format(*header))
    print("  " + "-" * (sum(col_widths) + 6))
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))

    print("\n--- QC IMPACT ---\n")
    print(f"  Raw cells (pre-QC):              {qc.get('n_cells_raw', 'N/A'):>10}")
    print(f"  Cells after Fernandes QC:         {qc.get('n_cells_fernandes', 'N/A'):>10}")
    print(f"  Cells after best-practice QC:     {qc.get('n_cells_bestpractice', 'N/A'):>10}")

    n_raw = qc.get("n_cells_raw", 0)
    n_fern = qc.get("n_cells_fernandes", 0)
    n_bp = qc.get("n_cells_bestpractice", 0)

    if all(isinstance(v, int) for v in [n_raw, n_fern, n_bp]):
        pct_fern = 100 * (n_raw - n_fern) / n_raw if n_raw > 0 else 0
        pct_bp = 100 * (n_raw - n_bp) / n_raw if n_raw > 0 else 0
        print(f"  Removed by Fernandes QC:          {n_raw-n_fern:>10,} ({pct_fern:.1f}%)")
        print(f"  Removed by best-practice QC:      {n_raw-n_bp:>10,} ({pct_bp:.1f}%)")
        if n_fern > n_bp:
            extra = n_fern - n_bp
            print(f"\n  *** KEY FINDING: Best-practice removes {extra:,} additional cells")
            print("      retained by Fernandes. These are likely dead/dying cells")
            print("      (10–40% MT reads). Inclusion biases the DE analysis. ***")

    print("\n--- BIOLOGICAL REPLICATES (untreated cells, per genotype) ---\n")
    if "note" in rep_df.columns:
        print(f"  {rep_df['note'].iloc[0]}")
    else:
        print(rep_df.to_string(index=False))
        min_reps = rep_df["n_replicates"].min() if "n_replicates" in rep_df.columns else 0
        if min_reps < 3:
            print(f"\n  *** KEY FINDING: Minimum replicates = {min_reps} (need >= 3 for valid")
            print("      pseudobulk DE). The study is statistically underpowered at")
            print("      the appropriate unit of replication (iPSC line, not cell). ***")

    print("\n--- INTERPRETATION ---")
    print("""
  1. PSEUDOREPLICATION: Wilcoxon DE inflates significance by treating cells
     as independent observations. The iPSC line is the true biological unit.

  2. UNDERPOWERED: Even ignoring pseudoreplication, the study may lack
     sufficient iPSC line replicates (n) for valid between-group comparisons.

  3. QC BIAS: The 40% MT cutoff retains potentially dead/dying cells,
     which may confound DE results (cells under stress have elevated MT%).

  4. scVI ADVANTAGE: By modelling batch/donor as a covariate and using
     posterior distributions, scVI provides a more principled DE framework
     than the cell-level Wilcoxon approach.

  References:
  - Murphy & Skene (2022) Nature Neurosci. — pseudoreplication in scRNA-seq
  - Squair et al. (2021) Nature Commun. — pseudobulk best practices
  - Murphy et al. (2023) eLife — diagnostic framework
  - Amezquita et al. (2020) Nature Methods — QC best practices
""")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
