"""
Download and load raw count matrices from ArrayExpress E-MTAB-9154.

Fetches Fernandes et al. (2020) iPSC-derived dopaminergic neuron scRNA-seq data
via EBI FTP. Data is stored as 6 TSV files (genes × cells) split across 3 zip
archives. Annotates each sample with genotype, condition, and batch metadata,
then concatenates into a single AnnData saved as processed/adata_raw.h5ad.

Accession : E-MTAB-9154 (ArrayExpress / EBI BioStudies)
FTP path   : ftp.ebi.ac.uk/pub/databases/arrayexpress/data/experiment/MTAB/E-MTAB-9154/

Study design (from SDRF):
  - 2 genotypes : KOLF2-WT ("WT"), KOLF2-AsynA53T ("SNCA_A53T")
  - 3 conditions: untreated, rotenone, tunicamycin
  - 1 iPSC line per genotype  ← KEY FINDING: n=1 biological replicate per group
"""

import ftplib
import io
import logging
import sys
import zipfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = PROJECT_ROOT / "processed"
DATA_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

FTP_HOST = "ftp.ebi.ac.uk"
FTP_DIR = "/pub/databases/arrayexpress/data/experiment/MTAB/E-MTAB-9154/"

# Maps TSV filename stem → (sample_id, genotype, condition)
# Derived from SDRF: HE_* = KOLF2-AsynA53T (heterozygous A53T), WT_* = KOLF2-WT
SAMPLE_MAP = {
    "HE_ROT":  ("A53T_ROT", "SNCA_A53T", "rotenone"),
    "HE_TUN":  ("A53T_TUN", "SNCA_A53T", "tunicamycin"),
    "HE_UNT":  ("A53T_UNT", "SNCA_A53T", "untreated"),
    "WT_ROT":  ("WT_ROT",   "WT",         "rotenone"),
    "WT_TUN":  ("WT_TUN",   "WT",         "tunicamycin"),
    "WT_UNT":  ("WT_UNT",   "WT",         "untreated"),
}

# The 3 processed zip archives and their contents (from SDRF)
ZIP_FILES = {
    "E-MTAB-9154.processed.1.zip": ["HE_ROT.tsv", "HE_TUN.tsv"],
    "E-MTAB-9154.processed.2.zip": ["HE_UNT.tsv", "WT_ROT.tsv"],
    "E-MTAB-9154.processed.3.zip": ["WT_TUN.tsv", "WT_UNT.tsv"],
}


def ftp_download(ftp_path: str, dest: Path) -> Path:
    """
    Download a single file from EBI FTP, skipping if already present.

    Args:
        ftp_path: Filename within FTP_DIR.
        dest: Local destination path.

    Returns:
        Path to the downloaded (or cached) file.
    """
    if dest.exists():
        log.info("  Cached: %s", dest.name)
        return dest

    log.info("  Downloading %s ...", ftp_path)
    ftp = ftplib.FTP(FTP_HOST)
    ftp.login()
    ftp.cwd(FTP_DIR)

    with open(dest, "wb") as fp:
        ftp.retrbinary(f"RETR {ftp_path}", fp.write)
    ftp.quit()
    log.info("  Saved to %s", dest)
    return dest


def load_tsv_as_adata(tsv_stream: io.IOBase, sample_id: str,
                       genotype: str, condition: str) -> ad.AnnData:
    """
    Parse a genes × cells TSV stream into an AnnData object.

    The TSV has gene names as the first column (row index) and cell barcodes
    as the header row. Values are raw integer counts stored as floats.

    Args:
        tsv_stream: File-like object for the TSV.
        sample_id: Sample identifier string.
        genotype: "WT" or "SNCA_A53T".
        condition: "untreated", "rotenone", or "tunicamycin".

    Returns:
        AnnData with shape (n_cells, n_genes) and obs metadata columns.
    """
    df = pd.read_csv(tsv_stream, sep="\t", index_col=0)
    # df is genes × cells; transpose to cells × genes for AnnData convention
    df = df.T
    df.index = [f"{sample_id}_{bc}" for bc in df.index]

    X = sp.csr_matrix(df.values.astype(np.int32))
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=df.index),
        var=pd.DataFrame(index=df.columns),
    )
    adata.var_names_make_unique()

    adata.obs["sample_id"] = sample_id
    adata.obs["genotype"] = genotype
    adata.obs["condition"] = condition
    adata.obs["batch"] = sample_id  # 1 iPSC line per sample → sample = batch

    log.info(
        "  Loaded %s: %d cells × %d genes (genotype=%s, condition=%s)",
        sample_id, adata.n_obs, adata.n_vars, genotype, condition,
    )
    return adata


def main():
    np.random.seed(42)

    adatas = []
    sample_records = []

    for zip_name, tsv_names in ZIP_FILES.items():
        zip_dest = DATA_DIR / zip_name
        ftp_download(zip_name, zip_dest)

        with zipfile.ZipFile(zip_dest) as zf:
            for tsv_name in tsv_names:
                stem = tsv_name.replace(".tsv", "")
                if stem not in SAMPLE_MAP:
                    log.warning("No metadata mapping for %s; skipping.", tsv_name)
                    continue

                sample_id, genotype, condition = SAMPLE_MAP[stem]
                log.info("Processing %s → sample_id=%s", tsv_name, sample_id)

                with zf.open(tsv_name) as f:
                    adata = load_tsv_as_adata(f, sample_id, genotype, condition)

                adatas.append(adata)
                sample_records.append({
                    "sample_id": sample_id,
                    "genotype": genotype,
                    "condition": condition,
                    "ipsc_line": "KOLF2-AsynA53T" if genotype == "SNCA_A53T" else "KOLF2-WT",
                    "n_cells": adata.n_obs,
                })

    if not adatas:
        log.error("No samples loaded.")
        sys.exit(1)

    # Print sample table
    sample_df = pd.DataFrame(sample_records).sort_values(["genotype", "condition"])
    log.info("\n\n=== SAMPLE TABLE ===\n%s\n", sample_df.to_string(index=False))

    # Concatenate — all samples share the same gene set so inner join is fine
    log.info("Concatenating %d samples...", len(adatas))
    adata_raw = ad.concat(adatas, join="inner")

    # Restore obs metadata (lost during concat)
    obs_meta = pd.concat([a.obs for a in adatas])[
        ["sample_id", "genotype", "condition", "batch"]
    ]
    for col in obs_meta.columns:
        adata_raw.obs[col] = obs_meta[col]

    log.info("\n=== RAW ADATA SUMMARY ===")
    log.info("Total cells pre-QC : %d", adata_raw.n_obs)
    log.info("Total genes         : %d", adata_raw.n_vars)

    per_sample = (
        adata_raw.obs.groupby(["sample_id", "genotype", "condition"])
        .size()
        .reset_index(name="n_cells")
    )
    log.info("\nPer-sample cell counts:\n%s\n", per_sample.to_string(index=False))

    log.info(
        "\n*** KEY FINDING: Only 1 iPSC line per genotype (KOLF2-WT and KOLF2-AsynA53T). "
        "This means n=1 biological replicate per genotype group. Pseudobulk DE is "
        "structurally impossible, and Wilcoxon cell-level DE has no valid replicate basis. "
        "The 'batch' covariate in scVI (sample_id) captures condition, not donor variability. ***"
    )

    out_path = PROCESSED_DIR / "adata_raw.h5ad"
    adata_raw.write_h5ad(out_path)
    log.info("Saved raw AnnData to %s", out_path)


if __name__ == "__main__":
    main()
