# Reanalysis of Fernandes et al. (2020): Pseudoreplication and Principled DE in iPSC scRNA-seq

CPSC 445 Final Project

Fernandes et al. (2020) profiled iPSC-derived dopaminergic neurons from a
wild-type and an isogenic SNCA-A53T Parkinson's disease line using scRNA-seq,
reporting hundreds of differentially expressed genes. This reanalysis
reproduces their clustering, then applies increasingly rigorous DE methods
to show that most of those findings are statistical artefacts.

**Dataset**: E-MTAB-9154 (ArrayExpress), 18,173 cells across 6 samples
(2 genotypes × 3 conditions: untreated, rotenone, tunicamycin).

---

## What the code does

| Script | Purpose |
|--------|---------|
| `download_data.py` | Fetch raw count matrices from EBI FTP and assemble into a single AnnData |
| `quality_control.py` | Compare Fernandes et al. QC thresholds vs best-practice (10% MT, Scrublet) |
| `clustering.py` | Reproduce original Scanpy clustering and run cell-level Wilcoxon DE |
| `pseudobulk_de.py` | Aggregate to the iPSC-line level and attempt edgeR DE |
| `scvi_de.py` | Train a scVI generative model and run posterior-based DE |
| `figures.py` | Murphy et al. diagnostic plots and method comparison figures |
| `summary.py` | Print a results table consolidating all DE method outcomes |

---

## Key Findings

1. **Pseudoreplication inflates DEGs.** The Wilcoxon test yields 525 DEGs across
   all clusters (FDR < 0.05), with DEG count trending positively with cluster
   size (Pearson r = 0.49) — a hallmark of spurious inflation. Permutation
   controls confirm the test produces abundant DEGs even under shuffled labels.

2. **n = 1 per genotype makes pseudobulk impossible.** The study used a single
   KOLF2-WT line and a single KOLF2-AsynA53T line. After pseudobulk aggregation,
   every cell-type group has exactly one biological replicate per genotype.
   edgeR cannot estimate dispersion and was not run; this is itself a key result.

3. **QC bias from a lenient mitochondrial cutoff.** The original 40% MT threshold
   retains 840 cells with MT% between 10–40% (mean 13.1%) that best-practice QC
   removes — preferentially from untreated samples (440 cells), which can
   confound the genotype comparison.

4. **scVI finds a largely non-overlapping gene set.** Applied to DAn1 untreated
   cells, scVI identifies 136 DEGs — sharing only 7 genes with the 107 Wilcoxon
   DAn1 hits. LFC correlation between methods is weak (Pearson r = 0.22). PD
   GWAS genes called significant by Wilcoxon (SNCA, MAPT) have negative Bayes
   factors under scVI, consistent with pseudoreplication artefact.

---

## References

- Fernandes et al. (2020) *Cell Reports* 33: 108263
- Murphy & Skene (2022) *Nature Commun.* 13: 7851 — pseudoreplication in scRNA-seq
- Murphy et al. (2023) *eLife* 12: RP90214 — diagnostic framework
- Squair et al. (2021) *Nature Commun.* 12: 5692 — pseudobulk best practices
- Lopez et al. (2018) *Nature Methods* 15: 1053–1058 — scVI
- Amezquita et al. (2020) *Nature Methods* 17: 137–145 — QC best practices
