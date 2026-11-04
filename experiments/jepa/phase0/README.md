# Phase 0: data, vocabulary, split, baselines

Everything phase 1 needs, built once and frozen. Inputs are the Xenium breast arrays from `bento3_ptv3.data.build_from_10x`
(`xyz.npy`, `gene.npy`, `nuc.npy`, `cell.npy`, `clust.npy`, `genes.json`, `nucleus_bnd.npz`, `cell_bnd.npz`), held in the private
HF repo `ckmah/bento3-ptv3-runs` under `data/xenium_breast_idc_v1`.

| Step | Script | Output |
| --- | --- | --- |
| Gene vocabulary | `python -m bento_jepa.vocab HGNC_TSV CELLWORLD_VOCAB OUT xenium_breast_idc=genes.json` | `vocab.json` (CellWorld's 19,228 tokens; new genes appended), `panel_<name>.json` |
| CellWorld checkpoints | see below | `small|base|large|huge/{model.pt,config.yaml}`, sha256 checked against the repo's `manifest.json` |
| Split | `python -m bento_jepa.split DATA OUT` | `split.json`, `role.npy` (-1 guard, 0 train, 1 validation, 2 probe-fit, 3 probe-test) |
| Baselines | `python -m bento_jepa.baselines DATA OUT` | `baselines.json` |

Inputs from outside: the HGNC complete set
(`https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt`) and CellWorld's
`data/gene_vocab.json` (github.com/UoM-HealthAI/CellWorld, MIT).

## CellWorld checkpoints

Public on Hugging Face at `Haiping-UoM/CellWorld`, although its `manifest.json` says `"intended_visibility": "private"`. Keep
copies private and ask the authors before publishing results built on them.

```bash
B=https://huggingface.co/Haiping-UoM/CellWorld/resolve/main
for s in small base large huge; do mkdir -p $s; curl -sSL -o $s/config.yaml $B/$s/config.yaml; curl -sSL -o $s/model.pt $B/$s/model.pt; done
```

| Scale | sha256 (first 12) | Size |
| --- | --- | --- |
| small | 6fa0e149a024 | 30 MB |
| base | 567f90bed469 | 110 MB |
| large | 77f1eb38eb8d | 249 MB |
| huge | 6001593e462a | 681 MB |

All four are the final step-10,000 checkpoints.

## Leakage

NexuST's Supplementary Table 1 (same group and corpus, HumanST-46M = CellWorld's 46M cells) lists **Xenium V1 FFPE Human Breast
IDC** (574,852 cells, 280 genes): the dataset our arrays come from (`Xenium_V1_FFPE_Human_Breast_IDC_outs.zip`; 571,833 cell ids
in our arrays). Its "Big 1", "Big 2", "With Addon" and ILC sections are in the corpus too. So CellWorld has seen this tissue, as
segmented cells: phase 1 picks the starting point only, and the benefit of starting from CellWorld is tested on sections it never
saw.

## Results (4 Nov 2026)

`results/` holds the small outputs: `vocab.json`, `panel_xenium_breast_idc.json`, `split.json` (block roles and the sha256 of
`role.npy`, which is 68 MB and goes to the private HF repo) and `baselines.json`.

| Baseline (probe-test) | Score |
| --- | --- |
| Subcellular comp20: nucleus vs cytoplasm | AUROC 0.824 |
| Subcellular comp20: signed distance to nucleus boundary | Spearman 0.635, R² 0.35 |
| Subcellular comp20: distance to cell boundary | Spearman 0.337, R² 0.11 |
| Cell type (26 clusters), own counts | balanced accuracy 0.840 |
| Niche composition (50 um), own counts | Pearson 0.728, R² 0.29 |
| Niche composition (50 um), 50 um gene composition | Pearson 0.850, R² 0.47 |
| Niche composition (50 um), CellCharter (12 niches) | Pearson 0.772, R² 0.33 |

The plan's "CellCompass" niche baseline is squidpy's CellCharter flavor (`calculate_niche(flavor="cellcharter")`); squidpy has
no method by that name. Baselines need about 13 min and under 15 GB of RAM on 4 CPUs.
