# Phase 0: data, vocabulary, split, baselines

Everything phase 1 needs, built once and frozen. Inputs are the Xenium breast arrays from `bento3_ptv3.data.build_from_10x`
(`xyz.npy`, `gene.npy`, `nuc.npy`, `cell.npy`, `clust.npy`, `genes.json`, `nucleus_bnd.npz`, `cell_bnd.npz`), held in the private
HF repo `ckmah/bento3-ptv3-runs` under `data/xenium_breast_idc_v1`.

| Step | Script | Output |
| --- | --- | --- |
| Gene vocabulary | `vocab.py HGNC_TSV CELLWORLD_VOCAB OUT xenium_breast_idc=genes.json` | `vocab.json` (CellWorld's 19,228 tokens; new genes appended), `panel_<name>.json` |
| CellWorld checkpoints | see below | `small|base|large|huge/{model.pt,config.yaml}`, sha256 checked against the repo's `manifest.json` |
| Split | `split.py DATA OUT` | `split.json`, `role.npy` (-1 guard, 0 train, 1 validation, 2 probe-fit, 3 probe-test) |
| Baselines | `baselines.py DATA OUT` | `baselines.json` |

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
