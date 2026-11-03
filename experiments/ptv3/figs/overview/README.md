# Overview figure

`overview.png` / `overview.pdf`: the planned design end to end (see the Panel-agnostic JEPA plan). Images are real data from
the 10x Xenium FFPE human breast IDC section; model panels are schematics; numbers are measured (molecules per patch, the
comp20 probe baselines). Panel e's colours are local gene composition, a stand-in for the field E(x), which is not trained yet.

Regenerate (needs the bento3_ptv3 data arrays in DATA, numpy, scipy, matplotlib, pandas, pyarrow):

```bash
python extract_local.py ../../bento3_ptv3/src/bento3_ptv3 DATA OUT
python extract2.py ../../bento3_ptv3/src/bento3_ptv3 DATA OUT
python overview_fig.py OUT
```
