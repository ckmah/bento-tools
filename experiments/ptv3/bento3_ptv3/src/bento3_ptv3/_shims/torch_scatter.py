# Minimal CPU shim: PTv3 only uses segment_csr(src, indptr, reduce)
import torch
def segment_csr(src, indptr, reduce="sum"):
    red = {"mean": "mean", "sum": "sum", "max": "max", "min": "min"}[reduce]
    return torch.segment_reduce(src, red, offsets=indptr, axis=0)
