# Minimal shim: PTv3 only uses segment_csr(src, indptr, reduce)
import torch
def segment_csr(src, indptr, reduce="sum"):
    """torch.segment_reduce over CSR segments. Reduces in fp32 and returns src's dtype: under bf16 autocast the pooled features
    arrive in bf16, and fp32 keeps sums/means over a segment exact enough and avoids relying on bf16 kernel coverage (CUDA backward)."""
    red = {"mean": "mean", "sum": "sum", "max": "max", "min": "min"}[reduce]
    out = torch.segment_reduce(src.float() if src.is_floating_point() else src, red, offsets=indptr, axis=0)
    return out.to(src.dtype)
