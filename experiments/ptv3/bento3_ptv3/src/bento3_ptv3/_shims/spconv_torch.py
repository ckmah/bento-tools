"""Pure-PyTorch submanifold sparse conv: the slice of spconv 2.x that PTv3 uses (SparseConvTensor, SubMConv3d,
modules.is_spconv_module). Used where no spconv wheel fits (CUDA 13 / Blackwell on molab). Same maths as spconv's
SubMConv3d: output only at active voxels, out[i] = bias + sum_k W_k x[neighbour_k(i)], neighbours within the same batch item.
Weight layout matches spconv 2.x: (out, k, k, k, in). Neighbour maps are cached per indice_key on the tensor, like spconv."""
import math, types, torch, torch.nn as nn

class SparseConvTensor:
    def __init__(self, features, indices, spatial_shape, batch_size, indice_dict=None):
        self.features, self.indices, self.spatial_shape, self.batch_size = features, indices, list(spatial_shape), batch_size
        self.indice_dict = {} if indice_dict is None else indice_dict
    def replace_feature(self, f):
        return SparseConvTensor(f, self.indices, self.spatial_shape, self.batch_size, self.indice_dict)

class SparseModule(nn.Module):
    pass

def _neighbours(t, k):
    """(K, N) index of the voxel at each kernel offset, -1 where empty. Hash = sorted int64 keys + searchsorted."""
    r = k // 2; ind = t.indices.long(); dev = ind.device
    dims = [t.batch_size] + [s + 2 * r + 1 for s in t.spatial_shape]
    def key(c):  # c: (..., 4) with batch first; spatial coords shifted by r so offsets stay non-negative
        return ((c[..., 0] * dims[1] + c[..., 1] + r) * dims[2] + c[..., 2] + r) * dims[3] + c[..., 3] + r
    sk, order = torch.sort(key(ind))
    rng = torch.arange(-r, r + 1, device=dev)
    off = torch.stack(torch.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)        # (K, 3), same order as weight k,k,k
    q = key(torch.cat([ind[None, :, :1].expand(len(off), -1, -1), ind[None, :, 1:] + off[:, None]], -1))  # (K, N)
    pos = torch.searchsorted(sk, q).clamp(max=len(sk) - 1)
    return torch.where(sk[pos] == q, order[pos], torch.full_like(pos, -1))

class SubMConv3d(SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, indice_key=None, **kw):
        super().__init__()
        assert stride == 1 and dilation == 1 and groups == 1, "shim supports plain submanifold conv only"
        self.k, self.indice_key = kernel_size, indice_key
        self.weight = nn.Parameter(torch.empty(out_channels, kernel_size, kernel_size, kernel_size, in_channels))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        fan_in = in_channels * kernel_size ** 3; b = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.weight, -b, b)                    # same scale as spconv's kaiming_uniform_(a=sqrt(5))
        if self.bias is not None: nn.init.uniform_(self.bias, -b, b)
    def forward(self, t):
        ck = (self.indice_key, self.k)
        pairs = t.indice_dict.get(ck)
        if pairs is None or pairs[0] != t.features.shape[0]:   # per tap: (output rows, input rows) where the neighbour exists
            nb = _neighbours(t, self.k); m = nb >= 0
            pairs = (t.features.shape[0], [(m[j].nonzero().squeeze(1), nb[j][m[j]]) for j in range(nb.shape[0])]); t.indice_dict[ck] = pairs
        x = t.features; W = self.weight.reshape(self.weight.shape[0], -1, x.shape[1])          # (out, K, in)
        # Under autocast the per-tap matmul runs in the autocast dtype (bf16 on molab) while x may be fp32 (e.g. straight out of a
        # LayerNorm), so the products and `out` can differ in dtype. Accumulate the taps in fp32 (index_add needs equal dtypes, and
        # fp32 keeps the up-to-27-term sum exact enough), add the bias in fp32, then return in the matmul's dtype, like a Linear would.
        out = x.new_zeros(x.shape[0], W.shape[0], dtype=torch.float32); res_dtype = None
        for j, (oi, ii) in enumerate(pairs[1]):   # each input row is the j-th neighbour of at most one output row: deterministic adds
            if len(oi):
                y = x[ii] @ W[:, j].t(); res_dtype = y.dtype
                out = out.index_add(0, oi, y.float())
        if res_dtype is None: res_dtype = (x[:0] @ W[:, 0].t()).dtype   # no active pairs: still report the dtype a matmul would give
        if self.bias is not None: out = out + self.bias.float()
        return t.replace_feature(out.to(res_dtype))

modules = types.SimpleNamespace(is_spconv_module=lambda m: isinstance(m, SparseModule))
