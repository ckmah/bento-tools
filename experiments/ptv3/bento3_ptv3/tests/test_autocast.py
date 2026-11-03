"""Regression: the molab run (CUDA, bf16 autocast) crashed in the sparse-conv shim (index_add: Float vs BFloat16).
CPU bf16 autocast takes the same dtype path, so these run without a GPU or the pretrained weights."""
import types, numpy as np, pytest, torch
from bento3_ptv3 import TrainConfig
from bento3_ptv3._shims.spconv_torch import SparseConvTensor, SubMConv3d
from bento3_ptv3.data import make_view
from bento3_ptv3.evaluate import collate
from bento3_ptv3.model import UtoniaLoRA
from bento3_ptv3.train import _amp

AC = dict(device_type="cpu", dtype=torch.bfloat16)


def _sparse(n=300, s=8, c=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    ind = torch.unique(torch.cat([torch.randint(0, 2, (n, 1), generator=g), torch.randint(0, s, (n, 3), generator=g)], 1).int(), dim=0)
    return SparseConvTensor(torch.randn(len(ind), c, generator=g), ind, [s] * 3, 2)


@pytest.mark.parametrize("in_dtype", [torch.float32, torch.bfloat16])
def test_subm_conv_under_bf16_autocast(in_dtype):
    torch.set_num_threads(1)   # multi-threaded CPU bf16 GEMM gave sporadic wrong blocks on the dev box (torch 2.15); 1 thread is exact
    torch.manual_seed(0); t = _sparse(); conv = SubMConv3d(16, 8, 3, indice_key="k")
    x = t.features.to(in_dtype)
    ref = conv(t.replace_feature(x.float())).features        # fp32 reference on the same (rounded) inputs, no autocast
    assert ref.dtype == torch.float32
    with torch.autocast(**AC):
        y = conv(t.replace_feature(x)).features
    assert y.dtype == torch.bfloat16 and torch.isfinite(y).all()
    assert torch.allclose(y.float(), ref, atol=5e-2, rtol=5e-2)


def test_amp_dict_cpu_opt_in():
    cpu = torch.device("cpu")
    assert not _amp(TrainConfig(amp="bf16"), cpu)["enabled"]
    assert _amp(TrainConfig(amp="bf16", amp_cpu=True), cpu) == dict(AC, enabled=True)


def _tiny(rng, n=600, G=12):
    return types.SimpleNamespace(xyz=np.c_[rng.uniform(0, 20, (n, 2)), rng.uniform(0, 5, n)].astype(np.float32), gene=rng.integers(0, G, n), zmin=0.0, G=G)


def test_tiny_forward_backward_bf16_autocast():
    torch.manual_seed(0); rng = np.random.default_rng(0)
    cfg = TrainConfig(objective="dino", prototypes=64, head_hidden=64, head_bottleneck=32, drop_path=0.0, win_um=20.0)
    n, G = 600, 12
    M = _tiny(rng, n, G)
    net = UtoniaLoRA(G, cfg); net.setup_trainable()
    views = [make_view(M, np.arange(i, i + 300), rng, cfg) for i in (0, 300)]
    b = collate(views, torch.device("cpu"))
    with torch.autocast(**AC):
        up, e = net.features(b); z = net.head.mlp(e).float(); logits = net.head.proto(torch.nn.functional.normalize(z, dim=-1)).float()
    assert up.dtype == torch.float32 and up.shape == (n, 972)
    loss = -torch.log_softmax(logits / 0.1, -1).mean()
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in net.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() and g.dtype == torch.float32 for g in grads)


def test_recon_view_and_head():
    """objective='recon': masked molecules leave the student input, one query point per masked patch takes their place, and every
    removed molecule is a target of exactly one query; the recon head gives count + layout + per-gene outputs per query."""
    from bento3_ptv3.data import make_recon_view, N_LAYOUT
    torch.manual_seed(0); rng = np.random.default_rng(1)
    cfg = TrainConfig(objective="recon", head_hidden=64, drop_path=0.0, win_um=20.0, mask_patch_um=(1.0, 3.0), mask_ratio=0.4)
    M = _tiny(rng); idx = np.arange(600)
    v = make_recon_view(M, idx, rng, cfg); nq = v["nq"]; n_in = len(v["gene"])
    assert nq > 0 and v["mask"][-nq:].all() and not v["mask"][:-nq].any()        # queries are the masked rows, at the end
    assert n_in - nq + len(v["tq"]) == len(idx)                                      # kept + removed = all molecules
    assert set(np.unique(v["tq"])) == set(range(nq))                                 # each query covers >= 1 removed molecule
    assert v["tlay"].min() >= 0 and v["tlay"].max() < N_LAYOUT
    assert len(np.unique(v["grid"], axis=0)) == n_in                                 # voxels stay unique
    net = UtoniaLoRA(M.G, cfg); net.setup_trainable()
    b = collate([v], torch.device("cpu"))
    with torch.autocast(**AC):
        _, e = net.features(b); out = net.recon(e[n_in - nq:]).float()
    assert out.shape == (nq, 1 + N_LAYOUT + M.G)
    out.pow(2).mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in net.recon.parameters())
