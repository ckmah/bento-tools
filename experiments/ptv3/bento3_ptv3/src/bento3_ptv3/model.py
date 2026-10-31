"""Pretrained Utonia-small PTv3 encoder (frozen) + LoRA on attention qkv/proj + a new input stem for genes; DINO-style head.
Trainable: gene embedding, stem linear (new: gene emb + coords -> 36), stem LayerNorm + mask token (pretrained init), LoRA A/B,
neck (972 -> emb_dim), projection head + prototypes. Everything else in the encoder is frozen at the Utonia weights."""
import math, os, hashlib, torch, torch.nn as nn, torch.nn.functional as F

UTONIA_SMALL = dict(order=("z", "z-trans", "hilbert", "hilbert-trans"), stride=(2, 2, 2, 2), enc_depths=(2, 2, 2, 6, 2),
                    enc_channels=(36, 72, 144, 288, 432), enc_num_head=(2, 4, 8, 16, 24), enc_patch_size=(1024,) * 5, mlp_ratio=4,
                    qkv_bias=True, pre_norm=True, shuffle_orders=True, enable_rpe=False, upcast_attention=False, upcast_softmax=False,
                    rope_base=10, enc_mode=True, traceable=True)


class LoRALinear(nn.Module):
    """y = W x + b + (alpha / r) * B A x. W, b frozen; B starts at zero, so step 0 is exactly the pretrained layer."""
    def __init__(s, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        s.base = base; s.r = r; s.scale = alpha / r
        for p in s.base.parameters(): p.requires_grad_(False)
        s.A = nn.Parameter(torch.empty(r, base.in_features)); s.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(s.A, a=math.sqrt(5))
    def forward(s, x):
        return s.base(x) + (x @ s.A.t().to(x.dtype)) @ s.B.t().to(x.dtype) * s.scale


def add_lora(module, targets, r, alpha):
    n = 0
    for name, child in list(module.named_modules()):
        for t in targets:
            sub = getattr(child, t, None)
            if name.endswith("attn") and isinstance(sub, nn.Linear):
                setattr(child, t, LoRALinear(sub, r, alpha)); n += 1
    return n


class BatchStd(nn.Module):
    """Standardise each channel with the statistics of the current batch (train and eval alike; no running stats), + affine.
    Same role as BatchNorm in SwAV / DINO-ResNet heads: per-molecule features share a large common component at init (frozen
    multi-level encoder features), and without this the prototype logits barely differ between molecules, so the Sinkhorn targets
    are near-uniform and training drifts to the uniform solution (seen in CPU smoke tests)."""
    def __init__(s, d):
        super().__init__(); s.w = nn.Parameter(torch.ones(d)); s.b = nn.Parameter(torch.zeros(d))
    def forward(s, x):
        x = x.float(); return (x - x.mean(0)) / (x.std(0) + 1e-5) * s.w + s.b


class DINOHead(nn.Module):
    def __init__(s, d_in, hidden, bott, K):
        super().__init__()
        s.mlp = nn.Sequential(nn.Linear(d_in, hidden), BatchStd(hidden), nn.GELU(), nn.Linear(hidden, hidden), BatchStd(hidden), nn.GELU(), nn.Linear(hidden, bott))
        s.proto = nn.utils.parametrizations.weight_norm(nn.Linear(bott, K, bias=False))
        s.proto.parametrizations.weight.original0.data.fill_(1.0); s.proto.parametrizations.weight.original0.requires_grad_(False)   # DINO norm_last_layer
    def forward(s, x):
        return s.proto(F.normalize(s.mlp(x), dim=-1))


class UtoniaLoRA(nn.Module):
    def __init__(s, G, cfg):
        super().__init__()
        from ._vendor.utonia.model import PointTransformerV3
        s.G, s.cfg = G, cfg; s.U_SCALE = 0.01 / cfg.ugrid_um
        s.emb = nn.Embedding(G, cfg.gene_dim)
        s.bb = PointTransformerV3(in_channels=cfg.gene_dim + 3, enable_flash=False, drop_path=cfg.drop_path, mask_token=True, **UTONIA_SMALL)
        C = sum(UTONIA_SMALL["enc_channels"])
        s.neck = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, cfg.emb_dim))
        s.head = DINOHead(cfg.emb_dim, cfg.head_hidden, cfg.head_bottleneck, cfg.prototypes)

    def load_pretrained(s, path):
        ck = torch.load(path, map_location="cpu", weights_only=True); src, dst = ck["state_dict"], s.bb.state_dict()
        for k in ["enc_depths", "enc_channels", "enc_num_head", "stride", "order"]:
            assert tuple(ck["config"][k]) == tuple(UTONIA_SMALL[k]), f"utonia config {k} differs"
        ok = {k: v for k, v in src.items() if k in dst and dst[k].shape == v.shape and not k.startswith("embedding.stem.linear.")}
        s.bb.load_state_dict(ok, strict=False)
        return dict(loaded_tensors=len(ok), ckpt_tensors=len(src), loaded_params=int(sum(v.numel() for v in ok.values())),
                    not_loaded=[k for k in src if k not in ok], fresh_in_backbone=[k for k in dst if k not in ok])

    def setup_trainable(s):
        """Freeze the encoder, then add LoRA (new trainable A/B) and unfreeze the stem. Call after load_pretrained."""
        for p in s.bb.parameters(): p.requires_grad_(False)
        n = add_lora(s.bb.enc, s.cfg.lora_targets, s.cfg.lora_rank, s.cfg.lora_alpha)
        for p in s.bb.embedding.parameters(): p.requires_grad_(True)   # stem linear (new) + LayerNorm + mask token
        return n

    def features(s, v):
        """v: collated batch (coord, grid, gene, mask, coord_n, offset). Returns (upcast 972-d feats, emb_dim-d embedding)."""
        feat = torch.cat([s.emb(v["gene"]), v["coord_n"]], 1)
        p = s.bb(dict(coord=v["coord"] * s.U_SCALE, grid_coord=v["grid"], feat=feat, offset=v["offset"], grid_size=s.cfg.ugrid_um * s.U_SCALE, mask=v["mask"]))
        while "pooling_parent" in p.keys():
            parent = p.pop("pooling_parent"); inv = p.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, p.feat[inv]], -1); p = parent
        up = p.feat.float(); return up, s.neck(up)


def param_report(net):
    tr = sum(p.numel() for p in net.parameters() if p.requires_grad); fr = sum(p.numel() for p in net.parameters() if not p.requires_grad)
    groups = {}
    for n, p in net.named_parameters():
        if not p.requires_grad: continue
        g = "lora" if (n.endswith(".A") or n.endswith(".B")) else n.split(".")[0] if not n.startswith("bb.") else "stem"
        groups[g] = groups.get(g, 0) + p.numel()
    return dict(trainable=tr, frozen=fr, total=tr + fr, trainable_frac=tr / (tr + fr), trainable_by_group=groups)


def fetch_utonia(cfg):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(cfg.utonia_repo, "utonia_small.pth", revision=cfg.utonia_rev, local_dir=os.path.join(cfg.data_dir, "utonia"))
    h = hashlib.sha256(open(p, "rb").read()).hexdigest(); assert h == cfg.utonia_sha256, "sha256 mismatch for utonia_small.pth"
    return p
