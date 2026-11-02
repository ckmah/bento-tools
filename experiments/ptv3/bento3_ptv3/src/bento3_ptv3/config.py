"""One dataclass holds every knob of a run. It is saved as config.json in the run dir and logged to Trackio."""
from dataclasses import dataclass, asdict, field, fields
import json


@dataclass
class TrainConfig:
    # run identity
    name: str = "lora_ssl"                 # run dir = out_dir/<name>_s<seed>
    seed: int = 0
    out_dir: str = "runs"
    data_dir: str = "data"
    # data: full Xenium section (10x FFPE Human Breast, pre-designed panel, sample 1 IDC/DCIS; 68.1M molecules, 280 genes)
    data_source: str = "hf"                # hf = prebuilt arrays from hf_data_repo; 10x = rebuild from the public 10x bundle
    hf_data_repo: str = "ckmah/bento3-ptv3-runs"
    hf_data_path: str = "data/xenium_breast_idc_v1"
    crop: tuple = ()                       # (x0, y0, side) um: train/eval inside this square only (smoke tests)
    split_seed: int = 0                    # held-out 1 mm blocks; fixed across training seeds
    block_um: float = 1000.0
    heldout_frac: float = 0.15
    # views (Sonata-style): two overlapping crops of one region, each rotated/flipped/scaled; the student view is masked
    win_um: float = 100.0
    shift_frac: float = 0.3                # view B = view A shifted by up to this fraction of win_um in x and y
    scale_range: tuple = (0.9, 1.1)
    jitter_um: float = 0.1
    mask_patch_um: float = 10.0            # student masks molecules in random mask_patch_um squares ...
    mask_ratio: float = 0.4                # ... covering this fraction of squares (pretrained Utonia mask token)
    min_mol: int = 500                     # skip windows with fewer molecules (glass)
    regions: int = 4                       # regions per step (each gives 2 views)
    ugrid_um: float = 0.5                  # 0.5 um voxel = Utonia's 0.01 grid unit (as in the nuspU arms)
    # model
    gene_dim: int = 32
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_targets: tuple = ("qkv", "proj")  # attention linears in every encoder block
    emb_dim: int = 64                      # neck output (per-molecule embedding saved for hierarchy work)
    head_hidden: int = 512
    head_bottleneck: int = 256
    prototypes: int = 1024
    drop_path: float = 0.1
    utonia_repo: str = "Pointcept/Utonia"
    utonia_rev: str = "9457f025e36c7001ee962a542f0d27a5b600e1e7"
    utonia_sha256: str = "3d7afb731771556ebdf20abd1b6a1f022d9b9f105db8446f5eff02037f7fceb9"
    # objective: teacher-student self-distillation (EMA teacher, centering + sharpening), loss on molecules masked in the student view
    t_temp: tuple = (0.04, 0.07)           # teacher temperature: warm up from [0] to [1] over t_temp_warmup of training
    t_temp_warmup: float = 0.3
    s_temp: float = 0.1
    teacher_norm: str = "sinkhorn"         # sinkhorn (Sonata / DINOv2) | center (DINO centring with center_momentum)
    center_momentum: float = 0.9
    ema: tuple = (0.994, 1.0)              # teacher EMA momentum, cosine from [0] to [1]
    koleo: float = 0.1                     # KoLeo weight on the student bottleneck (DINOv2); 0 = off
    koleo_max: int = 4096                  # rows that enter the KoLeo term (pairwise distances)
    match_max: int = 4096                  # matched molecules per view pair that enter the loss
    loss_on: str = "masked"                # masked = only molecules masked in the student view; all = every matched molecule
    # optimisation
    steps: int = 0                         # 0 = auto: fit train_hours (s/step measured over steps 50-250; fixed in the checkpoint)
    train_hours: float = 9.5               # training budget for steps=0 (molab hard-stops at 12 h: leaves time for download + eval)
    max_steps: int = 200000
    lr: float = 1e-3                       # head / neck / stem / gene embedding
    freeze_lora: bool = False              # diagnostic: keep LoRA B at zero (encoder exactly pretrained); head/neck/stem still train
    lora_lr: float = 5e-5                  # LoRA A/B (own param group; 0 = same as lr). 1e-3 blew the encoder up in lora_ssl_s0
    warmup: int = 1000
    weight_decay: float = 0.04
    clip: float = 3.0
    max_feat_abs: float = 1e4              # abort (before checkpointing) if max |encoder feature| exceeds this; init is ~4
    amp: str = "bf16"                      # none | bf16 (autocast on CUDA; on CPU only with amp_cpu)
    amp_cpu: bool = False                  # tests only: also autocast bf16 on CPU, to exercise the GPU dtype path without a GPU
    device: str = "auto"
    threads: int = 8
    loader_procs: int = 2                  # batch-building processes (fork; threads gave 1.0x: GIL). 0 = one prefetch thread
    prefetch: int = 6                      # batches built ahead in a thread (CUDA)
    log_every: int = 25
    ckpt_min: float = 15.0                 # checkpoint (and push to HF) every this many minutes ...
    ckpt_every: int = 5000                 # ... or every this many steps, whichever comes first
    session_hours: float = 11.25           # stop cleanly (checkpoint + push) after this long in one session; Run all resumes
    # eval: frozen-embedding linear probes on held-out blocks (probe-fit blocks vs probe-test blocks)
    viz_every: int = 500                   # interpretation panels on a fixed held-out tile -> run_dir/viz/ (0 = off)
    eval_every: int = 10000                # quick probe (nucleus, cluster) every N steps; 0 = only at start and end
    eval_tiles: int = 160                  # held-out 100 um tiles in the final eval (quick eval: eval_tiles // 4)
    eval_per_tile: int = 2500              # core molecules kept per tile
    probe_gene: bool = True                # final eval: masked-gene linear probe (own gene hidden) with abundance metrics
    # sync / tracking (all optional, never block training)
    hf_repo: str = "ckmah/bento3-ptv3-runs"
    hf_prefix: str = "lora"                # path in hf_repo
    trackio_project: str = "bento3-lora"
    trackio_space: str = ""                # "" = no Space publishing. trackio's static export (a background thread reading the
                                           # sqlite DB while training writes it) segfaulted the process; e.g. "ckmah/bento3-ptv3-trackio"
    trackio_publish_min: float = 5.0

    @property
    def run_name(self): return f"{self.name}_s{self.seed}"

    def to_dict(self): return asdict(self)

    def save(self, path): json.dump(self.to_dict(), open(path, "w"), indent=1)

    @classmethod
    def load(cls, path):
        d = json.load(open(path)); names = {f.name for f in fields(cls)}
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k in names})
