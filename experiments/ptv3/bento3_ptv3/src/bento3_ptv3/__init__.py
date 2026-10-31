"""bento3_ptv3: pretrained PTv3 (Utonia-small, frozen) + LoRA, self-distillation over Xenium molecules. See README.md."""
from .config import TrainConfig
from .train import train

__all__ = ["TrainConfig", "train"]
