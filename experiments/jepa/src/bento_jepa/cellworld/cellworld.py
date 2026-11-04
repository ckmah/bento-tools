from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from .encoder import SpatialTransformerStack
from .tokenizer import LinearCellTokenizer
from .pos_encoding import normalize_positions


class MetadataEmbedding(nn.Module):
    def __init__(
        self,
        num_organs: int,
        num_platforms: int,
        embed_dim: int,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.organ = nn.Embedding(
            num_organs,
            embed_dim,
            padding_idx=padding_idx,
        )
        self.platform = nn.Embedding(
            num_platforms,
            embed_dim,
            padding_idx=padding_idx,
        )

    def forward(
        self,
        organ_ids: torch.Tensor,
        platform_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.organ(organ_ids) + self.platform(platform_ids)


def _gather(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather cell positions: [B, N, K] and [B, S] -> [B, S, K]."""
    expanded_indices = indices.unsqueeze(-1).expand(
        -1,
        -1,
        tensor.shape[-1],
    )
    return torch.gather(tensor, 1, expanded_indices)


class CellWorld(nn.Module):
    """Canonical bag JEPA with a shared cell tokenizer."""

    def __init__(self, cfg):
        super().__init__()
        cfg = OmegaConf.create(cfg)
        model_cfg = cfg.model
        vocab_cfg = cfg.vocab
        embed_dim = model_cfg.embed_dim
        predictor_dim = model_cfg.predictor_dim
        spatial_cfg = model_cfg.spatial_encoding

        self.normalize_mode = spatial_cfg.normalize
        self.embed_dim = embed_dim
        self.predictor_dim = predictor_dim

        self.cell_encoder = LinearCellTokenizer(
            vocab_size=vocab_cfg.gene_vocab_size,
            embed_dim=embed_dim,
            padding_idx=vocab_cfg.gene_pad_id,
        )
        self.metadata = MetadataEmbedding(
            num_organs=vocab_cfg.organ_vocab_size,
            num_platforms=vocab_cfg.platform_vocab_size,
            embed_dim=embed_dim,
        )

        self.context_encoder = SpatialTransformerStack(
            dim=embed_dim,
            depth=model_cfg.encoder_depth,
            num_heads=model_cfg.encoder_heads,
            mlp_ratio=model_cfg.mlp_ratio,
            dropout=model_cfg.dropout,
            spatial_cfg=spatial_cfg,
        )
        self.target_encoder = SpatialTransformerStack(
            dim=embed_dim,
            depth=model_cfg.encoder_depth,
            num_heads=model_cfg.encoder_heads,
            mlp_ratio=model_cfg.mlp_ratio,
            dropout=model_cfg.dropout,
            spatial_cfg=spatial_cfg,
        )
        for target_parameter in self.target_encoder.parameters():
            target_parameter.requires_grad = False
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())

        self.ctx_to_pred = nn.Linear(embed_dim, predictor_dim)
        self.pred_to_target = nn.Linear(predictor_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))

        self.predictor = SpatialTransformerStack(
            dim=predictor_dim,
            depth=model_cfg.predictor_depth,
            num_heads=model_cfg.predictor_heads,
            mlp_ratio=model_cfg.mlp_ratio,
            dropout=model_cfg.dropout,
            spatial_cfg=spatial_cfg,
        )

    def featurize_cells(
        self,
        gene_ids: torch.Tensor,
        values: torch.Tensor,
        gene_offsets: torch.Tensor,
        num_cells: int,
        organ_ids: torch.Tensor,
        platform_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = organ_ids.shape[0]
        flat_cell_tokens = self.cell_encoder(
            gene_ids,
            values,
            gene_offsets,
        )
        cell_tokens = flat_cell_tokens.view(
            batch_size,
            num_cells,
            self.embed_dim,
        )
        metadata = self.metadata(
            organ_ids,
            platform_ids,
        )
        return cell_tokens + metadata.unsqueeze(1)

    def _build_tokens(self, batch: dict) -> torch.Tensor:
        return self.featurize_cells(
            batch["gene_ids"],
            batch["values"],
            batch["gene_offsets"],
            batch["coordinates"].shape[1],
            batch["organ_id"],
            batch["platform_id"],
        )

    def encode(self, batch: dict) -> torch.Tensor:
        tokens = self._build_tokens(batch)
        coordinates = normalize_positions(
            batch["coordinates"],
            self.normalize_mode,
        )
        return self.context_encoder(tokens, coordinates=coordinates)

    def forward(self, batch: dict) -> Dict[str, torch.Tensor]:
        context_indices = batch["context_idx"]
        target_indices = batch["target_idx"]
        num_context_cells = context_indices.shape[1]

        coordinates = normalize_positions(
            batch["coordinates"],
            self.normalize_mode,
        )
        context_coordinates = _gather(coordinates, context_indices)
        target_coordinates = _gather(coordinates, target_indices)

        tokens = self._build_tokens(batch)
        context_tokens = _gather(tokens, context_indices)
        context_repr = self.context_encoder(
            context_tokens,
            coordinates=context_coordinates,
        )

        with torch.no_grad():
            full_target_repr = self.target_encoder(
                tokens,
                coordinates=coordinates,
            )
            target_repr = _gather(full_target_repr, target_indices)

        partial_hint = self.featurize_cells(
            batch["target_partial_gene_ids"],
            batch["target_partial_values"],
            batch["target_partial_offsets"],
            target_indices.shape[1],
            batch["organ_id"],
            batch["platform_id"],
        )

        projected_context = self.ctx_to_pred(context_repr)
        target_queries = self.mask_token + self.ctx_to_pred(partial_hint)
        predictor_tokens = torch.cat(
            [projected_context, target_queries],
            dim=1,
        )
        predictor_coordinates = torch.cat(
            [context_coordinates, target_coordinates],
            dim=1,
        )

        predictor_output = self.predictor(
            predictor_tokens,
            coordinates=predictor_coordinates,
        )
        prediction = self.pred_to_target(
            predictor_output[:, num_context_cells:]
        )

        return {"loss": F.mse_loss(prediction, target_repr)}

    @torch.no_grad()
    def ema_update(self, tau: float):
        parameter_pairs = zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        )
        for context_parameter, target_parameter in parameter_pairs:
            target_parameter.data.mul_(tau).add_(
                context_parameter.data,
                alpha=1.0 - tau,
            )
