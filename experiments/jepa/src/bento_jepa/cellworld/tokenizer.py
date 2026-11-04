"""LinearCellTokenizer — value-weighted sum of gene embeddings.

Fused one-shot tokenizer: packed genes -> one token per cell, with no gene
positional encoding.
cell = Sum_g value[g] * gene_emb[gid].

Uses nn.EmbeddingBag(mode='sum') with per_sample_weights to fuse lookup +
weighted-sum without padded gene positions or a gene-embedding intermediate.
"""

import torch
import torch.nn as nn


class LinearCellTokenizer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 192,
                 padding_idx: int = 0):
        super().__init__()
        self.embed_dim = embed_dim
        self.gene_emb = nn.EmbeddingBag(vocab_size, embed_dim, mode='sum',
                                        padding_idx=padding_idx)

    def forward(
        self,
        gene_ids: torch.Tensor,
        values: torch.Tensor,
        gene_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate packed genes into one token per offset-defined bag."""
        gene_weights = values.to(self.gene_emb.weight.dtype)
        return self.gene_emb(
            gene_ids,
            gene_offsets,
            per_sample_weights=gene_weights,
        )
