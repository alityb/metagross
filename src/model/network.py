from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .state import FIELD_FEATURES, N_ACTIONS, POKEMON_DENSE_FEATURES, EncodedState, Vocabulary, build_vocabulary


LOGGER = logging.getLogger(__name__)


@dataclass
class PokeNetConfig:
    species_vocab: int
    move_vocab: int
    item_vocab: int
    ability_vocab: int
    d_model: int = 192
    nhead: int = 8
    num_layers: int = 3
    dim_feedforward: int = 384
    dropout: float = 0.1
    species_dim: int = 64
    move_dim: int = 32
    item_dim: int = 32
    ability_dim: int = 32
    entity_embedding_dim: int = 384
    action_dim: int = N_ACTIONS

    @classmethod
    def from_vocabulary(cls, vocab: Vocabulary) -> "PokeNetConfig":
        return cls(
            species_vocab=vocab.species_size,
            move_vocab=vocab.move_size,
            item_vocab=vocab.item_size,
            ability_vocab=vocab.ability_size,
        )


class PokeNet(nn.Module):
    def __init__(self, config: PokeNetConfig | None = None, vocab: Vocabulary | None = None):
        super().__init__()
        if config is None:
            config = PokeNetConfig.from_vocabulary(vocab or build_vocabulary("data/gen9_random_pool.json"))
        self.config = config
        self.vocab = vocab
        self.species_emb = nn.Embedding(config.species_vocab, config.entity_embedding_dim)
        self.move_emb = nn.Embedding(config.move_vocab, config.entity_embedding_dim)
        self.item_emb = nn.Embedding(config.item_vocab, config.entity_embedding_dim)
        self.ability_emb = nn.Embedding(config.ability_vocab, config.entity_embedding_dim)
        self.species_projection = nn.Linear(config.entity_embedding_dim, config.species_dim)
        self.move_projection = nn.Linear(config.entity_embedding_dim, config.move_dim)
        self.item_projection = nn.Linear(config.entity_embedding_dim, config.item_dim)
        self.ability_projection = nn.Linear(config.entity_embedding_dim, config.ability_dim)

        pokemon_raw = (
            config.species_dim   # 64
            + config.move_dim    # 32  (sum of 4 move slots)
            + config.move_dim    # 32  (last_move embedding — same projection)
            + config.item_dim    # 32
            + config.ability_dim # 32
            + POKEMON_DENSE_FEATURES  # 224
        )  # = 416
        self.pokemon_encoder = nn.Sequential(
            nn.Linear(pokemon_raw, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        self.field_encoder = nn.Sequential(
            nn.Linear(FIELD_FEATURES, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        head_input = config.d_model * 3
        self.policy_head = nn.Sequential(
            nn.Linear(head_input, config.d_model), nn.ReLU(),
            nn.Linear(config.d_model, config.action_dim),
        )
        # Ensemble of 4 value heads — take minimum over targets during training
        # (conservative Q-estimate, prevents overestimation spiral that caused
        # Phase 2 collapses). Matches Metamon NCritics=4 approach.
        # At inference: use mean over all 4 for best estimate.
        self.n_critics = 4
        self.value_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_input, config.d_model), nn.ReLU(),
                nn.Linear(config.d_model, 1), nn.Tanh(),
            )
            for _ in range(self.n_critics)
        ])
        self.reset_parameters()
        self._freeze_entity_embeddings()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _freeze_entity_embeddings(self) -> None:
        self.species_emb.weight.requires_grad = False
        self.move_emb.weight.requires_grad = False
        self.item_emb.weight.requires_grad = False
        self.ability_emb.weight.requires_grad = False

    def _encode(self, tensors: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared encoder: returns (head_state [B, 3*d_model], masked_logits [B, 14])."""
        species   = tensors["species_ids"].long()
        moves     = tensors["move_ids"].long()
        items     = tensors["item_ids"].long()
        abilities = tensors["ability_ids"].long()
        dense     = tensors["pokemon_dense"].float()
        field     = tensors["field"].float()
        active    = tensors["active_indices"].long()
        last_move = tensors["last_move_ids"].long()

        species_emb   = self.species_projection(self.species_emb(species))
        move_emb      = self.move_projection(self.move_emb(moves)).sum(dim=2)
        last_move_emb = self.move_projection(self.move_emb(last_move))
        item_emb      = self.item_projection(self.item_emb(items))
        ability_emb   = self.ability_projection(self.ability_emb(abilities))
        pokemon_raw   = torch.cat([species_emb, move_emb, last_move_emb,
                                   item_emb, ability_emb, dense], dim=-1)
        pokemon_tokens = self.pokemon_encoder(pokemon_raw)
        field_token    = self.field_encoder(field).unsqueeze(1)
        tokens         = torch.cat([field_token, pokemon_tokens], dim=1)
        encoded        = self.transformer(tokens)

        batch_index = torch.arange(encoded.shape[0], device=encoded.device)
        own_index   = active[:, 0].clamp(0, 5) + 1
        opp_index   = active[:, 1].clamp(0, 5) + 7
        head_state  = torch.cat(
            [encoded[:, 0], encoded[batch_index, own_index], encoded[batch_index, opp_index]],
            dim=-1,
        )
        logits = self.policy_head(head_state)
        mask = tensors.get("action_mask")
        if mask is not None:
            mask_bool = mask.bool()
            all_invalid = ~mask_bool.any(dim=1)
            if all_invalid.any():
                mask_bool[all_invalid, :4] = True
            logits = logits.masked_fill(~mask_bool, torch.finfo(logits.dtype).min)
        return head_state, logits

    def forward(self, batch: dict[str, Any] | EncodedState) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = self._to_tensors(batch)
        head_state, logits = self._encode(tensors)
        # Ensemble mean for inference (conservative individual estimates averaged)
        values = torch.stack([h(head_state).squeeze(-1) for h in self.value_heads], dim=1)
        return logits, values.mean(dim=1)

    def value_ensemble(self, batch: dict[str, Any] | EncodedState) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, values) where values.shape == (B, n_critics).

        PPO training uses values.min(dim=1).values as the conservative target
        to prevent overestimation spirals (Metamon NCritics=4 approach).
        """
        tensors = self._to_tensors(batch)
        head_state, logits = self._encode(tensors)
        values = torch.stack([h(head_state).squeeze(-1) for h in self.value_heads], dim=1)
        return logits, values

    def policy_value(self, batch: dict[str, Any] | EncodedState) -> tuple[torch.Tensor, torch.Tensor]:
        logits, value = self.forward(batch)
        return torch.softmax(logits, dim=-1), value

    def parameter_count(self, trainable_only: bool = True) -> int:
        params = self.parameters() if not trainable_only else (p for p in self.parameters() if p.requires_grad)
        return sum(parameter.numel() for parameter in params)

    def initialize_from_embeddings(self, path: str | Path) -> dict[str, int]:
        if self.vocab is None:
            raise ValueError("PokeNet.initialize_from_embeddings requires construction with a Vocabulary")
        payload = torch.load(path, map_location="cpu")
        groups = {
            "species": (self.species_emb, self.vocab.species),
            "moves": (self.move_emb, self.vocab.moves),
            "items": (self.item_emb, self.vocab.items),
            "abilities": (self.ability_emb, self.vocab.abilities),
        }
        copied: dict[str, int] = {}
        for group_name, (embedding, vocab) in groups.items():
            vectors = payload.get(group_name, {})
            count = 0
            with torch.no_grad():
                for name, index in vocab.items():
                    vector = vectors.get(name)
                    if vector is None:
                        continue
                    tensor = torch.as_tensor(vector, dtype=embedding.weight.dtype)
                    if tensor.numel() != self.config.entity_embedding_dim:
                        continue
                    embedding.weight[index].copy_(tensor.reshape(-1))
                    count += 1
            embedding.weight.requires_grad = False
            copied[group_name] = count
        LOGGER.info("Initialized frozen entity embeddings from %s: %s", path, copied)
        return copied

    def initialize_from_nebraskinator(self, path: str | Path) -> dict[str, int]:
        from .checkpoint import load_matching_weights

        stats = load_matching_weights(self, path, strict_shapes=True)
        if stats.get("matched", 0) == 0:
            LOGGER.warning("No Nebraskinator weights matched PokeNet shapes from %s", path)
        else:
            LOGGER.info("Transferred Nebraskinator weights from %s: %s", path, stats)
        return stats

    def _to_tensors(self, batch: dict[str, Any] | EncodedState) -> dict[str, torch.Tensor]:
        if isinstance(batch, EncodedState):
            batch = batch.as_batch()
        device = next(self.parameters()).device
        tensors: dict[str, torch.Tensor] = {}
        input_keys = {
            "species_ids",
            "move_ids",
            "item_ids",
            "ability_ids",
            "last_move_ids",
            "pokemon_dense",
            "field",
            "active_indices",
            "action_mask",
        }
        for key, value in batch.items():
            if key not in input_keys:
                continue
            if isinstance(value, torch.Tensor):
                tensors[key] = value.to(device)
            elif isinstance(value, np.ndarray):
                tensors[key] = torch.from_numpy(value).to(device)
            else:
                tensors[key] = torch.tensor(value, device=device)
        return tensors
