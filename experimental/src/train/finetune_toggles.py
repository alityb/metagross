"""Phase-1 ablation toggles for the metamon finetune pipeline.

Each toggle is independently switchable so the sweep changes ONE variable at a
time (AGENTS.md §6.8):

- Toggle A (rating conditioning): repurposes four dead early-gen format tokens
  (<gen1nu>, <gen1uu>, <gen1ou>, <gen1ubers>) — which never appear in
  gen9randombattle observations — as rating-band tokens at text position 0
  (where the constant format token normally sits, carrying zero information in
  single-format training). At play time the existing FORMAT_ALIASES mechanism
  injects the top band token: alias gen9randombattle -> "gen1ubers".
  Zero new vocabulary; pretrained embeddings get repurposed by the fine-tune.

- Toggle B (frozen-KL anchor): KLAnchoredFinetuneAgent adds
  coeff * KL(pi_hare || pi_base) at dataset states, reusing the frozen base
  snapshot MetamonFinetuneAgent already maintains. Coefficient deliberately
  biased LOW (default 0.02): the goal is to move past Kakuna, not preserve it.

Toggles C (filter shape) and D (HL-Gauss) are pure gin configs in train/gins/.
Toggle E (belief features) is SKIPPED: injecting new belief observations would
require new vocabulary/number channels, which the frozen 142M input layers
cannot accept without architecture surgery — meaningful plumbing, flagged per
instructions.
"""
from __future__ import annotations

import os

RATING_BANDS = [
    (0, 1500, "<gen1nu>"),
    (1500, 1900, "<gen1uu>"),
    (1900, 2200, "<gen1ou>"),
    (2200, 10_000, "<gen1ubers>"),
]
PLAY_TIME_BAND = "gen1ubers"  # alias target for high-rating prompting at eval


def band_word(rating: int | None) -> str:
    if rating is None:
        rating = 0  # unrated -> lowest band (conservative)
    for lo, hi, word in RATING_BANDS:
        if lo <= rating < hi:
            return word
    return RATING_BANDS[-1][2]


def rating_from_filename(filename: str) -> int | None:
    parts = os.path.basename(filename).split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def install_rating_conditioning() -> None:
    """Monkeypatch metamon's dataset wrapper to band-condition text position 0."""
    import metamon.rl.dataset_config as dc
    import metamon.rl.metamon_to_amago as m2a

    base_cls = m2a.MetamonAMAGODataset

    class RatingConditionedAMAGODataset(base_cls):
        def _band_token_id(self, filename: str) -> int:
            tokenizer = self.parsed_replay_dset.observation_space.tokenizer
            return int(tokenizer.tokenize(band_word(rating_from_filename(filename)))[0])

        def sample_random_trajectory(self):
            import random as _random

            dset = self.parsed_replay_dset
            i = _random.randrange(len(dset))
            filename = dset.filenames[i]
            rl_data = self._process_data(dset[i])
            tok = self._band_token_id(str(filename))
            # text_tokens: [T+1, L]; position 0 is the constant format token
            rl_data.obs["text_tokens"][:, 0] = tok
            return rl_data

    RatingConditionedAMAGODataset.__name__ = "RatingConditionedAMAGODataset"
    m2a.MetamonAMAGODataset = RatingConditionedAMAGODataset
    dc.MetamonAMAGODataset = RatingConditionedAMAGODataset
    print("TOGGLE_A rating conditioning installed "
          f"(bands={[(lo, hi, w) for lo, hi, w in RATING_BANDS]})", flush=True)


def install_binary_filter() -> None:
    """Register a binary advantage filter (1[A>0]) with the object API the
    MetamonFinetuneAgent expects (seq_enabled / set_mask), so ONLY the filter
    shape changes vs the IS/exp baseline."""
    import gin
    import metamon.rl.custom_agent as ca

    class BinaryAdvantageFilter:
        seq_enabled = False

        def __init__(self, threshold: float = 0.0, floor: float = 1e-7):
            self.threshold = threshold
            self.floor = floor
            self._mask = None

        def set_mask(self, mask):
            self._mask = None  # mask handled by the agent's loss masking

        def __call__(self, adv):
            return (adv > self.threshold).float().clamp_min(self.floor)

    gin.external_configurable(
        BinaryAdvantageFilter, name="BinaryAdvantageFilter", module="custom_agent"
    )
    ca.BinaryAdvantageFilter = BinaryAdvantageFilter
    print("TOGGLE_C binary advantage filter installed", flush=True)


def install_kl_agent() -> None:
    """Define KLAnchoredFinetuneAgent, register it with gin under
    custom_agent.KLAnchoredFinetuneAgent, and expose it on the module."""
    import gin
    import torch
    from torch.distributions import Distribution
    import amago
    import metamon.rl.custom_agent as ca

    # Metamon pads replay actions with sentinel rows. AMAGO masks them after
    # computing actor log-probs, but OneHotCategorical validates first. Disable
    # validation globally; every affected padded timestep is excluded by masks.
    Distribution.set_default_validate_args(False)

    class KLAnchoredFinetuneAgent(ca.MetamonFinetuneAgent):
        def __init__(self, *args, kl_anchor_coeff: float = 0.02, **kwargs):
            super().__init__(*args, **kwargs)
            self.kl_anchor_coeff = kl_anchor_coeff

        def forward(self, batch, log_step: bool):
            total_loss = super().forward(batch, log_step)
            if self.kl_anchor_coeff <= 0 or not self._checkpoint_loaded:
                return total_loss

            straight = {k: batch.obs[k] for k in self.pass_obs_keys_to_actor}
            # hare dist at dataset states (one extra encoder pass)
            o = self.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s)
            s_rep, _ = self.traj_encoder(
                seq=o, time_idxs=batch.time_idxs, hidden_state=None
            )
            hare_dist = self.actor(s_rep, straight_from_obs=straight)
            with torch.no_grad():
                o_b = self._base_tstep_encoder(obs=batch.obs, rl2s=batch.rl2s)
                s_b, _ = self._base_traj_encoder(
                    seq=o_b, time_idxs=batch.time_idxs, hidden_state=None
                )
                base_dist = self._base_actor(s_b, straight_from_obs=straight)

            eps = 1e-8
            p = hare_dist.probs.clamp_min(eps)
            q = base_dist.probs.clamp_min(eps)
            kl = (p * (p.log() - q.log())).sum(-1, keepdim=True)  # [B,L,G,1]
            mask = (~(batch.rl2s == self.pad_val).all(-1, keepdim=True)).bool()
            mask = mask[:, : kl.shape[1]]
            while mask.ndim < kl.ndim:
                mask = mask.unsqueeze(-2)
            kl_loss = amago.utils.masked_avg(kl, mask.expand_as(kl))
            if log_step:
                self.update_info["KL Anchor"] = kl_loss.detach()
            return total_loss + self.kl_anchor_coeff * kl_loss

    gin.external_configurable(
        KLAnchoredFinetuneAgent, name="KLAnchoredFinetuneAgent", module="custom_agent"
    )
    ca.KLAnchoredFinetuneAgent = KLAnchoredFinetuneAgent
    print("TOGGLE_B KL-anchored agent installed", flush=True)
