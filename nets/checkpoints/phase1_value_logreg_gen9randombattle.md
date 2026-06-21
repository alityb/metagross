# phase1_value_logreg_gen9randombattle

Format: `gen9randombattle`

Purpose: Phase 1 eval-hook integration gate. This is a tiny fixed-side logistic
value net trained with cross-entropy to output a direct side-one win probability
for non-terminal MCTS leaves.

Training data:

- Source: public Pokemon Showdown `gen9randombattle` replays fetched from replay search.
- Replays used: 200
- Examples: 8,594 fixed-side decision-point examples, doubled across `p1` and `p2` perspectives.
- Feature masking: public replay information only at each turn; no eventually revealed hidden info is included in features.
- Perspective: fixed side-one perspective; no to-move sign flip.

Model:

- Type: logistic value net
- Feature count: 16
- Loss: cross-entropy on eventual win/loss
- Export: `nets/checkpoints/phase1_value_logreg_gen9randombattle.txt`

Held-out metrics:

- Held-out examples: 1,718
- Held-out cross-entropy: 0.6446
- Held-out Brier: 0.2269
- Held-out accuracy: 61.8%
- Reliability bins: `nets/checkpoints/phase1_value_logreg_gen9randombattle.metrics.json`

Known limitation:

- Public replay features are masked and human-policy-labeled. They are not search-leaf determinizations, so a Phase 1 loss is confounded with target-policy mismatch.
