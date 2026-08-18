# v2 — Transformer + ELO Conditioning

**Notebook:** `chess_transformer_baseline_v2.ipynb`  
**Saved checkpoint:** `chess_gpt_baseline.pt` (overwrites v1)  
**Vocab file:** `chess_move_vocab.json`

---

## Motivation

v1 trains a single model for all players in the 2200–2600 ELO band. But a 2200-rated player and a 2600-rated player make very different decisions — the higher-rated player plays more "correct" moves more often, has a different tactical threshold, and plays different openings.

**Hypothesis**: Conditioning on ELO should allow the model to better predict which move *this specific strength player* would play, rather than averaging over the band.

This is the same motivation as Maia chess, which trains separate models per ELO range; here we try a continuous conditioning signal instead.

---

## Architecture — `ChessGPT` (v2) with ELO Projection

Same backbone as v1, with one addition:

| Component | Detail |
|---|---|
| Type | Decoder-only transformer |
| d_model | 256 |
| n_head | 4 |
| n_layer | 4 |
| **ELO conditioning** | Learned `nn.Linear(1, d_model)` projection of normalized avg ELO |
| ELO injection | `elo_emb` broadcast-added to all token positions: `x = x + elo_emb` |
| Weight tying | Yes (same as v1) |
| Init | GPT-2-style `N(0, 0.02)` |

### ELO Normalization
```python
avg_elo = (white_elo + black_elo) / 2
elo_norm = (avg_elo - min_elo) / (max_elo - min_elo)  # → [0, 1]
```
Using **average ELO** of both players as the conditioning signal. The normalized scalar is projected to `d_model` and added as a bias to every position's embedding.

---

## Training Improvements over v1

- **Extended training**: 30 epochs (vs. 5 in v1) with early stopping
- **LR Schedule**: Linear warmup → cosine decay stepped per batch  
  ```python
  WARMUP_STEPS = min(200, max(1, total_steps // 20))
  ```
  Warmup scales gracefully to small datasets (uses `min(200, ...)`)
- **Best-checkpoint tracking**: saves state dict of epoch with lowest val loss; restored before evaluation
- **Val/train PPL ratio**: logged each epoch as overfit diagnostic

---

## Data Changes

- Same Lichess Elite 2019-10 source, ELO 2200–2600
- DataLoader now also yields `elo_norm` per game alongside token sequences

---

## Evaluation Results

| Metric | Value |
|---|---|
| `top_1` | **~0.29** |
| `top_3` | not recorded |
| `unmasked_argmax_legal_rate` | not recorded |
| Train/test Top-1 gap | logged |

**~2.1 percentage point improvement** over v1's 0.2693, suggesting ELO conditioning provides a meaningful signal about playing style.

---

## Limitation: Averaging Both Players' ELO

Using the average ELO is a simplification. In any given position, it is **one player's turn**, and conditioning should ideally use:
- The current side-to-move's ELO (not the average)
- Potentially separate White and Black ELOs, allowing the model to understand relative strength

v3 and v4 address this via more structured meta-features.

---

## Comparison Table (from notebook)

| Model | Top-1 | Top-3 | Unmasked Legal |
|---|---|---|---|
| v1 (5 epochs) | 0.2693 | 0.4393 | 0.5393 |
| v2 (+ELO avg, 30 epochs) | ~0.29 | — | — |
