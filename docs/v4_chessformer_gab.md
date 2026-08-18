# v4 — Chessformer: Per-Square Tokens + Geometric Attention Bias

**Notebook:** `chess_transformer_v4.ipynb`  
**Saved checkpoint:** `chess_gpt_v4.pt`  
**Vocab file:** `chess_move_vocab_v4.json`

---

## Motivation

v3 fed a flat board vector into the model — losing all spatial structure. v4 makes a fundamental architectural shift:

> **Instead of tokenizing moves and feeding board as a side channel, tokenize the board positions themselves.**

Each of the 64 squares becomes a **token**, augmented with a piece embedding and board-state tokens. The model attends over board positions with geometry-aware biases. This is inspired by architectures like AlphaZero's policy head and transformer-based board game models like Leela Chess Zero's exploration of attention layers.

The central hypothesis: **if you give the attention mechanism explicit knowledge of relative square geometry, it can directly learn patterns like "a rook attacks all squares on its rank" without having to compose them through intermediate layers.**

---

## Architecture — `Chessformer` (v4)

### Token Groups

Each forward pass processes a **sequence of up to T game positions**, each position consisting of a group of tokens:

| Token Group | Count | Description |
|---|---|---|
| **Square tokens** | 64 | One per board square, embedding = `piece_emb(piece_id)` + `square_emb(square_index)` |
| **State token** | 1 | Global board state (turn, castling rights, en-passant) via `state_proj` |
| **Skill token** | 1 | Per-position meta features (ELO of both sides, clock, time control) via `skill_proj` |
| **History tokens** | `K_HISTORY` | Last K moves, each encoded as `hist_from_emb(from_sq) + hist_to_emb(to_sq) + hist_age_emb(age)` |

Total tokens per position: `64 + 1 + 1 + K_HISTORY`

### Piece Embedding (`piece_emb`)

```
PIECE_ID = {
    (WHITE, PAWN): 1, (WHITE, KNIGHT): 2, (WHITE, BISHOP): 3,
    (WHITE, ROOK): 4, (WHITE, QUEEN): 5, (WHITE, KING): 6,
    (BLACK, PAWN): 7, (BLACK, KNIGHT): 8, (BLACK, BISHOP): 9,
    (BLACK, ROOK): 10, (BLACK, QUEEN): 11, (BLACK, KING): 12,
    "empty": 0
}
```
12 piece types + empty square = 13 embeddings.

### State Features (`STATE_DIM`)
- Turn indicator (1.0 = White to move, 0.0 = Black)
- Kingside/queenside castling rights for both sides (4 floats)
- En-passant file as one-hot (8 floats)

### Meta / Skill Features (`META_DIM`)
- Side-to-move ELO normalized to [0, 1]
- Opponent ELO normalized to [0, 1]
- ELO difference (mover - opponent), clipped
- Clock time remaining for mover (log-normalized)
- Opponent clock time
- Time control base (log-normalized)
- Increment

---

## Geometric Attention Bias (GAB)

The key architectural innovation of v4.

### Motivation
Chess relationships are fundamentally geometric: a bishop attacks diagonally, a rook attacks on ranks/files, a knight jumps in L-shapes. These are all functions of `(delta_file, delta_rank)` between any two squares.

Standard self-attention learns these patterns implicitly through gradient descent on the (query × key) product. GAB makes them **explicit** by adding a learned bias to every attention logit based on the relative position of the two squares being attended to.

### Implementation

```python
# bucket index for each (from_sq, to_sq) pair
gap_index = delta_file * 15 + delta_rank  # in range 0..224 for (−7..7, −7..7)
# plus 4 special buckets: sq-to-state, sq-to-skill, sq-to-history, non-sq pairs

rel_bias = nn.Parameter(torch.zeros(n_head, N_GAB_BUCKETS))
# In forward: attn_logits += rel_bias[:, gap_index]
```

- 225 spatial buckets (15×15 grid of delta-file × delta-rank combinations) + 4 special-token buckets
- One bias parameter **per head per bucket** (e.g., `8 heads × 229 buckets = 1,832 parameters per layer`)
- Initialized to **zero** — training starts with standard attention and learns geometric structure from data
- Visualizable: after training, `model.blocks[0].attn.rel_bias` can be reshaped to `(n_head, 15, 15)` and plotted as a heatmap showing what spatial relationships each head has learned to attend to

### Pre-Layer Norm
v4 uses **pre-LN** (normalize before attention and MLP), not post-LN. This is more numerically stable for deep transformers and important because the square/state/skill/history tokens start with different magnitudes.

---

## Policy Head — Bilinear From-To Factorization

The policy head is a key design element shared with v5's CNN. Rather than predicting all 4,546 moves independently (which would require 4,546 output vectors), moves are factored as:

$$\text{logit}(f \to t) = \frac{q_f \cdot k_t}{\sqrt{d}} + \text{promo\_bias}[t, p]$$

```python
# q_proj: from-square query, k_proj: to-square key
q = self.q_proj(square_embeddings)  # (B, 64, d_head)
k = self.k_proj(square_embeddings)  # (B, 64, d_head)
logits_64x64 = (q @ k.transpose(-1, -2)) / sqrt(d)  # (B, 64, 64) = from × to
# gather only the (from, to) pairs that are in the vocabulary via MOVE_FT
logits_vocab = logits_64x64[:, MOVE_FT[:, 0], MOVE_FT[:, 1]]  # (B, 4546)
```

Why this works:
- Only `64 × d` parameters for the from-key and to-key projections, vs. `4546 × d` for a flat head
- The model explicitly knows that move `e2e4` involves square e2 and square e4
- Promotion moves are handled by a separate additive bias term (one per promotion piece type)

`<PAD>` and `<BOS>` tokens are forced to `-inf` in the logits.

---

## Training Hyperparameters

| Hyperparameter | Value |
|---|---|
| d_model | 256 |
| n_head | 8 |
| n_layer | 6 |
| K_HISTORY | 3 (last 3 moves) |
| Dropout | 0.1 |
| LR | 3e-4 |
| Weight decay | 0.01 |
| Batch size | Games per batch (collated by positions) |
| Epochs | Up to 30, with early stopping |
| WARMUP_STEPS | `min(200, total_steps // 20)` |
| Early stop patience | configurable |
| USE_GAB | True (main run) |
| USE_HISTORY | True (main run) |
| USE_SQUARE_AUX | True |

---

## Token Balance Check (Pre-Training Sanity Check)

Before training, v4 prints the std of each token group:
```
token group         std       vs square
square (x64)     0.XXXX       1.0x
state (x1)       0.XXXX       Yx
skill (x1)       0.XXXX       Yx
history (xK)     0.XXXX       Yx
```

**Why this matters**: v3 died because four additive terms were summed into one vector and one was 12× the others, swamping the signal. v4's groups are **separate tokens** (not summed), and pre-LN normalizes each token independently — so a scale gap cannot delete a channel, but it is still checked as a diagnostic.

---

## Evaluation Results

### Test Set (1,500 held-out games, 122,275 positions)

| Metric | Value |
|---|---|
| **top_1** | **0.4303** |
| **top_3** | **0.7096** |
| `unmasked_argmax_legal_rate` | **0.9817** |
| positions_evaluated | 122,275 |

### Train Set (same 1,500-game sample, for overfit check)

| Metric | Value |
|---|---|
| top_1 | 0.4504 |
| top_3 | 0.7266 |
| `unmasked_argmax_legal_rate` | 0.9825 |

**Train/test Top-1 gap: 0.0201** ✅ — minimal overfitting.

### Training Curve (5 epochs, all ran to completion)

| Epoch | Train PPL | Val PPL | Val/Train ratio |
|---|---|---|---|
| 1 | 56.9 | 21.5 | 0.38 |
| 2 | 16.3 | 10.3 | 0.63 |
| 3 | 10.0 | 7.7 | 0.77 |
| 4 | 8.0 | 6.8 | 0.85 |
| **5** | **7.2** | **6.5** | **0.91** |

Best val loss: **1.8770** (ppl 6.5) at epoch 5. The val/train PPL ratio climbing toward 1.0 is healthy — the model generalises well.

### Cross-Version Comparison (from notebook output)

| Model | Top-1 | Top-3 | Unmasked Legal |
|---|---|---|---|
| v1 (move tokens, 5 ep) | 0.2693 | 0.4393 | 0.5393 |
| v2 (+ELO avg) | 0.2900 | — | — |
| v3 (flat board+meta) | — | — | — |
| **v4 (squares+GAB)** | **0.4303** | **0.7096** | **0.9817** |

---

## What v4 Proved

1. **Per-square tokenization** is a strong inductive prior: each square carries its own piece identity, and attention directly models inter-square relationships
2. **GAB learns meaningful geometry**: the head-0 bias heatmap shows the model learns different attention patterns for different `(Δfile, Δrank)` offsets
3. **History tokens help**: the last K moves provide context about what just happened (piece just moved is often a strong candidate to move again or be captured)
4. **Bilinear policy head** drastically reduces output parameters while embedding structured knowledge about move representation

---

## Artifacts Saved per Run

```json
{
  "itos": [...],
  "arch": "chessformer-v4",
  "d_model": 256, "n_head": 8, "n_layer": 6,
  "use_gab": true,
  "use_history": true, "k_history": 3,
  "use_square_aux": true,
  "state_dim": 13, "meta_dim": 7, "aux_dim": ...,
  "min_elo": 2200, "max_elo": 2600,
  "has_clock_data": true
}
```
