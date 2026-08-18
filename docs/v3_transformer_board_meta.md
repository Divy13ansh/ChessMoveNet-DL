# v3 — Transformer + Flat Board Features + Meta Features

**Notebook:** `chess_transformer_v3.ipynb`  
**Saved checkpoint:** (separate from v1/v2, path configurable)  
**Vocab file:** (separate, includes board_dim/meta_dim config)

---

## Motivation

v1 and v2 are pure **sequence models** — the model must infer the board state entirely from the move history. The `unmasked_argmax_legal_rate` of ~54% in v1 shows it fails to track the board reliably.

v3 makes a fundamental change: **explicitly provide the board state as input** at every position, so the model doesn't have to infer it. The question becomes: how much does knowing the board state help?

A secondary question is whether structured **meta features** (ELO per-side, clock times, castling rights, etc.) provide further signal.

---

## Architecture — `ChessGPTv3`

Same transformer backbone, but augmented with two side channels:

| Component | Detail |
|---|---|
| Backbone | Decoder-only transformer, same dims as v2 |
| **Board features** | Flat vector of the full board state per position |
| **Meta features** | Scalar vector of game metadata per position |
| Feature injection | Board and meta features are projected to `d_model` and **added** to the token embedding at each position |
| Learned channel gains | `board_scale` and `meta_scale` — scalar parameters initialized to `EMB_INIT_STD`; the model learns how loudly to use each channel |

### Board Feature Vector (`BOARD_DIM` dimensions)
A flat float32 vector computed per position using `board_feature_vector(board)`:
- One-hot encoding of all 12 piece types × 64 squares (sparse but explicit)
- Turn-to-move indicator
- Castling rights (4 bits)
- En-passant square

### Meta Feature Vector (`META_DIM` dimensions)
Per-position scalar features:
- Normalized ELO of side-to-move and opponent (separate — not averaged like v2)
- Clock time remaining (normalized)
- Increment
- Game phase indicators

### Key Design: Separate ELO per Side
Unlike v2's averaged ELO, v3 gives the model:
- `meta[0]` = side-to-move's ELO (normalized)
- `meta[1]` = opponent's ELO (normalized)

This allows the model to distinguish "strong player to move" from "strong player as opponent".

### Learned Channel Gain Diagnostic
After training, `model.board_scale` and `model.meta_scale` are printed:
```
Learned channel gains (init {EMB_INIT_STD}):
  board_scale: X.XXXX  (Yx init)
  meta_scale:  X.XXXX  (Zx init)
```
- A gain growing well above init → model found the channel useful
- A gain shrinking toward 0 → model found the channel unhelpful

---

## Critical Bug That Was Fixed

v3 has an **assertion cell (Section 5)** that verifies the position alignment is correct. This is crucial:

> Position `t` must hold the board **before** move `t`, and `target[t]` must be move `t`.

An off-by-one error (position after move `t` paired with target `t`) would **leak the answer** and produce artificially large accuracy. The assertion replays the game independently and cross-checks the recorded board features. This bug check was carried forward to all subsequent versions.

---

## Training

Same protocol as v2:
- AdamW with linear warmup → cosine decay
- Early stopping with patience on val loss
- Best state dict restored before eval

---

## Evaluation Protocol

v3 introduces the **teacher-forced one-forward-pass-per-game** approach that becomes standard in v4 and v5:

```python
# One forward pass for the whole game
logits = model(x, pad_mask, board_feats=bf, meta_feats=mf)[0]  # (T, V)
probs = F.softmax(logits.float(), dim=-1).cpu().numpy()

for t in range(T):
    legal_ids = legal_per_pos[t]
    ranked = [legal_ids[i] for i in np.argsort(-probs[t][legal_ids])]
    ...
```

Previously (v1/v2), evaluation used **one forward pass per position** (O(n²) per game). v3 switches to one pass per game (O(n)), which is essential for larger eval sets.

---

## Results

| Model | Top-1 | Top-3 | Unmasked Legal |
|---|---|---|---|
| v1 (5 epochs) | 0.2693 | 0.4393 | 0.5393 |
| v2 (+ELO avg) | ~0.29 | — | — |
| v3 (board+meta) | (see notebook output) | (see notebook output) | (see notebook output) |

v3's actual numbers are referenced in v4's comparison table as a placeholder (`None`) suggesting the run wasn't completed or numbers weren't recorded — v3 served primarily as an architectural stepping stone toward v4.

---

## What v3 Revealed

1. **Flat board vectors can work** but add ~`BOARD_DIM × d_model` parameters in the projection layer
2. **Summing too many additive terms into one residual** is dangerous: if board scale >> token embedding scale, it can dominate and make other channels invisible. The `board_scale` / `meta_scale` learned gains address this.
3. The flat board representation treats all 64 squares identically as a vector — there's no concept of "proximity" or "which piece is on which square interacting with which other piece." This motivates v4's per-square tokenization approach.

---

## Limitation: Flat Board = No Spatial Structure

A flat vector loses the 8×8 spatial structure. A rook on a1 and a rook on h1 are just two separate entries in a 768-dim vector; the model can't easily learn that they're on the same rank. v4 addresses this by giving each square its own token.
