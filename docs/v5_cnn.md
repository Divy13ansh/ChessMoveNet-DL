# v5 — CNN Baseline: ResNet Trunk vs. Chessformer

**Notebook:** `chess_cnn_v5.ipynb`  
**Checkpoint path:** configurable (e.g., `chess_cnn_v5.pt`)

---

## Motivation

v4 (Chessformer + GAB) represents the transformer-based approach. v5 asks the competing question:

> **Can a well-tuned CNN match or beat the transformer, given the same data and same evaluation protocol?**

The hypothesis motivating v4's transformer approach is that chess requires **long-range geometric reasoning** — a rook's attack along a rank spans the full 8 squares, and a king-safety evaluation involves squares far from the king. The claim is that attention + GAB can represent these relationships more efficiently than convolution.

v5 tests this claim empirically. If a carefully tuned CNN matches v4, it suggests the **data size** is the binding constraint (not representational capacity or range), and convolution's translation equivariance is a strong enough prior.

---

## Input Representation — Bitboard Planes

v5 encodes the board as a **stack of 2D planes** (8×8 images), not as token sequences:

### Piece Planes (`PL_PIECES` offset)
- 12 planes: one per (color, piece_type) combination
- `plane[i, rank, file] = 1.0` if that piece is on that square, else `0.0`

### Side-to-Move Plane (`PL_STM`)
- 1 plane, all 1s if White to move, all 0s if Black to move

### Castling Rights (`PL_CASTLE`)
- 4 planes: kingside White, queenside White, kingside Black, queenside Black
- All 1s or all 0s per plane

### En-Passant Plane (`PL_EP`)
- 1 plane with a `1` at the target square for en passant, 0s elsewhere
- **Design note**: v5 uses a single-square encoding (1 hot on the target square), not the v3/v4 8-dim file vector. For a conv, the ep target must sit at the actual pixel a capturing pawn can reach, otherwise locality is wasted.

### History Planes (`PL_HIST`, optional)
- 2 planes per previous ply: `from_plane` and `to_plane`
- `K_HISTORY` previous moves → `2 × K_HISTORY` planes
- An all-zero pair encodes "no such move" (equivalent to v4's `NO_SQUARE` sentinel)

### Meta Features (`META_DIM = 7` scalars)
- Not stored as planes; broadcast in `forward` to avoid ~20% unnecessary host→device traffic
- Same 7 scalars as v4: side-to-move ELO, opponent ELO, ELO diff, both clocks, time control

### Total Channels
```
IN_CH = 12 (pieces) + 1 (turn) + 4 (castle) + 1 (ep) + 2*K_HISTORY (history)
```

### Canonical Orientation (`CANONICAL_ORIENTATION`)
When `True`: if it is Black's turn, the board is **flipped vertically and piece colors are swapped**, so that planes 0–5 are always the mover's own pieces and the target move is remapped via `MIRROR_MOVE`. This halves the function the network must learn (White-to-move and Black-to-move positions become the same distribution).

> **v4 cannot easily implement canonical orientation** (the square index embedding and history from/to embeddings are square-indexed, flipping would require remapping). This gives v5 a potential advantage, so ablation A1 runs v5 with `CANONICAL_ORIENTATION=False` as well.

---

## Architecture — ResNet Trunk + Bilinear Policy Head

### ResNet Trunk

Standard AlphaZero-shaped tower:

| Stage | Detail |
|---|---|
| Stem | `3×3 conv → BN → ReLU`, output channels = `CH` |
| Residual blocks | `N_BLOCKS` blocks, no downsampling |
| Each block | `3×3 conv → BN → ReLU → 3×3 conv → BN → (+residual) → ReLU` |
| Optional SE | Squeeze-Excitation block at the end of each residual block |
| Zero-init | Last BN in each block has `weight=0` (starts as identity) |

**No downsampling**: an 8×8 board has no scale hierarchy worth pooling over. Stride or pooling would discard exactly the square-identity information the policy head needs.

**Receptive field**: grows 2 squares per block. With 14 blocks, the receptive field covers the board several times — but a rook's a1–h1 relation requires ~4 layers to compose, while GAB gets it as a single learned bias.

### Squeeze-Excitation (SE) Block

```python
class SE(nn.Module):
    def forward(self, x):
        # Global average pool → squeeze to scalar per channel
        s = x.mean(dim=[-2, -1], keepdim=True)  # (B, C, 1, 1)
        # 2-layer MLP: C → C//ratio → C
        s = self.fc2(F.relu(self.fc1(s.flatten(1)))).reshape(B, C, 1, 1)
        return x * torch.sigmoid(s)  # per-channel gate
```

SE is the **only global pathway** in a pure conv stack. Without it, a 3×3 conv stack cannot combine information from opposite corners of the board in fewer than ~7 layers. With SE, the network can modulate all channels globally after each block. Ablation A3 isolates its contribution.

### Policy Head — Bilinear From-To (identical to v4)

$$\text{logit}(f \to t) = \frac{q_f \cdot k_t}{\sqrt{d}} + \text{promo\_bias}[t, p]$$

The CNN trunk produces `(B, CH, 8, 8)`. This is reshaped to 64 per-square vectors of dim `CH`. `q_proj` and `k_proj` produce from-key and to-key per square; the 64×64 outer product is gathered through `MOVE_FT` into the 4,546-way vocabulary.

**The policy head is identical to v4's** — this is by design, to isolate the trunk as the variable under test. Any accuracy delta between v4 and v5 comes from the trunk, not the head.

`POLICY_HEAD = "flat"` is an ablation option that uses a 1×1 conv → MLP → 4,546 logits flat head. The flat head has no knowledge of which squares are involved in a move and costs ~9.3M parameters vs. ~51k for the factored head (ablation A4).

---

## Training

### Differences from v4

| Aspect | v4 (Transformer) | v5 (CNN) |
|---|---|---|
| LR | `3e-4` | **`1e-3`** |
| Batch normalization | No (pre-LN transformer) | **Yes** |
| Mixed precision | Optional | **Yes (AMP)** |
| Weight decay target | All params | **Conv/linear only** (not BN scales/biases) |

**LR justification**: BatchNorm makes the loss surface far more forgiving, so `1e-3` is not a handicap — it is appropriate for a BN ResNet. Running the CNN at v4's `3e-4` would be a handicap for the CNN.

**Weight decay and BN**: decaying BN scales and biases is a known small regression (they are in a different regime from weight matrices) and costs nothing to avoid.

**Mixed precision**: convolutions benefit much more from AMP/fp16 than attention does; the `-inf` on special tokens survives fp16 correctly.

### Schedule
Same shape as v4: AdamW, linear warmup → cosine decay stepped per batch, grad clip at 1.0, early stopping, best state dict restored.

---

## Evaluation

Identical to v4's `evaluate_masked_topk`. One forward pass per game (not per position). The eval sample is `EVAL_MAX_GAMES = 1500`, identical to v4, for direct comparison.

Split fingerprints (hashes of game IDs) are printed at the end and must match v4's to ensure the comparison uses the exact same held-out games.

---

## Results

### Model Size

| Stat | Value |
|---|---|
| Input channels | 41 (12 pieces + 1 turn + 4 castle + 1 ep + 2×K_HISTORY×3) |
| Architecture | 14 blocks × 160 ch + SE |
| Parameters | **6.66M** (~1.02× v4's 6.53M) |
| Training positions | 977,679 (0.15 per parameter) |

### Test Set (1,500 held-out games, 122,275 positions — same split as v4)

| Metric | Value |
|---|---|
| **top_1** | **0.4470** |
| **top_3** | **0.7290** |
| `unmasked_argmax_legal_rate` | **0.9871** |
| positions_evaluated | 122,275 |
| 95% CI half-width | ±0.0028 |

### Train Set (same 1,500-game sample, for overfit check)

| Metric | Value |
|---|---|
| top_1 | 0.5848 |
| top_3 | 0.8402 |
| `unmasked_argmax_legal_rate` | 0.9926 |

**Train/test Top-1 gap: 0.1378** ⚠️ — significant overfit. Compare to v4's gap of 0.0201.

### Training Curve (6 epochs, best at epoch 5)

| Epoch | Train PPL | Val PPL | Val/Train ratio |
|---|---|---|---|
| 1 | 29.4 | 11.0 | 0.37 |
| 2 | 8.8 | 8.4 | 0.94 |
| 3 | 6.8 | 7.0 | 1.04 |
| 4 | 5.4 | 6.5 | 1.20 |
| **5** | **4.4** | **6.4** | **1.47** |
| 6 | 3.7 | 6.6 | 1.77 |

Best val loss: **1.8572** (ppl 6.4) at epoch 5. Val PPL starts rising at epoch 6. The val/train ratio climbs steeply — the CNN overfits much faster than the transformer.

### Final Head-to-Head: v5 vs. v4 (from notebook output)

```
model                           params    top-1    top-3
--------------------------------------------------------
v4  Chessformer + GAB            6.53M   0.4300       --
v5  CNN 14x160 +SE               6.66M   0.4470   0.7290

v5 - v4 Top-1: +0.0170  (CNN ahead)
95% CI half-width on v5's Top-1: ±0.0028
Delta (0.0170) is ~6× the CI → statistically resolvable.
```

**Split fingerprint verified** (same held-out set as v4):
- `train = b63dd8dc195f8a38`
- `test  = 70244c2f8aa8c234`

---

## Planned Ablations (from notebook)

| # | Flag | Change | Question |
|---|---|---|---|
| **A1** | `CANONICAL_ORIENTATION` | `False → True` | How much of the gap is orientation symmetry, not architecture? Expected to help CNN; not available to v4. |
| **A2** | `N_BLOCKS` | `14 → 8 → 20` | Is depth (receptive field) or capacity the binding constraint? |
| **A3** | `USE_SE` | `True → False` | How much does the global SE pathway contribute? (CNN's only whole-board channel) |
| **A4** | `POLICY_HEAD` | `"bilinear" → "flat"` | How much of the score is the factored head vs. trunk? Note: flat head ≈ +9.3M params, deliberately breaks parity. |
| **A5** | `DROPOUT` | `0.0 → 0.1` | Match v4's regularization exactly |
| **A6** | `USE_HISTORY_PLANES` | `True → False` | Does move history matter beyond the current position? |
| **A7** | `MAX_GAMES` | `15000 → 5000 / 30000` | **The decisive ablation**: plot Top-1 vs. training positions for both architectures |

### A7 is the Most Important Ablation

If the CNN's Top-1 curve is **flatter** as training data grows, it means:
- CNN has the better inductive prior at small data (translation equivariance, weight sharing)
- Transformer wins as data grows (more flexible, learns arbitrary patterns from data)
- The current data regime favors whichever prior is better at ~1M positions

This could reframe the entire comparison: the conclusion might be "CNN is better at this data scale; Transformer would win with 10× more data."

---

## Interpreting the Headline Comparison

### If CNN ≈ v4 (delta > -1%):
At ~1M training positions, neither model is capacity-limited. Convolution's translation equivariance is worth as much as GAB's long-range wiring. Weight sharing is a strong prior in the data-starved regime (~0.15 tokens per parameter, vs. compute-optimal ~20).

### If v4 wins clearly (delta < -2%):
Long-range reasoning is the story. Chess moves are dominated by sliding pieces and king safety — both are relations between distant squares. A 3×3 conv must compose ~4 layers to relate a1 to h1; GAB gets `(Δfile, Δrank)` as a single learned bias per head.

### Confounds to Rule Out

| Confound | Why it bites | Check |
|---|---|---|
| Learning rate | 1e-3 was chosen for BN ResNet, not tuned | 1-epoch sweep {3e-4, 1e-3, 3e-3} |
| Canonical orientation | v5 advantage not available to v4 | Quote both w/ and w/o for v5 (A1) |
| Regularization | v4 = dropout 0.1, v5 = BN | Compare val/train PPL ratio curves |
| Epochs run | Early stopping may fire differently | Print best epoch for both |
| Elo band | Both 2200–2600 | Keep same; do not mix |
