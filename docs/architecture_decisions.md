# Architecture Decisions & Lessons Learned

This document captures the non-obvious design choices, failure modes encountered, and the rationale behind them across all ChessMoveNet-DL experiments.

---

## 1. Legal-Move Masking: Evaluation Only, Not Training

**Decision**: Legal moves are never used during training; masking is evaluation-only.

**Rationale**: 
- Training with legal-move masking would require enumerating legal moves for every position in every batch — expensive and adds complexity
- The model learns the legal move structure implicitly from the data (moves in training games are always legal by definition)
- At evaluation time, masking is applied to get a fair Top-k metric against a well-defined legal action set

**Risk**: If the model learns to predict illegal moves (e.g., `e1e1`, moving to an occupied square), it wastes probability mass. This is tracked by `unmasked_argmax_legal_rate`. In v1, this is 54% — the model wastes probability on illegal moves nearly half the time.

---

## 2. Why `unmasked_argmax_legal_rate` Is NOT an Accuracy Metric

Frequently misread as "the model plays a legal move 54% of the time." Correct reading:

> **Of all positions evaluated, in what fraction was the model's own top pick (before any masking) already a legal move?**

In v1 (pure sequence model), this is 54% — the model's raw argmax is illegal about half the time. Legal-move masking boosts the apparent Top-1 from ~10-15% (what you'd get with random legal move from the model's arbitrary distribution) to the reported 26.9%.

In v3/v4/v5 (board is an explicit input), this number is near 1.0. **This is not an improvement in chess understanding** — it's structural: when you give the model the board, it knows which squares are occupied and naturally assigns higher probability to moves from occupied squares. The diagnostic is still useful: if it drops below ~0.95 in v4/v5, something is wrong.

---

## 3. Weight Initialization — GPT-2 Style

**Problem**: `nn.Embedding` default init is `N(0,1)`. With weight tying (output head = embedding matrix), this produces logits of magnitude ~`vocab_size` at step 0, causing loss to be already near `log(1)` for the highest-probability token. Effectively makes early training unstable.

**Fix**: `nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)`, the same as GPT-2. Also zero out the PAD embedding explicitly so padding doesn't influence the representation.

---

## 4. Summing Multiple Side Channels → Domination Failure in v3

**Problem**: v3 added board features and meta features to the token embedding via straight addition:
```python
x = tok_emb + pos_emb + board_proj + meta_proj
```
If any one of the four terms is 12× larger than the others in L2 norm, it drowns out all other signals. In practice, if the board feature projection is large, the model only sees the board and ignores the move sequence.

**Fix (v3)**: Learned scalar gains `board_scale` and `meta_scale` so the model can attenuate unhelpful channels.

**Fix (v4)**: Different token groups are **separate tokens in the sequence**, not summed vectors. Pre-LN normalizes each token independently before attention, so a scale gap between groups cannot delete a channel.

**Lesson**: When combining heterogeneous feature types, use separate tokens (or gated / scaled addition). Straight addition without normalization is brittle.

---

## 5. The Position Alignment Off-By-One Bug

**Critical**: If you pair the board state **after** move `t` with target `t`, you leak the answer. The model sees the resulting position and must predict the move that caused it — trivially easy.

**v3 explicitly adds an assertion cell** that replays the game independently and cross-checks that position `t` holds the board **before** move `t`. This check was propagated to v4 and v5.

**How to check**:
```python
# Sanity check: position t should be before move t
board = chess.Board()
for t, move in enumerate(game.mainline_moves()):
    assert board_feature_vector(board) == stored_board_feats[t]  # board before move t
    board.push(move)
```

Any off-by-one here produces unrealistically high accuracy — a red flag that should trigger immediate suspicion.

---

## 6. Zero-Initialization for Residual Branches

Both v4 and v5 use zero-initialization on the last layer of each residual block:

**v4 (transformer)**: GAB bias initialized to zero → attention starts as standard unbiased self-attention; geometric biases are learned from data
```python
nn.init.zeros_(self.rel_bias)
```

**v5 (CNN)**: Last BN in each residual block has `weight=0` → each block starts as identity
```python
nn.init.zeros_(self.bn2.weight)
```

**Effect**: Training begins at a well-conditioned, shallow function and gradually deepens. Without this, very deep networks can have poorly conditioned gradients at init.

---

## 7. Bilinear Policy Head — Parameter Efficiency

A flat policy head would need to learn a separate output vector for each of the 4,546 moves: **4,546 × d_model = ~1.17M parameters** just for the head.

The bilinear head factors this as:
```
logit(from→to) = q_from · k_to / sqrt(d_head)
```
where `q_proj` and `k_proj` are `(d_model, d_head)` matrices: **2 × d_model × d_head = ~131k parameters** (at d_head=256).

Beyond efficiency, the head **encodes the structure of moves**: the model knows that `e2e4` involves square e2 (from) and square e4 (to). This is not something a flat head learns automatically; it has to be discovered from patterns in the data.

**Promotions** are a special case — `e7e8` (rook) and `e7e8q` (pawn) are different vocabulary items but share the same from/to squares. A separate `promo_bias[to_square, piece_type]` additive term handles this.

---

## 8. One Forward Pass Per Game (Not Per Position) in Evaluation

**v1/v2**: Evaluation uses one forward pass per position (O(n²) per game — each position rebuilds the full sequence from scratch).

**v3/v4/v5**: Single forward pass for an entire game (teacher-forced), then extract per-position logits. This is:
- `O(n)` passes per game (one batch of all T positions)
- ~100× faster for large evaluation sets
- Equivalent numerically (teacher-forcing, not autoregressive generation)

The note in v1 says the O(n²) eval is "fine for a few thousand positions, but worth optimizing before scaling." v3 makes this optimization the standard.

---

## 9. LR Differences Between CNN and Transformer

| Model | LR | Reason |
|---|---|---|
| v1/v2/v3 | (varied) | — |
| v4 transformer | 3e-4 | Standard for pre-LN transformers |
| v5 CNN | 1e-3 | BatchNorm makes loss surface more forgiving |

Running the CNN at 3e-4 would be a handicap — it's not testing "CNN vs. transformer" but "undertrained CNN vs. well-trained transformer." The 1e-3 for v5 was chosen empirically as appropriate for a BN ResNet.

**For any ablation that changes architecture significantly**: re-check LR with a 1-epoch sweep over `{3e-4, 1e-3, 3e-3}` before drawing conclusions.

---

## 10. Split Fingerprint — Ensuring Fair Comparisons

Any accuracy comparison across model versions is only valid if they're evaluated on the **same held-out games**.

v4 and v5 save a split fingerprint (hash of the train/test game IDs) with every checkpoint:
```json
"split_fingerprint": {
  "train": "abc123...",
  "test":  "def456..."
}
```

v5 explicitly prints "these must match v4's" and shows both fingerprints at eval time. If fingerprints differ, the comparison is apples to oranges.

---

## 11. Weight Decay — Exclude BN Scales and Biases

In CNN training (v5), weight decay is applied only to conv and linear weight parameters:
```python
decay     = [p for n, p in model.named_parameters() if "weight" in n and "bn" not in n]
no_decay  = [p for n, p in model.named_parameters() if "bn" in n or "bias" in n]
optimizer = AdamW([
    {"params": decay,    "weight_decay": WD},
    {"params": no_decay, "weight_decay": 0.0},
])
```

Decaying BN scale/shift parameters is a known small regression (they operate in a different scale regime from weight matrices) and costs nothing to avoid.

---

## 12. Fine-Tuning: Measure Base Performance Before First Update

**Pattern established in v4_carlsen**:
```python
best_val_loss = run_epoch(val_loader, train=False)  # epoch 0
best_state_dict = current_weights  # start with pretrained
# ... training loop ...
# Only update best_state_dict if epoch beats epoch-0 val loss
```

If you initialize `best_val_loss = inf`, the first training epoch unconditionally writes a checkpoint — even if fine-tuning made things worse. If every epoch makes things worse, you end up saving a damaged checkpoint. Measuring the pretrained model first sets a proper baseline.

---

## 13. Data Scale vs. Model Capacity

The consistent theme across all experiments:

> At ~15,000 games (~1M positions), both models operate in a **severely data-starved regime**: ~0.15 tokens per parameter vs. compute-optimal ~20.

This has several implications:
- **Inductive priors matter more than capacity**: the model with the better prior for this task will win, not the model with more parameters
- **Overfitting is the primary risk**: val/train PPL ratio >> 1 → more data or regularization needed
- **Ablations on model size** may not be informative at current data scale — the binding constraint is data, not capacity

The most important next experiment is **Ablation A7**: training on 5k, 15k, and 30k games and plotting both architectures' Top-1 vs. training data size.

---

## 14. Canonical Orientation (v5 Advantage)

When `CANONICAL_ORIENTATION = True` in v5:
- Black-to-move positions are flipped vertically and piece colors are swapped
- Planes 0–5 are always the **mover's** own pieces
- Target moves are remapped through `MIRROR_MOVE`
- The network only needs to learn one orientation (mover's perspective)

This is a v5-specific advantage: v4's per-square token embedding is indexed by square number (a1=0, h8=63). Flipping would require remapping all 64 square embeddings, plus remapping history from/to squares — much more complex.

Because canonical orientation is an advantage for v5 that v4 doesn't share, the fair headline comparison is:
1. **v5 without canonical orientation** vs. v4 (true controlled comparison)
2. **v5 with canonical orientation** vs. v4 (v5 best possible)
