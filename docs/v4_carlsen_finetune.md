# Carlsen Fine-Tune — Fine-Tuning v4 on a Single Player

**Notebook:** `chess_transformer_v4_carlsen.ipynb`  
**Base checkpoint:** `chess_gpt_v4.pt`  
**Output checkpoint:** (configurable, e.g., `chess_gpt_v4_carlsen.pt`)

---

## Motivation

v4 trains on the general Lichess Elite population (ELO 2200–2600). Can we fine-tune the model to predict the moves of a **specific elite player** (Magnus Carlsen)?

This is interesting for several reasons:
1. **Player fingerprinting**: different top-level players have distinct styles (sharp vs. positional, specific opening repertoires, endgame tendencies)
2. **Transfer learning test**: does a model trained on ~2200–2600 players transfer usefully to a 2800+ GOAT?
3. **Data efficiency**: Carlsen has a finite number of recorded games — does fine-tuning on a small dataset overfit?

The question is whether the base v4 model's representations of piece interactions generalize to capture individual playing style.

---

## Data

| Setting | Value |
|---|---|
| PGN source | Carlsen's games (separate PGN file, not Lichess Elite) |
| `CARLSEN_ONLY` flag | If `True`, only trains on positions where Carlsen is to move; if `False`, trains on both sides |
| `TRAIN_BOTH_SIDES` | Controls whether Black positions (where opponent moves) are included |
| Time control filter | Configurable; similar to base model |
| Train/test split | Game-level split, same as base model convention |

### Important: Different Population, Non-Comparable Top-1
> **Note**: The Carlsen fine-tune's Top-1 accuracy (~0.45) is **not comparable** to v4's Lichess Elite Top-1. Predicting a single elite player is a different (and easier) task than predicting the average 2200–2600 human. Elite play is more predictable because top players play more "correct" chess; the distribution over moves is sharper.

---

## Fine-Tuning Setup

### Starting Point
Loads `chess_gpt_v4.pt` as the base checkpoint with **strict=True** weight loading (crashes rather than silently mismatching architecture).

### Key Fine-Tuning Decision: Measure Base Performance First
```python
# Measure pretrained model BEFORE any weight updates
base_val_loss = run_epoch(val_loader, train=False)
print(f"Epoch 0 (pretrained, no fine-tuning) | val ppl {math.exp(base_val_loss):.2f}")

best_val_loss = base_val_loss  # fine-tuning must BEAT this to save a checkpoint
```

**Why this matters**: if you initialize `best_val_loss = inf`, epoch 1 unconditionally overwrites the base weights. A single bad epoch would permanently damage the checkpoint with no recovery path. By measuring the base model first, the fine-tuned checkpoint is only saved if it actually improves on the pretrained model.

### Hyperparameters
| Hyperparameter | Value |
|---|---|
| LR | Lower than base (exact value in notebook) — standard fine-tuning practice |
| Epochs | Short run with early stopping |
| Early stop patience | Same as base |
| Per-epoch top-1 eval | Optional (`EVAL_GAMES_PER_EPOCH` flag) — allows tracking whether Top-1 improves before the full eval |

---

## Evaluation Protocol

Same legal-move-masked Top-k evaluation as v4. Key comparisons:

1. **Base v4 on Carlsen test set**: how well does the general model predict Carlsen without any fine-tuning?
2. **Fine-tuned model on Carlsen test set**: does fine-tuning help?
3. **Overfit gap**: train Top-1 vs. test Top-1 (with very little data, this gap is the main risk)

### Summary Output
```
SUMMARY  -- all on the SAME held-out Carlsen positions
  Top-1   base X.XXXX  ->  fine-tuned Y.YYYY   (+Z.ZZZZ)
  Top-3   base X.XXXX  ->  fine-tuned Y.YYYY   (+Z.ZZZZ)
  Legal-argmax  X.XXXX  ->  Y.YYYY
  Overfit gap (train - test Top-1): +Z.ZZZZ
```

---

## Interactive Move Prediction

The notebook includes a `top_moves()` function that accepts an arbitrary board position and returns the model's predicted top moves with probabilities:

```python
def top_moves(board, elo_w=MAX_ELO, elo_b=MAX_ELO, n=5):
    """Masked softmax over legal moves for an arbitrary board (no move history)."""
    ...

# Example usage
for line in ([], ["e2e4"], ["d2d4", "g8f6"]):
    b = chess.Board()
    for u in line: b.push(chess.Move.from_uci(u))
    label = " ".join(line) or "start"
    print(f"{label:<16} " + "  ".join(f"{s} {p:.3f}" for s, p in top_moves(b)))
```

This is a qualitative check: does the model's probability distribution look plausible for Carlsen's style at common opening positions?

---

## Results and Interpretation

The fine-tune is evaluated relative to the pretrained base:

| Scenario | Interpretation |
|---|---|
| Fine-tuned beats base | Fine-tuning captures Carlsen-specific style; transfer worked |
| Fine-tuned ≈ base | Style differences between Carlsen and the training population are too subtle to extract with this data volume |
| Fine-tuned worse than base (reverts to pretrained) | Overfitting on small dataset; base model generalizes better even to Carlsen |

### Key Risk: Small Dataset Overfitting
With a single player's game history (~hundreds to low thousands of games), the fine-tune dataset is tiny relative to the base model's training data. The `best_val_loss >= base_val_loss` check at the end explicitly prints a warning if no epoch beat the pretrained model.

---

## Saving

```json
{
  "finetuned_from": "chess_gpt_v4.pt",
  "finetuned_on": "carlsen_games.pgn",
  "finetune_player": "Carlsen,M",
  "finetune_scope": "all_moves | carlsen_moves_only",
  "finetune_games": N,
  "finetune_positions": N,
  "finetune_lr": LR,
  "finetune_epochs_run": N,
  "finetune_best_val_loss": X.XXXX,
  "finetune_top1_base": X.XXXX,
  "finetune_top1_test": X.XXXX,
  "finetune_top3_test": X.XXXX,
  "finetune_tc_base": TC_BASE
}
```

Weights are downloaded immediately if on Colab (runtime dies on disconnect), or left in-place otherwise.
