# Project Overview — ChessMoveNet-DL

## What Is This Project?

ChessMoveNet-DL is a deep learning research project aimed at **predicting which chess move a human player will play** given the full game context. This is sometimes called "move mimicry" or "human move prediction" — it's not about finding the *best* chess move (Stockfish does that), but about predicting what a *specific class of human* would play.

---

## Data Source

**Lichess Elite Database — October 2019**

- Source: `lichess_db_standard_rated_2019-10.pgn.zst` (Lichess public PGN dumps)
- Filter: ELO band **2200–2600** (strong club players / near-titled)
- Format: Standard PGN with headers including `WhiteElo`, `BlackElo`, `TimeControl`
- `MAX_GAMES` typically set to ~15,000 games per experiment (a small slice of a much larger file)
- Clock data (`%clk` annotations) is used where available, treated as optional

### Data Split
- **Game-level split**: entire games go to train or test — no position-level leakage
- Roughly 80/10/10 or 90/10 train/val/test depending on version
- A **split fingerprint** (hash of game IDs) is saved with checkpoints so that cross-version comparisons use identical held-out sets

---

## Vocabulary / Tokenization

All models share the same move vocabulary:

- **UCI notation**: moves like `e2e4`, `g1f3`, `e7e8q` (promotion)
- Vocabulary built from all moves seen in the training corpus
- **Size**: ~4,546 unique move tokens (varies slightly by version — v4 uses 4,546 precisely)
- Special tokens: `<BOS>` (start of game), `<PAD>` (padding)
- Saved as `chess_move_vocab.json` / `chess_move_vocab_v4.json`

---

## Evaluation Protocol (Consistent Across All Versions)

All experiments use the same evaluation methodology to ensure comparability:

### Legal-Move-Masked Top-k

1. Get the model's **raw, unmasked** distribution over all ~4,546 tokens for the next move
2. Use `python-chess` to enumerate all **truly legal moves** for that board position (replaying the real game, not the model's belief)
3. **Mask** the distribution to legal moves only, renormalize
4. Check whether the human-played move lands in the **Top-1** or **Top-3** of the masked distribution
5. Separately track `unmasked_argmax_legal_rate` — whether the model's own top pick was already legal before masking (a board-tracking quality diagnostic)

### Evaluation Set Size
- Typically: first **1,500 test games** → ~120,000 positions
- Standard error on Top-1: approximately ±0.15% — sufficient to resolve 1–2% gaps
- Same sample used for train-set overfit-gap check

### Key Metrics Reported
| Metric | Description |
|---|---|
| `top_1` | Human move ranked #1 among legal moves |
| `top_3` | Human move ranked in top 3 among legal moves |
| `unmasked_argmax_legal_rate` | Fraction of positions where raw argmax was already legal |
| `positions_evaluated` | Number of positions in the eval loop |
| val/train ppl ratio | Perplexity ratio as overfit diagnostic |

---

## Training Protocol (Common Across All Versions)

- **Loss**: `F.cross_entropy` over full vocabulary — **no legality masking during training**
- **Optimizer**: AdamW
- **LR Schedule**: Linear warmup (≤200 steps or 5% of total steps) → cosine decay stepped **per batch**
- **Grad clipping**: `clip_grad_norm` at 1.0
- **Early stopping**: patience on validation loss; best checkpoint restored before evaluation
- **Evaluation**: Teacher-forced (real game moves, not model's own predictions)

---

## Checkpoints and Artifacts

| File | Contents |
|---|---|
| `chess_gpt_baseline.pt` | v1/v2 transformer weights |
| `chess_gpt_v4.pt` | v4 Chessformer weights |
| `chess_move_vocab.json` | v1/v2/v3 vocabulary + config |
| `chess_move_vocab_v4.json` | v4 vocabulary + architecture config |
