# v1 — Transformer Baseline (Pure Move-Token GPT)

**Notebook:** `chess_transformer_baseline.ipynb`  
**Saved checkpoint:** `chess_gpt_baseline.pt`  
**Vocab file:** `chess_move_vocab.json`

---

## Motivation

Establish the simplest possible DL baseline: treat a chess game as a **sequence of move tokens** and train a causal (decoder-only) transformer to predict the next move, identical in spirit to a language model.

The question is: **does a pure sequence model, with no board representation at all, learn anything useful about chess?**

---

## Architecture — `ChessGPT` (v1)

| Component | Detail |
|---|---|
| Type | Decoder-only transformer (GPT-style) |
| Input | Token sequence of UCI move strings |
| Context length (`BLOCK_SIZE`) | 160 plies (truncates long games, losing opening context) |
| Token embedding | `nn.Embedding(vocab_size, d_model)` with `padding_idx=PAD_ID` |
| Position embedding | Learned absolute position embedding |
| Attention | Causal self-attention (masked so token `t` only sees tokens `0..t-1`) |
| Architecture | `d_model=256`, `n_head=4`, `n_layer=4` |
| Weight tying | Output head weights tied to token embedding (GPT-2 style) |
| Init | GPT-2-style small-std init (`N(0, 0.02)`) to avoid huge logits at step 0 |
| Parameters | ~few million |

### Initialization Note
Standard `nn.Embedding` default is `N(0,1)`, which, combined with weight tying, produces huge-magnitude logits and destabilizes training from step 1. GPT-2-style `N(0, 0.02)` was adopted.

---

## Data Representation

- Each game is tokenized as: `[BOS, move1_id, move2_id, ..., moveN_id]`
- Input `x = [BOS, ..., moveN-1]`, target `y = [move1, ..., moveN]` (shifted by 1)
- No board state — the model must infer board state purely from move history
- `BLOCK_SIZE=160` truncates long games

---

## Training

- **Epochs**: 5
- **Optimizer**: AdamW (no explicit LR schedule in this version)
- **Loss**: Cross-entropy on all non-pad tokens

---

## Evaluation Results

| Metric | Value |
|---|---|
| `top_1` | **0.2693** |
| `top_3` | **0.4393** |
| `unmasked_argmax_legal_rate` | **0.5393** |

**Key observation**: The `unmasked_argmax_legal_rate` of ~54% tells the real story — the model's raw top pick is illegal about half the time. This means **masking is doing significant heavy lifting**: without legal-move masking, Top-1 would be far lower. The 26.9% Top-1 achieved here is the model learning game patterns from sequence alone, but it barely tracks the board.

---

## Known Limitations

1. **No ELO conditioning** — one model for the entire 2200–2600 band
2. **No board state** — the model has to infer position from move history alone; the 54% unmasked legal rate confirms it struggles to track the board reliably
3. **Truncated context** — `BLOCK_SIZE=160` loses opening context for long games; costs O(n²) attention to fix
4. **5 epochs only** — undertrained; later versions train up to 30 epochs with early stopping
5. **No LR schedule** — added in v2 onward

---

## Next Steps (Recorded in Notebook)

- Add ELO conditioning (→ v2)
- Add explicit board state as input (→ v3)
- Switch to per-square tokenization (→ v4)
- Use a CNN trunk instead of attention (→ v5)
