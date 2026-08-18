# Results Summary — All Experiments

> **Evaluation protocol**: Legal-move-masked Top-k, teacher-forced on held-out games. All numbers on Lichess Elite 2019-10, ELO 2200–2600, unless noted. 14,945 games total; 3,000 held-out test games (1,500 used for eval). See [overview.md](./overview.md) for full protocol.

---

## Complete Results Table

| Model | Train Top-1 | **Test Top-1** | Test Top-3 | Train-Test Gap | Notes |
|---|---|---|---|---|---|
| Random baseline | — | ~0.034 | ~0.102 | — | avg 29.5 legal moves |
| Logistic Regression | 0.236 | 0.194 | — | 0.042 | ✅ |
| SGD Classifier | 0.232 | 0.192 | — | 0.040 | ✅ |
| XGBoost Classifier | 0.328 | 0.254 | — | 0.074 | ⚠️ overfit |
| LightGBM Classifier | 0.344 | 0.278 | — | 0.066 | ⚠️ overfit |
| LightGBM Ranker v1 | 0.344 | 0.288 | — | 0.056 | ⚠️ overfit |
| LightGBM Ranker v2 | 0.354 | 0.322 | — | 0.032 | ✅ |
| **LightGBM Ranker v3** | 0.382 | **0.332** | **0.544** | 0.050 | ✅ ← ML best |
| DL v1 — GPT (5 ep) | — | **0.2693** | **0.4393** | — | unmasked legal: 0.5393 |
| DL v2 — GPT + ELO (30 ep) | — | **~0.29** | — | — | — |
| DL v3 — flat board + meta | — | not recorded | — | — | — |
| **DL v4 — Chessformer + GAB** | **0.4504** | **0.4303** | **0.7096** | **0.0201** | ✅ unmasked: 0.9817 |
| **DL v5 — CNN 14×160 + SE** | **0.5848** | **0.4470** | **0.7290** | **0.1378** | ⚠️ gap; unmasked: 0.9871 |
| Carlsen fine-tune (v4 base) | — | ~0.45 | — | — | ⚠️ Carlsen only, not comparable |

> **Carlsen note**: ~0.45 Top-1 is on Carlsen's games only, not the general test set. Not comparable to rows above.

> **Random baseline**: with mean 29.5 legal moves per position, uniformly random → Top-1 ≈ 1/29.5 ≈ 3.4%.

> **v4 vs. v5 same split**: confirmed via fingerprint — `train=b63dd8dc195f8a38`, `test=70244c2f8aa8c234`.

---

## Final v4 vs. v5 Head-to-Head

```
model                        params    top-1    top-3
──────────────────────────────────────────────────────
v4  Chessformer + GAB         6.53M   0.4303   0.7096
v5  CNN 14×160 +SE            6.66M   0.4470   0.7290

v5 − v4 Top-1: +0.0170  (CNN ahead)
95% CI half-width on v5 Top-1: ±0.0028
Delta (0.0170) > 6× CI → statistically resolvable, CNN wins.
```

**The CNN beats the Transformer at this data scale.** At ~1M training positions both models are data-starved (~0.15 tokens/param vs. compute-optimal ~20). The CNN's inductive bias — translation equivariance, weight sharing — is the better prior in this regime. However, the CNN's **overfit gap is much larger** (0.1378 vs. 0.0201), suggesting it would benefit from more regularization or more data.

---

## Training Curves Summary

### v4 — Chessformer (5 epochs, best at ep 5)

| Epoch | Train PPL | Val PPL | Val/Train ratio |
|---|---|---|---|
| 1 | 56.9 | 21.5 | 0.38 |
| 2 | 16.3 | 10.3 | 0.63 |
| 3 | 10.0 | 7.7 | 0.77 |
| 4 | 8.0 | 6.8 | 0.85 |
| **5** | **7.2** | **6.5** | **0.91** |

Best val loss: **1.8770** (ppl 6.5) at epoch 5. Clean convergence — val/train ratio steadily approaches 1.

### v5 — CNN (6 epochs, best at ep 5)

| Epoch | Train PPL | Val PPL | Val/Train ratio |
|---|---|---|---|
| 1 | 29.4 | 11.0 | 0.37 |
| 2 | 8.8 | 8.4 | 0.94 |
| 3 | 6.8 | 7.0 | 1.04 |
| 4 | 5.4 | 6.5 | 1.20 |
| **5** | **4.4** | **6.4** | **1.47** |
| 6 | 3.7 | 6.6 | 1.77 |

Best val loss: **1.8572** (ppl 6.4) at epoch 5. Val PPL starts rising at epoch 6. Val/train ratio climbs sharply → the CNN overfits faster than the transformer.

---

## DL Progression

```
v1: 0.2693  ■─────────────────────────────────────────────────
v2: 0.2900  ─■────────────────────────────────────────────────
v3:  n/a
ML: 0.332   ─────────■  ← ML ceiling (LightGBM Ranker v3)
v4: 0.4303  ─────────────────────────────────────■  (transformer)
v5: 0.4470  ───────────────────────────────────────■  (CNN, winner)
```

---

## Key Observations

### 1. CNN beats Transformer at this data scale (+1.7pp)
v5 (0.4470) > v4 (0.4303) — statistically significant (delta > 6× CI half-width). Translation equivariance is a better inductive bias than learned attention when you have ~1M positions and ~6.5M parameters.

### 2. Both DL models crush the ML ceiling (+10pp)
v4 and v5 both far exceed the ML best (0.332). The structured board representation (per-square tokens for v4, bitboard planes for v5) is what makes the difference — DL v1/v2 with no board state couldn't beat ML.

### 3. CNN overfits badly
v5 train/test gap = 0.1378 vs v4's 0.0201. The CNN memorizes training positions faster. Mitigations to try: canonical orientation, dropout (A5), more data (A7). The val/train PPL ratio reaches 1.77 by epoch 6 — a clear overfit signal.

### 4. Board tracking is solved in v4/v5
`unmasked_argmax_legal_rate`: v1 = 0.5393, v4 = 0.9817, v5 = 0.9871. With the board as explicit input, the model's raw argmax is legal >98% of the time. Legal-move masking at eval is now just a minor refinement, not heavy lifting.

### 5. Ranking >> Classification (ML side)
LightGBM Ranker v1 (0.288) > LightGBM Classifier (0.278) on same features. LambdaRank is better aligned with the task.

### 6. DL v1 < ML ceiling — board state is essential
v1 GPT (0.2693) is below LightGBM Ranker v3 (0.332). Pure sequence models are not automatically superior. The board structure is what gets DL above ML.

---

## Parameter Counts

| Model | Params |
|---|---|
| DL v4 — Chessformer + GAB | **6.53M** |
| DL v5 — CNN 14×160 + SE | **6.66M** (~1.02× v4) |

Training positions: **977,679** (12k games × ~81.5 avg moves). Tokens/param: **0.15** (compute-optimal ≈ 20).

---

## Statistical Notes

- v5 Top-1 95% CI: **±0.0028** (122,275 positions evaluated)
- v4 vs. v5 delta: **+0.0170** → ~**6× the CI** → statistically resolvable
- Split fingerprint verified identical: both evaluated on the **exact same 1,500 test games**
- Two runs of same model on different seeds differ by ~0.5% — treat deltas < 1% with caution

---

## Open Experiments / Remaining Ablations

| Experiment | Status | Expected Insight |
|---|---|---|
| v3 full run + numbers | Not recorded | Flat board vs. per-square tokens |
| v5 A1: `CANONICAL_ORIENTATION=True` | Planned | May close v5 overfit gap; v4 can't do this easily |
| v5 A2: depth sweep N_BLOCKS = 8/14/20 | Planned | Receptive field vs. capacity |
| v5 A3: `USE_SE=False` | Planned | Value of global pooling (SE) in CNN |
| v5 A4: flat policy head | Planned | Head vs. trunk contribution |
| v5 A7: training size sweep (5k/15k/30k) | **Most important** | Do CNN and Xfmr converge as data grows? |
| v4 + more data / more epochs | Open | Does transformer close the gap with more data? |
| ELO bands (Maia-style separate models) | Open | vs. continuous ELO conditioning |
| Linear probe: hidden states → board state | Open | Karvonen-style internal representation analysis |
