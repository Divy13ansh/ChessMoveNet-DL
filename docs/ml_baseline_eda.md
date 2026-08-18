# ML Baseline — EDA, Dataset & Classical ML Approach

**Notebook:** `chess_human_move_predictor_v4.ipynb`  
**Approach:** Classical ML (Logistic Regression, SGD, LightGBM Classifier, XGBoost, LightGBM Ranker)  
**Context:** Previous semester project; the foundation that the DL work (v1–v5) builds on and seeks to beat.

---

## Data Source

Same as the DL experiments:
- **Lichess Elite Database — October 2019**
- **ELO filter**: 2200–2600
- **Games loaded**: 15,000
- **Games after filtering** (move count 8–200): **14,945**

---

## Game-Level EDA

### Basic Statistics

| Statistic | Value |
|---|---|
| Total games | 14,945 |
| Total positions extracted | **1,218,514** |
| Avg positions per game | **81.2** |
| Min move count (filter) | 8 |
| Max move count (filter) | 200 |

### ELO Distribution

| Statistic | White ELO | Black ELO | Avg ELO | ELO Diff |
|---|---|---|---|---|
| Mean | 2420.1 | 2420.1 | 2420.1 | 0.04 |
| Std | 86.1 | 86.5 | 57.3 | 129.1 |
| Min | 2200 | 2200 | 2300.5 | −388 |
| 25th pct | 2373 | 2374 | 2379 | −88 |
| Median | 2425 | 2425 | 2414 | −1 |
| 75th pct | 2476 | 2476 | 2456.5 | +89 |
| Max | 2600 | 2600 | 2592 | +383 |

> ELO difference is nearly symmetrically distributed around 0 (median = −1), confirming the matchmaking system is roughly balanced.

### Game Results

| Result | Count | Percentage |
|---|---|---|
| White wins (1-0) | 7,251 | **48.5%** |
| Black wins (0-1) | 6,545 | **43.8%** |
| Draw (1/2-1/2) | 1,149 | **7.7%** |

### Move Count Distribution

| Percentile | Plies |
|---|---|
| Min | 8 |
| 25th | 57 |
| Median | 77 |
| 75th | 103 |
| Max | 200 |

### Termination Types

| Type | Count |
|---|---|
| Normal | 11,187 (74.8%) |
| Time forfeit | 3,758 (25.2%) |

### Top 10 Openings (ECO)

| Opening | Count |
|---|---|
| Indian Game | 214 |
| Modern Defense | 182 |
| Sicilian Defense: Closed Variation | 146 |
| Trompowsky Attack | 121 |
| Queen's Pawn | 120 |
| Scandinavian Defense: Mieses-Kotroc Variation | 117 |
| King's Indian Attack | 109 |
| Queen's Pawn Game: Mason Attack | 107 |
| Zukertort Opening: Sicilian Invitation | 107 |
| Horwitz Defense | 102 |

---

## Move-Level EDA

### Legal Move Statistics (per position)

| Statistic | Value |
|---|---|
| Mean legal moves | **29.5** |
| Std | 11.75 |
| Min | 1 |
| 25th pct | 22 |
| Median | 31 |
| 75th pct | 38 |
| Max | 79 |

### Move Vocabulary

| Statistic | Value |
|---|---|
| Unique moves played across all 1.2M positions | **1,871** |
| Top 10 moves cover | 9.09% of all positions |

### Top 10 Most Frequently Played Moves

| Rank | Move | Count | % of All Positions |
|---|---|---|---|
| 1 | g1f3 (Nf3) | 13,028 | 1.07% |
| 2 | g8f6 (Nf6) | 12,997 | 1.07% |
| 3 | d2d4 (d4) | 12,515 | 1.03% |
| 4 | e8g8 (O-O, Black) | 11,688 | 0.96% |
| 5 | e1g1 (O-O, White) | 11,482 | 0.94% |
| 6 | b1c3 (Nc3) | 11,060 | 0.91% |
| 7 | e2e4 (e4) | 10,706 | 0.88% |
| 8 | e7e6 (e6) | 9,457 | 0.78% |
| 9 | b8c6 (Nc6) | 9,219 | 0.76% |
| 10 | g7g6 (g6) | 8,611 | 0.71% |

> The extreme long-tail distribution of moves — top 10 cover only ~9% — motivates the ranking approach over naive classification.

### Move Frequency Distribution (long tail)

- 28 moves appear exactly once  
- 10 moves appear exactly twice  
- The distribution is extremely heavy-tailed: a handful of common developing moves dominate

---

## Position Feature Statistics

Computed on all 1,218,514 positions:

| Feature | Mean | Std | Min | Median | Max |
|---|---|---|---|---|---|
| White material (pawns=1, Q=9, etc.) | 26.1 | 11.8 | 0 | 29 | 46 |
| Black material | 26.1 | 11.8 | 0 | 29 | 45 |
| Material balance (W−B) | −0.02 | 2.5 | −45 | 0 | 31 |
| Total material | 52.2 | 23.4 | 1 | 59 | 81 |
| White pieces (count) | 11.2 | 4.1 | 1 | 11 | — |
| Black pieces (count) | 11.2 | 4.1 | 1 | 11 | — |
| Total pieces | 22.3 | 8.0 | 3 | 22 | — |
| Halfmove clock | 1.87 | 3.42 | 0 | — | — |

> Material balance is nearly zero on average (symmetric games), with rare extreme swings (±45 = one side has nearly a queen extra).

---

## Feature Engineering (802-dim total → 762 live)

Three encoding functions produce the full feature vector:

### 1. `board_to_array(board)` — 768 dims

```
64 squares × 12 piece types (6 white + 6 black) = 768 one-hot binary features
```

For each square: `[WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK]` — all 0 if empty.

**Design note**: The old encoding used 128 one-hot from/to-square columns, treating `from_square=4` as if it has a magnitude relationship. This encoding avoids fake ordinality.

### 2. `move_to_semantic_array(board, move)` — 24 dims

```
Moving piece type one-hot (6)      = 6
Moving piece color (1)             = 1  → OHE (2)
From-square rank / file             = 2
To-square rank / file               = 2
Rank distance / File distance       = 2
is_capture (1)                      = 1
Captured piece type one-hot (6)    = 6
gives_check (1)                    = 1
is_promotion (1)                   = 1
is_castling (1)                    = 1
                                -------
Total                              = 24
```

### 3. `get_metadata_features_fast(board)` — 10 dims

```
Castling rights (4): KS-W, QS-W, KS-B, QS-B
Side to move (1)
En-passant square (1)
Mobility: num legal moves (1)
Material counts (3)
```

### Dead Feature Audit

| Feature Pool | Count |
|---|---|
| Total raw features | 802 |
| All-zero features | 40 |
| Constant features | 40 (same set) |
| **Live features (used in models)** | **762** |

The 40 dead features are board squares that **never** contain certain piece types in the dataset (e.g., a king is never on a1 or h8 in the 2200–2600 ELO corpus, certain squares can never have certain pieces due to the starting position and game structure).

---

## Dataset Construction

### Strategy: Negative Sampling

For each position in each game:
- **1 positive sample**: the move actually played (`label = 1`)
- **N negative samples**: randomly chosen legal but unplayed moves (`label = 0`)

Two datasets built:

| Dataset | Negatives/position | Used for | Shape |
|---|---|---|---|
| `df_ranker` | 15 | LightGBM Ranker | **(6,086,182, 762)** |
| `df_dataset_balanced` | 3 | Classifier models | **(3,115,830, 762)** |

> `df_ranker` uses 15 negatives to give the ranker sufficient contrast between candidates. The ranker needs to see many negative alternatives to learn fine-grained ranking.

### Balanced Dataset Stats

```
Rows:              3,115,830
Unique positions:    778,958
Positive labels:     791,425  (25.4%)
Negative labels:   2,324,405  (74.6%)
Samples/position:       ~4.0  (always exactly 1 pos + 3 neg)
```

### Train / Test Split

| Set | Games | Rows (ranker) | Rows (balanced) |
|---|---|---|---|
| Train | **12,000** (80%) | — | — |
| Test | **3,000** (20%) | — | — |

> **Critical**: game-level split — all positions from a given game go entirely to train or test. No position leakage.

### Train / Validation Sub-split (within train)

Using `GroupShuffleSplit` with `game_id` as grouping key (10% val):

| Pipeline | Train rows | Val rows | Train games | Val games | Overlap |
|---|---|---|---|---|---|
| Ranker | 5,477,085 | 609,097 | 4,500 | 500 | **0** |
| Classifier | 2,804,267 | 311,563 | 8,991 | 999 | **0** |

Ranker group sizes: 48–3,480 positions per game (reflects variable game length × 16 samples/position).

---

## Models Trained

### Model Progression

```
Logistic Regression     (linear baseline)
       ↓
SGD Classifier          (online linear, large-scale)
       ↓
LightGBM Classifier     (gradient boosting, binary CE)
       ↓
XGBoost Classifier      (gradient boosting, binary CE)
       ↓
LightGBM Ranker v1      (LambdaRank, 300 trees)
       ↓
LightGBM Ranker v2      (tuned: more leaves, regularization)
       ↓
LightGBM Ranker v3      (hyperparameter search, 20 iterations)
```

### Evaluation Protocol

- **Top-1**: human move ranked #1 among all legal moves for that position
- **Top-3**: human move in top 3
- **Gap**: train Top-1 − test Top-1 (>0.05 = overfit warning)
- Evaluation on held-out `test_games` (3,000 games, never seen during training)

---

## Results — All Models

| Model | Train Top-1 | Test Top-1 | Gap | Status |
|---|---|---|---|---|
| **LightGBM Ranker v3** | 0.382 | **0.332** | 0.050 | ✅ ok |
| **LightGBM Ranker v2** | 0.354 | **0.322** | 0.032 | ✅ ok |
| LightGBM Ranker v1 | 0.344 | 0.288 | 0.056 | ⚠️ overfit |
| LightGBM Classifier | 0.344 | 0.278 | 0.066 | ⚠️ overfit |
| XGBoost Classifier | 0.328 | 0.254 | 0.074 | ⚠️ overfit |
| Logistic Regression | 0.236 | 0.194 | 0.042 | ✅ ok |
| SGD Classifier | 0.232 | 0.192 | 0.040 | ✅ ok |

**Best ML model**: LightGBM Ranker v3 — **Test Top-1 = 0.332, Test Top-3 = 0.544**

### Logistic Regression (Baseline Detail)
- Val Top-1: 0.2557, Val Top-3: 0.7499
- Test Top-1: 0.194, Gap: 0.042

### LightGBM Ranker v3 Config (Best)
```python
LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=300,
    num_leaves=255,
    learning_rate=0.05,
    min_data_in_leaf=120,
    feature_fraction=0.5,
    bagging_fraction=0.8,
    lambda_l1=0.1,
    lambda_l2=0.0,
    bagging_freq=1
)
```

---

## Key Observations from ML Baseline

1. **Ranking >> Classification**: LightGBM Ranker v1 (0.288) outperforms LightGBM Classifier (0.278) using the same features. The ranking formulation (LambdaRank) is more appropriate for this task — we want to rank legal moves, not binary-classify each one independently.

2. **More negatives = better ranker**: v1 (ranker) used `df_ranker` with 15 negatives. The extra contrast between candidates helps the ranker learn finer distinctions.

3. **Tree models overfit**: Both LightGBM Classifier (gap=0.066) and XGBoost (gap=0.074) exceed the overfit threshold. The ranker v2 and v3 add regularization (`min_data_in_leaf`, `feature_fraction`, L1) to bring the gap below 0.05.

4. **Linear models are safe but weak**: Logistic Regression (0.194) and SGD (0.192) don't overfit but plateau far below tree models — the 762-dim feature space has complex interactions they can't capture.

5. **ML ceiling**: Best ML = **0.332** Top-1. The DL experiments (v1–v5) were explicitly motivated to beat this.

---

## ML vs. DL Comparison (Cross-Experiment)

| System | Test Top-1 | Notes |
|---|---|---|
| Random move | ~1/29.5 ≈ 3.4% | Avg legal moves = 29.5 |
| Logistic Regression | 0.194 | ML baseline |
| LightGBM Ranker v3 | **0.332** | Best ML |
| DL v1 (GPT, 5 ep) | 0.2693 | Worse than best ML |
| DL v2 (+ELO, 30 ep) | ~0.29 | Near ML territory |
| DL v3 (+flat board) | (not recorded) | — |
| DL v4 (Chessformer+GAB) | (see checkpoint) | Goal: beat 0.332 |
| DL v5 (CNN+SE) | (see checkpoint) | Goal: beat 0.332 |

> v1's 0.2693 is **below** the best ML model. The DL work is not automatically superior — it's only in v4/v5 with explicit board structure that DL is expected to become competitive or better.
