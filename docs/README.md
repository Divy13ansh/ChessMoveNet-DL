# ChessMoveNet-DL — Experimentation Documentation

> **Project Goal:** Predict the human-played chess move given the current board position and game context, using deep learning. The primary metric is **legal-move-masked Top-1 accuracy** (does the model rank the actual human move first among all legal moves?).

---

## Documentation Index

| File | Contents |
|---|---|
| [README.md](./README.md) | This index |
| [overview.md](./overview.md) | Project background, data, vocabulary, evaluation protocol |
| [ml_baseline_eda.md](./ml_baseline_eda.md) | **Previous semester ML work** — EDA, feature engineering, ML results |
| [v1_transformer_baseline.md](./v1_transformer_baseline.md) | Decoder-only transformer — pure move-token GPT |
| [v2_transformer_elo.md](./v2_transformer_elo.md) | +ELO conditioning on the move-token transformer |
| [v3_transformer_board_meta.md](./v3_transformer_board_meta.md) | +Flat board feature vector + meta features |
| [v4_chessformer_gab.md](./v4_chessformer_gab.md) | Chessformer: per-square tokens + Geometric Attention Bias |
| [v4_carlsen_finetune.md](./v4_carlsen_finetune.md) | Fine-tuning v4 on Magnus Carlsen's games |
| [v5_cnn.md](./v5_cnn.md) | CNN baseline — ResNet trunk vs. transformer |
| [architecture_diagrams.md](./architecture_diagrams.md) | Mermaid diagrams for all architectures |
| [results_summary.md](./results_summary.md) | All results, ablations, and cross-version comparison |
| [architecture_decisions.md](./architecture_decisions.md) | Key design decisions and lessons learned |

---

## Quick Numbers Reference

| Model | Test Top-1 | Test Top-3 |
|---|---|---|
| Random baseline | ~0.034 | ~0.102 |
| Logistic Regression | 0.194 | — |
| SGD Classifier | 0.192 | — |
| LightGBM Classifier | 0.278 | — |
| XGBoost Classifier | 0.254 | — |
| LightGBM Ranker v1 | 0.288 | — |
| LightGBM Ranker v2 | 0.322 | — |
| **LightGBM Ranker v3 (ML Best)** | **0.332** | **0.544** |
| DL v1 — GPT (5 ep) | 0.2693 | 0.4393 |
| DL v2 — GPT + ELO (30 ep) | ~0.29 | — |
| DL v3 — flat board + meta | not recorded | — |
| **DL v4 — Chessformer + GAB** | **0.4303** | **0.7096** |
| **DL v5 — CNN 14×160 + SE** | **0.4470** | **0.7290** |
