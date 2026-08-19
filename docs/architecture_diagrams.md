# Architecture Diagrams — ChessMoveNet-DL

Mermaid diagrams for every major architecture. Use these as visual references alongside the per-version docs.

---

## Overall Experiment Lineage

```mermaid
graph TD
    DATA["Lichess Elite 2019-10\n14,945 games · ELO 2200–2600\n1,218,514 positions"]
    
    DATA --> ML["ML Baseline\nchess_human_move_predictor_v4.ipynb"]
    DATA --> DL["DL Experiments"]
    
    ML --> LR["Logistic Regression\nTop-1: 0.194"]
    ML --> SGD["SGD Classifier\nTop-1: 0.192"]
    ML --> LGBC["LightGBM Classifier\nTop-1: 0.278"]
    ML --> XGB["XGBoost Classifier\nTop-1: 0.254"]
    ML --> RNK1["LightGBM Ranker v1\nTop-1: 0.288"]
    ML --> RNK2["LightGBM Ranker v2\nTop-1: 0.322"]
    ML --> RNK3["LightGBM Ranker v3\nTop-1: 0.332 ← ML Best"]
    
    DL --> V1["v1 · Pure GPT\nTop-1: 0.2693"]
    V1 --> V2["v2 · + ELO Conditioning\nTop-1: ~0.29"]
    V2 --> V3["v3 · + Flat Board + Meta\n(numbers not recorded)"]
    V3 --> V4["v4 · Chessformer + GAB\n(see checkpoint)"]
    V4 --> CAR["Carlsen Fine-tune\n~0.45 on Carlsen only"]
    V4 --> V5["v5 · CNN ResNet + SE\n(vs v4 head-to-head)"]
```

---

## v1 / v2 — GPT Architecture

```mermaid
graph TD
    subgraph INPUT["Input (per game)"]
        TOK["Move Tokens\nBOS, e2e4, g8f6, ..."]
        ELO2["Avg ELO (v2 only)\nnormalized to 0–1"]
    end

    subgraph EMBED["Embeddings"]
        TE["Token Embedding\nd_model=256"]
        PE["Position Embedding\nlearned absolute"]
        EP["ELO Projection\nLinear(1→256)\nv2 only"]
    end

    subgraph ADD["Fusion"]
        SUM["Element-wise Sum\n+ broadcast ELO (v2)"]
    end

    subgraph XFMR["Transformer Blocks × 4"]
        LN1["LayerNorm"]
        ATTN["Causal Self-Attention\n4 heads, block_size=160"]
        LN2["LayerNorm"]
        MLP["MLP: 256→1024→256\nGELU"]
    end

    subgraph HEAD["Output Head"]
        LNF["LayerNorm"]
        LIN["Linear(256 → 4546)\nweight-tied to tok_emb"]
        SM["Softmax → Next Move"]
    end

    TOK --> TE
    TOK --> PE
    ELO2 --> EP
    TE --> SUM
    PE --> SUM
    EP --> SUM
    SUM --> LN1
    LN1 --> ATTN
    ATTN --> LN2
    LN2 --> MLP
    MLP --> LNF
    LNF --> LIN
    LIN --> SM
```

---

## v3 — Transformer + Flat Board Features

```mermaid
graph TD
    subgraph INPUT["Inputs (per position)"]
        TOK3["Move Token Sequence\nUCI ids, block_size=160"]
        BF["Board Feature Vector\n~768-dim flat float32"]
        MF["Meta Features\n7-dim: ELO×2, clock×2, etc."]
    end

    subgraph PROJ["Projections"]
        TE3["Token Embedding\n256-dim"]
        PE3["Position Embedding\n256-dim"]
        BP["Board Projection\nLinear(768→256)"]
        MP["Meta Projection\nLinear(7→256)"]
        BS["board_scale\nlearned scalar"]
        MS["meta_scale\nlearned scalar"]
    end

    subgraph FUSE["Fusion (Additive)"]
        ADD3["x = tok_emb + pos_emb\n  + board_scale × board_proj\n  + meta_scale × meta_proj"]
    end

    subgraph BLOCKS["Transformer Blocks × 4"]
        BLK3["Pre-LN Transformer Block\nd_model=256, 4 heads"]
    end

    subgraph HEAD3["Output"]
        LNF3["LayerNorm → Linear(256→4546)"]
    end

    TOK3 --> TE3
    TOK3 --> PE3
    BF --> BP --> BS
    MF --> MP --> MS
    TE3 --> ADD3
    PE3 --> ADD3
    BS --> ADD3
    MS --> ADD3
    ADD3 --> BLK3
    BLK3 --> LNF3
```

> ⚠️ **v3 failure mode**: if `board_scale` >> `tok_emb`, board features dominate and the model ignores move sequence. Learned gains address but don't fully solve this; v4 uses separate tokens instead.

---

## v4 — Chessformer Architecture

```mermaid
graph TD

    %% =========================================================
    %% INPUT TOKENIZATION
    %% =========================================================

    subgraph BOARD["Board Tokens — 64"]
        PIE["piece_emb(piece_id)<br/>13 piece types → d_model"]
        SQE["square_emb(sq_index)<br/>64 squares → d_model"]
        SUM["Board Token<br/>piece_emb + square_emb"]
    end

    PIE --> SUM
    SQE --> SUM


    subgraph OTHER["Other Tokens — 3"]
        STA["State Token ×1<br/>state_proj(13-dim)<br/>turn + castling + en-passant"]
        SKL["Skill Token ×1<br/>skill_proj(7-dim)<br/>ELO ×2 + clocks + TC"]
        HST["History Tokens ×3<br/>hist_from_emb + hist_to_emb + hist_age_emb<br/>last K moves"]
    end


    %% =========================================================
    %% TOKEN SEQUENCE
    %% =========================================================

    subgraph SEQ["Input Sequence — 67 Tokens"]
        TOK["[sq_0 ... sq_63 | state | skill | hist_0 | hist_1 | hist_2]"]
    end

    SUM --> TOK
    STA --> TOK
    SKL --> TOK
    HST --> TOK


    %% =========================================================
    %% TRANSFORMER
    %% =========================================================

    subgraph TRANSFORMER["Transformer Encoder — 6 × Pre-LN Blocks"]

        X["Input X<br/>(67 × d_model)"]

        LN1["LayerNorm"]

        subgraph ATTENTION["Multi-Head Self-Attention — 8 Heads"]
            QKV["Q, K, V Projections"]
            SCORES["Attention Scores<br/>QKᵀ / √d_head"]

            GAB["Geometric Attention Bias<br/>rel_bias[head, Δfile × 15 + Δrank]<br/>229 buckets / layer"]

            ADD["Add GAB"]
            SOFTMAX["Softmax"]
            AV["Attention × V"]
            OPROJ["Output Projection"]
        end

        RES1["Residual Add"]

        LN2["LayerNorm"]

        MLP["MLP<br/>256 → 1024 → 256<br/>GELU"]

        RES2["Residual Add"]

        XOUT["Output X<br/>(67 × d_model)"]

        X --> LN1
        LN1 --> QKV
        QKV --> SCORES
        GAB --> ADD
        SCORES --> ADD
        ADD --> SOFTMAX
        SOFTMAX --> AV
        AV --> OPROJ
        OPROJ --> RES1

        X --> RES1

        RES1 --> LN2
        LN2 --> MLP
        MLP --> RES2
        RES1 --> RES2

        RES2 --> XOUT

    end

    TOK --> X

    XOUT -.->|"repeat ×6"| X


    %% =========================================================
    %% POLICY HEAD
    %% =========================================================

    subgraph POLICY["Bilinear Policy Head"]

        RESHAPE["Select square tokens<br/>sq_0 ... sq_63<br/>(64 × d_model)"]

        QPROJ["q_proj<br/>(64, d_head)<br/>from-square"]

        KPROJ["k_proj<br/>(64, d_head)<br/>to-square"]

        OUTER["Bilinear / Outer Product<br/>Q × Kᵀ<br/>(64 × 64)"]

        GATHER["Gather via MOVE_FT<br/>→ 4546 legal move logits"]

        PROMO["Promotion Bias<br/>+ promo_bias[to_sq, piece]"]

        LOGITS["Final Move Logits<br/>(4546,)"]

        RESHAPE --> QPROJ
        RESHAPE --> KPROJ

        QPROJ --> OUTER
        KPROJ --> OUTER

        OUTER --> GATHER
        GATHER --> PROMO
        PROMO --> LOGITS

    end

    XOUT --> RESHAPE
```

### GAB Detail

```mermaid
graph LR
    subgraph GAB_DETAIL["Geometric Attention Bias (GAB)"]
        FROM["from-square (file, rank)"]
        TO["to-square (file, rank)"]
        DELTA["Δfile = to_file - from_file\nΔrank = to_rank - from_rank"]
        BUCKET["bucket = Δfile×15 + Δrank\n(225 spatial + 4 special = 229 total)"]
        BIAS["rel_bias[head, bucket]\nlearned, init=0"]
        ADD["attn_logit += bias"]
    end

    FROM --> DELTA
    TO --> DELTA
    DELTA --> BUCKET --> BIAS --> ADD
```

---

## v5 — CNN Architecture

```mermaid
graph TD
    subgraph INPUT5["Input Tensor (B, IN_CH, 8, 8)"]
        PLANES["Bitboard Planes:\n· 12 piece planes (one-hot per piece type)\n· 1 side-to-move plane\n· 4 castling rights planes\n· 1 en-passant plane\n· 2×K_HISTORY history planes\n= IN_CH total channels"]
        META5["Meta scalars (broadcast in forward):\n7 scalars: ELO×2, diff, clock×2, TC base"]
    end

    subgraph STEM["Stem"]
        CONV1["Conv2d(IN_CH, CH, 3×3, pad=1)\n→ BatchNorm → ReLU"]
    end

    subgraph RESBLOCK["Residual Block (×N_BLOCKS=14)"]
        C1["Conv2d(CH, CH, 3×3, pad=1) → BN → ReLU"]
        C2["Conv2d(CH, CH, 3×3, pad=1) → BN\n(last BN weight init=0 → starts as identity)"]
        RES["+ residual → ReLU"]
        SE_BLOCK["SE Block (USE_SE=True):\nglobal avg pool → FC(CH→CH/r) → ReLU\n→ FC(CH/r→CH) → Sigmoid → channel gate"]
    end

    subgraph POLICY5["Bilinear Policy Head (same as v4)"]
        RESHAPE5["(B, CH, 8, 8) → (B, 64, CH)"]
        QK5["q_proj(CH→d_head), k_proj(CH→d_head)"]
        OUTER5["Outer product (64×64)"]
        GATHER5["Gather via MOVE_FT → (4546,)"]
        PROMO5["+ promo_bias for promotions"]
    end

    PLANES --> CONV1
    META5 -.broadcast.-> CONV1
    CONV1 --> C1 --> C2 --> RES --> SE_BLOCK
    SE_BLOCK --> RESHAPE5
    RESHAPE5 --> QK5 --> OUTER5 --> GATHER5 --> PROMO5
```

---

## Bilinear Policy Head (Shared: v4 + v5)

```mermaid
graph LR
    subgraph HEAD_DETAIL["Bilinear Move Scoring"]
        SQ["64 per-square vectors\n(from transformer or CNN trunk)"]
        QP["q_proj → query per square\nshape: 64 × d_head"]
        KP["k_proj → key per square\nshape: 64 × d_head"]
        DOT["logit(f→t) = q_f · k_t / sqrt(d_head)\n64×64 = 4096 raw logits"]
        FT["MOVE_FT lookup table\nmaps (from_sq, to_sq) → vocab_id\nfor all 4,546 valid moves"]
        VOCAB["4,546 vocab logits"]
        PB["+ promo_bias[to_sq, piece]\nfor promotion moves only"]
        OUT["Final logits → softmax → Top-k"]
    end

    SQ --> QP
    SQ --> KP
    QP --> DOT
    KP --> DOT
    DOT --> FT --> VOCAB --> PB --> OUT
```

**Why bilinear?** A flat head needs `4546 × d_model` ≈ 1.17M params. Bilinear factorization costs `2 × d_model × d_head` ≈ 131k params and bakes in the knowledge that moves are (from-square, to-square) pairs.

---

## ML Baseline Architecture

```mermaid
graph TD
    subgraph ML_INPUT["Per-candidate-move features (762 live dims)"]
        B762["board_to_array: 768-dim → (40 dead dropped)\n64 squares × 12 piece types (one-hot)"]
        M24["move_to_semantic_array: 24-dim\npiece type OHE (6) + color (2) + from/to rank/file (4)\n+ distances (2) + capture (7) + check/promo/castle (3)"]
        META10["get_metadata_features_fast: 10-dim\ncastling (4) + turn (1) + en-passant (1)\n+ mobility + material (3)"]
    end

    subgraph SAMPLING["Negative Sampling"]
        POS["1 positive (played move) label=1"]
        NEG15["15 random legal negatives (ranker)\nor 3 (classifier) label=0"]
    end

    subgraph MODELS["Models"]
        LR2["Logistic Regression\nTest Top-1: 0.194"]
        SGD2["SGD Classifier\nTest Top-1: 0.192"]
        LGBC2["LightGBM Classifier\nTest Top-1: 0.278"]
        XGB2["XGBoost Classifier\nTest Top-1: 0.254"]
        RNK["LightGBM Ranker v3\nLambdaRank · NDCG\nTest Top-1: 0.332 ← best"]
    end

    B762 --> ML_STACK["Feature matrix (762-dim)"]
    M24 --> ML_STACK
    META10 --> ML_STACK
    POS --> ML_STACK
    NEG15 --> ML_STACK
    ML_STACK --> LR2
    ML_STACK --> SGD2
    ML_STACK --> LGBC2
    ML_STACK --> XGB2
    ML_STACK --> RNK
```

---

## Evaluation Flow (Consistent Across All Models)

```mermaid
sequenceDiagram
    participant Game as Held-out Game
    participant Chess as python-chess
    participant Model
    participant Eval as Evaluator

    loop For each position t in game
        Game->>Chess: replay to position t (real moves)
        Chess->>Eval: legal_moves(board_t)
        Game->>Model: board/tokens at position t
        Model->>Eval: raw logits over all 4546 tokens
        Eval->>Eval: mask to legal moves only
        Eval->>Eval: rank legal moves by masked prob
        Eval->>Eval: check if played_move in Top-1 / Top-3
        Note over Eval: Also record: was raw argmax legal? (unmasked_argmax_legal_rate)
    end
    Eval->>Eval: aggregate Top-1, Top-3, legal_rate over all positions
```
