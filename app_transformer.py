"""Flask server for the v4 Chessformer.

v4 is an *encoder over one position*, not a sequence model: 64 square tokens plus a
state token, a skill token and K history tokens. There is no BLOCK_SIZE, no <BOS>
prefix and no growing token sequence -- the whole game history enters only through
the K_HISTORY from/to tokens. The model and feature code below are copied verbatim
from chess_transformer_v4.ipynb (sections 4, 5, 7, 8); they must stay in sync or the
checkpoint will silently mispredict.
"""
import json
import math
import os

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[+] Using device: {device}")

# Overridable so a fine-tuned checkpoint (e.g. finetune_carlsen.py's) can be served
# without editing this file. Both must come from the same training run.
VOCAB_PATH = os.environ.get("CHESS_VOCAB_PATH", "chess_move_vocab_v4.json")
MODEL_PATH = os.environ.get("CHESS_MODEL_PATH", "chess_gpt_v4.pt")

# ── Config, read from the artifact the notebook wrote ────────────────────────
# Defaults mirror the notebook's Section 1 so the app still stands up if a field
# is missing from an older vocab file.
CFG = {
    "d_model": 256, "n_head": 8, "n_layer": 8,
    "use_gab": True, "use_history": True, "k_history": 8,
    "use_square_aux": False,
    "state_dim": 13, "meta_dim": 7, "aux_dim": 4,
    "min_elo": 2200, "max_elo": 2600,
    "has_clock_data": False,
}

itos, stoi = [], {}

if os.path.exists(VOCAB_PATH):
    with open(VOCAB_PATH, "r") as f:
        vocab_data = json.load(f)
    itos = vocab_data["itos"]
    stoi = {tok: i for i, tok in enumerate(itos)}
    arch = vocab_data.get("arch", "unknown")
    if arch != "chessformer-v4":
        print(f"[!] '{VOCAB_PATH}' says arch={arch!r}, expected 'chessformer-v4'. "
              f"This server only speaks v4.")
    for key in CFG:
        if key in vocab_data:
            CFG[key] = vocab_data[key]
    print(f"[+] Vocabulary loaded: {len(itos)} tokens, arch={arch}")
    print(f"[+] Config: d_model={CFG['d_model']} n_layer={CFG['n_layer']} "
          f"n_head={CFG['n_head']} gab={CFG['use_gab']} "
          f"history={CFG['use_history']}(K={CFG['k_history']}) aux={CFG['use_square_aux']}")
else:
    print(f"[!] Vocabulary not found at '{VOCAB_PATH}'")

VOCAB_SIZE = len(itos)

D_MODEL   = CFG["d_model"]
N_HEAD    = CFG["n_head"]
N_LAYER   = CFG["n_layer"]
USE_GAB   = CFG["use_gab"]
USE_HISTORY = CFG["use_history"]
K_HISTORY = CFG["k_history"]
USE_SQUARE_AUX = CFG["use_square_aux"]
STATE_DIM = CFG["state_dim"]
META_DIM  = CFG["meta_dim"]
AUX_DIM   = CFG["aux_dim"]
MIN_ELO, MAX_ELO = CFG["min_elo"], CFG["max_elo"]
HAS_CLOCK_DATA = CFG["has_clock_data"]

ELO_SPAN = max(1, MAX_ELO - MIN_ELO)
_TC_LOG_MAX = math.log1p(10800.0)
NO_SQUARE = 64          # sentinel index for "no move here yet" in history
N_SPECIAL = 2           # <PAD>, <BOS> -- kept for index parity with v1-v3
N_PIECE_IDS = 13

# The training set is Lichess Elite blitz; 180+0 is the modal time control. The model
# saw a non-zero tc_base on essentially every game (the Elite DB keeps the TimeControl
# header even though it strips %clk), so feeding 0.0 here would be off-distribution.
DEFAULT_TC_BASE = 180.0
DEFAULT_TC_INC = 0.0

# ── Move-space decomposition (notebook section 4) ────────────────────────────
PROMOTION_PIECES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
PROMO_INDEX = {pc: i for i, pc in enumerate(PROMOTION_PIECES)}
N_PROMO = len(PROMOTION_PIECES)

MOVE_FROM  = np.zeros(VOCAB_SIZE, dtype=np.int64)
MOVE_TO    = np.zeros(VOCAB_SIZE, dtype=np.int64)
MOVE_PROMO = np.full(VOCAB_SIZE, -1, dtype=np.int64)

for _i in range(N_SPECIAL, VOCAB_SIZE):
    _mv = chess.Move.from_uci(itos[_i])
    MOVE_FROM[_i] = _mv.from_square
    MOVE_TO[_i] = _mv.to_square
    if _mv.promotion is not None:
        MOVE_PROMO[_i] = PROMO_INDEX[_mv.promotion]

MOVE_FT       = MOVE_FROM * 64 + MOVE_TO
MOVE_PROMO_FT = np.where(MOVE_PROMO >= 0, MOVE_TO * N_PROMO + np.maximum(MOVE_PROMO, 0), 0)
IS_PROMO      = MOVE_PROMO >= 0

# piece id: 0 = empty, 1-6 = white P N B R Q K, 7-12 = black P N B R Q K
PIECE_ID = {}
for _ci, _color in enumerate((chess.WHITE, chess.BLACK)):
    for _pi, _pt in enumerate((chess.PAWN, chess.KNIGHT, chess.BISHOP,
                               chess.ROOK, chess.QUEEN, chess.KING)):
        PIECE_ID[(_color, _pt)] = 1 + _ci * 6 + _pi


# ── Model (notebook sections 7-8) ────────────────────────────────────────────
def build_gab_index(n_square, n_global):
    """(N, N) bucket index. Square pairs -> (df, dr) bucket; anything else -> 3 extra buckets."""
    N = n_square + n_global
    idx = np.zeros((N, N), dtype=np.int64)
    n_geo = 15 * 15
    for i in range(N):
        for j in range(N):
            if i < n_square and j < n_square:
                df = chess.square_file(j) - chess.square_file(i)
                dr = chess.square_rank(j) - chess.square_rank(i)
                idx[i, j] = (df + 7) * 15 + (dr + 7)
            elif i < n_square:
                idx[i, j] = n_geo          # square -> global
            elif j < n_square:
                idx[i, j] = n_geo + 1      # global -> square
            else:
                idx[i, j] = n_geo + 2      # global -> global
    return idx, n_geo + 3


class GeometricSelfAttention(nn.Module):
    """Bidirectional self-attention over board tokens, with a learned per-head bias
    on the geometric relationship between every pair of squares."""

    def __init__(self, d_model, n_head, n_buckets, dropout=0.1, use_gab=True):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.use_gab = use_gab

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rel_bias = nn.Parameter(torch.zeros(n_head, n_buckets))

    def forward(self, x, gab_index):
        B, N, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, N, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_head, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if self.use_gab:
            att = att + self.rel_bias[:, gab_index].unsqueeze(0)

        att = self.dropout(F.softmax(att, dim=-1))
        out = (att @ v).transpose(1, 2).contiguous().view(B, N, C)
        return self.proj(out)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_head, n_buckets, dropout=0.1, use_gab=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = GeometricSelfAttention(d_model, n_head, n_buckets, dropout, use_gab)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, gab_index):
        x = x + self.attn(self.ln1(x), gab_index)
        x = x + self.mlp(self.ln2(x))
        return x


class Chessformer(nn.Module):
    """Encoder over 64 square tokens + global tokens, with a bilinear from/to policy head."""

    def __init__(self, d_model=D_MODEL, n_head=N_HEAD, n_layer=N_LAYER, dropout=0.1,
                 use_gab=USE_GAB, use_history=USE_HISTORY, k_history=K_HISTORY,
                 use_aux=USE_SQUARE_AUX):
        super().__init__()
        self.use_history = use_history
        self.use_aux = use_aux
        self.k_history = k_history if use_history else 0

        n_global = 2 + self.k_history
        gab_idx, n_buckets = build_gab_index(64, n_global)
        self.register_buffer("gab_index", torch.from_numpy(gab_idx))
        self.n_tokens = 64 + n_global

        self.piece_emb  = nn.Embedding(N_PIECE_IDS, d_model)
        self.square_emb = nn.Embedding(64, d_model)
        if use_aux:
            self.aux_proj = nn.Linear(AUX_DIM, d_model)

        self.state_proj = nn.Linear(STATE_DIM, d_model)
        self.skill_proj = nn.Linear(META_DIM, d_model)

        if use_history:
            self.hist_from_emb = nn.Embedding(65, d_model)
            self.hist_to_emb   = nn.Embedding(65, d_model)
            self.hist_age_emb  = nn.Embedding(k_history, d_model)

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, n_head, n_buckets, dropout, use_gab) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.promo_head = nn.Linear(d_model, N_PROMO)
        self.d_head_scale = 1.0 / math.sqrt(d_model)

        self.register_buffer("move_ft", torch.from_numpy(MOVE_FT))
        self.register_buffer("move_promo_ft", torch.from_numpy(MOVE_PROMO_FT))
        self.register_buffer("is_promo", torch.from_numpy(IS_PROMO))

    def forward(self, piece_ids, state, meta, hist_from=None, hist_to=None, aux=None):
        B = piece_ids.shape[0]

        sq = self.piece_emb(piece_ids) + self.square_emb.weight.unsqueeze(0)
        if self.use_aux and aux is not None:
            sq = sq + self.aux_proj(aux)

        tokens = [sq, self.state_proj(state).unsqueeze(1), self.skill_proj(meta).unsqueeze(1)]

        if self.use_history:
            age = self.hist_age_emb.weight.unsqueeze(0)
            tokens.append(self.hist_from_emb(hist_from) + self.hist_to_emb(hist_to) + age)

        x = self.drop(torch.cat(tokens, dim=1))
        for block in self.blocks:
            x = block(x, self.gab_index)
        x = self.ln_f(x)

        sq_out = x[:, :64]
        q = self.q_proj(sq_out)
        k = self.k_proj(sq_out)
        from_to = (q @ k.transpose(1, 2) * self.d_head_scale).reshape(B, 64 * 64)
        promo = self.promo_head(sq_out).reshape(B, 64 * N_PROMO)

        logits = from_to[:, self.move_ft]
        logits = logits + torch.where(
            self.is_promo.unsqueeze(0),
            promo[:, self.move_promo_ft],
            torch.zeros((), device=logits.device, dtype=logits.dtype),
        )
        logits[:, :N_SPECIAL] = float("-inf")
        return logits


# ── Load model ───────────────────────────────────────────────────────────────
model = None
if os.path.exists(MODEL_PATH) and VOCAB_SIZE > 0:
    try:
        model = Chessformer().to(device)
        state_dict = torch.load(MODEL_PATH, map_location=device)

        # Strict on purpose. A v1/v2/v3 checkpoint has entirely different keys, and
        # loading one non-strictly would leave a randomly initialised model quietly
        # serving predictions.
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())
        print(f"[+] Model loaded from {MODEL_PATH} ({n_params / 1e6:.2f}M params, "
              f"{model.n_tokens} tokens/position)")
    except Exception as e:
        print(f"[!] Failed to load model: {e}")
        print(f"[!] '{MODEL_PATH}' must be a v4 (Chessformer) checkpoint. v1/v2/v3 "
              f"checkpoints are not loadable here.")
        import traceback
        traceback.print_exc()
        model = None
else:
    print(f"[!] Model file not found at '{MODEL_PATH}' or vocabulary not loaded")


# ── Feature construction (notebook section 5, single-position form) ──────────
def _square_aux(board, out):
    """(64, AUX_DIM) binary per-square spatial features."""
    out[:] = 0.0
    for j, color in enumerate((chess.WHITE, chess.BLACK)):
        for sq in chess.scan_forward(board.occupied_co[color]):
            for tgt in chess.scan_forward(board.attacks_mask(sq)):
                out[tgt, j] = 1.0
    for mv in board.legal_moves:
        out[mv.from_square, 2] = 1.0
        out[mv.to_square, 3] = 1.0
    return out


def build_position_features(board, elo_white, elo_black, tc_base, tc_inc):
    """One position -> the batch-of-1 tensors Chessformer.forward expects.

    Mirrors extract_game_positions() for a single t. Two deliberate differences,
    both because inference starts from an arbitrary FEN rather than ply 0 of a
    replayed game:
      * side to move comes from board.turn, not from ply parity;
      * history comes from board.move_stack, which is only populated if the caller
        supplied a move history that actually reaches this position.
    """
    piece_ids = np.zeros((1, 64), dtype=np.int64)
    for sq, piece in board.piece_map().items():
        piece_ids[0, sq] = PIECE_ID[(piece.color, piece.piece_type)]

    state = np.zeros((1, STATE_DIM), dtype=np.float32)
    state[0, 0] = 1.0 if board.turn == chess.WHITE else 0.0
    state[0, 1] = float(board.has_kingside_castling_rights(chess.WHITE))
    state[0, 2] = float(board.has_queenside_castling_rights(chess.WHITE))
    state[0, 3] = float(board.has_kingside_castling_rights(chess.BLACK))
    state[0, 4] = float(board.has_queenside_castling_rights(chess.BLACK))
    if board.ep_square is not None:
        state[0, 5 + chess.square_file(board.ep_square)] = 1.0

    e_w = (elo_white - MIN_ELO) / ELO_SPAN
    e_b = (elo_black - MIN_ELO) / ELO_SPAN
    e_w = min(1.0, max(0.0, e_w))
    e_b = min(1.0, max(0.0, e_b))

    meta = np.zeros((1, META_DIM), dtype=np.float32)
    white_to_move = board.turn == chess.WHITE
    meta[0, 0] = e_w if white_to_move else e_b
    meta[0, 1] = e_b if white_to_move else e_w
    # meta[2:4] are the lagged clock readings. The Elite database strips %clk, so the
    # model was trained with has_clk=0 throughout and never learned to use them --
    # feeding anything but zero here would be off-distribution.
    meta[0, 4] = 1.0 if HAS_CLOCK_DATA else 0.0
    meta[0, 5] = min(1.0, math.log1p(tc_base) / _TC_LOG_MAX) if tc_base else 0.0
    meta[0, 6] = min(1.0, tc_inc / 60.0)

    feats = {
        "piece_ids": torch.from_numpy(piece_ids),
        "state": torch.from_numpy(state),
        "meta": torch.from_numpy(meta),
    }

    if USE_HISTORY:
        hist_from = np.full((1, K_HISTORY), NO_SQUARE, dtype=np.int64)
        hist_to   = np.full((1, K_HISTORY), NO_SQUARE, dtype=np.int64)
        stack = board.move_stack
        for k in range(K_HISTORY):
            j = len(stack) - 1 - k
            if j < 0:
                break
            hist_from[0, k] = stack[j].from_square
            hist_to[0, k] = stack[j].to_square
        feats["hist_from"] = torch.from_numpy(hist_from)
        feats["hist_to"] = torch.from_numpy(hist_to)

    if USE_SQUARE_AUX:
        aux = np.zeros((1, 64, AUX_DIM), dtype=np.float32)
        _square_aux(board, aux[0])
        feats["aux"] = torch.from_numpy(aux)

    return {k: v.to(device) for k, v in feats.items()}


def board_with_history(fen, move_history):
    """Rebuild a board that carries its move_stack, so the history tokens are real.

    Returns (board, history_used). If the supplied history does not replay to the
    supplied FEN we trust the FEN and drop the history rather than feed the model
    moves from a different game -- the model then sees the same all-sentinel history
    it sees at ply 0.
    """
    target = chess.Board(fen)

    replayed = chess.Board()
    ok = True
    for uci in move_history:
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError:
            ok = False
            break
        if mv not in replayed.legal_moves:
            ok = False
            break
        replayed.push(mv)

    # epd() compares placement/turn/castling/ep and ignores the move counters, which
    # are not v4 features.
    if ok and replayed.epd() == target.epd():
        return replayed, True
    return target, False


@torch.no_grad()
def predict_logits(board, elo_white, elo_black, tc_base, tc_inc):
    """(VOCAB_SIZE,) raw logits for the side to move. One forward pass, one position."""
    model.eval()
    feats = build_position_features(board, elo_white, elo_black, tc_base, tc_inc)
    return model(**feats)[0].float().cpu().numpy()


def rank_legal_moves(board, logits):
    """Legal-move-masked softmax. Masking before the softmax rather than renormalising
    afterwards -- if the model puts nearly all its mass on illegal moves the legal
    probabilities underflow to zero in float32 and the renormalisation divides by 0."""
    legal = list(board.legal_moves)
    if not legal:
        return [], None

    ids = np.array([stoi[m.uci()] for m in legal], dtype=np.int64)
    sub = logits[ids]
    sub = sub - sub.max()
    probs = np.exp(sub)
    probs /= probs.sum()

    order = np.argsort(-probs)
    ranked = [{"move": legal[i], "uci": legal[i].uci(), "prob": float(probs[i])} for i in order]

    # The v1-v4 board-tracking diagnostic: does the unmasked argmax land on a legal
    # move? In v4 the board is an input, so this should sit very high.
    argmax_is_legal = bool(int(logits.argmax()) in set(ids.tolist()))
    return ranked, argmax_is_legal


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/best-move", methods=["POST"])
def best_move():
    """Predict the move a human of the given rating would play.

    Expects JSON:
    {
        "fen": "...",
        "move_history": ["e2e4", "c7c5", ...],   // UCI moves from the start position
        "elo": 2400,                             // both sides, unless overridden
        "white_elo": 2400, "black_elo": 2400,    // optional, per side
        "tc_base": 180, "tc_inc": 0              // optional time control in seconds
    }
    """
    data = request.get_json(force=True)
    fen = data.get("fen", chess.STARTING_FEN)
    move_history = data.get("move_history", []) or []
    elo = data.get("elo", 2400)
    elo_white = data.get("white_elo", elo)
    elo_black = data.get("black_elo", elo)
    tc_base = float(data.get("tc_base", DEFAULT_TC_BASE))
    tc_inc = float(data.get("tc_inc", DEFAULT_TC_INC))

    try:
        board = chess.Board(fen)
    except ValueError as e:
        return jsonify({"error": f"Invalid FEN: {e}"}), 400

    if board.is_game_over():
        return jsonify({"is_game_over": True, "result": board.result(), "legal_moves": []})

    if model is None:
        legal = sorted(board.legal_moves, key=lambda m: m.uci())
        if not legal:
            return jsonify({"error": "No legal moves available"}), 400
        best = legal[0]
        return jsonify({
            "best_move": best.uci(),
            "uci": best.uci(),
            "san": board.san(best),
            "from_sq": chess.square_name(best.from_square),
            "to_sq": chess.square_name(best.to_square),
            "legal_moves": [m.uci() for m in board.legal_moves],
            "is_game_over": False,
            "model_loaded": False,
            "top5": [],
        })

    board, history_used = board_with_history(fen, move_history)

    try:
        logits = predict_logits(board, elo_white, elo_black, tc_base, tc_inc)
    except Exception as e:
        print(f"[!] Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    ranked, argmax_is_legal = rank_legal_moves(board, logits)
    if not ranked:
        return jsonify({"error": "No legal moves available"}), 400

    n_hist = min(len(board.move_stack), K_HISTORY) if USE_HISTORY else 0
    print("\n" + "=" * 60)
    print(f"  FEN        : {board.fen()}")
    print(f"  To move    : {'white' if board.turn == chess.WHITE else 'black'}")
    print(f"  ELO        : W {elo_white} / B {elo_black}")
    print(f"  History    : {n_hist}/{K_HISTORY} tokens"
          f"{'' if history_used else '  (history did not match the FEN, dropped)'}")
    print(f"  Legal moves: {len(ranked)}   argmax legal: {argmax_is_legal}")
    print("-" * 60)
    print("  RANK  SAN        UCI       PROBABILITY")
    for i, item in enumerate(ranked[:10]):
        san = board.san(item["move"])
        print(f"  {i+1:>3}.  {san:<10} {item['uci']:<8}  {item['prob']:.6f}{'  <' if i == 0 else ''}")
    print("=" * 60 + "\n")

    top5 = [
        {"rank": i + 1, "uci": it["uci"], "san": board.san(it["move"]), "score": it["prob"]}
        for i, it in enumerate(ranked[:5])
    ]

    best = ranked[0]["move"]
    return jsonify({
        "best_move": best.uci(),
        "uci": best.uci(),
        "san": board.san(best),
        "from_sq": chess.square_name(best.from_square),
        "to_sq": chess.square_name(best.to_square),
        "legal_moves": [m.uci() for m in board.legal_moves],
        "is_game_over": False,
        "model_loaded": True,
        "top5": top5,
        "history_used": history_used,
        "history_tokens": n_hist,
        "argmax_is_legal": argmax_is_legal,
    })


@app.route("/api/legal-moves", methods=["POST"])
def legal_moves():
    data = request.get_json(force=True)
    fen = data.get("fen", chess.STARTING_FEN)
    try:
        board = chess.Board(fen)
    except ValueError as e:
        return jsonify({"error": f"Invalid FEN: {e}"}), 400

    moves = [{
        "uci": m.uci(),
        "from_sq": chess.square_name(m.from_square),
        "to_sq": chess.square_name(m.to_square),
        "san": board.san(m),
    } for m in board.legal_moves]
    return jsonify({"legal_moves": moves, "fen": board.fen()})


@app.route("/api/apply-move", methods=["POST"])
def apply_move():
    data = request.get_json(force=True)
    fen = data.get("fen", chess.STARTING_FEN)
    uci = data.get("uci", "")

    try:
        board = chess.Board(fen)
    except ValueError as e:
        return jsonify({"error": f"Invalid FEN: {e}"}), 400

    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return jsonify({"error": f"Invalid UCI move: {uci}"}), 400

    if move not in board.legal_moves:
        return jsonify({"error": "Illegal move"}), 400

    san = board.san(move)
    board.push(move)
    is_over = board.is_game_over()
    return jsonify({
        "new_fen": board.fen(),
        "san": san,
        "uci": uci,
        "is_game_over": is_over,
        "result": board.result() if is_over else None,
    })


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "vocab_size": VOCAB_SIZE,
        "arch": "chessformer-v4",
        "n_layer": N_LAYER,
        "d_model": D_MODEL,
        "n_head": N_HEAD,
        "use_gab": USE_GAB,
        "k_history": K_HISTORY if USE_HISTORY else 0,
        "use_square_aux": USE_SQUARE_AUX,
        "tokens_per_position": model.n_tokens if model is not None else None,
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
                    if model is not None else None,
        "elo_range": [MIN_ELO, MAX_ELO],
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)