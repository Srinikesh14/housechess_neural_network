"""FastAPI endpoints for evaluating and playing through chess positions."""

from __future__ import annotations

import os
from pathlib import Path

import chess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.encoder import INPUT_SIZE, LEGACY_INPUT_SIZE, encode_board, encode_board_v1
from backend.mlp import MLP

app = FastAPI(title="Chess ANN", description="Educational NumPy chess evaluator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_VERSION = os.environ.get("CHESS_ANN_MODEL_VERSION", "v1").lower()
if MODEL_VERSION not in ("v1", "v2", "wsp"):
    raise ValueError("CHESS_ANN_MODEL_VERSION must be 'v1', 'v2', or 'wsp'.")
MODEL_PATHS = {
    "v1": Path(__file__).with_name("models") / "model_v1_original.npz",
    "v2": Path(__file__).with_name("models") / "model_v2_stockfish.npz",
}
if MODEL_VERSION == "wsp":
    wsp_model_path = os.environ.get("CHESS_ANN_WSP_MODEL_PATH")
    if not wsp_model_path:
        raise ValueError(
            "Set CHESS_ANN_WSP_MODEL_PATH to an explicitly accepted WSP model."
        )
    MODEL_PATHS["wsp"] = Path(wsp_model_path)
if MODEL_VERSION == "v1" and not MODEL_PATHS["v1"].is_file():
    MODEL_PATHS["v1"] = Path(__file__).with_name("model.npz")
MODEL_PATH = MODEL_PATHS[MODEL_VERSION]
if MODEL_PATH.is_file():
    model = MLP.load(MODEL_PATH)
    expected_size = LEGACY_INPUT_SIZE if MODEL_VERSION == "v1" else INPUT_SIZE
    if model.W1.shape[0] != expected_size:
        raise ValueError(
            f"Selected {MODEL_VERSION} model must use {expected_size} input features, "
            f"received {model.W1.shape[0]}."
        )
    model_status = {
        "v1": "trained",
        "v2": "trained-v2",
        "wsp": "accepted-wsp",
    }[MODEL_VERSION]
else:
    if MODEL_VERSION in ("v2", "wsp"):
        raise FileNotFoundError(
            f"Selected {MODEL_VERSION} model not found at {MODEL_PATH}; "
            "provide the model before selecting this version."
        )
    model = MLP(seed=7)
    model_status = "untrained-random-initialization"


class PositionRequest(BaseModel):
    fen: str = Field(default_factory=lambda: chess.STARTING_FEN)


class MoveRequest(PositionRequest):
    uci: str


def parse_board(fen: str) -> chess.Board:
    try:
        return chess.Board(fen)
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=f"Invalid FEN: {error}") from error


def encode_for_model(board: chess.Board):
    if model.W1.shape[0] == LEGACY_INPUT_SIZE:
        return encode_board_v1(board)
    if model.W1.shape[0] == INPUT_SIZE:
        return encode_board(board)
    raise ValueError(f"Unsupported model input dimension: {model.W1.shape[0]}")


def get_evaluation(board: chess.Board) -> tuple[float, float]:
    white_score = model.forward(encode_for_model(board))
    turn_score = white_score if board.turn == chess.WHITE else -white_score
    return white_score, turn_score


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": model_status}


@app.post("/api/evaluate")
def evaluate_position(request: PositionRequest) -> dict[str, float | str]:
    board = parse_board(request.fen)
    white_score, turn_score = get_evaluation(board)
    return {
        "white_score": white_score,
        "side_to_move_score": turn_score,
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
    }


@app.post("/api/legal-moves")
def legal_moves(request: PositionRequest) -> dict[str, list[dict[str, str | None]]]:
    board = parse_board(request.fen)
    moves = [
        {
            "uci": move.uci(),
            "san": board.san(move),
            "from": chess.square_name(move.from_square),
            "to": chess.square_name(move.to_square),
            "promotion": chess.piece_symbol(move.promotion) if move.promotion else None,
        }
        for move in board.legal_moves
    ]
    return {"moves": moves}


@app.post("/api/play")
def play_move(request: MoveRequest) -> dict[str, object]:
    board = parse_board(request.fen)
    try:
        move = chess.Move.from_uci(request.uci)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=f"Invalid UCI move: {error}") from error
    if move not in board.legal_moves:
        raise HTTPException(status_code=422, detail="That move is not legal in this position.")

    san = board.san(move)
    captured_piece = board.piece_at(move.to_square)
    if board.is_en_passant(move):
        captured_square = move.to_square - 8 if board.turn == chess.WHITE else move.to_square + 8
        captured_piece = board.piece_at(captured_square)
    captured_color = (
        "white" if captured_piece.color == chess.WHITE else "black"
    ) if captured_piece else None
    board.push(move)
    return {
        "fen": board.fen(),
        "san": san,
        "game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
        "turn": "white" if board.turn == chess.WHITE else "black",
        "captured_piece": captured_piece.symbol() if captured_piece else None,
        "captured_color": captured_color,
        "is_check": board.is_check(),
        "is_checkmate": board.is_checkmate(),
    }


@app.post("/api/best-move")
def best_move(request: PositionRequest) -> dict[str, str | float]:
    board = parse_board(request.fen)
    legal = list(board.legal_moves)
    if not legal:
        raise HTTPException(status_code=409, detail="There are no legal moves in this position.")

    moving_color = board.turn
    selected_move: chess.Move | None = None
    selected_score: float | None = None
    for move in legal:
        board.push(move)
        white_score = model.forward(encode_for_model(board))
        mover_score = white_score if moving_color == chess.WHITE else -white_score
        board.pop()
        if selected_score is None or mover_score > selected_score:
            selected_move = move
            selected_score = mover_score

    assert selected_move is not None and selected_score is not None
    return {
        "uci": selected_move.uci(),
        "san": board.san(selected_move),
        "predicted_mover_score": selected_score,
    }
