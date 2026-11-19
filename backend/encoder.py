"""Convert chess positions into the project's 773-value input representation."""

from __future__ import annotations

import chess
import numpy as np
from numpy.typing import NDArray

INPUT_SIZE = 773
PIECE_FEATURES = 12 * 64
METADATA_FEATURES = 5


def encode_board(board: chess.Board) -> NDArray[np.float32]:
    """Encode a python-chess board as 12 piece planes and 5 binary features.

    The piece planes are ordered white pawn through king, then black pawn
    through king. Within each plane, python-chess square indices are used
    (a1=0 through h8=63). Metadata is side-to-move followed by white
    king-side, white queen-side, black king-side, and black queen-side
    castling rights.
    """
    features = np.zeros(INPUT_SIZE, dtype=np.float32)

    for square, piece in board.piece_map().items():
        color_offset = 0 if piece.color == chess.WHITE else 6
        plane = color_offset + piece.piece_type - 1
        features[plane * 64 + square] = 1.0

    features[PIECE_FEATURES] = float(board.turn == chess.WHITE)
    features[PIECE_FEATURES + 1] = float(board.has_kingside_castling_rights(chess.WHITE))
    features[PIECE_FEATURES + 2] = float(board.has_queenside_castling_rights(chess.WHITE))
    features[PIECE_FEATURES + 3] = float(board.has_kingside_castling_rights(chess.BLACK))
    features[PIECE_FEATURES + 4] = float(board.has_queenside_castling_rights(chess.BLACK))
    return features


def encode_fen(fen: str) -> NDArray[np.float32]:
    """Parse a FEN string and encode its position."""
    return encode_board(chess.Board(fen))
