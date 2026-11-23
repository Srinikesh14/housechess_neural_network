"""Convert chess positions into the project's 838-value input representation."""

from __future__ import annotations

import chess
import numpy as np
from numpy.typing import NDArray

LEGACY_INPUT_SIZE = 773
INPUT_SIZE = 838
PIECE_FEATURES = 12 * 64
METADATA_OFFSET = PIECE_FEATURES
EN_PASSANT_OFFSET = LEGACY_INPUT_SIZE
HALFMOVE_CLOCK_OFFSET = EN_PASSANT_OFFSET + 64


def encode_board_v1(board: chess.Board) -> NDArray[np.float32]:
    """Encode the legacy 773-value representation for original models."""
    features = np.zeros(LEGACY_INPUT_SIZE, dtype=np.float32)

    for square, piece in board.piece_map().items():
        color_offset = 0 if piece.color == chess.WHITE else 6
        plane = color_offset + piece.piece_type - 1
        features[plane * 64 + square] = 1.0

    features[METADATA_OFFSET] = float(board.turn == chess.WHITE)
    features[METADATA_OFFSET + 1] = float(board.has_kingside_castling_rights(chess.WHITE))
    features[METADATA_OFFSET + 2] = float(board.has_queenside_castling_rights(chess.WHITE))
    features[METADATA_OFFSET + 3] = float(board.has_kingside_castling_rights(chess.BLACK))
    features[METADATA_OFFSET + 4] = float(board.has_queenside_castling_rights(chess.BLACK))
    return features


def encode_board(board: chess.Board) -> NDArray[np.float32]:
    """Encode a board as 12 piece planes plus 70 state features.

    The piece planes are ordered white pawn through king, then black pawn
    through king. Within each plane, python-chess square indices are used
    (a1=0 through h8=63). Metadata is side-to-move followed by white
    king-side, white queen-side, black king-side, and black queen-side
    castling rights. The next 64 values are a one-hot en-passant target
    square (all zero when absent), followed by the halfmove clock clipped
    to 100 and divided by 100.
    """
    features = np.zeros(INPUT_SIZE, dtype=np.float32)
    features[:LEGACY_INPUT_SIZE] = encode_board_v1(board)
    if board.ep_square is not None:
        features[EN_PASSANT_OFFSET + board.ep_square] = 1.0
    features[HALFMOVE_CLOCK_OFFSET] = min(board.halfmove_clock, 100) / 100.0
    return features


def encode_fen(fen: str) -> NDArray[np.float32]:
    """Parse a FEN string and encode its position."""
    return encode_board(chess.Board(fen))
