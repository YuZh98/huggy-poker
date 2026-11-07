"""Vectorized Short Deck PLO5 evaluator. Scores are dense ranks (higher is better), not evaluator.py ints."""
from itertools import combinations
from pathlib import Path

import numpy as np

_data = np.load(Path(__file__).resolve().parent / "tables" / "eval.npz")
PT = _data["pt"]
PAIR_IDX = _data["pair_idx"].astype(np.intp)
TRI_IDX = _data["tri_idx"].astype(np.intp)
RAW_SCORES = _data["raw_scores"]

_P = np.array(list(combinations(range(5), 2)))
_T = np.array(list(combinations(range(5), 3)))


def pair_indices(hole: np.ndarray) -> np.ndarray:
    """(..., 5) hole cards -> (..., 10) pair-table rows."""
    return PAIR_IDX[hole[..., _P[:, 0]], hole[..., _P[:, 1]]]


def triple_indices(board: np.ndarray) -> np.ndarray:
    """(..., 5) board cards -> (..., 10) triple-table columns."""
    return TRI_IDX[board[..., _T[:, 0]], board[..., _T[:, 1]], board[..., _T[:, 2]]]


def best_scores(pairs: np.ndarray, triples: np.ndarray) -> np.ndarray:
    """pairs (N,10) or (10,), triples (N,10) -> (N,) best dense score."""
    if pairs.ndim == 1:
        pairs = pairs[None, :]
    return PT[pairs[:, :, None], triples[:, None, :]].reshape(len(triples), -1).max(axis=1)


def score(hole, board) -> int:
    """Single-hand dense score; board must have 5 cards."""
    h = np.asarray(hole)
    b = np.asarray(board)
    return int(best_scores(pair_indices(h), triple_indices(b)[None, :])[0])
