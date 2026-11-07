"""Card features for the CFR abstraction; identical code is used offline (training data) and online (bot)."""
from __future__ import annotations

import numpy as np

import fasteval as fe
from strategy import PF_EQ, pf_class

DECK = np.arange(36)
POST_SAMPLES = 400


def eq_random(hole, board, rng: np.random.Generator, n: int = POST_SAMPLES) -> float:
    """Equity vs a uniformly random opponent hand (Monte Carlo over opp hand + runout)."""
    hole = np.asarray(hole, dtype=np.int64)
    if len(board) == 0:
        return float(PF_EQ[pf_class(hole)])
    board = np.asarray(board, dtype=np.int64)
    unknown = np.setdiff1d(DECK, np.concatenate([hole, board]))
    need = 5 + (5 - len(board))
    draws = unknown[rng.random((n, len(unknown))).argsort(axis=1)[:, :need]]
    boards = np.concatenate([np.broadcast_to(board, (n, len(board))), draws[:, 5:]], axis=1)
    tri = fe.triple_indices(boards)
    me = fe.best_scores(fe.pair_indices(hole), tri)
    them = fe.best_scores(fe.pair_indices(draws[:, :5]), tri)
    return float(((me > them) + 0.5 * (me == them)).mean())


def bucket(eq: float, thresholds: np.ndarray) -> int:
    return int(np.searchsorted(thresholds, eq, side="right"))
