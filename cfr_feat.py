"""Card features for the CFR abstraction; identical code is used offline (training data) and online (bot)."""
from __future__ import annotations

import numpy as np

import fasteval as fe
from strategy import PF_EQ, pf_class, random_prefix

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
    draws = unknown[random_prefix(rng, n, len(unknown), need)]
    boards = np.concatenate([np.broadcast_to(board, (n, len(board))), draws[:, 5:]], axis=1)
    tri = fe.triple_indices(boards)
    me = fe.best_scores(fe.pair_indices(hole), tri)
    them = fe.best_scores(fe.pair_indices(draws[:, :5]), tri)
    return float(((me > them) + 0.5 * (me == them)).mean())


def bucket(eq: float, thresholds: np.ndarray) -> int:
    return int(np.searchsorted(thresholds, eq, side="right"))


GROUP = 5
LAYOUT = np.array([[240, 1, 1], [20, 3, 4], [20, 3, 4], [60, 1, 4]])  # per street: equity bins, potential bins, textures


def texture(board) -> int:
    """0 plain, +1 flush possible (3+ of a suit), +2 paired."""
    if len(board) == 0:
        return 0
    b = np.asarray(board, dtype=np.int64)
    flush = np.bincount(b & 3, minlength=4).max() >= 3
    paired = np.bincount(b >> 2, minlength=9).max() >= 2
    return int(flush) + 2 * int(paired)


def features(hole, board, rng: np.random.Generator, n: int = POST_SAMPLES) -> tuple[float, float, int]:
    """(equity vs random, potential, texture). Potential = between-runout variance of hand strength (E[HS^2]-E[HS]^2,
    binomial-noise corrected), from the same n draws grouped by runout (GROUP opponent hands per runout)."""
    hole = np.asarray(hole, dtype=np.int64)
    nb = len(board)
    if nb == 0:
        return float(PF_EQ[pf_class(hole)]), 0.0, 0
    tex = texture(board)
    if nb == 5:
        return eq_random(hole, board, rng, n), 0.0, tex
    board = np.asarray(board, dtype=np.int64)
    unknown = np.setdiff1d(DECK, np.concatenate([hole, board]))
    m = max(2, n // GROUP)
    n = m * GROUP
    draws = unknown[random_prefix(rng, n, len(unknown), 10 - nb)]
    run = draws[::GROUP, 5:]                                    # (m, 5-nb) one runout per group
    boards = np.concatenate([np.broadcast_to(board, (m, nb)), run], axis=1)
    tri = fe.triple_indices(boards)
    me = fe.best_scores(fe.pair_indices(hole), tri)             # (m,)
    opp = draws[:, :5]
    g = np.repeat(np.arange(m), GROUP)
    valid = ~(opp[:, :, None] == run[g][:, None, :]).any(axis=(1, 2))
    them = fe.best_scores(fe.pair_indices(opp), fe.triple_indices(boards[g]))
    win = (me[g] > them) + 0.5 * (me[g] == them)
    cnt = np.bincount(g, weights=valid, minlength=m)
    hs_sum = np.bincount(g, weights=win * valid, minlength=m)
    ok = cnt >= 2
    if not ok.any():
        return eq_random(hole, board, rng, n), 0.0, tex
    hs = hs_sum[ok] / cnt[ok]
    eq = float(hs_sum.sum() / max(1.0, cnt.sum()))
    var = float(hs.var() - np.mean(hs * (1 - hs) / (cnt[ok] - 1)))
    return eq, max(0.0, var), tex


def bucket2(eq: float, var: float, tex: int, s: int, thr_eq, thr_var) -> int:
    e_n, v_n, t_n = LAYOUT[s]
    e = int(np.searchsorted(thr_eq, eq, side="right"))
    v = int(np.searchsorted(thr_var, var, side="right")) if v_n > 1 else 0
    return (e * v_n + v) * t_n + (tex if t_n > 1 else 0)
