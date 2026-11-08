"""Equity-vs-inferred-range strategy for heads-up Short Deck PLO5."""
from __future__ import annotations

import hashlib
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

import fasteval as fe
from game import Action, Street

RAISES = (Action.RAISE_HALF, Action.RAISE_POT)

_pf = np.load(Path(__file__).resolve().parent / "tables" / "preflop.npz")
PF_KEY = _pf["key"]
PF_EQ = _pf["eq"].astype(np.float64)
_counts = np.bincount(PF_KEY, minlength=len(PF_EQ)).astype(np.float64)
_order = np.argsort(PF_EQ)
_cum = np.cumsum(_counts[_order]) / _counts.sum()
PF_PCT = np.empty_like(PF_EQ)
PF_PCT[_order] = _cum - _counts[_order] / (2 * _counts.sum())

BINOM = np.zeros((37, 6), dtype=np.int64)
for n in range(37):
    BINOM[n, 0] = 1
    for k in range(1, 6):
        BINOM[n, k] = BINOM[n - 1, k - 1] + BINOM[n - 1, k] if n else 0
DECK = np.arange(36)

DEFAULT_PARAMS = dict(
    samples=400,
    # range model
    prior_n=8.0,
    prior_agg=(0.35, 0.35, 0.35, 0.35),
    prior_fold=0.4,
    agg_tighten=0.5,
    floor=0.03,
    width=0.146,
    call_range_scale=0.669,
    # policy
    call_margin=-0.014,
    raise_edge=0.106,
    raise_pot_eq=0.596,
    bet_half_eq=0.535,
    bet_pot_eq=0.55,
    pf_open_eq=0.45,
    pf_open_pot_eq=0.554,
    pf_limp_eq=0.341,
    bluff_fold_rate=0.496,
    bluff_freq=0.674,
    bluff_size_pot=0.0,
    time_bank=3.5,
    call_overhead=0.0011,
    grace=0.001,
    dead_streak=8,
    pool_mult=6.0,
    size_cond=1,
    draw_mix=0.0,
    mdf_slack=0.983,
    oop_pen=0.007,
    style_min_acts=40,
    route_min_faced=60,
    route_fold=0.45,
    route_agg=0.18,
    route_hyst=0.04,
    style_lo=0.12,
    style_hi=0.50,
    style_hyst=0.05,
    ip_bonus=0.06,
    mdf_relax=0.065,
    half_q_cap=1.0,
    mdf_slack_flop=2.0,
    mdf_slack_turn=2.0,
    mdf_slack_river=2.0,
)


# ----------------------------------------------------------------------------- helpers

def pf_class(hands: np.ndarray) -> np.ndarray:
    s = np.sort(hands, axis=-1)
    return PF_KEY[sum(BINOM[s[..., i], i + 1] for i in range(5))]


def soft_step(pct: np.ndarray, q: float, floor: float, width: float) -> np.ndarray:
    return floor + (1 - floor) * np.clip((pct - q) / width + 0.5, 0.0, 1.0)


def pct_rank(x: np.ndarray) -> np.ndarray:
    """Within-sample percentile (mid-rank, ties broken by order) using a single sort."""
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return (r + 0.5) / len(x)


def made_pct(opp: np.ndarray, board: tuple[int, ...]) -> np.ndarray:
    """Within-sample percentile of each opp hand's current made strength (flop/turn/river)."""
    b = np.array(board)
    pairs = fe.pair_indices(opp)  # (N,10)
    tris = np.array([fe.TRI_IDX[x, y, z] for x, y, z in combinations(b.tolist(), 3)])
    strength = fe.PT[pairs[:, :, None], tris[None, None, :]].reshape(len(opp), -1).max(axis=1)
    return pct_rank(strength)


# ----------------------------------------------------------------------------- opponent model

def new_model() -> dict:
    z = [0, 0, 0, 0]
    return dict(act=list(z), agg=list(z), pot=list(z), faced=list(z), folded=list(z), streak=0)


def agg_rate(m: dict, s: int, p: dict) -> float:
    return (m["agg"][s] + p["prior_n"] * p["prior_agg"][s]) / (m["act"][s] + p["prior_n"])


def pot_rate(m: dict, s: int, p: dict) -> float:
    return (m["pot"][s] + p["prior_n"] * p["prior_agg"][s] * 0.5) / (m["act"][s] + p["prior_n"])


def fold_rate(m: dict, s: int, p: dict) -> float:
    return (m["folded"][s] + p["prior_n"] * p["prior_fold"]) / (m["faced"][s] + p["prior_n"])


def ingest_history(obs, memory: dict) -> None:
    """Update opponent stats from actions not yet seen this hand."""
    if memory.get("hand") != obs.hand_index:
        memory["hand"] = obs.hand_index
        memory["seen"] = 0
        memory["pending"] = None
    m = memory["opp"]
    hist = obs.action_history
    for k in range(memory["seen"], len(hist)):
        a = hist[k]
        if a.is_me:
            if a.action in RAISES:
                memory["pending"] = int(a.street)
            continue
        s = int(a.street)
        m["act"][s] += 1
        if a.action in RAISES:
            m["agg"][s] += 1
        if a.action == Action.RAISE_POT:
            m["pot"][s] += 1
        if a.action in RAISES or a.amount > 0:
            m["streak"] = 0
        if memory["pending"] is not None:
            m["faced"][memory["pending"]] += 1
            memory["pending"] = None
    memory["seen"] = len(hist)


def on_hand_end(result, memory: dict) -> None:
    if result.hand_index == memory.get("num_hands", 1000) - 1:
        print(f"[huggy] half end: internal think {memory.get('think', 0.0):.2f}s over {memory.get('calls', 0)} "
              f"decisions, style {memory.get('style')}", file=sys.stderr, flush=True)
    m = memory["opp"]
    opp_folded = not result.went_to_showdown and result.winner == "me"
    if opp_folded:
        m["streak"] += 1
    if memory.get("hand") != result.hand_index:
        return
    pend = memory.get("pending")
    if pend is not None:
        m["faced"][pend] += 1
        m["act"][pend] += 1
        if opp_folded:
            m["folded"][pend] += 1


# ----------------------------------------------------------------------------- equity

def rollout_pct(opp: np.ndarray, board_s: tuple[int, ...], extra: np.ndarray) -> np.ndarray:
    """Percentile of opp strength on an independent random completion of board_s (draw-aware)."""
    k = 5 - len(board_s)
    full = np.concatenate([np.broadcast_to(np.array(board_s, dtype=np.int64), (len(opp), len(board_s))),
                           extra[:, :k]], axis=1)
    strength = fe.best_scores(fe.pair_indices(opp), fe.triple_indices(full))
    return pct_rank(strength)


def opp_weights(obs, opp: np.ndarray, extra: np.ndarray, memory: dict, p: dict) -> np.ndarray:
    m = memory["opp"]
    w = np.ones(len(opp))
    by_street_raises = [0, 0, 0, 0]
    by_street_called_bet = [False] * 4
    last_pot = [False] * 4
    last_me_raise = [False] * 4
    for a in obs.action_history:
        s = int(a.street)
        if a.is_me:
            last_me_raise[s] = a.action in RAISES
        else:
            if a.action in RAISES:
                by_street_raises[s] += 1
                last_pot[s] = a.action == Action.RAISE_POT
            elif last_me_raise[s]:
                by_street_called_bet[s] = True
            last_me_raise[s] = False
    visible = len(obs.board)
    for s in range(4):
        k = by_street_raises[s]
        if not k and not by_street_called_bet[s]:
            continue
        if s == 0:
            pct = PF_PCT[pf_class(opp)]
        else:
            nb = 3 + (s - 1)
            if nb > visible:
                continue
            pct = made_pct(opp, obs.board[:nb])
            if p["draw_mix"] > 0 and nb < 5:
                pct = (1 - p["draw_mix"]) * pct + p["draw_mix"] * rollout_pct(opp, obs.board[:nb], extra)
        if k:
            rate = pot_rate(m, s, p) if p["size_cond"] and last_pot[s] else agg_rate(m, s, p)
            q = 1 - rate * p["agg_tighten"] ** (k - 1)
            if s > 0 and not last_pot[s]:
                q = min(q, p["half_q_cap"])
        else:
            q = fold_rate(m, s, p) * p["call_range_scale"]
        w *= soft_step(pct, q, p["floor"], p["width"])
    return w


def equity(obs, memory: dict, p: dict, n: int) -> float:
    """Equity vs inferred range. Weights are computed on a larger pool, then n hands are
    resampled with replacement in proportion to weight (unbiased, avoids ESS collapse)."""
    hole = np.array(obs.hole_cards)
    board = np.array(obs.board, dtype=np.int64)
    unknown = np.setdiff1d(DECK, np.concatenate([hole, board]))
    need = 5 + (5 - len(board))
    rng = memory["rng"]
    m = max(n, int(n * p["pool_mult"]))
    perm = unknown[random_prefix(rng, m, len(unknown), min(len(unknown), need + 5))]
    w = opp_weights(obs, perm[:, :5], perm[:, need:], memory, p)
    if m > n:
        idx = rng.choice(m, size=n, replace=True, p=w / w.sum())
        perm, w = perm[idx], np.ones(n)
    opp = perm[:, :5]
    boards = np.concatenate([np.broadcast_to(board, (len(perm), len(board))), perm[:, 5:need]], axis=1)
    tri = fe.triple_indices(boards)
    me = fe.best_scores(fe.pair_indices(hole), tri)
    them = fe.best_scores(fe.pair_indices(opp), tri)
    outcome = (me > them) + 0.5 * (me == them)
    return float((w * outcome).sum() / w.sum())


# ----------------------------------------------------------------------------- policy

def legal_raise(obs, prefer: Action) -> Action | None:
    if prefer in obs.legal_actions:
        return prefer
    for a in RAISES:
        if a in obs.legal_actions:
            return a
    return None


_BANK_ROWS, _BANK_K = 50000, 15
_bank_rng = np.random.default_rng(20261107)
PREFIX_BANK = {n: _bank_rng.random((_BANK_ROWS, n)).argsort(axis=1)[:, :_BANK_K].astype(np.int8)
               for n in range(21, 32)}


def random_prefix(rng, rows: int, n: int, k: int) -> np.ndarray:
    """Each row: first k entries of a uniform random permutation of range(n), drawn from a precomputed bank."""
    bank = PREFIX_BANK.get(n)
    if bank is None or k > _BANK_K:
        return rng.random((rows, n)).argsort(axis=1)[:, :k]
    return bank[rng.integers(0, _BANK_ROWS, rows), :k].astype(np.intp)


def charged(elapsed: float, p: dict) -> float:
    """Judge-style charge for one request: compute + estimated HTTP overhead, minus the 1 ms grace."""
    return max(0.0, elapsed + p["call_overhead"] - p["grace"])


def obs_seed(obs) -> int:
    key = repr((obs.hand_index, obs.hole_cards, obs.board, obs.pot, obs.to_call,
                tuple((int(a.street), a.is_me, int(a.action), a.amount) for a in obs.action_history)))
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little")


def decide(obs, memory: dict, p: dict) -> Action:
    t0 = time.perf_counter()
    memory["rng"] = np.random.default_rng(obs_seed(obs))
    ingest_history(obs, memory)
    used = memory.get("think", 0.0)
    scale = memory.get("bank_scale", 1.0)
    bank = p["time_bank"] * scale
    pace = bank * (obs.hand_index + 1) / memory.get("num_hands", 1000) + 0.5 * scale
    if used < pace:
        n = p["samples"]
    elif used < bank:
        n = p["samples"] // 4
    elif used < bank + 1.0 * scale:
        n = 40
    else:
        n = 0

    opp_raised_pf = any((not a.is_me) and a.action in RAISES for a in obs.action_history)
    if obs.street == Street.PREFLOP and not opp_raised_pf:
        eq = float(PF_EQ[pf_class(np.array(obs.hole_cards))])
    elif n:
        eq = equity(obs, memory, p, n)
    else:
        eq = float(PF_EQ[pf_class(np.array(obs.hole_cards))]) if obs.street == Street.PREFLOP else 0.5

    action = policy(obs, eq, memory, p)
    if action in RAISES:
        memory["pending"] = int(obs.street)
    if obs.to_call > 0 and obs.street != Street.PREFLOP:
        memory["defense"][0] += 1
        memory["defense"][1] += action == Action.FOLD
        cell = memory["defense_ss"].setdefault(defense_key(obs), [0, 0])
        cell[0] += 1
        cell[1] += action == Action.FOLD
    memory["think"] = used + charged(time.perf_counter() - t0, p)
    memory["calls"] = memory.get("calls", 0) + 1
    memory["last_eq"] = eq
    return action


def defense_key(obs) -> tuple:
    """(street, size class) of the bet we face: 0 = up to ~half pot, 1 = larger."""
    return int(obs.street), 0 if obs.to_call <= 0.42 * obs.pot else 1


def policy(obs, eq: float, memory: dict, p: dict) -> Action:
    c, pot = obs.to_call, obs.pot
    if obs.street != Street.PREFLOP:
        eq += p["ip_bonus"] if obs.is_button else -p["oop_pen"]
    can_raise = obs.legal_actions and RAISES[0] in obs.legal_actions
    rng = memory["rng"]

    if can_raise and memory["opp"]["streak"] >= p["dead_streak"]:
        return legal_raise(obs, Action.RAISE_HALF)

    if obs.street == Street.PREFLOP and obs.raises_this_street == 0:
        if can_raise and eq >= p["pf_open_pot_eq"]:
            return legal_raise(obs, Action.RAISE_POT)
        if can_raise and eq >= p["pf_open_eq"]:
            return legal_raise(obs, Action.RAISE_HALF)
        if c == 0:
            return Action.CHECK_CALL
        return Action.CHECK_CALL if eq >= p["pf_limp_eq"] else Action.FOLD

    if c > 0:
        need = c / (pot + c)
        if can_raise and eq >= max(p["raise_pot_eq"], need + p["raise_edge"]):
            return legal_raise(obs, Action.RAISE_POT)
        if eq >= need + p["call_margin"]:
            return Action.CHECK_CALL
        if obs.street != Street.PREFLOP:
            faced, folded = memory["defense"]
            fold_rate_us = (folded + 2.0) / (faced + 6.0)
            if fold_rate_us > c / pot + p["mdf_slack"] and eq >= need - p["mdf_relax"]:
                return Action.CHECK_CALL
            faced_s, folded_s = memory["defense_ss"].get(defense_key(obs), (0, 0))
            slack_s = p[("mdf_slack_flop", "mdf_slack_turn", "mdf_slack_river")[int(obs.street) - 1]]
            if (folded_s + 2.0) / (faced_s + 6.0) > c / pot + slack_s and eq >= need - p["mdf_relax"]:
                return Action.CHECK_CALL
        return Action.FOLD

    if can_raise:
        if eq >= p["bet_pot_eq"]:
            return legal_raise(obs, Action.RAISE_POT)
        if eq >= p["bet_half_eq"]:
            return legal_raise(obs, Action.RAISE_HALF)
        s = int(obs.street)
        if s > 0 and fold_rate(memory["opp"], s, p) >= p["bluff_fold_rate"] and rng.random() < p["bluff_freq"]:
            if p["bluff_size_pot"] > 0 and rng.random() < p["bluff_size_pot"]:
                return legal_raise(obs, Action.RAISE_POT)
            return legal_raise(obs, Action.RAISE_HALF)
    return Action.CHECK_CALL


# ----------------------------------------------------------------------------- bot interface

STYLE_FILE = Path(__file__).resolve().parent / "tables" / "styles.json"


def match_settings(config) -> dict:
    """Time-bank scale and hands per half from the judge's AgentConfig (game.py marks both provisional)."""
    try:
        bank = float(getattr(config, "time_bank_seconds", 8.0) or 8.0)
        hands = int(getattr(config, "num_hands", 1000) or 1000)
    except (TypeError, ValueError):
        bank, hands = 8.0, 1000
    return {"bank_scale": max(0.05, bank / 8.0), "num_hands": max(1, hands)}


def classify_style(m: dict, current: str | None, p: dict) -> str | None:
    """Opponent style from overall raise rate, with hysteresis around the thresholds."""
    acts = sum(m["act"])
    if acts < p["style_min_acts"]:
        return current
    agg = sum(m["agg"]) / acts
    lo, hi, h = p["style_lo"], p["style_hi"], p["style_hyst"]
    if current == "pass" and agg < lo + h or current != "pass" and agg < lo - (h if current else 0):
        return "pass"
    if current == "aggr" and agg > hi - h or current != "aggr" and agg > hi + (h if current else 0):
        return "aggr"
    return "eq"


def make_bot(styles: dict | None = None, **overrides):
    """styles: {"pass"|"eq"|"aggr": param overrides}; defaults to tables/styles.json when present."""
    p = {**DEFAULT_PARAMS, **overrides}
    if styles is None and STYLE_FILE.exists():
        import json
        styles = json.loads(STYLE_FILE.read_text())
    style_params = {k: {**p, **v} for k, v in (styles or {}).items()}

    def new_half(config, memory: dict) -> None:
        memory.clear()
        memory["opp"] = new_model()
        memory["defense"] = [0, 0]
        memory["defense_ss"] = {}
        memory["rng"] = np.random.default_rng()
        memory["think"] = 0.0
        memory["style"] = None
        memory.update(match_settings(config))

    def choose_action(obs, memory: dict) -> Action:
        if style_params:
            memory["style"] = classify_style(memory["opp"], memory["style"], p)
        return decide(obs, memory, style_params.get(memory["style"], p))

    return new_half, choose_action, on_hand_end, p
