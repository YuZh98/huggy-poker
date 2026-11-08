"""Implicit-modelling portfolio on top of the release-5 stop-loss hybrid (Bard, Johanson & Bowling 2013).

Heuristic by default, exactly as cfr_bot.make_bot(hybrid="stoploss"). A selector scores every observed opponent action
under K+1 node-level action models (index 0 = blueprint / unknown class) and keeps running log-likelihoods. CFR mode
turns on (never off) when the stop-loss fires or, with sel_early, when a non-blueprint model leads decisively at a
decision with hand_index >= stop_hmin. In CFR mode the leading model's response table replaces the blueprint; the row
lookup, purification and sampling are cfr_bot._choose's.

models npz: names (K strings, or K+1 with the class-0 label first), P float32 [K+1, N_nodes, 4], optional node_obs.
Names without a response table are absorbers: they compete in the likelihood so that a field opponent is claimed by
its own cluster rather than by a hard opponent's model; only hard names with a table can trigger the early switch.
With models=None, K = 0 and the bot is release 5.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cfr_bot  # noqa: E402
import cfr_feat  # noqa: E402
import strategy  # noqa: E402
from game import BIG_BLIND, Action  # noqa: E402

TREE = cfr_bot.TREE
ACTS = cfr_bot.ACTS
N_NODES = len(TREE.player)
LEGAL = np.array([[c >= 0 for c in ch] for ch in TREE.children], dtype=np.float64)
LOG_FLOOR = 1e-3
MAX_TABLES = 10


def _normalize(rows: np.ndarray) -> np.ndarray:
    """Rows to distributions; empty rows become uniform over the node's legal actions."""
    out = np.array(rows, dtype=np.float64, copy=True)
    empty = out.sum(-1) <= 0
    out[empty] = LEGAL[empty]
    return out / np.maximum(out.sum(-1, keepdims=True), 1e-12)


def blueprint_marginal(strat: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Node-level action distribution of a [N, B, 4] table; buckets weighted by `weights` [N, B] (uniform if None or
    all-zero at a node)."""
    if weights is None:
        return _normalize(strat.mean(axis=1, dtype=np.float64))
    w = np.asarray(weights, dtype=np.float64)
    w = np.where(w.sum(-1, keepdims=True) > 0, w, 1.0)
    w /= w.sum(-1, keepdims=True)
    return _normalize(np.einsum("nb,nba->na", w, strat, dtype=np.float64))


def _load_table(path):
    d = np.load(path)
    return d["strat"].astype(np.float32), d["thr"], d


def make_bot(tables: dict, models=None, blueprint=ROOT / "tables" / "cfr.npz", purify: float = 0.15,
             hybrid: str | None = "stoploss", stop_bb: float = 120.0, stop_hmin: int = 100,
             sel_min_acts: int = 100, sel_lead: float = math.log(20), sel_early: bool = True, sel_tail: bool = True, sel_max_net: float = math.inf,
             hard_names: set[str] | None = None, heur=None, **heur_overrides):
    """tables: {name: npz path} response tables sharing the blueprint's bucket thresholds; models: npz with names/P
    (names without a table play the blueprint in CFR mode); hard_names: classes allowed to trigger the early switch
    (default: the names that have a table; a leading absorber keeps the heuristic); sel_max_net: the early switch
    fires only while our net in the half is <= this many bb (chip guard; inf = off); sel_tail: also score the opponent's
    hand-ending action (fold vs call/check), inferred in on_hand_end from our last action and the result."""
    assert hybrid in (None, "stoploss"), hybrid
    assert len(tables) <= MAX_TABLES, "K <= 10 tables"
    H = heur or strategy
    strat_bp, thr, data = _load_table(blueprint)
    assert strat_bp.shape[0] == N_NODES, strat_bp.shape
    v2 = "layout" in data.files
    if v2:
        lay = data["layout"]
        thr_e = [thr[s, : lay[s, 0] - 1] for s in range(4)]
        thr_v = [data["thr_var"][s, : max(0, lay[s, 1] - 1)] for s in range(4)]
    strats = [strat_bp]
    tab_index: dict[str, int] = {}
    loaded: dict[str, int] = {str(Path(blueprint).resolve()): 0}  # identical paths share one array
    for name, path in tables.items():
        key = str(Path(path).resolve())
        if key not in loaded:
            s, t, d = _load_table(path)
            assert s.shape == strat_bp.shape, (name, s.shape)
            assert ("layout" in d.files) == v2 and np.allclose(t, thr), f"table {name}: thresholds differ from blueprint"
            loaded[key] = len(strats)
            strats.append(s)
        tab_index[name] = loaded[key]
    if models is None:
        names: list[str] = []
        P = blueprint_marginal(strat_bp)[None]
    else:
        z = np.load(models, allow_pickle=False)
        names = [str(x) for x in z["names"]]
        P = np.array(z["P"], dtype=np.float64)
        if len(names) == P.shape[0]:
            names = names[1:]
        assert P.shape == (len(names) + 1, N_NODES, 4), P.shape
        empty = P.sum(-1) <= 0
        P[0] = _normalize(P[0])
        for k in range(1, len(names) + 1):
            P[k][empty[k]] = P[0][empty[k]]
    K = len(names)
    logP = np.ascontiguousarray(np.log(np.maximum(P, LOG_FLOOR)).transpose(1, 2, 0))  # [N, 4, K+1]
    tab_of = [0] + [tab_index.get(n, 0) for n in names]
    if hard_names is None:
        hard_names = set(tab_index)
    hard = [False] + [n in hard_names and n in tab_index for n in names]
    h_new_half, h_choose, h_hand_end, hp = H.make_bot(styles={}, **heur_overrides)

    def new_half(config, memory: dict) -> None:
        h_new_half(config, memory)
        memory["cfr_cache"] = {}
        memory["cfr_stats"] = [0, 0]
        memory["cfr_on"] = False
        memory["cfr_switch_hand"] = None
        memory["sel_switch"] = False
        memory["net_bb"] = 0.0
        memory["sel_ll"] = np.zeros(K + 1)
        memory["sel_n"] = 0
        memory["sel_hand"] = None
        memory["sel_seen"] = 0
        memory["sel_key"] = ()
        memory["sel_last"] = None
        memory["sel_table_uses"] = 0

    # ------------------------------------------------------------------ selector

    def sel_ingest(obs, memory: dict, key: tuple) -> None:
        """Score the opponent's actions not yet seen this hand; node = tree node of the history prefix."""
        if memory["sel_hand"] != obs.hand_index:
            memory["sel_hand"] = obs.hand_index
            memory["sel_seen"] = 0
        opp = 1 if obs.is_button else 0
        hist = obs.action_history
        ll = memory["sel_ll"]
        for k in range(memory["sel_seen"], len(hist)):
            a = hist[k]
            if a.is_me:
                continue
            node = TREE.key.get(key[:k])
            if node is None or TREE.player[node] != opp:
                continue
            ll += logP[node, int(a.action)]
            memory["sel_n"] += 1
        memory["sel_seen"] = len(hist)

    def sel_tail_ingest(result, memory: dict) -> None:
        """The opponent's hand-ending action never reaches action_history: after our last action it folded
        (we won without showdown) or called/checked (showdown); a raise would have given us another decision.
        A hand in which we never acted and won is the button's open-fold at the root."""
        opp_folded = not result.went_to_showdown and result.winner == "me"
        last = memory["sel_last"]
        if last is None or last[0] != result.hand_index:
            if opp_folded and memory["sel_hand"] != result.hand_index:
                memory["sel_ll"] += logP[0, 0]
                memory["sel_n"] += 1
            return
        memory["sel_last"] = None
        node = TREE.key.get(last[1])
        if node is None or TREE.player[node] != last[2]:
            return
        memory["sel_ll"] += logP[node, 0 if opp_folded else 1]
        memory["sel_n"] += 1

    def leader(memory: dict) -> int:
        if K == 0 or memory["sel_n"] < sel_min_acts:
            return 0
        ll = memory["sel_ll"]
        k = 1 + int(np.argmax(ll[1:]))
        if ll[k] - ll[0] < sel_lead:
            return 0
        for j in range(1, K + 1):
            if j != k and ll[j] >= ll[k]:
                return 0
        return k

    # ------------------------------------------------------------------ routing

    def use_heuristic(obs, memory: dict) -> bool:
        if hybrid is None:
            return False
        if not memory["cfr_on"] and sel_early and obs.hand_index >= stop_hmin and memory["net_bb"] <= sel_max_net \
                and hard[leader(memory)]:
            memory["cfr_on"] = True
            memory["cfr_switch_hand"] = obs.hand_index
            memory["sel_switch"] = True
        return not memory["cfr_on"]

    def choose_action(obs, memory: dict) -> Action:
        """Charges the whole decision (CFR path and any heuristic fallback) to memory["think"]."""
        t0 = time.perf_counter()
        before = memory.get("think", 0.0)
        try:
            action = _choose(obs, memory)
            memory["sel_last"] = (obs.hand_index, memory["sel_key"] + (int(action),), 1 if obs.is_button else 0)
            return action
        finally:
            memory["think"] = before + H.charged(time.perf_counter() - t0, hp)

    def _choose(obs, memory: dict) -> Action:
        H.ingest_history(obs, memory)
        key = tuple(int(a.action) for a in obs.action_history)
        memory["sel_key"] = key
        sel_ingest(obs, memory, key)
        if use_heuristic(obs, memory):
            memory["cfr_stats"][1] += 1
            return h_choose(obs, memory)
        if obs.legal_actions and Action.RAISE_HALF in obs.legal_actions \
                and memory["opp"]["streak"] >= hp["dead_streak"]:
            return Action.RAISE_HALF
        node = TREE.key.get(key)
        me = 0 if obs.is_button else 1
        if node is None or TREE.player[node] != me:
            memory["cfr_stats"][1] += 1
            return h_choose(obs, memory)
        lead = leader(memory)
        ti = tab_of[lead]
        memory["sel_table_uses"] += ti != 0
        strat = strats[ti]
        s = int(obs.street)
        ck = (obs.hand_index, s)
        rng = np.random.default_rng(H.obs_seed(obs))
        if ck not in memory["cfr_cache"]:
            used = memory.get("think", 0.0)
            bank = hp["time_bank"] * memory.get("bank_scale", 1.0)
            n = cfr_feat.POST_SAMPLES if used < bank else 100 if used < bank + memory.get("bank_scale", 1.0) else 30
            feat = cfr_feat.features if v2 else cfr_feat.eq_random
            memory["cfr_cache"] = {ck: feat(obs.hole_cards, obs.board, rng, n)}
        if v2:
            eq, var, tex = memory["cfr_cache"][ck]
            b = cfr_feat.bucket2(eq, var, tex, s, thr_e[s], thr_v[s])
        else:
            b = cfr_feat.bucket(memory["cfr_cache"][ck], thr[s])
        row = strat[node, b]
        legal = [int(a) for a in obs.legal_actions]
        w = np.array([row[a] for a in legal], dtype=np.float64)
        if purify > 0 and w.sum() > 0:
            w = w / w.sum()
            kept = np.where(w >= purify, w, 0.0)
            w = kept if kept.sum() > 0 else (w == w.max()).astype(np.float64)
        if w.sum() <= 0:
            memory["cfr_stats"][1] += 1
            return h_choose(obs, memory)
        memory["cfr_stats"][0] += 1
        a = legal[int(rng.choice(len(legal), p=w / w.sum()))]
        if a in (Action.RAISE_HALF, Action.RAISE_POT):
            memory["pending"] = s
        return ACTS[a]

    def on_hand_end(result, memory: dict) -> None:
        h_hand_end(result, memory)
        if sel_tail:
            sel_tail_ingest(result, memory)
        memory["net_bb"] += result.my_net / BIG_BLIND
        h = result.hand_index + 1
        if hybrid == "stoploss" and not memory["cfr_on"] and h >= stop_hmin and memory["net_bb"] <= -stop_bb:
            memory["cfr_on"] = True
            memory["cfr_switch_hand"] = h
        if result.hand_index == memory.get("num_hands", 1000) - 1:
            ll = memory["sel_ll"]
            k = leader(memory)
            best = 1 + int(np.argmax(ll[1:])) if K else 0
            trigger = "selector" if memory["sel_switch"] else "stoploss" if memory["cfr_switch_hand"] is not None else "none"
            label = ["blueprint"] + names
            top3 = ", ".join(f"{label[j]} {ll[j] - ll[0]:+.1f}" for j in np.argsort(-ll)[:3])
            print(f"[huggy] router: net {memory['net_bb']:+.0f} bb, cfr switch at hand {memory['cfr_switch_hand']} "
                  f"({trigger}), leader {label[k]} ll-lead {ll[best] - ll[0]:.1f} n {memory['sel_n']}, "
                  f"cfr/heuristic decisions {memory['cfr_stats']}, model-table decisions {memory['sel_table_uses']}, "
                  f"top3 [{top3}]", file=sys.stderr, flush=True)

    return new_half, choose_action, on_hand_end, hp
