"""CFR blueprint policy: exact tree node from action history + equity bucket -> average strategy.

Falls back to the heuristic strategy (strategy.py) on any mismatch or unvisited infoset.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import cfr_feat
import cfr_tree
import strategy
from game import Action

TREE = cfr_tree.build()
ACTS = (Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT)


def make_bot(table: str | Path = Path(__file__).resolve().parent / "tables" / "cfr.npz",
             purify: float = 0.0, hybrid: str | None = None, **heur_overrides):
    """purify: drop actions with probability < purify (>= 0.5 plays the argmax).
    hybrid: None (CFR only), "cfr_default" (heuristic vs passive/aggressive opponents),
            "heur_default" (CFR vs equity-style opponents),
            "evidence" (heuristic by default; CFR once the opponent folds to our bets often AND is not passive —
                        thresholds chosen on the training pool, see docs/PLAN.md)."""
    data = np.load(table)
    strat = data["strat"].astype(np.float32)
    thr = data["thr"]
    h_new_half, h_choose, h_hand_end, hp = strategy.make_bot(styles={}, **heur_overrides)

    def new_half(config, memory: dict) -> None:
        h_new_half(config, memory)
        memory["cfr_cache"] = {}
        memory["cfr_stats"] = [0, 0]

    def evidence_route(memory: dict) -> bool:
        """True = heuristic. Switch to CFR on fold-rate >= hi_fold and raise-rate >= hi_agg; hysteresis on exit."""
        m = memory["opp"]
        faced, acts = sum(m["faced"]), sum(m["act"])
        if faced < hp["route_min_faced"] or acts < hp["style_min_acts"]:
            return True
        fold, agg = sum(m["folded"]) / faced, sum(m["agg"]) / acts
        if memory.get("route_cfr"):
            memory["route_cfr"] = fold >= hp["route_fold"] - hp["route_hyst"] and agg >= hp["route_agg"] - hp["route_hyst"]
        else:
            memory["route_cfr"] = fold >= hp["route_fold"] and agg >= hp["route_agg"]
        return not memory["route_cfr"]

    def use_heuristic(memory: dict) -> bool:
        if hybrid is None:
            return False
        if hybrid == "evidence":
            return evidence_route(memory)
        memory["style"] = strategy.classify_style(memory["opp"], memory.get("style"), hp)
        if hybrid == "cfr_default":
            return memory["style"] in ("pass", "aggr")
        return memory["style"] != "eq"

    def choose_action(obs, memory: dict) -> Action:
        """Charges the whole decision (CFR path and any heuristic fallback) to memory["think"]."""
        t0 = time.perf_counter()
        before = memory.get("think", 0.0)
        try:
            return _choose(obs, memory)
        finally:
            memory["think"] = before + strategy.charged(time.perf_counter() - t0, hp)

    def _choose(obs, memory: dict) -> Action:
        strategy.ingest_history(obs, memory)
        if use_heuristic(memory):
            memory["cfr_stats"][1] += 1
            return h_choose(obs, memory)
        if obs.legal_actions and Action.RAISE_HALF in obs.legal_actions \
                and memory["opp"]["streak"] >= hp["dead_streak"]:
            return Action.RAISE_HALF
        node = TREE.key.get(tuple(int(a.action) for a in obs.action_history))
        me = 0 if obs.is_button else 1
        if node is None or TREE.player[node] != me:
            memory["cfr_stats"][1] += 1
            return h_choose(obs, memory)
        s = int(obs.street)
        ck = (obs.hand_index, s)
        rng = np.random.default_rng(strategy.obs_seed(obs))
        if ck not in memory["cfr_cache"]:
            used = memory.get("think", 0.0)
            bank = hp["time_bank"] * memory.get("bank_scale", 1.0)
            n = cfr_feat.POST_SAMPLES if used < bank else 100 if used < bank + memory.get("bank_scale", 1.0) else 30
            memory["cfr_cache"] = {ck: cfr_feat.eq_random(obs.hole_cards, obs.board, rng, n)}
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

    return new_half, choose_action, on_hand_end, hp
