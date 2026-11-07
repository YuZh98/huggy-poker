"""CFR blueprint policy: exact tree node from action history + equity bucket -> average strategy.

Falls back to the heuristic strategy (strategy.py) on any mismatch or unvisited infoset.
"""
from __future__ import annotations

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
            "heur_default" (CFR vs equity-style opponents)."""
    data = np.load(table)
    strat = data["strat"].astype(np.float32)
    thr = data["thr"]
    h_new_half, h_choose, h_hand_end, hp = strategy.make_bot(styles={}, **heur_overrides)

    def new_half(config, memory: dict) -> None:
        h_new_half(config, memory)
        memory["cfr_cache"] = {}
        memory["cfr_stats"] = [0, 0]

    def use_heuristic(memory: dict) -> bool:
        if hybrid is None:
            return False
        memory["style"] = strategy.classify_style(memory["opp"], memory.get("style"), hp)
        if hybrid == "cfr_default":
            return memory["style"] in ("pass", "aggr")
        return memory["style"] != "eq"

    def choose_action(obs, memory: dict) -> Action:
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
            memory["cfr_cache"] = {ck: cfr_feat.eq_random(obs.hole_cards, obs.board, rng)}
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
