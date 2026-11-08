"""CFR blueprint policy: exact tree node from action history + equity bucket -> average strategy.

Falls back to the heuristic strategy (strategy.py) on any mismatch or unvisited infoset.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import cfr_feat
import cfr_tree
import strategy
from game import BIG_BLIND, Action

TREE = cfr_tree.build()
ACTS = (Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT)


def make_bot(table: str | Path = Path(__file__).resolve().parent / "tables" / "cfr.npz",
             purify: float = 0.0, hybrid: str | None = None, stop_bb: float = 120.0, stop_hmin: int = 100, stop_k: float = 0.0, heur=None,
             **heur_overrides):
    """purify: drop actions with probability < purify (>= 0.5 plays the argmax).
    heur: heuristic module (default strategy.py); it must provide make_bot, classify_style, charged, ingest_history, obs_seed.
    hybrid: None (CFR only), "cfr_default" (heuristic vs passive/aggressive opponents),
            "heur_default" (CFR vs equity-style opponents),
            "evidence" (heuristic by default; CFR once the opponent folds to our bets often AND is not passive —
                        thresholds chosen on the training pool, see docs/PLAN.md),
            "delayed" (heuristic by default; from hand 400, at every 100-hand checkpoint, switch to CFR for the rest of
                       the half if the opponent's check-raise rate >= 0.09 over >= 100 opportunities AND its raise
                       rate >= 0.30 — the signature of equilibrium-type opponents; never switches back),
            "stoploss" (heuristic by default; switch to CFR for the rest of the half once our cumulative net in this
                        half is <= -stop_bb big blinds at hand >= stop_hmin; never switches back; with stop_k > 0 the
                        threshold is instead -stop_k * sd * sqrt(h), sd = running per-hand SD of our net)."""
    H = heur or strategy
    data = np.load(table)
    strat = data["strat"].astype(np.float32)
    thr = data["thr"]
    v2 = "layout" in data.files
    if v2:
        lay = data["layout"]
        thr_e = [thr[s, : lay[s, 0] - 1] for s in range(4)]
        thr_v = [data["thr_var"][s, : max(0, lay[s, 1] - 1)] for s in range(4)]
    h_new_half, h_choose, h_hand_end, hp = H.make_bot(styles={}, **heur_overrides)

    def new_half(config, memory: dict) -> None:
        h_new_half(config, memory)
        memory["cfr_cache"] = {}
        memory["cfr_stats"] = [0, 0]
        memory["xr"] = dict(n=0, y=0, pending=None, hand=None, seen=0, checked=set())
        memory["cfr_on"] = False
        memory["cfr_switch_hand"] = None
        memory["next_check"] = 400
        memory["net_bb"] = 0.0
        memory["net_sq"] = 0.0

    def track_check_raise(obs, memory: dict) -> None:
        """Opportunity = we bet/raised on a street where the opponent had checked; resolved by its next action there."""
        x = memory["xr"]
        if x["hand"] != obs.hand_index:
            x.update(hand=obs.hand_index, seen=0, checked=set(), pending=None)
        hist = obs.action_history
        for k in range(x["seen"], len(hist)):
            a = hist[k]
            if a.is_me:
                continue
            st = int(a.street)
            if x["pending"] is not None and st == x["pending"]:
                x["n"] += 1
                x["y"] += a.action in (Action.RAISE_HALF, Action.RAISE_POT)
                x["pending"] = None
            if a.action == Action.CHECK_CALL and a.amount == 0:
                x["checked"].add(st)
        x["seen"] = len(hist)

    def maybe_switch(obs, memory: dict) -> None:
        if memory["cfr_on"] or obs.hand_index < memory["next_check"]:
            return
        memory["next_check"] = (obs.hand_index // 100 + 1) * 100
        x, m = memory["xr"], memory["opp"]
        acts = sum(m["act"])
        rr = sum(m["agg"]) / acts if acts else 0.0
        if x["n"] >= 100 and x["y"] / x["n"] >= 0.09 and rr >= 0.30:
            memory["cfr_on"] = True
            memory["cfr_switch_hand"] = obs.hand_index

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
        if hybrid in ("delayed", "stoploss"):
            return not memory["cfr_on"]
        if hybrid == "evidence":
            return evidence_route(memory)
        memory["style"] = H.classify_style(memory["opp"], memory.get("style"), hp)
        if hybrid == "cfr_default":
            return memory["style"] in ("pass", "aggr")
        return memory["style"] != "eq"

    def choose_action(obs, memory: dict) -> Action:
        """Charges the whole decision (CFR path and any heuristic fallback) to memory["think"]."""
        t0 = time.perf_counter()
        before = memory.get("think", 0.0)
        try:
            action = _choose(obs, memory)
            if hybrid == "delayed" and action in (Action.RAISE_HALF, Action.RAISE_POT) \
                    and int(obs.street) in memory["xr"]["checked"]:
                memory["xr"]["pending"] = int(obs.street)
            return action
        finally:
            memory["think"] = before + H.charged(time.perf_counter() - t0, hp)

    def _choose(obs, memory: dict) -> Action:
        H.ingest_history(obs, memory)
        if hybrid == "delayed":
            track_check_raise(obs, memory)
            maybe_switch(obs, memory)
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
        if hybrid == "stoploss":
            x = result.my_net / BIG_BLIND
            memory["net_bb"] += x
            memory["net_sq"] += x * x
            h = result.hand_index + 1
            if stop_k > 0:
                sd = max(0.0, memory["net_sq"] / h - (memory["net_bb"] / h) ** 2) ** 0.5
                limit = stop_k * sd * h ** 0.5
            else:
                limit = stop_bb
            if not memory["cfr_on"] and h >= stop_hmin and memory["net_bb"] <= -limit:
                memory["cfr_on"] = True
                memory["cfr_switch_hand"] = result.hand_index + 1
            if result.hand_index == memory.get("num_hands", 1000) - 1:
                print(f"[huggy] router: net {memory['net_bb']:+.0f} bb, cfr switch at hand {memory['cfr_switch_hand']}, "
                      f"cfr/heuristic decisions {memory['cfr_stats']}", file=sys.stderr, flush=True)
        if hybrid == "delayed":
            x = memory["xr"]
            if x["hand"] == result.hand_index and x["pending"] is not None:
                x["n"] += 1
                x["pending"] = None
            if result.hand_index == memory.get("num_hands", 1000) - 1:
                print(f"[huggy] router: check-raise {x['y']}/{x['n']}, cfr switch at hand {memory['cfr_switch_hand']}, "
                      f"cfr/heuristic decisions {memory['cfr_stats']}", file=sys.stderr, flush=True)

    return new_half, choose_action, on_hand_end, hp
