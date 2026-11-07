"""Exact betting tree of HU Short Deck PLO5 (card-independent), built from game.py semantics.

Player 0 = button (small blind), player 1 = big blind. Actions are indexed by game.Action value.
A node is keyed by the tuple of action values taken so far in the hand.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from game import Action, Deal, Game

ACTIONS = (Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT)
_DUMMY = Deal(hole_cards=((0, 1, 2, 3, 4), (5, 6, 7, 8, 9)), board=(10, 11, 12, 13, 14))


@dataclass
class Tree:
    player: list = field(default_factory=list)    # -1 for terminal
    street: list = field(default_factory=list)
    children: list = field(default_factory=list)  # 4 child ids, -1 if illegal
    fold_by: list = field(default_factory=list)   # terminal: player who folded, -1 showdown
    stake: list = field(default_factory=list)     # terminal: chips each side risks (button view)
    key: dict = field(default_factory=dict)       # action tuple -> node id


def build() -> Tree:
    t = Tree()

    def rec(game: Game, hist: tuple) -> int:
        nid = len(t.player)
        t.key[hist] = nid
        t.player.append(-1)
        t.street.append(int(game.street))
        t.children.append([-1, -1, -1, -1])
        t.fold_by.append(-1)
        t.stake.append(0)
        if game.is_terminal:
            last = game.history[-1]
            if last.action == Action.FOLD:
                folder = 0 if last.seat == game.button_seat else 1
                t.fold_by[nid] = folder
                t.stake[nid] = game.contributed[last.seat]
            else:
                assert game.contributed[0] == game.contributed[1]
                t.stake[nid] = game.contributed[0]
            return nid
        t.player[nid] = 0 if game.to_act == game.button_seat else 1
        for a in game.legal_actions():
            child = game.clone()
            child.apply(a)
            t.children[nid][int(a)] = rec(child, hist + (int(a),))
        return nid

    rec(Game(_DUMMY, button_seat=0), ())
    return t


if __name__ == "__main__":
    import time
    t0 = time.time()
    tr = build()
    dec = sum(p >= 0 for p in tr.player)
    print(f"nodes {len(tr.player)}, decision {dec}, terminal {len(tr.player) - dec}, {time.time() - t0:.1f}s")
    for s in range(4):
        print("street", s, sum(1 for p, st in zip(tr.player, tr.street) if p >= 0 and st == s))
