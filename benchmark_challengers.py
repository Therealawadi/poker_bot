"""Local independent sparring; not part of the submitted bot."""
import random
import statistics
from time import perf_counter
from card_utils import full_deck, evaluate
from engine import play_hand
from model import Model
from player_action import PlayerAction


class Pressure:
    def seed(self, hand, player_id):
        self.hand, self.player_id = hand, player_id

    def run(self, actions, state):
        if state.can_raise:
            return PlayerAction.raise_to(self.player_id, min(state.max_raise_to, max(state.min_raise_to, state.current_bet + state.pot)))
        return PlayerAction.call(self.player_id)


class SmallBets(Pressure):
    def run(self, actions, state):
        if state.can_raise:
            return PlayerAction.raise_to(self.player_id, state.min_raise_to)
        return PlayerAction.call(self.player_id)


class Rock(Pressure):
    def run(self, actions, state):
        high, low = sorted([c.value for c in self.hand], reverse=True)
        if state.street == 'preflop':
            strong = (high == low and low >= 10) or (high == 14 and low == 13)
            if not strong:
                return PlayerAction.call(self.player_id) if state.call_amount == 0 else PlayerAction.fold(self.player_id)
        return PlayerAction.raise_to(self.player_id, state.max_raise_to) if state.can_raise else PlayerAction.call(self.player_id)


class Trapper(Pressure):
    def run(self, actions, state):
        if state.street == 'preflop':
            return PlayerAction.call(self.player_id)
        rank = evaluate(self.hand + list(state.board))
        connects = self.hand[0].value == self.hand[1].value or any(c.value == h.value for c in state.board for h in self.hand)
        if rank[0] >= 1 and connects:
            if state.street in ('turn', 'river') and state.can_raise:
                return PlayerAction.raise_to(self.player_id, min(state.max_raise_to, max(state.min_raise_to, state.current_bet + state.pot)))
            return PlayerAction.call(self.player_id)
        return PlayerAction.call(self.player_id) if not state.call_amount else PlayerAction.fold(self.player_id)



def main():
    for cls in (Pressure, SmallBets, Rock, Trapper):
        outcomes = [[] for _ in range(250)]
        faults = 0
        slowest = 0
        for seat in (0, 1):
            ours, theirs = Model(), cls()
            for deal in range(250):
                deck = full_deck()
                random.Random(53317 + deal).shuffle(deck)
                players = [ours, theirs] if seat == 0 else [theirs, ours]
                start = perf_counter()
                result = play_hand(players, deck, [500, 500], button=deal % 2,
                                   teams=['us', cls.__name__] if seat == 0 else [cls.__name__, 'us'])
                slowest = max(slowest, perf_counter() - start)
                faults += int(seat in result.faults)
                outcomes[deal].append(result.net[seat] / 20 * 100)
        values = [statistics.mean(x) for x in outcomes]
        mean = statistics.mean(values)
        ci = 1.96 * statistics.stdev(values) / len(values)**0.5
        print(f'{cls.__name__}: {mean:+.1f} +/- {ci:.1f} bb/100; faults={faults}; 500 hands', flush=True)


if __name__ == "__main__":
    main()
