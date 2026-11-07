"""Offline comparison against the preserved previous strategy; not a bot dependency."""
import argparse
import json
import random
import statistics
from pathlib import Path

from arena.dummies import DUMMIES
from benchmark_challengers import Pressure, SmallBets, Rock, Trapper
from card_utils import full_deck
from engine import play_hand
from poker_previous_model import Model as Old
from model import Model as New


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deals', type=int, default=100)
    parser.add_argument('--seed', type=int, default=7193)
    parser.add_argument('--out', default='strategy_comparison.json')
    parser.add_argument('--opponents', nargs='+')
    args = parser.parse_args()
    choices = dict(DUMMIES)
    choices.update(pressure=Pressure, smallbets=SmallBets, rock=Rock, trap=Trapper, previous=Old)
    results = {}
    for name, cls in choices.items():
        if args.opponents and name not in args.opponents:
            continue
        samples = []
        for version in (Old, New):
            paired = [[] for _ in range(args.deals)]
            faults = []
            for seat in (0, 1):
                ours = version()
                theirs = cls(seed=args.seed) if name in DUMMIES else cls()
                for deal in range(args.deals):
                    deck = full_deck()
                    random.Random(args.seed + deal).shuffle(deck)
                    players = [ours, theirs] if seat == 0 else [theirs, ours]
                    teams = ['us', 'opponent'] if seat == 0 else ['opponent', 'us']
                    result = play_hand(players, deck, [500, 500], button=deal % 2, teams=teams)
                    if seat in result.faults:
                        faults.append(result.faults[seat])
                    paired[deal].append(result.net[seat] * 5)
            samples.append([statistics.mean(x) for x in paired])
            assert not faults, (name, faults[:3])
        changes = [new - old for old, new in zip(*samples)]
        results[name] = dict(old=statistics.mean(samples[0]), new=statistics.mean(samples[1]),
                             improvement=statistics.mean(changes),
                             ci95=1.96 * statistics.stdev(changes) / len(changes)**0.5,
                             hands_per_version=2*args.deals, faults=0)
        print(name, json.dumps(results[name]), flush=True)
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
