"""Regression checks for the team's own strategy and evaluator."""

import pickle
import random

import pytest

from card_utils import Card, evaluate, full_deck
from game_state import GameState, GameStateView
from model import Model
from poker_strength import COMBOS, card_id, rank_hand


def cards(text):
    return [Card.parse(c) for c in text.split()]


def ids(text):
    return tuple(card_id(c) for c in cards(text))


@pytest.mark.parametrize('size', [5, 6, 7])
def test_fast_evaluator_matches_organisers(size):
    rng = random.Random(8821 + size)
    deck = full_deck()
    for _ in range(2000):
        hand = rng.sample(range(52), size)
        reference = evaluate([deck[c] for c in hand])
        expected = reference[0] << 20
        for i, value in enumerate(reference[1:]):
            expected |= value << (16 - 4 * i)
        assert rank_hand(hand) == expected


@pytest.mark.parametrize('hand,category', [
    ('AS 2S 3S 4S 5S KD QH', 8),
    ('AS AC AH AD KS KC KH', 7),
    ('AS AC AH KS KC KH 2D', 6),
    ('AS KS 9S 7S 5S 3S 2S', 5),
    ('AS 2C 3D 4H 5S 6D KH', 4),
    ('2S 2C 3D 3H 4S 4D AH', 2),
])
def test_tricky_hand_categories(hand, category):
    assert rank_hand(ids(hand)) >> 20 == category


def make_model(hole):
    model = Model()
    model.seed(cards(hole), 0)
    model._begin(GameStateView(GameState([500, 500], teams=['us', 'them'])))
    return model


def test_river_nuts_is_exact():
    model = make_model('AS KS')
    equity = model._equity([1], ids('QS JS TS 2C 3D'), random.Random(4), 1)
    assert equity == 1.0


def test_royal_flush_on_board_splits_exactly():
    model = make_model('2C 3D')
    equity = model._equity([1], ids('AS KS QS JS TS'), random.Random(4), 1)
    assert equity == 0.5


def test_river_equity_respects_weighted_range_and_blockers():
    model = make_model('AC AD')
    model.ranges[1] = [float(hand == ids('KC KD')) for hand in COMBOS]
    assert model._equity([1], ids('2C 3D 7H 8S 9C'), random.Random(4), 10) == 1


def test_three_players_split_royal_flush():
    model = Model()
    model.seed(cards('2C 3D'), 0)
    model._begin(GameStateView(GameState([500]*3, teams=['us', 'a', 'b'])))
    equity = model._equity([1, 2], ids('AS KS QS JS TS'), random.Random(9), 200)
    assert equity == pytest.approx(1/3)


def test_learning_survives_pickle_and_seat_changes():
    model = make_model('AC KD')
    model.profiles['them']['pre'] = 20
    model.profiles['them']['pfr'] = 18
    restored = pickle.loads(pickle.dumps(model))
    restored.seed(cards('9C 9D'), 1)
    restored._begin(GameStateView(GameState([500]*2, teams=['them', 'us'])))
    assert restored.profiles['them']['pre'] == 20
    assert restored._style(0)[1] > 0.7


def test_never_folds_a_free_check():
    model = Model()
    model.seed(cards('2C 7D'), 1)
    state = GameState([500, 500])
    from player_action import PlayerAction
    limp = PlayerAction.call(0)
    state.process_action(limp)
    action = model.run([limp], GameStateView(state))
    assert action.kind != 'fold'
    state.validate_action(action)


def river_state(board):
    from player_action import PlayerAction
    state = GameState([500, 500], teams=['us', 'them'])
    state.setup_hand([500, 500], button=1, teams=['us', 'them'])
    incoming = iter(cards(board))
    while len(state.board) < 5:
        action = (PlayerAction.reveal_card(next(incoming)) if state.is_street_closed
                  else PlayerAction.call(state.to_act))
        state.process_action(action)
    assert state.to_act == 0
    return state


def test_limp_then_reraise_counts_once_as_preflop_raise():
    from player_action import PlayerAction
    model = Model()
    model.seed(cards('AC KD'), 1)
    state = GameState([500, 500], teams=['them', 'us'])
    actions = [PlayerAction.call(0), PlayerAction.raise_to(1, 60),
               PlayerAction.raise_to(0, 160), PlayerAction.call(1)]
    for action in actions:
        state.process_action(action)
    model._observe(actions, GameStateView(state))
    stats = model.profiles['them']
    assert stats['pre'] == stats['vpip'] == stats['pfr'] == 1


def test_river_avoids_betting_when_only_better_hands_continue():
    model = make_model('9C 9D')
    state = river_state('2C 5D 7H JS KH')
    weak = tuple(sorted(ids('3C 4C')))
    strong = tuple(sorted(ids('KC KD')))
    model.ranges[1] = [3.0 if h == weak else 1.0 if h == strong else 0.0 for h in COMBOS]
    model.profiles['them'].update(pre=100, faced=100, folds=65)
    action = model._river_action(GameStateView(state), 1, 0.75, 0.9)
    assert action.kind == 'call'


def test_river_nuts_extracts_maximum_from_observed_calling_station():
    model = make_model('AS KS')
    state = river_state('QS JS TS 2C 3D')
    model.profiles['them'].update(pre=100, faced=100, folds=0, pfr=0)
    action = model._river_action(GameStateView(state), 1, 1.0, 0.9)
    assert action.kind == 'raise'
    assert action.amount == state.max_raise_to
    state.validate_action(action)


def test_river_can_bluff_an_observed_frequent_folder():
    model = make_model('2H 7D')
    state = river_state('3C 4D 9H KC QS')
    model.profiles['them'].update(pre=300, faced=300, folds=300)
    action = model._river_action(GameStateView(state), 1, 0.10, 0.0)
    assert action.kind == 'raise'
    state.validate_action(action)
