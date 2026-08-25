from dataclasses import dataclass
from types import SimpleNamespace

from belief.public_reveal_mask import (
    from_live_belief_tracker,
    from_public_events,
    from_replay_facts,
    from_transformer_tracker,
    information_fractions,
    replay_reveal_snapshots,
)


def _move(name):
    return SimpleNamespace(id=name)


def _state(hidden_move="SECRET"):
    hidden = SimpleNamespace(
        id="HIDDENMON",
        moves=[_move(hidden_move), _move("NONE"), _move("NONE"), _move("NONE")],
        item="SECRETITEM",
        ability="SECRETABILITY",
    )
    none = SimpleNamespace(
        id="NONE",
        moves=[_move("NONE") for _ in range(4)],
        item="NONE",
        ability="NONE",
    )
    return SimpleNamespace(side_two=SimpleNamespace(pokemon=[hidden] + [none] * 5))


@dataclass
class Event:
    actor: str
    species: str
    kind: str
    move_id: str = ""
    stable_identity_evidence: bool = False


def test_hidden_completion_alone_cannot_change_mask():
    state = _state("FIRSTSECRET")
    assert from_public_events(state, ()) == 0
    state.side_two.pokemon[0].moves[0].id = "DIFFERENTSECRET"
    state.side_two.pokemon[0].item = "OTHERITEM"
    state.side_two.pokemon[0].ability = "OTHERABILITY"
    assert from_public_events(state, ()) == 0


def test_typed_history_marks_only_matching_stable_public_facts():
    events = [
        Event("opponent", "hiddenmon", "switch"),
        Event("opponent", "hiddenmon", "move", move_id="secret"),
        Event("opponent", "hiddenmon", "item", stable_identity_evidence=False),
        Event("opponent", "hiddenmon", "ability", stable_identity_evidence=True),
    ]
    fractions = information_fractions(from_public_events(_state(), events))
    assert fractions == (1 / 6, 1 / 24, 0.0, 1 / 6)


def test_live_tracker_uses_reveal_sets_not_sampled_truth():
    belief = SimpleNamespace(
        revealed_moves={"secret"},
        revealed_item=None,
        revealed_ability="secretability",
    )
    tracker = SimpleNamespace(_opponent_mons={"hiddenmon": belief})
    fractions = information_fractions(from_live_belief_tracker(_state(), tracker))
    assert fractions == (1 / 6, 1 / 24, 0.0, 1 / 6)


def test_transformer_tracker_and_engine_mask_share_one_public_contract():
    pokemon = SimpleNamespace(
        name="hiddenmon",
        moves=[SimpleNamespace(name="secret")],
        ability="unknownability",
    )
    tracker = SimpleNamespace(
        opponent_team=[SimpleNamespace(pokemon=pokemon, revealed_item="leftovers")]
    )
    fractions = information_fractions(from_transformer_tracker(_state(), tracker))
    assert fractions == (1 / 6, 1 / 24, 1 / 6, 0.0)


def test_replay_snapshots_are_start_of_turn_and_observer_relative():
    log = "\n".join(
        (
            "|player|p1|Alice|",
            "|player|p2|Bob|",
            "|switch|p1a: Hero|Hero, L80|100/100",
            "|switch|p2a: Hiddenmon|Hiddenmon, L80|100/100",
            "|-start|p2a: Hiddenmon|ability: Secret Ability",
            "|turn|1",
            "|move|p2a: Hiddenmon|Secret|p1a: Hero",
            "|-heal|p2a: Hiddenmon|100/100|[from] item: Leftovers",
            "|turn|2",
        )
    )
    snapshots = replay_reveal_snapshots(log, "Alice")
    assert snapshots[1].species == frozenset({"hiddenmon"})
    assert snapshots[1].moves == ()
    assert snapshots[1].abilities == frozenset({"hiddenmon"})
    assert snapshots[2].moves == (("hiddenmon", ("secret",)),)
    assert snapshots[2].items == frozenset({"hiddenmon"})
    assert information_fractions(from_replay_facts(_state(), snapshots[2])) == (
        1 / 6,
        1 / 24,
        1 / 6,
        1 / 6,
    )


def test_replay_snapshot_does_not_count_other_players_private_facts():
    log = "\n".join(
        (
            "|player|p1|Alice|",
            "|player|p2|Bob|",
            "|switch|p1a: Hiddenmon|Hiddenmon, L80|100/100",
            "|switch|p2a: Other|Other, L80|100/100",
            "|move|p1a: Hiddenmon|Secret|p2a: Other",
            "|turn|1",
        )
    )
    facts = replay_reveal_snapshots(log, "Alice")[1]
    assert facts.species == frozenset({"other"})
    assert facts.moves == ()
