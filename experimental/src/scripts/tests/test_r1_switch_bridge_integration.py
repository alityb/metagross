from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "experimental" / "src"))
sys.path.insert(
    0,
    str(
        ROOT
        / "experimental/engine/pe_v3_learned_priors/poke-engine-py/python"
    ),
)

try:
    import metamon.rl.pretrained as pretrained
    from metamon.env.metamon_battle import MetamonBackendBattle
    from metamon.interface import (
        UniversalAction,
        UniversalState,
        consistent_move_order,
        consistent_pokemon_order,
    )
    import poke_engine
    from poke_engine import Move as EngineMove
    from poke_engine import Pokemon as EnginePokemon
    from poke_engine import Side as EngineSide
    from poke_engine import State as EngineState
except ModuleNotFoundError as exc:  # pragma: no cover - system Python lacks Metamon
    raise unittest.SkipTest("requires the repository Metamon environment") from exc

from scripts.prior_server import (  # noqa: E402
    fresh_observation_space,
    observation_history,
    player_information_state,
)
from scripts.r1_public_events import (  # noqa: E402
    BOOSTED_DOUBLE_SWITCH_CERTIFICATE,
    DECLARATIVE_BOOST_CERTIFICATE,
    LEFTOVERS_ACTIVATION_CERTIFICATE,
    MIXED_BOOST_SWITCH_CERTIFICATE,
    PublicEventProjectionError,
    PublicSwitchEvent,
    R1SwitchTracker,
    SILENT_MECHANICS_CERTIFICATE,
    project_information_set_basic_move,
    project_information_set_switch,
)
from scripts.verify_r1_policy_snapshots import infer_snapshots  # noqa: E402


PLAYER_MOVES = {
    "charmander": ["seismictoss", "recover", "swordsdance", "thunderwave"],
    "bulbasaur": ["seismictoss", "recover", "swordsdance", "thunderwave"],
}


def request(active: str, *, active_hp: int = 100, seismic_pp: int = 32) -> dict:
    pokemon = []
    for name in ("charmander", "bulbasaur"):
        pokemon.append(
            {
                "ident": f"p1: {name.title()}",
                "details": f"{name.title()}, L80",
                "condition": f"{active_hp if name == active else 100}/100",
                "active": name == active,
                "moves": PLAYER_MOVES[name],
                "baseAbility": "noability",
                "item": "",
                "teraType": "Normal",
            }
        )
    active_moves = [
        {
            "move": move,
            "id": move,
            "pp": seismic_pp if move == "seismictoss" else 16,
            "maxpp": 32 if move == "seismictoss" else 16,
            "target": "normal",
            "disabled": False,
        }
        for move in PLAYER_MOVES[active]
    ]
    return {
        "active": [{"moves": active_moves, "canTerastallize": "Normal"}],
        "side": {"name": "Alice", "id": "p1", "pokemon": pokemon},
        "rqid": 1,
    }


def silent_request(*, calm_mind_pp: int = 32) -> dict:
    moves = ["calmmind", "recover", "tackle", "thunderwave"]
    max_pp = {"calmmind": 32, "recover": 16, "tackle": 56, "thunderwave": 32}
    return {
        "active": [
            {
                "moves": [
                    {
                        "move": move,
                        "id": move,
                        "pp": calm_mind_pp if move == "calmmind" else max_pp[move],
                        "maxpp": max_pp[move],
                        "target": "self" if move in {"calmmind", "recover"} else "normal",
                        "disabled": False,
                    }
                    for move in moves
                ],
                "canTerastallize": "Normal",
            }
        ],
        "side": {
            "name": "Alice",
            "id": "p1",
            "pokemon": [
                {
                    "ident": "p1: Charmander",
                    "details": "Charmander, L50",
                    "condition": "100/100",
                    "active": True,
                    "moves": moves,
                    "baseAbility": "protosynthesis",
                    "item": "leftovers",
                    "teraType": "Normal",
                }
            ],
        },
        "rqid": 1,
    }


def silent_request_with_reserve(
    *, calm_mind_pp: int = 32, trapped: bool = False
) -> dict:
    payload = silent_request(calm_mind_pp=calm_mind_pp)
    payload["active"][0]["trapped"] = trapped
    payload["side"]["pokemon"].append(
        {
            "ident": "p1: Bulbasaur",
            "details": "Bulbasaur, L50",
            "condition": "100/100",
            "active": False,
            "moves": ["gigadrain", "protect", "sludgebomb", "synthesis"],
            "baseAbility": "overgrow",
            "item": "leftovers",
            "teraType": "Grass",
        }
    )
    return payload


def bulbasaur_request_with_reserve(*, trapped: bool = False) -> dict:
    moves = ["gigadrain", "protect", "sludgebomb", "synthesis"]
    max_pp = {"gigadrain": 16, "protect": 16, "sludgebomb": 16, "synthesis": 8}
    return {
        "active": [
            {
                "moves": [
                    {
                        "move": move,
                        "id": move,
                        "pp": max_pp[move],
                        "maxpp": max_pp[move],
                        "target": "self" if move in {"protect", "synthesis"} else "normal",
                        "disabled": False,
                    }
                    for move in moves
                ],
                "canTerastallize": "Grass",
                "trapped": trapped,
            }
        ],
        "side": {
            "name": "Alice",
            "id": "p1",
            "pokemon": [
                {
                    "ident": "p1: Charmander",
                    "details": "Charmander, L50",
                    "condition": "100/100",
                    "active": False,
                    "moves": ["calmmind", "recover", "tackle", "thunderwave"],
                    "baseAbility": "protosynthesis",
                    "item": "leftovers",
                    "teraType": "Normal",
                },
                {
                    "ident": "p1: Bulbasaur",
                    "details": "Bulbasaur, L50",
                    "condition": "100/100",
                    "active": True,
                    "moves": moves,
                    "baseAbility": "overgrow",
                    "item": "leftovers",
                    "teraType": "Grass",
                },
            ],
        },
        "rqid": 2,
    }


def feed(battle, lines):
    for line in lines:
        battle.parse_message(line.split("|"))


def mask_and_names(battle, state):
    illegal = np.ones(13, dtype=bool)
    for action in UniversalAction.definitely_valid_actions(state, battle):
        illegal[action.action_idx] = False
    names = {}
    moves = consistent_move_order(list(battle.active_pokemon.moves.values()))[:4]
    switches = consistent_pokemon_order(
        [pokemon for pokemon in battle.team.values() if not pokemon.fainted and not pokemon.active]
    )[:5]
    for index, move in enumerate(moves):
        names[move.id] = index
        names[f"{move.id}-tera"] = index + 9
    for index, pokemon in enumerate(switches):
        normalized = "".join(character for character in pokemon.name.lower() if character.isalnum())
        names[f"switch {normalized}"] = index + 4
    return illegal.tolist(), names


class R1SwitchBridgeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = pretrained.LocalFinetunedModel(
            base_model=pretrained.Kakuna,
            amago_ckpt_dir=str(ROOT / "srcs" / "models"),
            model_name="randbats_exit_r1",
            default_checkpoint=5,
        )

    def test_projected_switch_matches_showdown_protocol_and_checkpoint(self):
        battle = MetamonBackendBattle(
            "battle-gen9randombattle-switchtest",
            "Alice",
            logging.getLogger("r1-switch-integration"),
            save_replays=False,
            gen=9,
        )
        feed(
            battle,
            [
                "|player|p1|Alice",
                "|player|p2|Bob",
                "|gen|9",
                "|tier|[Gen 9] Random Battle",
                "|poke|p1|Charmander, L80",
                "|poke|p1|Bulbasaur, L80",
                "|poke|p2|Squirtle, L80",
                "|poke|p2|Pikachu, L80",
                "|start",
                "|switch|p1a: Charmander|Charmander, L80|100/100",
                "|switch|p2a: Squirtle|Squirtle, L80|100/100",
                "|turn|1",
            ],
        )
        battle.parse_request(request("charmander"))
        observation_space = fresh_observation_space(self.model.observation_space)
        source_state = UniversalState.from_Battle(battle)
        source_observation = observation_space.state_to_obs(source_state)
        source_history = observation_history(observation_space)
        source_snapshot = {
            "text_tokens": source_observation["text_tokens"].tolist(),
            "numbers": source_observation["numbers"].tolist(),
            "player_information_state": player_information_state(battle, source_state),
            "player_observation_history": source_history,
            "continuation_observation_history": source_history,
        }
        tracker = R1SwitchTracker.from_snapshot(source_snapshot, self.model.observation_space)
        projected = tracker.apply_switch_events(
            [
                PublicSwitchEvent("self", "bulbasaur", 80, 1.0, "nostatus", True),
                PublicSwitchEvent("opponent", "pikachu", 80, 1.0, "nostatus", False),
            ]
        )

        feed(
            battle,
            [
                "|switch|p1a: Bulbasaur|Bulbasaur, L80|100/100",
                "|switch|p2a: Pikachu|Pikachu, L80|100/100",
                "|turn|2",
            ],
        )
        battle.parse_request(request("bulbasaur"))
        target_state = UniversalState.from_Battle(battle)
        target_observation = observation_space.state_to_obs(target_state)
        target_illegal, target_names = mask_and_names(battle, target_state)

        self.assertEqual(projected["text_tokens"], target_observation["text_tokens"].tolist())
        self.assertTrue(
            np.array_equal(
                np.asarray(projected["numbers"], dtype=np.float32),
                np.asarray(target_observation["numbers"], dtype=np.float32),
            ),
            [
                (index, projected_value, target_value)
                for index, (projected_value, target_value) in enumerate(
                    zip(projected["numbers"], target_observation["numbers"].tolist())
                )
                if projected_value != target_value
            ],
        )
        self.assertEqual(projected["illegal_actions"], target_illegal)
        self.assertEqual(projected["name_table"], target_names)

        experiment = self.model.initialize_agent(checkpoint=5, log=False)
        agent = experiment.policy
        agent.eval()
        device = next(agent.parameters()).device
        projected_snapshot = {**projected, "probs": [0.0] * 13}
        target_snapshot = {
            "text_tokens": target_observation["text_tokens"].tolist(),
            "numbers": target_observation["numbers"].tolist(),
            "illegal_actions": target_illegal,
            "probs": [0.0] * 13,
        }
        probabilities = infer_snapshots(
            agent, [projected_snapshot, target_snapshot], device
        )
        self.assertTrue(np.array_equal(probabilities[0], probabilities[1]))

    def test_basic_move_projection_matches_protocol_request_and_checkpoint(self):
        battle = MetamonBackendBattle(
            "battle-gen9randombattle-movetest",
            "Alice",
            logging.getLogger("r1-move-integration"),
            save_replays=False,
            gen=9,
        )
        feed(
            battle,
            [
                "|player|p1|Alice",
                "|player|p2|Bob",
                "|gen|9",
                "|tier|[Gen 9] Random Battle",
                "|poke|p1|Charmander, L50",
                "|poke|p1|Bulbasaur, L50",
                "|poke|p2|Squirtle, L50",
                "|poke|p2|Pikachu, L50",
                "|start",
                "|switch|p1a: Charmander|Charmander, L50|100/100",
                "|switch|p2a: Squirtle|Squirtle, L50|100/100",
                "|turn|1",
            ],
        )
        battle.parse_request(request("charmander"))
        observation_space = fresh_observation_space(self.model.observation_space)
        source_state = UniversalState.from_Battle(battle)
        source_observation = observation_space.state_to_obs(source_state)
        source_history = observation_history(observation_space)
        source_snapshot = {
            "text_tokens": source_observation["text_tokens"].tolist(),
            "numbers": source_observation["numbers"].tolist(),
            "player_information_state": player_information_state(battle, source_state),
            "player_observation_history": source_history,
            "continuation_observation_history": source_history,
        }
        tracker = R1SwitchTracker.from_snapshot(source_snapshot, self.model.observation_space)

        def engine_pokemon(name, speed):
            return EnginePokemon(
                id=name,
                level=50,
                hp=100,
                maxhp=100,
                speed=speed,
                ability="none",
                item="none",
                moves=[
                    EngineMove(id="seismictoss", pp=32),
                    EngineMove(id="recover"),
                    EngineMove(id="swordsdance"),
                    EngineMove(id="thunderwave"),
                ],
            )

        mechanical = EngineState(
            side_one=EngineSide(
                pokemon=[
                    engine_pokemon("charmander", 200),
                    engine_pokemon("bulbasaur", 100),
                ]
            ),
            side_two=EngineSide(pokemon=[engine_pokemon("squirtle", 50)]),
        )
        projection = project_information_set_basic_move(
            poke_engine,
            [mechanical],
            "seismictoss",
            "seismictoss",
            0.5,
        )
        self.assertEqual(len(projection.observation_classes), 1)
        projected = tracker.apply_basic_move_class(projection.observation_classes[0])

        feed(
            battle,
            [
                "|move|p1a: Charmander|Seismic Toss|p2a: Squirtle",
                "|-damage|p2a: Squirtle|50/100",
                "|move|p2a: Squirtle|Seismic Toss|p1a: Charmander",
                "|-damage|p1a: Charmander|50/100",
                "|turn|2",
            ],
        )
        battle.parse_request(
            request("charmander", active_hp=50, seismic_pp=31)
        )
        target_state = UniversalState.from_Battle(battle)
        target_observation = observation_space.state_to_obs(target_state)
        target_illegal, target_names = mask_and_names(battle, target_state)

        self.assertEqual(projected["text_tokens"], target_observation["text_tokens"].tolist())
        self.assertTrue(
            np.array_equal(
                np.asarray(projected["numbers"], dtype=np.float32),
                np.asarray(target_observation["numbers"], dtype=np.float32),
            ),
            [
                (index, projected_value, target_value)
                for index, (projected_value, target_value) in enumerate(
                    zip(projected["numbers"], target_observation["numbers"].tolist())
                )
                if projected_value != target_value
            ],
        )
        self.assertEqual(projected["illegal_actions"], target_illegal)
        self.assertEqual(projected["name_table"], target_names)

        experiment = self.model.initialize_agent(checkpoint=5, log=False)
        agent = experiment.policy
        agent.eval()
        device = next(agent.parameters()).device
        projected_snapshot = {**projected, "probs": [0.0] * 13}
        target_snapshot = {
            "text_tokens": target_observation["text_tokens"].tolist(),
            "numbers": target_observation["numbers"].tolist(),
            "illegal_actions": target_illegal,
            "probs": [0.0] * 13,
        }
        probabilities = infer_snapshots(
            agent, [projected_snapshot, target_snapshot], device
        )
        self.assertTrue(np.array_equal(probabilities[0], probabilities[1]))

    def test_silent_item_ability_projection_matches_protocol_and_checkpoint(self):
        battle = MetamonBackendBattle(
            "battle-gen9randombattle-silentmechanicstest",
            "Alice",
            logging.getLogger("r1-silent-mechanics-integration"),
            save_replays=False,
            gen=9,
        )
        feed(
            battle,
            [
                "|player|p1|Alice",
                "|player|p2|Bob",
                "|gen|9",
                "|tier|[Gen 9] Random Battle",
                "|poke|p1|Charmander, L50",
                "|poke|p2|Squirtle, L50",
                "|start",
                "|switch|p1a: Charmander|Charmander, L50|100/100",
                "|switch|p2a: Squirtle|Squirtle, L50|100/100",
                "|-boost|p1a: Charmander|spa|1",
                "|-boost|p1a: Charmander|spd|1",
                "|turn|1",
            ],
        )
        battle.parse_request(silent_request())
        observation_space = fresh_observation_space(self.model.observation_space)
        source_state = UniversalState.from_Battle(battle)
        source_observation = observation_space.state_to_obs(source_state)
        source_history = observation_history(observation_space)
        source_snapshot = {
            "text_tokens": source_observation["text_tokens"].tolist(),
            "numbers": source_observation["numbers"].tolist(),
            "player_information_state": player_information_state(battle, source_state),
            "player_observation_history": source_history,
            "continuation_observation_history": source_history,
        }
        tracker = R1SwitchTracker.from_snapshot(
            source_snapshot, self.model.observation_space
        )

        def engine_pokemon(name, ability, move, speed):
            return EnginePokemon(
                id=name,
                level=50,
                hp=100,
                maxhp=100,
                speed=speed,
                ability=ability,
                item="leftovers",
                moves=[
                    EngineMove(id=move, pp=32),
                    EngineMove(id="recover", pp=16),
                    EngineMove(id="tackle", pp=56),
                    EngineMove(id="thunderwave", pp=32),
                ],
            )

        mechanical = EngineState(
            side_one=EngineSide(
                pokemon=[
                    engine_pokemon(
                        "charmander", "protosynthesis", "calmmind", 200
                    )
                ],
                special_attack_boost=1,
                special_defense_boost=1,
            ),
            side_two=EngineSide(
                pokemon=[engine_pokemon("squirtle", "sapsipper", "bulkup", 50)]
            ),
        )
        projection = project_information_set_basic_move(
            poke_engine,
            [mechanical],
            "calmmind",
            "bulkup",
            0.5,
        )
        self.assertEqual(projection.certificate, SILENT_MECHANICS_CERTIFICATE)
        projected = tracker.apply_basic_move_class(projection.observation_classes[0])

        feed(
            battle,
            [
                "|move|p1a: Charmander|Calm Mind|p1a: Charmander",
                "|-boost|p1a: Charmander|spa|1",
                "|-boost|p1a: Charmander|spd|1",
                "|move|p2a: Squirtle|Bulk Up|p2a: Squirtle",
                "|-boost|p2a: Squirtle|atk|1",
                "|-boost|p2a: Squirtle|def|1",
                "|turn|2",
            ],
        )
        battle.parse_request(silent_request(calm_mind_pp=31))
        target_state = UniversalState.from_Battle(battle)
        target_observation = observation_space.state_to_obs(target_state)
        target_illegal, target_names = mask_and_names(battle, target_state)

        self.assertEqual(projected["text_tokens"], target_observation["text_tokens"].tolist())
        self.assertTrue(
            np.array_equal(
                np.asarray(projected["numbers"], dtype=np.float32),
                np.asarray(target_observation["numbers"], dtype=np.float32),
            ),
            [
                (index, projected_value, target_value)
                for index, (projected_value, target_value) in enumerate(
                    zip(projected["numbers"], target_observation["numbers"].tolist())
                )
                if projected_value != target_value
            ],
        )
        self.assertEqual(projected["illegal_actions"], target_illegal)
        self.assertEqual(projected["name_table"], target_names)
        self.assertEqual(target_state.opponent_active_pokemon.item, "unknownitem")
        self.assertEqual(target_state.opponent_active_pokemon.ability, "unknownability")

        experiment = self.model.initialize_agent(checkpoint=5, log=False)
        agent = experiment.policy
        agent.eval()
        device = next(agent.parameters()).device
        probabilities = infer_snapshots(
            agent,
            [
                {**projected, "probs": [0.0] * 13},
                {
                    "text_tokens": target_observation["text_tokens"].tolist(),
                    "numbers": target_observation["numbers"].tolist(),
                    "illegal_actions": target_illegal,
                    "probs": [0.0] * 13,
                },
            ],
            device,
        )
        self.assertTrue(np.array_equal(probabilities[0], probabilities[1]))

    def test_leftovers_activation_matches_protocol_and_checkpoint(self):
        battle = MetamonBackendBattle(
            "battle-gen9randombattle-leftoverstest",
            "Alice",
            logging.getLogger("r1-leftovers-integration"),
            save_replays=False,
            gen=9,
        )
        feed(
            battle,
            [
                "|player|p1|Alice",
                "|player|p2|Bob",
                "|gen|9",
                "|tier|[Gen 9] Random Battle",
                "|poke|p1|Charmander, L50",
                "|poke|p2|Squirtle, L50",
                "|start",
                "|switch|p1a: Charmander|Charmander, L50|100/100",
                "|switch|p2a: Squirtle|Squirtle, L50|50/100",
                "|turn|1",
            ],
        )
        battle.parse_request(silent_request())
        observation_space = fresh_observation_space(self.model.observation_space)
        source_state = UniversalState.from_Battle(battle)
        source_observation = observation_space.state_to_obs(source_state)
        source_history = observation_history(observation_space)
        source_snapshot = {
            "text_tokens": source_observation["text_tokens"].tolist(),
            "numbers": source_observation["numbers"].tolist(),
            "player_information_state": player_information_state(battle, source_state),
            "player_observation_history": source_history,
            "continuation_observation_history": source_history,
        }
        tracker = R1SwitchTracker.from_snapshot(
            source_snapshot, self.model.observation_space
        )

        def engine_pokemon(name, ability, move, speed, hp):
            return EnginePokemon(
                id=name,
                level=50,
                hp=hp,
                maxhp=100,
                speed=speed,
                ability=ability,
                item="leftovers",
                moves=[
                    EngineMove(id=move, pp=32),
                    EngineMove(id="recover", pp=16),
                    EngineMove(id="tackle", pp=56),
                    EngineMove(id="thunderwave", pp=32),
                ],
            )

        mechanical = EngineState(
            side_one=EngineSide(
                pokemon=[
                    engine_pokemon(
                        "charmander", "protosynthesis", "calmmind", 200, 100
                    )
                ]
            ),
            side_two=EngineSide(
                pokemon=[
                    engine_pokemon("squirtle", "sapsipper", "bulkup", 50, 50)
                ]
            ),
        )
        projection = project_information_set_basic_move(
            poke_engine,
            [mechanical],
            "calmmind",
            "bulkup",
            0.5,
        )
        self.assertEqual(
            projection.certificate, LEFTOVERS_ACTIVATION_CERTIFICATE
        )
        projected = tracker.apply_basic_move_class(projection.observation_classes[0])

        feed(
            battle,
            [
                "|move|p1a: Charmander|Calm Mind|p1a: Charmander",
                "|-boost|p1a: Charmander|spa|1",
                "|-boost|p1a: Charmander|spd|1",
                "|move|p2a: Squirtle|Bulk Up|p2a: Squirtle",
                "|-boost|p2a: Squirtle|atk|1",
                "|-boost|p2a: Squirtle|def|1",
                "|-heal|p2a: Squirtle|56/100|[from] item: Leftovers",
                "|turn|2",
            ],
        )
        battle.parse_request(silent_request(calm_mind_pp=31))
        target_state = UniversalState.from_Battle(battle)
        target_observation = observation_space.state_to_obs(target_state)
        target_illegal, target_names = mask_and_names(battle, target_state)

        self.assertEqual(projected["text_tokens"], target_observation["text_tokens"].tolist())
        self.assertTrue(
            np.array_equal(
                np.asarray(projected["numbers"], dtype=np.float32),
                np.asarray(target_observation["numbers"], dtype=np.float32),
            )
        )
        self.assertEqual(projected["illegal_actions"], target_illegal)
        self.assertEqual(projected["name_table"], target_names)
        self.assertEqual(target_state.opponent_active_pokemon.item, "leftovers")
        self.assertEqual(target_state.opponent_active_pokemon.ability, "unknownability")
        self.assertEqual(
            tracker.public_opponent_registry()["squirtle"]["revealed_item"],
            "leftovers",
        )

        experiment = self.model.initialize_agent(checkpoint=5, log=False)
        agent = experiment.policy
        agent.eval()
        device = next(agent.parameters()).device
        probabilities = infer_snapshots(
            agent,
            [
                {**projected, "probs": [0.0] * 13},
                {
                    "text_tokens": target_observation["text_tokens"].tolist(),
                    "numbers": target_observation["numbers"].tolist(),
                    "illegal_actions": target_illegal,
                    "probs": [0.0] * 13,
                },
            ],
            device,
        )
        self.assertTrue(np.array_equal(probabilities[0], probabilities[1]))

    def test_declarative_boost_projection_matches_protocol(self):
        battle = MetamonBackendBattle(
            "battle-gen9randombattle-declarativeboosttest",
            "Alice",
            logging.getLogger("r1-declarative-boost-integration"),
            save_replays=False,
            gen=9,
        )
        feed(
            battle,
            [
                "|player|p1|Alice",
                "|player|p2|Bob",
                "|gen|9",
                "|tier|[Gen 9] Random Battle",
                "|poke|p1|Charmander, L50",
                "|poke|p2|Squirtle, L50",
                "|start",
                "|switch|p1a: Charmander|Charmander, L50|100/100",
                "|switch|p2a: Squirtle|Squirtle, L50|100/100",
                "|turn|1",
            ],
        )
        battle.parse_request(silent_request())
        observation_space = fresh_observation_space(self.model.observation_space)
        source_state = UniversalState.from_Battle(battle)
        source_observation = observation_space.state_to_obs(source_state)
        source_history = observation_history(observation_space)
        tracker = R1SwitchTracker.from_snapshot(
            {
                "text_tokens": source_observation["text_tokens"].tolist(),
                "numbers": source_observation["numbers"].tolist(),
                "player_information_state": player_information_state(
                    battle, source_state
                ),
                "player_observation_history": source_history,
                "continuation_observation_history": source_history,
            },
            self.model.observation_space,
        )

        def engine_pokemon(name, ability, item, move, speed):
            return EnginePokemon(
                id=name,
                level=50,
                hp=100,
                maxhp=100,
                speed=speed,
                ability=ability,
                item=item,
                moves=[
                    EngineMove(id=move, pp=32),
                    EngineMove(id="recover", pp=16),
                    EngineMove(id="tackle", pp=56),
                    EngineMove(id="thunderwave", pp=32),
                ],
            )

        mechanical = EngineState(
            side_one=EngineSide(
                pokemon=[
                    engine_pokemon(
                        "charmander",
                        "protosynthesis",
                        "leftovers",
                        "calmmind",
                        200,
                    )
                ]
            ),
            side_two=EngineSide(
                pokemon=[
                    engine_pokemon(
                        "squirtle", "technician", "widelens", "nastyplot", 50
                    )
                ]
            ),
        )
        projection = project_information_set_basic_move(
            poke_engine, [mechanical], "calmmind", "nastyplot", 0.5
        )
        self.assertEqual(projection.certificate, DECLARATIVE_BOOST_CERTIFICATE)
        projected = tracker.apply_basic_move_class(projection.observation_classes[0])

        feed(
            battle,
            [
                "|move|p1a: Charmander|Calm Mind|p1a: Charmander",
                "|-boost|p1a: Charmander|spa|1",
                "|-boost|p1a: Charmander|spd|1",
                "|move|p2a: Squirtle|Nasty Plot|p2a: Squirtle",
                "|-boost|p2a: Squirtle|spa|2",
                "|turn|2",
            ],
        )
        battle.parse_request(silent_request(calm_mind_pp=31))
        target_state = UniversalState.from_Battle(battle)
        target_observation = observation_space.state_to_obs(target_state)
        target_illegal, target_names = mask_and_names(battle, target_state)

        self.assertEqual(projected["text_tokens"], target_observation["text_tokens"].tolist())
        self.assertTrue(
            np.array_equal(
                np.asarray(projected["numbers"], dtype=np.float32),
                np.asarray(target_observation["numbers"], dtype=np.float32),
            )
        )
        self.assertEqual(projected["illegal_actions"], target_illegal)
        self.assertEqual(projected["name_table"], target_names)
        self.assertEqual(target_state.opponent_active_pokemon.item, "unknownitem")
        self.assertEqual(target_state.opponent_active_pokemon.ability, "unknownability")

    def test_mixed_boost_switch_projects_certified_trapping_legality(self):
        battle = MetamonBackendBattle(
            "battle-gen9randombattle-mixedboostswitchtest",
            "Alice",
            logging.getLogger("r1-mixed-boost-switch-integration"),
            save_replays=False,
            gen=9,
        )
        feed(
            battle,
            [
                "|player|p1|Alice",
                "|player|p2|Bob",
                "|gen|9",
                "|tier|[Gen 9] Random Battle",
                "|poke|p1|Charmander, L50",
                "|poke|p1|Bulbasaur, L50",
                "|poke|p2|Squirtle, L50",
                "|poke|p2|Gothitelle, L50",
                "|start",
                "|switch|p1a: Charmander|Charmander, L50|100/100",
                "|switch|p2a: Gothitelle|Gothitelle, L50|80/100",
                "|switch|p2a: Squirtle|Squirtle, L50|100/100",
                "|turn|1",
            ],
        )
        battle.parse_request(silent_request_with_reserve())
        observation_space = fresh_observation_space(self.model.observation_space)
        source_state = UniversalState.from_Battle(battle)
        source_observation = observation_space.state_to_obs(source_state)
        source_history = observation_history(observation_space)
        tracker = R1SwitchTracker.from_snapshot(
            {
                "text_tokens": source_observation["text_tokens"].tolist(),
                "numbers": source_observation["numbers"].tolist(),
                "player_information_state": player_information_state(
                    battle, source_state
                ),
                "player_observation_history": source_history,
                "continuation_observation_history": source_history,
            },
            self.model.observation_space,
        )

        mechanical = EngineState(
            side_one=EngineSide(
                pokemon=[
                    EnginePokemon(
                        id="charmander",
                        level=50,
                        hp=100,
                        maxhp=100,
                        speed=200,
                        ability="protosynthesis",
                        item="leftovers",
                        moves=[
                            EngineMove(id="calmmind", pp=32),
                            EngineMove(id="recover", pp=16),
                            EngineMove(id="tackle", pp=56),
                            EngineMove(id="thunderwave", pp=32),
                        ],
                    ),
                    EnginePokemon(
                        id="bulbasaur",
                        level=50,
                        hp=100,
                        maxhp=100,
                        moves=[
                            EngineMove(id="gigadrain"),
                            EngineMove(id="protect"),
                            EngineMove(id="sludgebomb"),
                            EngineMove(id="synthesis"),
                        ],
                    ),
                ]
            ),
            side_two=EngineSide(
                pokemon=[
                    EnginePokemon(
                        id="squirtle",
                        level=50,
                        hp=100,
                        maxhp=100,
                        moves=[
                            EngineMove(id="tackle"),
                            EngineMove(id="protect"),
                            EngineMove(id="rest"),
                            EngineMove(id="sleeptalk"),
                        ],
                    ),
                    EnginePokemon(
                        id="gothitelle",
                        level=50,
                        hp=80,
                        maxhp=100,
                        ability="shadowtag",
                        moves=[
                            EngineMove(id="psychic"),
                            EngineMove(id="protect"),
                            EngineMove(id="rest"),
                            EngineMove(id="sleeptalk"),
                        ],
                    ),
                ]
            ),
        )
        projection = project_information_set_basic_move(
            poke_engine,
            [mechanical],
            "calmmind",
            "switch gothitelle",
            0.5,
            public_opponent=tracker.public_opponent_registry(),
        )
        self.assertEqual(projection.certificate, MIXED_BOOST_SWITCH_CERTIFICATE)
        side_two_projection = project_information_set_basic_move(
            poke_engine,
            [mechanical],
            "calmmind",
            "switch gothitelle",
            0.5,
            observer_side="SideTwo",
            public_opponent={},
        )
        self.assertEqual(
            side_two_projection.certificate, MIXED_BOOST_SWITCH_CERTIFICATE
        )
        self.assertTrue(
            any(
                isinstance(event, PublicSwitchEvent) and event.actor == "self"
                for event in side_two_projection.observation_classes[0].events
            )
        )
        projected = tracker.apply_basic_move_class(projection.observation_classes[0])

        feed(
            battle,
            [
                "|switch|p2a: Gothitelle|Gothitelle, L50|80/100",
                "|move|p1a: Charmander|Calm Mind|p1a: Charmander",
                "|-boost|p1a: Charmander|spa|1",
                "|-boost|p1a: Charmander|spd|1",
                "|turn|2",
            ],
        )
        battle.parse_request(
            silent_request_with_reserve(calm_mind_pp=31, trapped=True)
        )
        target_state = UniversalState.from_Battle(battle)
        target_observation = observation_space.state_to_obs(target_state)
        target_illegal, target_names = mask_and_names(battle, target_state)

        self.assertEqual(projected["text_tokens"], target_observation["text_tokens"].tolist())
        self.assertTrue(
            np.array_equal(
                np.asarray(projected["numbers"], dtype=np.float32),
                np.asarray(target_observation["numbers"], dtype=np.float32),
            ),
            [
                (index, projected_value, target_value)
                for index, (projected_value, target_value) in enumerate(
                    zip(projected["numbers"], target_observation["numbers"].tolist())
                )
                if projected_value != target_value
            ],
        )
        self.assertEqual(projected["illegal_actions"], target_illegal)
        self.assertEqual(projected["name_table"], target_names)
        self.assertEqual(target_state.opponent_active_pokemon.name, "gothitelle")
        self.assertEqual(target_state.opponent_active_pokemon.hp_pct, 0.8)

    def test_boosted_double_switch_projection_matches_protocol(self):
        battle = MetamonBackendBattle(
            "battle-gen9randombattle-boosteddoubleswitchtest",
            "Alice",
            logging.getLogger("r1-boosted-double-switch-integration"),
            save_replays=False,
            gen=9,
        )
        feed(
            battle,
            [
                "|player|p1|Alice",
                "|player|p2|Bob",
                "|gen|9",
                "|tier|[Gen 9] Random Battle",
                "|poke|p1|Charmander, L50",
                "|poke|p1|Bulbasaur, L50",
                "|poke|p2|Squirtle, L50",
                "|poke|p2|Gothitelle, L50",
                "|start",
                "|switch|p1a: Charmander|Charmander, L50|100/100",
                "|switch|p2a: Gothitelle|Gothitelle, L50|80/100",
                "|switch|p2a: Squirtle|Squirtle, L50|100/100",
                "|-boost|p1a: Charmander|spa|2",
                "|-boost|p1a: Charmander|spd|2",
                "|turn|1",
            ],
        )
        battle.parse_request(silent_request_with_reserve())
        observation_space = fresh_observation_space(self.model.observation_space)
        source_state = UniversalState.from_Battle(battle)
        source_observation = observation_space.state_to_obs(source_state)
        source_history = observation_history(observation_space)
        tracker = R1SwitchTracker.from_snapshot(
            {
                "text_tokens": source_observation["text_tokens"].tolist(),
                "numbers": source_observation["numbers"].tolist(),
                "player_information_state": player_information_state(
                    battle, source_state
                ),
                "player_observation_history": source_history,
                "continuation_observation_history": source_history,
            },
            self.model.observation_space,
        )

        mechanical = EngineState(
            side_one=EngineSide(
                pokemon=[
                    EnginePokemon(
                        id="charmander",
                        level=50,
                        moves=[
                            EngineMove(id="calmmind", pp=32),
                            EngineMove(id="recover", pp=16),
                            EngineMove(id="tackle", pp=56),
                            EngineMove(id="thunderwave", pp=32),
                        ],
                    ),
                    EnginePokemon(
                        id="bulbasaur",
                        level=50,
                        moves=[
                            EngineMove(id="gigadrain", pp=16),
                            EngineMove(id="protect", pp=16),
                            EngineMove(id="sludgebomb", pp=16),
                            EngineMove(id="synthesis", pp=8),
                        ],
                    ),
                ],
                special_attack_boost=2,
                special_defense_boost=2,
            ),
            side_two=EngineSide(
                pokemon=[
                    EnginePokemon(
                        id="squirtle",
                        level=50,
                        moves=[
                            EngineMove(id="tackle"),
                            EngineMove(id="protect"),
                            EngineMove(id="rest"),
                            EngineMove(id="sleeptalk"),
                        ],
                    ),
                    EnginePokemon(
                        id="gothitelle",
                        level=50,
                        hp=80,
                        maxhp=100,
                        ability="shadowtag",
                        moves=[
                            EngineMove(id="psychic"),
                            EngineMove(id="protect"),
                            EngineMove(id="rest"),
                            EngineMove(id="sleeptalk"),
                        ],
                    ),
                ]
            ),
        )
        projection = project_information_set_switch(
            poke_engine,
            [mechanical],
            "switch bulbasaur",
            "switch gothitelle",
            0.5,
            public_opponent=tracker.public_opponent_registry(),
        )
        self.assertEqual(
            projection.certificate, BOOSTED_DOUBLE_SWITCH_CERTIFICATE
        )
        side_two_projection = project_information_set_switch(
            poke_engine,
            [mechanical],
            "switch bulbasaur",
            "switch gothitelle",
            0.5,
            observer_side="SideTwo",
        )
        self.assertEqual(
            side_two_projection.certificate, BOOSTED_DOUBLE_SWITCH_CERTIFICATE
        )
        self.assertEqual(side_two_projection.cleared_self_boosts, ())
        active_record = next(record for record in tracker.player_team if record.active)
        active_record.pokemon.spa_boost = 1
        with self.assertRaisesRegex(
            PublicEventProjectionError, "PUBLIC_PRESTATE_MISMATCH"
        ):
            tracker.apply_switch_projection(projection)
        active_record.pokemon.spa_boost = 2
        projected = tracker.apply_switch_projection(projection)

        feed(
            battle,
            [
                "|switch|p2a: Gothitelle|Gothitelle, L50|80/100",
                "|switch|p1a: Bulbasaur|Bulbasaur, L50|100/100",
                "|turn|2",
            ],
        )
        battle.parse_request(bulbasaur_request_with_reserve(trapped=True))
        target_state = UniversalState.from_Battle(battle)
        target_observation = observation_space.state_to_obs(target_state)
        target_illegal, target_names = mask_and_names(battle, target_state)

        self.assertEqual(projected["text_tokens"], target_observation["text_tokens"].tolist())
        self.assertTrue(
            np.array_equal(
                np.asarray(projected["numbers"], dtype=np.float32),
                np.asarray(target_observation["numbers"], dtype=np.float32),
            )
        )
        self.assertEqual(projected["illegal_actions"], target_illegal)
        self.assertEqual(projected["name_table"], target_names)


if __name__ == "__main__":
    unittest.main()
