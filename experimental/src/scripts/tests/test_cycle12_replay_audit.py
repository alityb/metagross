from experimental.src.scripts import cycle12_replay_audit as v12


COMMIT = "f8ac14003a5f27e1bdc8d8c59608a773c1cb96e5"
PLAYERS = ["|player|p1|Alice", "|player|p2|Bob"]


def canon(extra):
    return v12.canonical_public_lines(
        [*PLAYERS, *extra, "|move|p1a: A|Tackle|p2a: B", "|win|Alice"],
        inputlog="", showdown_commit=COMMIT,
    )


def test_exact_transport_grammars_drop_without_dropping_mechanics():
    transport = [
        "||Invite sent to Bob!",
        "|hidelines|unlink|bob",
        "|-message|Bob forfeited.",
        v12.MODERATED_CHAT_LINE,
        "|error|'Minior-Shield' doesn't match any Pokémon, item, move, ability or nature. (Check your spelling?)",
    ]
    result = canon(transport)
    assert all(line not in result for line in transport)
    assert "|move|p1a: A|Tackle|p2a: B" in result
    assert "|win|Alice" in result


def test_invite_forfeit_and_unlink_require_authenticated_player():
    rows = ["||Invite sent to Eve!", "|hidelines|unlink|eve", "|-message|Eve forfeited."]
    result = canon(rows)
    assert all(line in result for line in rows)


def test_near_miss_raw_error_and_message_are_preserved():
    rows = [
        '|raw|<div class="broadcast-red"><strong>Different message</strong></div>',
        "|error|A mechanical-looking error",
        "|-message|Bob lost due to inactivity.",
        "|-damage|p2a: B|0 fnt",
    ]
    result = canon(rows)
    assert all(line in result for line in rows)
