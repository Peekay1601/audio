from dubbing_studio_mcp.planning import build_turns, guess_characters, normalize_sound_cues


def w(text, start, end, spk):
    return {"text": text, "start": start, "end": end, "type": "word", "speaker_id": spk}


def test_turns_split_on_speaker_and_gap():
    words = [
        w("hi", 0.0, 0.3, "a"), {"text": " ", "type": "spacing", "start": 0.3, "end": 0.4},
        w("there", 0.4, 0.8, "a"),
        w("yo", 1.0, 1.2, "b"),
        w("later", 3.0, 3.4, "b"),  # gap > 0.7 -> new turn
    ]
    turns = build_turns(words, pad=0.1, duration=4.0)
    assert [(t["speaker"], t["text"]) for t in turns] == [("a", "hi there"), ("b", "yo"), ("b", "later")]
    # padding never crosses into the neighbouring turn
    assert turns[0]["end"] <= turns[1]["start"] + 1e-9
    assert turns[0]["start"] == 0.0
    assert turns[2]["end"] == 3.5


def test_turns_respect_max_len():
    words = [w(f"w{i}", i * 0.5, i * 0.5 + 0.4, "a") for i in range(20)]
    turns = build_turns(words, max_len=3.0)
    assert len(turns) > 1
    assert all(t["end"] - t["start"] <= 3.0 + 0.25 for t in turns)


def test_guess_characters():
    script = "INT. HOUSE - DAY\n\nMAYA\nHello.\n\nARJUN (V.O.)\nHi.\n\nMAYA\nBye.\n\nCUT TO:\n"
    assert guess_characters(script) == {"MAYA": 2, "ARJUN": 1}
    assert guess_characters("Maya: hello\nArjun: hi\nMaya: bye") == {"MAYA": 2, "ARJUN": 1}


def test_sound_cue_defaults():
    cues = normalize_sound_cues(
        [{"layer": "ambience", "start": 0, "end": 50, "prompt": "rain"}, {"start": 2, "duration": 1, "prompt": "hit"}],
        video_duration=40,
    )
    assert cues[0]["duration"] == 40 and cues[0]["loop"] is True
    assert cues[1]["layer"] == "sfx" and cues[1]["loop"] is False
