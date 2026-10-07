"""End-to-end run of every tool against synthetic media and a fake ElevenLabs client."""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from dubbing_studio_mcp import media, server

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True)


def tone_mp3(freq: int, seconds: float) -> bytes:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}",
         "-f", "mp3", "-"],
        capture_output=True,
        check=True,
    )
    return proc.stdout


class FakeElevenLabs:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def speech_to_text(self, path, **kw):
        self.calls.append(("stt", kw))
        words = []
        # speaker_0 talks 0.5-2.5s, speaker_1 talks 3.5-5.0s, speaker_0 again 6.0-7.0s
        for spk, start, end, text in [
            ("speaker_0", 0.5, 2.5, "Where were you last night"),
            ("speaker_1", 3.5, 5.0, "Nowhere you need to know about"),
            ("speaker_0", 6.0, 7.0, "Tell me"),
        ]:
            toks = text.split()
            step = (end - start) / len(toks)
            for i, t in enumerate(toks):
                words.append({"text": t, "start": start + i * step, "end": start + (i + 1) * step,
                              "type": "word", "speaker_id": spk})
        return {"language_code": "eng", "text": "...", "words": words}

    async def design_voice(self, description, **kw):
        self.calls.append(("design", {"description": description, **kw}))
        audio = base64.b64encode(tone_mp3(300, 1)).decode()
        return {"text": "preview", "previews": [
            {"audio_base_64": audio, "generated_voice_id": f"gen{i}", "media_type": "audio/mpeg", "duration_secs": 1}
            for i in range(3)
        ]}

    async def my_voices(self, **kw):
        return [{"voice_id": "mine1", "name": "My Narrator", "labels": {"accent": "indian"}, "preview_url": None}]

    async def shared_voices(self, **kw):
        self.calls.append(("shared", kw))
        return [{"voice_id": "lib1", "public_owner_id": "owner1", "name": "Gruff Man", "gender": "male",
                 "age": "old", "accent": "british", "descriptive": "gruff", "use_case": "characters",
                 "preview_url": "https://example.invalid/p.mp3"}]

    async def download(self, url):
        return tone_mp3(500, 1)

    async def create_voice_from_preview(self, name, description, generated_voice_id):
        self.calls.append(("create_voice", {"name": name, "gid": generated_voice_id}))
        return {"voice_id": f"voice_{generated_voice_id}"}

    async def add_shared_voice(self, owner, voice_id, new_name):
        self.calls.append(("add_shared", {"owner": owner, "voice_id": voice_id}))
        return {"voice_id": f"added_{voice_id}"}

    async def speech_to_speech(self, voice_id, path, **kw):
        self.calls.append(("sts", {"voice_id": voice_id, **kw}))
        return tone_mp3(800, media.probe(path)["duration"])

    async def isolate_voice(self, path):
        return Path(path).read_bytes()

    async def sound_effect(self, text, duration_seconds=None, **kw):
        self.calls.append(("sfx", {"text": text, "duration_seconds": duration_seconds, **kw}))
        return tone_mp3(200, duration_seconds or 2)

    async def compose_music(self, prompt=None, music_length_ms=None, **kw):
        self.calls.append(("music", {"prompt": prompt, "music_length_ms": music_length_ms, **kw}))
        return tone_mp3(440, (music_length_ms or 3000) / 1000)


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    root = tmp_path / "MyFilm"
    server.create_project(str(root), "My Film")
    # 8s video: red 4s then blue 4s (one hard cut at 4s), with a 1 kHz tone.
    ff("-f", "lavfi", "-i", "color=red:s=320x240:d=4:r=25", "-f", "lavfi", "-i", "color=blue:s=320x240:d=4:r=25",
       "-f", "lavfi", "-i", "sine=frequency=1000:duration=8",
       "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-map", "2:a",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(root / "01_input/video/film.mp4"))
    ff("-f", "lavfi", "-i", "sine=frequency=220:duration=8", str(root / "01_input/dubbing/dialogue.wav"))
    (root / "01_input/script/script.fountain").write_text(
        "INT. KITCHEN - NIGHT\n\nMAYA\nWhere were you last night?\n\nARJUN\nNowhere you need to know about.\n\n"
        "MAYA\nTell me.\n"
    )
    return root


async def test_full_pipeline(project: Path):
    fake = FakeElevenLabs()
    server.set_client(fake)
    p = str(project)

    out = await server.analyze_project(p)
    assert "MAYA" in out and "ARJUN" in out
    analysis = json.loads((project / "02_analysis/analysis.json").read_text())
    assert len(analysis["scenes"]) == 2
    assert abs(analysis["scenes"][1]["start"] - 4.0) < 0.2
    assert len(analysis["audio_energy_db_per_second"]) >= 8

    frames = await server.view_frames(p, timestamps=[1.0])
    assert any(isinstance(x, server.Image) for x in frames)

    out = await server.transcribe_dubbing(p, keyterms=["Maya", "Arjun"])
    assert "speaker_0" in out and "speaker_1" in out
    server.assign_characters(p, [
        {"file": "dialogue.wav", "speaker": "speaker_0", "character": "Maya"},
        {"file": "dialogue.wav", "speaker": "speaker_1", "character": "ARJUN"},
    ])

    server.save_voice_prompts(p, [
        {"character": "MAYA", "prompt": "Woman, early 30s, Indian English accent, warm but tense.", "gender": "female"},
        {"character": "ARJUN", "prompt": "Man, late 30s, low gravelly voice, evasive.", "gender": "male"},
    ])
    await server.design_voice_options(p, "MAYA")
    await server.find_library_voices(p, "ARJUN", gender="male")
    sheet = server.list_voice_options(p)
    assert "D01" in sheet and "L01" in sheet
    assert (project / "03_voices/VOICE_OPTIONS.md").exists()
    assert list((project / "03_voices/voice_options/maya").glob("D0*_designed.mp3"))

    # Conversion is blocked until every character is approved.
    blocked = await server.convert_dialogue(p)
    assert blocked.startswith("Blocked")
    assert not any(c[0] == "sts" for c in fake.calls)

    await server.approve_voice(p, "MAYA", option_id="D02", stability=0.4)
    arjun_opts = json.loads((project / "03_voices/voice_options/arjun/options.json").read_text())
    lib = next(o for o in arjun_opts if o["source"] == "voice_library")
    await server.approve_voice(p, "ARJUN", option_id=lib["option_id"])
    approved = json.loads((project / "03_voices/approved_voices.json").read_text())
    assert approved["MAYA"]["voice_id"] == "voice_gen1"
    assert approved["ARJUN"]["voice_id"] == "added_lib1"

    out = await server.convert_dialogue(p)
    assert "Converted 3/3" in out, out
    sts = [c for c in fake.calls if c[0] == "sts"]
    assert {c[1]["voice_id"] for c in sts} == {"voice_gen1", "added_lib1"}
    assert any(c[1].get("voice_settings") == {"stability": 0.4} for c in sts)
    for stem in ["03_voices/stems/maya.wav", "03_voices/stems/arjun.wav",
                 "03_voices/stems/original/maya.wav", "03_voices/dialogue_full.wav"]:
        assert abs(media.probe(project / stem)["duration"] - 8.0) < 0.1, stem
    # Arjun's stem is silent where Maya speaks and active where he speaks.
    arjun = media.decode(project / "03_voices/stems/arjun.wav", mono=True)
    sr = media.SAMPLE_RATE
    assert abs(arjun[int(1.0 * sr):int(2.0 * sr)]).max() < 1e-4
    assert abs(arjun[int(4.0 * sr):int(4.5 * sr)]).max() > 0.05

    # Re-running reuses converted segments.
    n = len(sts)
    await server.convert_dialogue(p)
    assert len([c for c in fake.calls if c[0] == "sts"]) == n

    server.save_soundscape_plan(p, [
        {"id": "AMB1", "layer": "ambience", "start": 0, "duration": 8, "prompt": "quiet kitchen night hum, fridge"},
        {"id": "SFX1", "layer": "sfx", "start": 4.0, "duration": 1.0, "prompt": "door slam, wooden, small room"},
    ])
    out = await server.generate_soundscape(p)
    assert "2/2" in out, out
    assert abs(media.probe(project / "04_soundscape/soundscape_full.wav")["duration"] - 8.0) < 0.1
    assert (project / "04_soundscape/stems/ambience.wav").exists()
    sfx = [c for c in fake.calls if c[0] == "sfx"]
    assert sfx[0][1]["loop"] is True and sfx[1][1]["loop"] is False

    server.save_music_plan(p, [{"id": "M1", "start": 1.0, "end": 7.5, "prompt": "tense low strings, 70 BPM"}])
    out = await server.generate_music(p)
    assert "1/1" in out
    music = [c for c in fake.calls if c[0] == "music"][0][1]
    assert music["music_length_ms"] == 6500
    m = media.decode(project / "05_music/music_full.wav", mono=True)
    assert abs(m[: int(0.9 * sr)]).max() < 1e-4  # silent before the cue starts

    out = await server.render_preview(p)
    assert (project / "06_final/final_mix.wav").exists()
    assert media.probe(project / "06_final/preview.mp4")["has_video"]

    status = server.project_status(p)
    assert "APPROVED" in status


def test_bad_plans_are_rejected(project: Path):
    import asyncio

    server.set_client(FakeElevenLabs())
    asyncio.run(server.analyze_project(str(project)))
    with pytest.raises(ValueError):
        server.save_soundscape_plan(str(project), [{"layer": "sfx", "start": 1, "prompt": "x"}])
    with pytest.raises(ValueError):
        server.save_music_plan(str(project), [{"start": 5, "end": 2, "prompt": "x"}])
