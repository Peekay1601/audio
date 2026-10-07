"""Request shapes sent by the real client (mocked transport, no network)."""

import json

import httpx

from dubbing_studio_mcp.elevenlabs import ElevenLabs


def make_client(handler, retries_ok=True):
    c = ElevenLabs(api_key="k")
    c._client = httpx.AsyncClient(base_url="https://api.test", transport=httpx.MockTransport(handler),
                                  headers={"xi-api-key": "k"})
    return c


async def test_speech_to_text_multipart(tmp_path):
    f = tmp_path / "a.wav"
    f.write_bytes(b"RIFFdata")
    seen = {}

    def handler(req: httpx.Request):
        seen["path"] = req.url.path
        seen["body"] = req.read().decode(errors="replace")
        seen["key"] = req.headers["xi-api-key"]
        return httpx.Response(200, json={"words": []})

    c = make_client(handler)
    await c.speech_to_text(f, num_speakers=2, keyterms=["Maya", "Arjun"])
    assert seen["path"] == "/v1/speech-to-text" and seen["key"] == "k"
    body = seen["body"]
    assert 'name="diarize"' in body and 'name="model_id"' in body
    assert body.count('name="keyterms"') == 2
    assert 'name="file"; filename="a.wav"' in body


async def test_retry_then_success_reopens_file(tmp_path):
    f = tmp_path / "a.wav"
    f.write_bytes(b"x" * 10)
    hits = []

    def handler(req: httpx.Request):
        hits.append(req.read())
        if len(hits) == 1:
            return httpx.Response(429, json={"detail": "busy"})
        return httpx.Response(200, content=b"AUDIO")

    import dubbing_studio_mcp.elevenlabs as mod

    async def no_sleep(_):
        return None

    orig = mod.asyncio.sleep
    mod.asyncio.sleep = no_sleep
    try:
        c = make_client(handler)
        out = await c.speech_to_speech("v1", f, voice_settings={"stability": 0.5})
    finally:
        mod.asyncio.sleep = orig
    assert out == b"AUDIO" and len(hits) == 2
    assert b"x" * 10 in hits[1]  # file body re-sent on retry


async def test_json_bodies():
    bodies = {}

    def handler(req: httpx.Request):
        bodies[req.url.path] = (json.loads(req.read() or b"{}"), dict(req.url.params))
        if req.url.path == "/v1/text-to-voice/design":
            return httpx.Response(200, json={"previews": [], "text": ""})
        return httpx.Response(200, content=b"A")

    c = make_client(handler)
    await c.sound_effect("rain", duration_seconds=12, loop=True, prompt_influence=0.5)
    await c.compose_music(prompt="strings", music_length_ms=6500)
    await c.design_voice("old man")
    sfx, sfx_q = bodies["/v1/sound-generation"]
    assert sfx == {"text": "rain", "duration_seconds": 12, "loop": True, "prompt_influence": 0.5,
                   "model_id": "eleven_text_to_sound_v2"}
    assert "output_format" in sfx_q
    music, _ = bodies["/v1/music"]
    assert music["prompt"] == "strings" and music["music_length_ms"] == 6500 and music["force_instrumental"] is True
    design, _ = bodies["/v1/text-to-voice/design"]
    assert design["voice_description"] == "old man" and design["auto_generate_text"] is True


async def test_error_surfaces_body():
    import pytest

    c = make_client(lambda req: httpx.Response(401, json={"detail": "invalid_api_key"}))
    with pytest.raises(Exception, match="invalid_api_key"):
        await c.my_voices()
