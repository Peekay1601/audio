"""Thin async client for the ElevenLabs REST endpoints this server uses."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import httpx

BASE_URL = (os.environ.get("ELEVENLABS_BASE_URL") or "https://api.elevenlabs.io")

DEFAULT_OUTPUT_FORMAT = (os.environ.get("ELEVENLABS_OUTPUT_FORMAT") or "mp3_44100_128")
STT_MODEL = (os.environ.get("ELEVENLABS_STT_MODEL") or "scribe_v2")
STS_MODEL = (os.environ.get("ELEVENLABS_STS_MODEL") or "eleven_multilingual_sts_v2")
TTV_MODEL = (os.environ.get("ELEVENLABS_TTV_MODEL") or "eleven_multilingual_ttv_v2")
SFX_MODEL = (os.environ.get("ELEVENLABS_SFX_MODEL") or "eleven_text_to_sound_v2")
# Empty = let ElevenLabs pick its default music model.
MUSIC_MODEL = (os.environ.get("ELEVENLABS_MUSIC_MODEL") or "")

RETRY_STATUS = {429, 500, 502, 503, 504}


class ElevenLabsError(RuntimeError):
    pass


def _drop_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


class ElevenLabs:
    def __init__(self, api_key: str | None = None, timeout: float = 600.0):
        self.api_key = api_key or (os.environ.get("ELEVENLABS_API_KEY") or "")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if not self.api_key:
            raise ElevenLabsError(
                "No ElevenLabs API key. Add it in Claude Desktop: Settings -> Extensions -> ElevenLabs Dubbing Studio."
            )
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=BASE_URL,
                headers={"xi-api-key": self.api_key},
                timeout=httpx.Timeout(self._timeout, connect=30.0),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        delay = 2.0
        for attempt in range(5):
            # Files must be re-opened on every attempt, so callers pass a factory.
            files_factory = kwargs.pop("files_factory", None)
            req_kwargs = dict(kwargs)
            opened: list[Any] = []
            if files_factory is not None:
                files, opened = files_factory()
                req_kwargs["files"] = files
                kwargs["files_factory"] = files_factory
            try:
                resp = await self._http().request(method, url, **req_kwargs)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                if attempt == 4:
                    raise ElevenLabsError(f"{method} {url} failed: {exc}") from exc
                resp = None
            finally:
                for fh in opened:
                    fh.close()
            if resp is not None and resp.status_code < 400:
                return resp
            if resp is not None and (resp.status_code not in RETRY_STATUS or attempt == 4):
                raise ElevenLabsError(f"{method} {url} -> HTTP {resp.status_code}: {resp.text[:1500]}")
            await asyncio.sleep(delay)
            delay *= 2
        raise ElevenLabsError(f"{method} {url} failed after retries")

    @staticmethod
    def _file_factory(field: str, path: Path):
        def factory():
            fh = open(path, "rb")
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            return {field: (path.name, fh, mime)}, [fh]

        return factory

    # ----- voices -------------------------------------------------------
    async def my_voices(self, search: str | None = None, page_size: int = 30) -> list[dict]:
        resp = await self._request(
            "GET", "/v2/voices", params=_drop_none({"search": search, "page_size": page_size})
        )
        return resp.json().get("voices", [])

    async def shared_voices(
        self,
        search: str | None = None,
        gender: str | None = None,
        age: str | None = None,
        accent: str | None = None,
        language: str | None = None,
        use_cases: str | None = None,
        page_size: int = 20,
    ) -> list[dict]:
        params = _drop_none(
            {
                "search": search,
                "gender": gender,
                "age": age,
                "accent": accent,
                "language": language,
                "use_cases": use_cases,
                "page_size": page_size,
            }
        )
        resp = await self._request("GET", "/v1/shared-voices", params=params)
        return resp.json().get("voices", [])

    async def design_voice(
        self,
        description: str,
        text: str | None = None,
        model_id: str | None = None,
        guidance_scale: float | None = None,
        loudness: float | None = None,
        seed: int | None = None,
    ) -> dict:
        body = _drop_none(
            {
                "voice_description": description,
                "model_id": model_id or TTV_MODEL,
                "text": text,
                "auto_generate_text": None if text else True,
                "guidance_scale": guidance_scale,
                "loudness": loudness,
                "seed": seed,
            }
        )
        resp = await self._request(
            "POST", "/v1/text-to-voice/design", params={"output_format": DEFAULT_OUTPUT_FORMAT}, json=body
        )
        return resp.json()

    async def create_voice_from_preview(self, name: str, description: str, generated_voice_id: str) -> dict:
        resp = await self._request(
            "POST",
            "/v1/text-to-voice",
            json={
                "voice_name": name,
                "voice_description": description,
                "generated_voice_id": generated_voice_id,
            },
        )
        return resp.json()

    async def add_shared_voice(self, public_owner_id: str, voice_id: str, new_name: str) -> dict:
        resp = await self._request(
            "POST", f"/v1/voices/add/{public_owner_id}/{voice_id}", json={"new_name": new_name}
        )
        return resp.json()

    async def download(self, url: str) -> bytes:
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as c:
            r = await c.get(url)
            r.raise_for_status()
            return r.content

    # ----- speech -------------------------------------------------------
    async def speech_to_text(
        self,
        path: Path,
        num_speakers: int | None = None,
        language_code: str | None = None,
        keyterms: list[str] | None = None,
    ) -> dict:
        data: dict[str, Any] = _drop_none(
            {
                "model_id": STT_MODEL,
                "diarize": "true",
                "timestamps_granularity": "word",
                "tag_audio_events": "true",
                "num_speakers": str(num_speakers) if num_speakers else None,
                "language_code": language_code,
            }
        )
        if keyterms:
            # httpx sends a list value as a repeated multipart field.
            data["keyterms"] = list(keyterms)
        resp = await self._request(
            "POST", "/v1/speech-to-text", data=data, files_factory=self._file_factory("file", path)
        )
        return resp.json()

    async def speech_to_speech(
        self,
        voice_id: str,
        path: Path,
        remove_background_noise: bool = False,
        voice_settings: dict | None = None,
        model_id: str | None = None,
        seed: int | None = None,
    ) -> bytes:
        data = _drop_none(
            {
                "model_id": model_id or STS_MODEL,
                "remove_background_noise": "true" if remove_background_noise else "false",
                "voice_settings": json.dumps(voice_settings) if voice_settings else None,
                "seed": str(seed) if seed is not None else None,
            }
        )
        resp = await self._request(
            "POST",
            f"/v1/speech-to-speech/{voice_id}",
            params={"output_format": DEFAULT_OUTPUT_FORMAT},
            data=data,
            files_factory=self._file_factory("audio", path),
        )
        return resp.content

    async def isolate_voice(self, path: Path) -> bytes:
        resp = await self._request(
            "POST", "/v1/audio-isolation", files_factory=self._file_factory("audio", path)
        )
        return resp.content

    # ----- sound & music ------------------------------------------------
    async def sound_effect(
        self,
        text: str,
        duration_seconds: float | None = None,
        loop: bool = False,
        prompt_influence: float | None = None,
    ) -> bytes:
        body = _drop_none(
            {
                "text": text,
                "duration_seconds": duration_seconds,
                "loop": loop or None,
                "prompt_influence": prompt_influence,
                "model_id": SFX_MODEL,
            }
        )
        resp = await self._request(
            "POST", "/v1/sound-generation", params={"output_format": DEFAULT_OUTPUT_FORMAT}, json=body
        )
        return resp.content

    async def compose_music(
        self,
        prompt: str | None = None,
        music_length_ms: int | None = None,
        composition_plan: dict | None = None,
        force_instrumental: bool = True,
        seed: int | None = None,
    ) -> bytes:
        body = _drop_none(
            {
                "prompt": None if composition_plan else prompt,
                "composition_plan": composition_plan,
                "music_length_ms": None if composition_plan else music_length_ms,
                "force_instrumental": force_instrumental if not composition_plan else None,
                "model_id": MUSIC_MODEL or None,
                "seed": seed,
            }
        )
        resp = await self._request("POST", "/v1/music", json=body)
        return resp.content
