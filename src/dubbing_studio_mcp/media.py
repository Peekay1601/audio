"""ffmpeg / numpy helpers: probing, cutting, frame grabs and timeline mixing."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 48_000
CHANNELS = 2


class MediaError(RuntimeError):
    pass


def _bin(name: str) -> str:
    """Locate ffmpeg: FFMPEG_PATH, then PATH, then the copy bundled by imageio-ffmpeg."""
    if name == "ffmpeg" and os.environ.get("FFMPEG_PATH"):
        return os.environ["FFMPEG_PATH"]
    path = shutil.which(name)
    if path:
        return path
    if name == "ffmpeg":
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:  # pragma: no cover - depends on platform wheel
            raise MediaError(f"ffmpeg not found and the bundled copy failed to load: {exc}") from exc
    raise MediaError(f"'{name}' was not found on PATH.")


def run(args: list[str], input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(args, input=input_bytes, capture_output=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="replace")[-1500:]
        raise MediaError(f"{Path(args[0]).name} failed: {tail}")
    return proc


def probe(path: Path) -> dict:
    try:
        _bin("ffprobe")
    except MediaError:
        return _probe_with_ffmpeg(path)
    proc = run(
        [_bin("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    )
    info = json.loads(proc.stdout)
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    fps = None
    if video and video.get("avg_frame_rate", "0/0") != "0/0":
        num, den = video["avg_frame_rate"].split("/")
        fps = round(float(num) / float(den), 3) if float(den) else None
    return {
        "duration": float(info.get("format", {}).get("duration", 0.0)),
        "has_video": video is not None,
        "has_audio": audio is not None,
        "width": video.get("width") if video else None,
        "height": video.get("height") if video else None,
        "fps": fps,
        "audio_channels": audio.get("channels") if audio else None,
        "audio_sample_rate": int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None,
    }


def _probe_with_ffmpeg(path: Path) -> dict:
    """Fallback when ffprobe isn't installed: parse the banner of ``ffmpeg -i``."""
    proc = subprocess.run([_bin("ffmpeg"), "-hide_banner", "-i", str(path)], capture_output=True)
    text = proc.stderr.decode(errors="replace")
    if "Invalid data found" in text or "No such file" in text:
        raise MediaError(f"Cannot read {path.name}: {text[-500:]}")
    duration = 0.0
    m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if m:
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    video = re.search(r"Stream #\S+.*?: Video: .*", text)
    audio = re.search(r"Stream #\S+.*?: Audio: .*", text)
    width = height = fps = None
    if video:
        size = re.search(r", (\d{2,5})x(\d{2,5})", video.group(0))
        if size:
            width, height = int(size.group(1)), int(size.group(2))
        rate = re.search(r"([\d.]+) fps", video.group(0))
        fps = float(rate.group(1)) if rate else None
    channels = rate_hz = None
    if audio:
        hz = re.search(r"(\d+) Hz", audio.group(0))
        rate_hz = int(hz.group(1)) if hz else None
        line = audio.group(0)
        channels = 1 if "mono" in line else 2 if "stereo" in line else None
        ch = re.search(r"(\d+) channels", line)
        channels = int(ch.group(1)) if ch else channels
    return {
        "duration": duration,
        "has_video": video is not None,
        "has_audio": audio is not None,
        "width": width,
        "height": height,
        "fps": fps,
        "audio_channels": channels,
        "audio_sample_rate": rate_hz,
    }


def extract_audio(src: Path, out: Path, start: float | None = None, end: float | None = None) -> Path:
    """Decode (a slice of) any media file to 48k WAV."""
    out.parent.mkdir(parents=True, exist_ok=True)
    args = [_bin("ffmpeg"), "-y", "-v", "error"]
    if start is not None:
        args += ["-ss", f"{max(start, 0):.3f}"]
    if end is not None:
        args += ["-t", f"{end - max(start or 0.0, 0):.3f}"]
    args += ["-i", str(src), "-vn", "-ac", "2", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(out)]
    run(args)
    return out


def detect_scenes(video: Path, threshold: float = 0.3) -> list[float]:
    """Return timestamps (seconds) of hard cuts using ffmpeg's scene score."""
    proc = subprocess.run(
        [
            _bin("ffmpeg"),
            "-hide_banner",
            "-i",
            str(video),
            "-an",
            "-vf",
            f"select='gt(scene,{threshold})',showinfo",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise MediaError(proc.stderr.decode(errors="replace")[-1500:])
    times = [float(m) for m in re.findall(r"pts_time:([0-9.]+)", proc.stderr.decode(errors="replace"))]
    return sorted(set(round(t, 3) for t in times))


def build_scenes(cuts: list[float], duration: float, min_len: float = 0.5) -> list[dict]:
    bounds = [0.0] + [c for c in cuts if 0 < c < duration] + [duration]
    scenes: list[dict] = []
    for start, end in zip(bounds, bounds[1:]):
        if scenes and end - start < min_len:
            scenes[-1]["end"] = round(end, 3)
            continue
        scenes.append({"start": round(start, 3), "end": round(end, 3)})
    for i, s in enumerate(scenes, 1):
        s["id"] = i
        s["duration"] = round(s["end"] - s["start"], 3)
    return scenes


def grab_frame(video: Path, t: float, out: Path, width: int = 768) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            _bin("ffmpeg"),
            "-y",
            "-v",
            "error",
            "-ss",
            f"{max(t, 0):.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-vf",
            f"scale={width}:-2",
            "-q:v",
            "4",
            str(out),
        ]
    )
    return out


def loudness_profile(src: Path, window: float = 1.0) -> list[float]:
    """Per-window RMS level in dBFS — a cheap 'energy curve' of the edit."""
    audio = decode(src, mono=True)
    hop = int(SAMPLE_RATE * window)
    levels = []
    for i in range(0, len(audio), hop):
        chunk = audio[i : i + hop]
        rms = float(np.sqrt(np.mean(chunk**2))) if len(chunk) else 0.0
        levels.append(round(20 * np.log10(rms) if rms > 1e-6 else -120.0, 1))
    return levels


# ----- PCM in / out ------------------------------------------------------
def decode(src: Path, mono: bool = False) -> np.ndarray:
    ch = 1 if mono else CHANNELS
    proc = run(
        [_bin("ffmpeg"), "-v", "error", "-i", str(src), "-vn", "-f", "f32le", "-ac", str(ch), "-ar", str(SAMPLE_RATE), "-"]
    )
    data = np.frombuffer(proc.stdout, dtype=np.float32)
    return data if mono else data.reshape(-1, CHANNELS)


def write_wav(out: Path, audio: np.ndarray) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    run(
        [
            _bin("ffmpeg"),
            "-y",
            "-v",
            "error",
            "-f",
            "f32le",
            "-ac",
            str(CHANNELS),
            "-ar",
            str(SAMPLE_RATE),
            "-i",
            "-",
            "-c:a",
            "pcm_s24le",
            str(out),
        ],
        input_bytes=audio.tobytes(),
    )
    return out


# ----- timeline ----------------------------------------------------------
@dataclass
class Clip:
    path: Path
    start: float  # position on the video timeline, seconds
    gain_db: float = 0.0
    fade_in: float = 0.0
    fade_out: float = 0.0
    length: float | None = None  # trim (or loop, if ``loop``) to this many seconds
    loop: bool = False


def _shape(audio: np.ndarray, clip: Clip) -> np.ndarray:
    if clip.length is not None:
        n = int(round(clip.length * SAMPLE_RATE))
        if clip.loop and 0 < len(audio) < n:
            audio = np.concatenate([audio] * (n // len(audio) + 1))
        audio = audio[:n]
    audio = audio * (10 ** (clip.gain_db / 20))
    n = len(audio)
    if clip.fade_in > 0 and n:
        k = min(n, int(clip.fade_in * SAMPLE_RATE))
        audio[:k] *= np.linspace(0, 1, k, dtype=np.float32)[:, None]
    if clip.fade_out > 0 and n:
        k = min(n, int(clip.fade_out * SAMPLE_RATE))
        audio[n - k :] *= np.linspace(1, 0, k, dtype=np.float32)[:, None]
    return audio


def render_timeline(clips: list[Clip], duration: float, out: Path) -> Path:
    """Place clips at their timeline positions into one WAV of ``duration`` seconds."""
    total = int(round(duration * SAMPLE_RATE))
    mix = np.zeros((total, CHANNELS), dtype=np.float32)
    for clip in clips:
        audio = _shape(decode(clip.path).copy(), clip)
        start = int(round(clip.start * SAMPLE_RATE))
        if start >= total or not len(audio):
            continue
        end = min(total, start + len(audio))
        mix[start:end] += audio[: end - start]
    peak = float(np.max(np.abs(mix))) if total else 0.0
    if peak > 0.99:  # soft safety: never clip the delivered stem
        mix *= 0.99 / peak
    return write_wav(out, mix)


def sum_stems(stems: list[tuple[Path, float]], out: Path) -> Path:
    """Sum full-length stems with per-stem gain (dB)."""
    arrays = [decode(p) * (10 ** (g / 20)) for p, g in stems]
    n = max(len(a) for a in arrays)
    mix = np.zeros((n, CHANNELS), dtype=np.float32)
    for a in arrays:
        mix[: len(a)] += a
    peak = float(np.max(np.abs(mix))) if n else 0.0
    if peak > 0.99:
        mix *= 0.99 / peak
    return write_wav(out, mix)


def mux_preview(video: Path, audio: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            _bin("ffmpeg"),
            "-y",
            "-v",
            "error",
            "-i",
            str(video),
            "-i",
            str(audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "256k",
            "-shortest",
            str(out),
        ]
    )
    return out
