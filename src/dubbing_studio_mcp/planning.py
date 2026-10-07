"""Pure logic: script reading, dialogue turn building and cue-plan validation."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any


# ----- script ------------------------------------------------------------
def read_script(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        return "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    if ext == ".docx":
        import docx

        return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    if ext == ".fdx":  # Final Draft XML
        import xml.etree.ElementTree as ET

        lines = []
        for para in ET.parse(path).getroot().iter("Paragraph"):
            text = "".join(t.text or "" for t in para.iter("Text"))
            kind = para.get("Type", "")
            lines.append(text.upper() if kind == "Character" else text)
        return "\n".join(lines)
    return path.read_text(errors="replace")


CHARACTER_CUE = re.compile(r"^\s{0,40}([A-Z][A-Z0-9 .'\-]{1,40}?)(\s*\((?:V\.O\.|O\.S\.|O\.C\.|CONT'D|.*?)\))?\s*:?\s*$")
INLINE_CUE = re.compile(r"^\s*([A-Z][A-Za-z0-9 .'\-]{1,30}):\s+\S")
NOT_CHARACTERS = {
    "INT", "EXT", "CUT TO", "FADE IN", "FADE OUT", "DISSOLVE TO", "THE END", "CONTINUED", "MORE",
    "SMASH CUT TO", "MATCH CUT TO", "TITLE", "SUPER", "INTERCUT", "BACK TO", "END",
}


def guess_characters(script: str) -> dict[str, int]:
    """Heuristic speaking-character count from screenplay or 'NAME: line' formats."""
    counts: dict[str, int] = defaultdict(int)
    for line in script.splitlines():
        m = CHARACTER_CUE.match(line) or INLINE_CUE.match(line)
        if not m:
            continue
        name = m.group(1).strip().rstrip(".").upper()
        if name.startswith(("INT", "EXT")) or name in NOT_CHARACTERS or len(name) < 2:
            continue
        counts[name] += 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# ----- dialogue turns ----------------------------------------------------
def build_turns(
    words: list[dict],
    max_gap: float = 0.7,
    max_len: float = 30.0,
    pad: float = 0.12,
    duration: float | None = None,
) -> list[dict]:
    """Group diarized words into single-speaker turns.

    A new turn starts when the speaker changes, the silence gap exceeds
    ``max_gap`` or the turn would exceed ``max_len`` seconds. Turns are padded
    by ``pad`` seconds without crossing into a neighbouring turn.
    """
    spoken = [w for w in words if w.get("type", "word") == "word" and w.get("start") is not None]
    turns: list[dict] = []
    for w in spoken:
        speaker = w.get("speaker_id") or "speaker_0"
        start, end = float(w["start"]), float(w.get("end") or w["start"])
        cur = turns[-1] if turns else None
        if (
            cur
            and cur["speaker"] == speaker
            and start - cur["end"] <= max_gap
            and end - cur["start"] <= max_len
        ):
            cur["end"] = end
            cur["text"] += " " + w["text"].strip()
        else:
            turns.append({"speaker": speaker, "start": start, "end": end, "text": w["text"].strip()})

    raw = [(t["start"], t["end"]) for t in turns]
    for i, t in enumerate(turns):
        start, end = raw[i]
        lo = raw[i - 1][1] if i else 0.0
        hi = raw[i + 1][0] if i + 1 < len(raw) else (duration if duration else end + pad)
        t["start"] = round(max(start - pad, min(lo, start), 0.0), 3)
        t["end"] = round(min(end + pad, max(hi, end)), 3)
        t["index"] = i
    return turns


def speaker_summary(turns: list[dict], samples: int = 4) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for t in turns:
        s = out.setdefault(t["speaker"], {"turns": 0, "seconds": 0.0, "sample_lines": []})
        s["turns"] += 1
        s["seconds"] = round(s["seconds"] + t["end"] - t["start"], 1)
        if len(s["sample_lines"]) < samples and len(t["text"]) > 12:
            s["sample_lines"].append(f"[{t['start']:.1f}s] {t['text'][:160]}")
    return out


# ----- cue plans ---------------------------------------------------------
SOUND_LAYERS = {"ambience", "room_tone", "sfx", "foley", "transition"}


def _num(cue: dict, key: str, default: float | None = None) -> float | None:
    v = cue.get(key, default)
    return None if v is None else float(v)


def normalize_sound_cues(cues: list[dict[str, Any]], video_duration: float) -> list[dict]:
    out, errors = [], []
    for i, c in enumerate(cues, 1):
        cid = str(c.get("id") or f"S{i:03d}")
        start = _num(c, "start", 0.0) or 0.0
        dur = _num(c, "duration")
        if dur is None and c.get("end") is not None:
            dur = float(c["end"]) - start
        layer = str(c.get("layer", "sfx")).lower()
        if not c.get("prompt"):
            errors.append(f"{cid}: missing prompt")
        if dur is None or dur <= 0:
            errors.append(f"{cid}: needs duration (or end) > 0")
            continue
        if layer not in SOUND_LAYERS:
            errors.append(f"{cid}: layer must be one of {sorted(SOUND_LAYERS)}")
        if start >= video_duration:
            errors.append(f"{cid}: starts after the video ends ({video_duration:.1f}s)")
        dur = min(dur, video_duration - start)
        loop = bool(c.get("loop", layer in {"ambience", "room_tone"} or dur > 30))
        out.append(
            {
                "id": cid,
                "layer": layer,
                "start": round(start, 3),
                "duration": round(dur, 3),
                "prompt": c.get("prompt", ""),
                "loop": loop,
                "prompt_influence": _num(c, "prompt_influence", 0.4),
                "gain_db": _num(c, "gain_db", -6.0 if layer in {"ambience", "room_tone"} else 0.0),
                "fade_in": _num(c, "fade_in", 0.5 if loop else 0.02),
                "fade_out": _num(c, "fade_out", 0.8 if loop else 0.05),
                "scene": c.get("scene"),
                "notes": c.get("notes", ""),
            }
        )
    if errors:
        raise ValueError("Sound plan has problems:\n- " + "\n- ".join(errors))
    return out


def normalize_music_cues(cues: list[dict[str, Any]], video_duration: float) -> list[dict]:
    out, errors = [], []
    for i, c in enumerate(cues, 1):
        cid = str(c.get("id") or f"M{i:02d}")
        start = _num(c, "start", 0.0) or 0.0
        end = _num(c, "end")
        if end is None and c.get("duration") is not None:
            end = start + float(c["duration"])
        if end is None or end <= start:
            errors.append(f"{cid}: needs end (or duration) after start")
            continue
        end = min(end, video_duration)
        if not c.get("prompt") and not c.get("composition_plan"):
            errors.append(f"{cid}: needs a prompt or a composition_plan")
        if end - start > 600:
            errors.append(f"{cid}: longer than 600s; split it into several cues")
        out.append(
            {
                "id": cid,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "prompt": c.get("prompt", ""),
                "composition_plan": c.get("composition_plan"),
                "force_instrumental": bool(c.get("force_instrumental", True)),
                "gain_db": _num(c, "gain_db", -8.0),
                "fade_in": _num(c, "fade_in", 1.0),
                "fade_out": _num(c, "fade_out", 2.0),
                "sync_points": c.get("sync_points", []),
                "notes": c.get("notes", ""),
            }
        )
    if errors:
        raise ValueError("Music plan has problems:\n- " + "\n- ".join(errors))
    return out


def fmt_tc(seconds: float) -> str:
    m, s = divmod(max(seconds, 0.0), 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:06.3f}"
