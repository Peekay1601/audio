"""Project folder layout and on-disk state.

Everything the server produces lives inside one project folder that the user
chooses. Each stage writes plain JSON / Markdown / audio files so the user can
inspect, edit or re-run any step by hand.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mxf"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".aif", ".aiff", ".opus"}
SCRIPT_EXTS = {".txt", ".md", ".fountain", ".srt", ".vtt", ".fdx", ".pdf", ".docx"}

LAYOUT = {
    "input_video": "01_input/video",
    "input_script": "01_input/script",
    "input_dubbing": "01_input/dubbing",
    "analysis": "02_analysis",
    "frames": "02_analysis/frames",
    "transcripts": "02_analysis/transcripts",
    "voices": "03_voices",
    "voice_options": "03_voices/voice_options",
    "voice_segments": "03_voices/segments",
    "voice_stems": "03_voices/stems",
    "soundscape": "04_soundscape",
    "soundscape_cues": "04_soundscape/cues",
    "soundscape_stems": "04_soundscape/stems",
    "music": "05_music",
    "music_cues": "05_music/cues",
    "final": "06_final",
}

README_TEXT = """# {name}

Audio post project managed by the ElevenLabs Dubbing Studio MCP server.

## Put your files here
- `01_input/video/`   -> the full video render (one file)
- `01_input/script/`  -> the script (.txt, .md, .fountain, .fdx, .pdf, .docx, .srt)
- `01_input/dubbing/` -> the dubbing audio: one full mix, or one file per character

## What gets generated
- `02_analysis/`   scene list, key frames, diarized transcripts
- `03_voices/`     voice prompts, voice options to approve, converted dialogue per character
- `04_soundscape/` sound design plan + every generated ambience / SFX / foley cue
- `05_music/`      music plan + every generated music cue, laid out to the edit
- `06_final/`      combined preview mix and preview video
"""


def slugify(value: str) -> str:
    value = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE).strip().lower()
    return re.sub(r"[\s-]+", "_", value) or "untitled"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_project_path(path: str) -> Path:
    """Expand ``~`` and resolve relative paths against AUDIO_PROJECTS_ROOT (if set)."""
    p = Path(os.path.expandvars(os.path.expanduser(path)))
    if not p.is_absolute():
        root = os.environ.get("AUDIO_PROJECTS_ROOT")
        p = Path(os.path.expanduser(root)) / p if root else Path.cwd() / p
    return p.resolve()


@dataclass
class Project:
    root: Path

    # ----- construction -------------------------------------------------
    @classmethod
    def create(cls, path: str, name: str | None = None) -> "Project":
        root = resolve_project_path(path)
        root.mkdir(parents=True, exist_ok=True)
        project = cls(root)
        for rel in LAYOUT.values():
            (root / rel).mkdir(parents=True, exist_ok=True)
        if not project.state_file.exists():
            project.write_json("project.json", {"name": name or root.name, "created": now_iso()})
        readme = root / "README.md"
        if not readme.exists():
            readme.write_text(README_TEXT.format(name=name or root.name))
        return project

    @classmethod
    def open(cls, path: str) -> "Project":
        root = resolve_project_path(path)
        project = cls(root)
        if not project.state_file.exists():
            raise FileNotFoundError(
                f"No project at {root}. Run create_project first (it makes the folder structure)."
            )
        # Re-create any folder the user may have deleted.
        for rel in LAYOUT.values():
            (root / rel).mkdir(parents=True, exist_ok=True)
        return project

    # ----- paths --------------------------------------------------------
    @property
    def state_file(self) -> Path:
        return self.root / "project.json"

    def dir(self, key: str) -> Path:
        return self.root / LAYOUT[key]

    def rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    # ----- json helpers -------------------------------------------------
    def read_json(self, rel: str, default: Any = None) -> Any:
        f = self.root / rel
        if not f.exists():
            return default
        return json.loads(f.read_text())

    def write_json(self, rel: str, data: Any) -> Path:
        f = self.root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(f.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        tmp.replace(f)
        return f

    def write_text(self, rel: str, text: str) -> Path:
        f = self.root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
        return f

    def update_state(self, **values: Any) -> dict:
        state = self.read_json("project.json", {})
        state.update(values)
        self.write_json("project.json", state)
        return state

    # ----- inputs -------------------------------------------------------
    @staticmethod
    def _files(folder: Path, exts: set[str]) -> list[Path]:
        if not folder.exists():
            return []
        return sorted(
            p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts and not p.name.startswith(".")
        )

    def video_files(self) -> list[Path]:
        return self._files(self.dir("input_video"), VIDEO_EXTS)

    def script_files(self) -> list[Path]:
        return self._files(self.dir("input_script"), SCRIPT_EXTS)

    def dubbing_files(self) -> list[Path]:
        return self._files(self.dir("input_dubbing"), AUDIO_EXTS | VIDEO_EXTS)

    def video(self) -> Path:
        files = self.video_files()
        if not files:
            raise FileNotFoundError(f"No video found. Put the full render in {self.dir('input_video')}")
        return files[0]

    def dubbing_file(self, name: str) -> Path:
        for f in self.dubbing_files():
            if f.name == name or f.stem == name:
                return f
        raise FileNotFoundError(
            f"Dubbing file '{name}' not found. Available: {[f.name for f in self.dubbing_files()]}"
        )
