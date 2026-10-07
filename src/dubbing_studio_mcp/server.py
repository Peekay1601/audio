"""ElevenLabs Dubbing Studio — MCP server.

Claude (the MCP client) does the creative work: it watches key frames, reads the
script and the diarized dubbing transcript, then writes voice, sound-design and
music prompts. This server is the hands: it organises the project folder, talks
to ElevenLabs, and lays every generated file onto the video's timeline.
"""

from __future__ import annotations

import asyncio
import base64
import csv
import io
import os
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import Context, Image, MCPServer

from . import media
from .elevenlabs import ElevenLabs
from .planning import (
    build_turns,
    fmt_tc,
    guess_characters,
    normalize_music_cues,
    normalize_sound_cues,
    speaker_summary,
)
from .planning import read_script as parse_script_file
from .project import Project, now_iso, slugify

INSTRUCTIONS = """\
Audio post-production for a finished video edit, powered by ElevenLabs.
Folder stages: 01_input (user uploads video, script, dubbing) -> 02_analysis -> 03_voices
-> 04_soundscape -> 05_music -> 06_final. Use the `audio_post_workflow` prompt for the full
step-by-step. Golden rules:
1. Never convert dialogue until the user has explicitly approved a voice for that character.
2. Always LOOK at the video (view_frames) and READ the script before writing any prompt.
3. Show the user every plan (voice prompts, sound plan, music plan) and the saved file paths.
"""

mcp = MCPServer("elevenlabs-dubbing-studio", instructions=INSTRUCTIONS)
_client: ElevenLabs | None = None
_sem = asyncio.Semaphore(int(os.environ.get("ELEVENLABS_CONCURRENCY") or 3))


def el() -> ElevenLabs:
    global _client
    if _client is None:
        _client = ElevenLabs()
    return _client


def set_client(client: Any) -> None:
    """Swap the ElevenLabs client (used by tests)."""
    global _client
    _client = client


async def _limited(coro):
    async with _sem:
        return await coro


async def _progress(ctx: Context | None, done: int, total: int, message: str) -> None:
    if ctx is None:
        return
    try:
        await ctx.report_progress(done, total, message)
    except Exception:  # progress is best-effort; never fail a job over it
        pass


def _ext_for_output() -> str:
    from .elevenlabs import DEFAULT_OUTPUT_FORMAT

    return ".wav" if DEFAULT_OUTPUT_FORMAT.startswith(("pcm", "wav")) else ".mp3"


def _write_audio(path: Path, data: bytes) -> Path:
    """Save API audio. Raw PCM output is wrapped into a WAV so every file plays."""
    from .elevenlabs import DEFAULT_OUTPUT_FORMAT

    path.parent.mkdir(parents=True, exist_ok=True)
    if DEFAULT_OUTPUT_FORMAT.startswith("pcm_") and not data[:4] == b"RIFF":
        import wave

        rate = int(DEFAULT_OUTPUT_FORMAT.split("_")[1])
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(data)
    else:
        path.write_bytes(data)
    return path


def _analysis(project: Project) -> dict:
    data = project.read_json("02_analysis/analysis.json")
    if not data:
        raise RuntimeError("Run analyze_project first.")
    return data


def _cast(project: Project) -> dict:
    return project.read_json("03_voices/cast.json", {"files": {}})


def _characters(project: Project) -> list[str]:
    names: set[str] = set()
    for f in _cast(project)["files"].values():
        names.update(v for v in f.get("speakers", {}).values() if v)
        if f.get("character"):
            names.add(f["character"])
    names.update(project.read_json("03_voices/voice_prompts.json", {}).get("characters", {}).keys())
    names.discard("SKIP")
    return sorted(names)


# =========================================================================
# 1. Project setup & analysis
# =========================================================================
@mcp.tool()
def create_project(project_path: str, name: str | None = None) -> str:
    """Create (or repair) the project folder structure at `project_path`.

    Relative paths are resolved against AUDIO_PROJECTS_ROOT if it is set.
    After this, the user uploads: the full video render to 01_input/video, the
    script to 01_input/script and the dubbing audio to 01_input/dubbing.
    """
    p = Project.create(project_path, name)
    return (
        f"Project ready at {p.root}\n\n"
        f"Upload your files:\n"
        f"  video   -> {p.dir('input_video')}\n"
        f"  script  -> {p.dir('input_script')}\n"
        f"  dubbing -> {p.dir('input_dubbing')}  (one full dialogue mix, or one file per character)\n\n"
        f"Outputs will land in 03_voices/, 04_soundscape/, 05_music/ and 06_final/. "
        f"Call analyze_project when the uploads are in place."
    )


@mcp.tool()
def project_status(project_path: str) -> str:
    """Show what's uploaded, what's done, and the next step."""
    p = Project.open(project_path)
    lines = [f"Project: {p.root}"]
    lines.append(f"Video:   {[f.name for f in p.video_files()] or 'MISSING'}")
    lines.append(f"Script:  {[f.name for f in p.script_files()] or 'MISSING'}")
    lines.append(f"Dubbing: {[f.name for f in p.dubbing_files()] or 'MISSING'}")
    analysis = p.read_json("02_analysis/analysis.json")
    lines.append(f"Analysis: {'done' if analysis else 'not run'}")
    transcripts = sorted(x.name for x in p.dir("transcripts").glob("*.turns.json"))
    lines.append(f"Transcripts: {transcripts or 'none'}")
    cast = _cast(p)["files"]
    lines.append(f"Cast mapping: {cast or 'not set'}")
    prompts = p.read_json("03_voices/voice_prompts.json", {}).get("characters", {})
    approved = p.read_json("03_voices/approved_voices.json", {})
    for ch in _characters(p):
        n_opts = len(p.read_json(f"03_voices/voice_options/{slugify(ch)}/options.json", []))
        state = f"APPROVED -> {approved[ch]['name']} ({approved[ch]['voice_id']})" if ch in approved else "awaiting approval"
        lines.append(f"  - {ch}: prompt={'yes' if ch in prompts else 'no'}, options={n_opts}, {state}")
    lines.append(f"Dialogue stems: {sorted(x.name for x in p.dir('voice_stems').glob('*.wav')) or 'none'}")
    lines.append(f"Soundscape plan: {'yes' if p.read_json('04_soundscape/soundscape_plan.json') else 'no'}; "
                 f"mix: {'yes' if (p.dir('soundscape') / 'soundscape_full.wav').exists() else 'no'}")
    lines.append(f"Music plan: {'yes' if p.read_json('05_music/music_plan.json') else 'no'}; "
                 f"mix: {'yes' if (p.dir('music') / 'music_full.wav').exists() else 'no'}")
    return "\n".join(lines)


@mcp.tool()
def list_projects() -> str:
    """List the projects in the projects folder chosen in the extension settings."""
    root = os.environ.get("AUDIO_PROJECTS_ROOT")
    if not root:
        return "No projects folder configured; pass full paths to create_project."
    base = Path(os.path.expanduser(root))
    projects = sorted(d.name for d in base.iterdir() if (d / "project.json").exists()) if base.exists() else []
    return f"Projects folder: {base}\n" + ("\n".join(f"  - {n}" for n in projects) or "  (none yet)")


@mcp.tool()
def open_in_finder(project_path: str, item: str = "") -> str:
    """Open a project folder (or a file inside it, e.g. a voice preview) on the user's
    computer — in Finder / File Explorer, or the default audio player for a file.

    `item` is relative to the project, e.g. "01_input/video",
    "03_voices/VOICE_OPTIONS.md" or "03_voices/voice_options/maya/D02_designed.mp3".
    Use it so the user can drop in uploads or listen to voice options without a terminal.
    """
    import subprocess
    import sys

    p = Project.open(project_path)
    target = (p.root / item).resolve() if item else p.root
    if p.root not in (target, *target.parents):
        raise ValueError("item must be inside the project folder")
    if not target.exists():
        raise FileNotFoundError(f"{target} does not exist")
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(target)])
    elif sys.platform == "win32":
        os.startfile(str(target))  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", str(target)])
    return f"Opened {target}"


@mcp.tool()
async def analyze_project(project_path: str, scene_threshold: float = 0.3, ctx: Context | None = None) -> str:
    """Analyse the uploads: video specs, scene cuts + key frames, audio energy curve,
    the script (and a guess at its speaking characters) and the dubbing files.

    Returns the scene list and the script so you can start planning. Use
    view_frames to actually look at the scenes before writing any prompt.
    """
    p = Project.open(project_path)
    video = p.video()
    info = await asyncio.to_thread(media.probe, video)
    await _progress(ctx, 1, 4, "detecting scene cuts")
    cuts = await asyncio.to_thread(media.detect_scenes, video, scene_threshold)
    scenes = media.build_scenes(cuts, info["duration"])

    await _progress(ctx, 2, 4, "grabbing key frames")
    for s in scenes:
        mid = (s["start"] + s["end"]) / 2
        frame = p.dir("frames") / f"scene_{s['id']:03d}.jpg"
        if not frame.exists():
            await asyncio.to_thread(media.grab_frame, video, mid, frame)
        s["frame"] = p.rel(frame)

    await _progress(ctx, 3, 4, "measuring audio energy")
    energy: list[float] = []
    ref_audio = p.dir("analysis") / "video_audio.wav"
    if info["has_audio"]:
        await asyncio.to_thread(media.extract_audio, video, ref_audio)
        energy = await asyncio.to_thread(media.loudness_profile, ref_audio)

    script_text, script_file, characters = "", None, {}
    if p.script_files():
        script_file = p.script_files()[0]
        script_text = await asyncio.to_thread(parse_script_file, script_file)
        characters = guess_characters(script_text)
        p.write_text("02_analysis/script.txt", script_text)

    dubbing = []
    for f in p.dubbing_files():
        d = await asyncio.to_thread(media.probe, f)
        dubbing.append({"file": f.name, "duration": round(d["duration"], 3), "channels": d["audio_channels"]})

    analysis = {
        "analyzed_at": now_iso(),
        "video": video.name,
        "duration": round(info["duration"], 3),
        "fps": info["fps"],
        "resolution": f"{info['width']}x{info['height']}",
        "scenes": scenes,
        "audio_energy_db_per_second": energy,
        "script_file": script_file.name if script_file else None,
        "script_characters_guess": characters,
        "dubbing_files": dubbing,
    }
    p.write_json("02_analysis/analysis.json", analysis)
    await _progress(ctx, 4, 4, "done")

    out = [
        f"Video: {video.name} — {fmt_tc(info['duration'])} @ {info['fps']} fps, {analysis['resolution']}",
        f"Scenes ({len(scenes)}):",
    ]
    for s in scenes:
        out.append(f"  #{s['id']:>3}  {fmt_tc(s['start'])} -> {fmt_tc(s['end'])}  ({s['duration']:.1f}s)")
    out.append(f"Dubbing files: {dubbing or 'NONE — upload to 01_input/dubbing'}")
    if script_file:
        out.append(f"Script: {script_file.name}; likely speaking characters: {characters or 'unclear'}")
        out.append("---- SCRIPT (first 15k chars; use read_script for the rest) ----")
        out.append(script_text[:15000])
    else:
        out.append("Script: MISSING — upload it to 01_input/script")
    out.append("\nNext: view_frames to watch the scenes, then transcribe_dubbing.")
    return "\n".join(out)


@mcp.tool()
def read_script(project_path: str, offset: int = 0, max_chars: int = 40000) -> str:
    """Return the script text (paged by characters)."""
    p = Project.open(project_path)
    cached = p.root / "02_analysis/script.txt"
    if cached.exists():
        text = cached.read_text()
    elif p.script_files():
        text = parse_script_file(p.script_files()[0])
    else:
        return "No script uploaded."
    chunk = text[offset : offset + max_chars]
    more = len(text) - (offset + len(chunk))
    return chunk + (f"\n\n[... {more} more chars; call again with offset={offset + len(chunk)}]" if more > 0 else "")


@mcp.tool(structured_output=False)
async def view_frames(
    project_path: str,
    scene_ids: list[int] | None = None,
    timestamps: list[float] | None = None,
    max_frames: int = 12,
) -> list[Any]:
    """Look at the video. Returns key frames for the given scene ids (default: an even
    spread of scenes) and/or exact timestamps in seconds. Call it several times to
    cover the whole edit before writing sound or music prompts.
    """
    p = Project.open(project_path)
    analysis = _analysis(p)
    video = p.video()
    scenes = {s["id"]: s for s in analysis["scenes"]}
    picks: list[tuple[str, Path]] = []
    if not scene_ids and not timestamps:
        ids = sorted(scenes)
        step = max(1, len(ids) // max_frames)
        scene_ids = ids[::step][:max_frames]
    for sid in scene_ids or []:
        s = scenes.get(sid)
        if s:
            picks.append((f"scene {sid}  {fmt_tc(s['start'])}-{fmt_tc(s['end'])}", p.root / s["frame"]))
    for t in timestamps or []:
        f = p.dir("frames") / f"t_{t:09.3f}.jpg"
        if not f.exists():
            await asyncio.to_thread(media.grab_frame, video, t, f)
        picks.append((f"t={fmt_tc(t)}", f))
    result: list[Any] = []
    for label, path in picks[:max_frames]:
        result.append(label)
        result.append(Image(path=path))
    return result or ["No frames matched."]


# =========================================================================
# 2. Dialogue: transcription & cast
# =========================================================================
@mcp.tool()
async def transcribe_dubbing(
    project_path: str,
    files: list[str] | None = None,
    num_speakers: int | None = None,
    language_code: str | None = None,
    keyterms: list[str] | None = None,
    max_turn_seconds: float = 30.0,
    ctx: Context | None = None,
) -> str:
    """Transcribe the dubbing files with ElevenLabs Scribe (speaker diarization +
    word timestamps) and split them into single-speaker turns.

    `num_speakers`: how many distinct voices are in a file (helps diarization).
    `keyterms`: character names / unusual words from the script to improve accuracy.
    Returns, per file and speaker, sample lines so you can match speakers to the
    script's characters, then call assign_characters.
    """
    p = Project.open(project_path)
    targets = [p.dubbing_file(f) for f in files] if files else p.dubbing_files()
    if not targets:
        raise RuntimeError(f"No dubbing files in {p.dir('input_dubbing')}")
    out = []
    for i, src in enumerate(targets, 1):
        await _progress(ctx, i - 1, len(targets), f"transcribing {src.name}")
        wav = p.dir("transcripts") / f"{src.stem}.source.wav"
        if not wav.exists():
            await asyncio.to_thread(media.extract_audio, src, wav)
        duration = (await asyncio.to_thread(media.probe, wav))["duration"]
        raw = await el().speech_to_text(wav, num_speakers=num_speakers, language_code=language_code, keyterms=keyterms)
        p.write_json(f"02_analysis/transcripts/{src.stem}.json", raw)
        turns = build_turns(raw.get("words", []), max_len=max_turn_seconds, duration=duration)
        p.write_json(f"02_analysis/transcripts/{src.stem}.turns.json", {"file": src.name, "duration": duration, "turns": turns})
        summary = speaker_summary(turns)
        out.append(f"== {src.name} ({fmt_tc(duration)}, {len(turns)} turns, language={raw.get('language_code')})")
        for spk, s in summary.items():
            out.append(f"  {spk}: {s['turns']} turns, {s['seconds']}s")
            out.extend(f"      {line}" for line in s["sample_lines"])
    await _progress(ctx, len(targets), len(targets), "done")
    out.append(
        "\nMatch each speaker_id to a script character (or tell the user if unsure), then call "
        "assign_characters. If a file contains only one character, assign the whole file."
    )
    return "\n".join(out)


@mcp.tool()
def assign_characters(project_path: str, assignments: list[dict[str, Any]]) -> str:
    """Map dubbing speakers to characters.

    Each item: {"file": "<dubbing file>", "speaker": "speaker_0", "character": "MAYA"}.
    Omit "speaker" to say the whole file is one character. Optional
    "offset_seconds" if the file does not start at 00:00 of the video.
    Use character "SKIP" for speakers that should be left out.
    """
    p = Project.open(project_path)
    cast = _cast(p)
    for a in assignments:
        src = p.dubbing_file(a["file"])
        entry = cast["files"].setdefault(src.name, {"speakers": {}, "character": None, "offset_seconds": 0.0})
        character = str(a["character"]).strip().upper()
        if a.get("speaker"):
            entry["speakers"][a["speaker"]] = character
        else:
            entry["character"] = character
        if "offset_seconds" in a:
            entry["offset_seconds"] = float(a["offset_seconds"])
    p.write_json("03_voices/cast.json", cast)
    lines = ["Cast saved to 03_voices/cast.json:"]
    for f, e in cast["files"].items():
        who = e["character"] or ", ".join(f"{k}->{v}" for k, v in e["speakers"].items())
        lines.append(f"  {f}: {who} (offset {e['offset_seconds']}s)")
    lines.append(f"Characters: {_characters(p)}")
    return "\n".join(lines)


# =========================================================================
# 3. Voices: prompts, options, approval
# =========================================================================
@mcp.tool()
def save_voice_prompts(project_path: str, voices: list[dict[str, Any]]) -> str:
    """Save the voice prompt you wrote for each character.

    Each item: {"character", "prompt", "preview_text"?, "gender"?, "age"?,
    "accent"?, "language"?, "notes"?}.
    A strong voice prompt covers: age, gender, accent/region, timbre (deep, raspy,
    breathy, bright), pace, energy and emotional baseline, and how the character
    should feel on screen — grounded in what you SAW in the frames and READ in the
    script. `preview_text` (100-1000 chars) should be an in-character line from the
    script so previews can be judged in context.
    """
    p = Project.open(project_path)
    data = p.read_json("03_voices/voice_prompts.json", {"characters": {}})
    for v in voices:
        name = str(v["character"]).strip().upper()
        data["characters"][name] = {k: v[k] for k in v if k != "character"} | {"updated_at": now_iso()}
    p.write_json("03_voices/voice_prompts.json", data)
    md = ["# Voice prompts\n"]
    for name, v in data["characters"].items():
        md.append(f"## {name}\n\n{v.get('prompt', '')}\n")
        for k in ("gender", "age", "accent", "language", "notes"):
            if v.get(k):
                md.append(f"- **{k}**: {v[k]}")
        if v.get("preview_text"):
            md.append(f"\n> {v['preview_text']}\n")
    p.write_text("03_voices/voice_prompts.md", "\n".join(md))
    return (
        f"Saved {len(voices)} voice prompt(s) to 03_voices/voice_prompts.json (+ .md). Next: for each character "
        f"call design_voice_options and/or find_library_voices, then list_voice_options for the user to choose."
    )


def _options_path(character: str) -> str:
    return f"03_voices/voice_options/{slugify(character)}/options.json"


def _add_options(p: Project, character: str, new: list[dict]) -> list[dict]:
    opts = p.read_json(_options_path(character), [])
    existing = {o["option_id"] for o in opts}
    for o in new:
        if o["option_id"] not in existing:
            opts.append(o)
    p.write_json(_options_path(character), opts)
    return opts


@mcp.tool()
async def design_voice_options(
    project_path: str,
    character: str,
    prompt: str | None = None,
    preview_text: str | None = None,
    guidance_scale: float | None = None,
    seed: int | None = None,
    model_id: str | None = None,
) -> str:
    """Generate custom voice candidates (ElevenLabs Voice Design — usually 3 per call)
    from the character's saved voice prompt (or `prompt` to override). Preview audio
    is saved under 03_voices/voice_options/<character>/ for the user to listen to.
    Call again with a tweaked prompt/seed for more options.
    """
    p = Project.open(project_path)
    name = character.strip().upper()
    saved = p.read_json("03_voices/voice_prompts.json", {"characters": {}})["characters"].get(name, {})
    description = prompt or saved.get("prompt")
    if not description:
        raise RuntimeError(f"No voice prompt for {name}. Call save_voice_prompts first or pass `prompt`.")
    text = preview_text or saved.get("preview_text")
    if text and not 100 <= len(text) <= 1000:
        text = None  # API requires 100-1000 chars; let it auto-generate instead
    resp = await el().design_voice(description, text=text, model_id=model_id, guidance_scale=guidance_scale, seed=seed)
    folder = p.root / "03_voices/voice_options" / slugify(name)
    n0 = len(p.read_json(_options_path(name), []))
    new = []
    for i, prev in enumerate(resp.get("previews", []), 1):
        oid = f"D{n0 + i:02d}"
        f = folder / f"{oid}_designed{'.wav' if 'wav' in prev.get('media_type', '') else '.mp3'}"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(base64.b64decode(prev["audio_base_64"]))
        new.append(
            {
                "option_id": oid,
                "source": "designed",
                "generated_voice_id": prev["generated_voice_id"],
                "name": f"{name.title()} {oid}",
                "description": description,
                "preview_file": p.rel(f),
                "preview_text": resp.get("text"),
            }
        )
    _add_options(p, name, new)
    return "\n".join(
        [f"Designed {len(new)} voice option(s) for {name}:"]
        + [f"  {o['option_id']}: {p.root / o['preview_file']}" for o in new]
        + ["Ask the user to listen and pick one, then call approve_voice."]
    )


@mcp.tool()
async def find_library_voices(
    project_path: str,
    character: str,
    search: str | None = None,
    gender: str | None = None,
    age: str | None = None,
    accent: str | None = None,
    language: str | None = None,
    use_cases: str | None = None,
    include_my_voices: bool = True,
    limit: int = 6,
) -> str:
    """Find existing ElevenLabs voices (your own + the public Voice Library) that fit
    a character, and download their previews as voice options.

    Filters: gender ('male'/'female'/'neutral'), age ('young'/'middle_aged'/'old'),
    accent (e.g. 'american', 'british', 'indian'), language (ISO code e.g. 'en','hi'),
    use_cases (e.g. 'characters_animation', 'narrative_story', 'conversational').
    """
    p = Project.open(project_path)
    name = character.strip().upper()
    found: list[dict] = []
    if include_my_voices:
        for v in await el().my_voices(search=search, page_size=limit):
            found.append(
                {
                    "source": "my_voice",
                    "voice_id": v["voice_id"],
                    "name": v.get("name"),
                    "description": v.get("description") or ", ".join(f"{k}: {x}" for k, x in (v.get("labels") or {}).items()),
                    "preview_url": v.get("preview_url"),
                }
            )
    shared = await el().shared_voices(
        search=search, gender=gender, age=age, accent=accent, language=language, use_cases=use_cases, page_size=limit
    )
    for v in shared:
        found.append(
            {
                "source": "voice_library",
                "voice_id": v["voice_id"],
                "public_owner_id": v.get("public_owner_id"),
                "name": v.get("name"),
                "description": v.get("description")
                or f"{v.get('gender')}, {v.get('age')}, {v.get('accent')}, {v.get('descriptive')}, {v.get('use_case')}",
                "preview_url": v.get("preview_url"),
            }
        )
    folder = p.root / "03_voices/voice_options" / slugify(name)
    opts = p.read_json(_options_path(name), [])
    known = {o.get("voice_id") for o in opts}
    n0 = len(opts)
    new = []
    for v in found:
        if v["voice_id"] in known or len(new) >= limit * 2:
            continue
        oid = f"L{n0 + len(new) + 1:02d}"
        v["option_id"] = oid
        if v.get("preview_url"):
            try:
                f = folder / f"{oid}_{slugify(v['name'] or 'voice')}.mp3"
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(await el().download(v["preview_url"]))
                v["preview_file"] = p.rel(f)
            except Exception as exc:  # a missing preview shouldn't lose the option
                v["preview_error"] = str(exc)
        new.append(v)
    _add_options(p, name, new)
    if not new:
        return "No new voices matched. Loosen the filters or change `search`."
    return "\n".join(
        [f"Added {len(new)} library voice option(s) for {name}:"]
        + [
            f"  {o['option_id']}: {o['name']} [{o['source']}] — {o['description']}\n"
            f"        preview: {p.root / o['preview_file'] if o.get('preview_file') else o.get('preview_url')}"
            for o in new
        ]
    )


@mcp.tool()
def list_voice_options(project_path: str, character: str | None = None) -> str:
    """List every voice option per character (with preview file paths) for the user to
    approve. Also writes 03_voices/VOICE_OPTIONS.md as an approval sheet.
    """
    p = Project.open(project_path)
    approved = p.read_json("03_voices/approved_voices.json", {})
    chars = [character.strip().upper()] if character else _characters(p)
    lines = ["# Voice options — pick one per character\n"]
    for ch in chars:
        lines.append(f"## {ch}" + (f"  — APPROVED: {approved[ch]['option_id']} {approved[ch]['name']}" if ch in approved else ""))
        opts = p.read_json(_options_path(ch), [])
        if not opts:
            lines.append("  (no options yet — run design_voice_options / find_library_voices)")
        for o in opts:
            preview = str(p.root / o["preview_file"]) if o.get("preview_file") else o.get("preview_url", "-")
            lines.append(f"- **{o['option_id']}** {o['name']} [{o['source']}] — {o.get('description', '')[:140]}\n  - preview: {preview}")
        lines.append("")
    text = "\n".join(lines)
    if not character:
        p.write_text("03_voices/VOICE_OPTIONS.md", text)
    return text + "\nThe user approves with e.g. 'MAYA -> D02'; then call approve_voice."


@mcp.tool()
async def approve_voice(
    project_path: str,
    character: str,
    option_id: str | None = None,
    voice_id: str | None = None,
    stability: float | None = None,
    similarity_boost: float | None = None,
    style: float | None = None,
    use_speaker_boost: bool | None = None,
) -> str:
    """Record the USER's choice of voice for a character. Only call this after the user
    has explicitly picked an option. Designed voices are saved to the user's
    ElevenLabs account; Voice Library voices are added to it.

    Pass `voice_id` instead of `option_id` to use any voice already in the account.
    Optional voice settings (0-1) tune the speech-to-speech conversion.
    """
    p = Project.open(project_path)
    name = character.strip().upper()
    if not option_id and not voice_id:
        raise ValueError("Pass option_id (from list_voice_options) or voice_id.")
    record: dict[str, Any] = {"approved_at": now_iso()}
    if voice_id:
        record |= {"voice_id": voice_id, "name": voice_id, "source": "direct", "option_id": "-"}
    else:
        opt = next((o for o in p.read_json(_options_path(name), []) if o["option_id"] == option_id), None)
        if not opt:
            raise ValueError(f"{option_id} is not an option for {name}. See list_voice_options.")
        project_name = p.read_json("project.json", {}).get("name", p.root.name)
        if opt["source"] == "designed":
            created = await el().create_voice_from_preview(
                f"{project_name} - {name.title()}", opt["description"][:1000], opt["generated_voice_id"]
            )
            vid = created["voice_id"]
        elif opt["source"] == "voice_library":
            added = await el().add_shared_voice(opt["public_owner_id"], opt["voice_id"], f"{project_name} - {name.title()}")
            vid = added.get("voice_id", opt["voice_id"])
        else:
            vid = opt["voice_id"]
        record |= {"voice_id": vid, "name": opt["name"], "source": opt["source"], "option_id": option_id}
    settings = {
        k: v
        for k, v in {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
            "use_speaker_boost": use_speaker_boost,
        }.items()
        if v is not None
    }
    if settings:
        record["voice_settings"] = settings
    approved = p.read_json("03_voices/approved_voices.json", {})
    approved[name] = record
    p.write_json("03_voices/approved_voices.json", approved)
    pending = [c for c in _characters(p) if c not in approved]
    return f"{name} -> {record['name']} (voice_id {record['voice_id']}). Still awaiting approval: {pending or 'none'}."


# =========================================================================
# 4. Dialogue conversion
# =========================================================================
def _dialogue_jobs(p: Project, characters: set[str] | None, files: set[str] | None) -> list[dict]:
    jobs = []
    for fname, entry in _cast(p)["files"].items():
        if files and fname not in files:
            continue
        src = p.dubbing_file(fname)
        turns_doc = p.read_json(f"02_analysis/transcripts/{src.stem}.turns.json")
        if not turns_doc:
            raise RuntimeError(f"{fname} has no transcript yet — run transcribe_dubbing first.")
        for t in turns_doc["turns"]:
            ch = entry.get("character") or entry["speakers"].get(t["speaker"])
            if not ch or ch == "SKIP" or (characters and ch not in characters):
                continue
            jobs.append(
                {
                    "file": fname,
                    "source_wav": p.dir("transcripts") / f"{src.stem}.source.wav",
                    "character": ch,
                    "turn": t,
                    "offset": float(entry.get("offset_seconds", 0.0)),
                }
            )
    return jobs


@mcp.tool()
async def convert_dialogue(
    project_path: str,
    characters: list[str] | None = None,
    files: list[str] | None = None,
    remove_background_noise: bool = False,
    isolate_voice_first: bool = False,
    regenerate: bool = False,
    ctx: Context | None = None,
) -> str:
    """Re-voice the dubbing: every turn is cut out, sent through ElevenLabs
    speech-to-speech with the character's APPROVED voice (keeping the original
    performance, timing and emotion), and placed back at its exact timecode.

    Writes per character, all full-length and in sync with the video:
      03_voices/stems/<character>.wav            converted voice
      03_voices/stems/original/<character>.wav   the untouched original, split out
      03_voices/dialogue_full.wav                all converted characters together
    Already-converted segments are reused (safe to re-run after an interruption).
    """
    p = Project.open(project_path)
    analysis = _analysis(p)
    approved = p.read_json("03_voices/approved_voices.json", {})
    wanted = {c.strip().upper() for c in characters} if characters else None
    jobs = _dialogue_jobs(p, wanted, set(files) if files else None)
    if not jobs:
        return "Nothing to convert. Check assign_characters / the characters filter."
    missing = sorted({j["character"] for j in jobs} - set(approved))
    if missing:
        return (
            f"Blocked: no approved voice yet for {missing}. Show the user list_voice_options and wait "
            f"for their pick before converting."
        )

    ext = _ext_for_output()
    done = 0

    async def convert(job: dict) -> dict:
        nonlocal done
        ch, t = job["character"], job["turn"]
        folder = p.dir("voice_segments") / slugify(ch)
        stem = f"{Path(job['file']).stem}_{t['index']:04d}"
        src = folder / f"{stem}_orig.wav"
        out = folder / f"{stem}_conv{ext}"
        if not src.exists():
            await asyncio.to_thread(media.extract_audio, job["source_wav"], src, t["start"], t["end"])
        if regenerate or not out.exists():
            send = src
            if isolate_voice_first:
                iso = folder / f"{stem}_iso.mp3"
                if regenerate or not iso.exists():
                    iso.write_bytes(await _limited(el().isolate_voice(src)))
                send = iso
            a = approved[ch]
            data = await _limited(
                el().speech_to_speech(
                    a["voice_id"], send, remove_background_noise=remove_background_noise, voice_settings=a.get("voice_settings")
                )
            )
            _write_audio(out, data)
        done += 1
        await _progress(ctx, done, len(jobs), f"{ch}: {t['text'][:40]}")
        return {**job, "orig": src, "conv": out, "timeline_start": job["offset"] + t["start"]}

    results = await asyncio.gather(*(convert(j) for j in jobs), return_exceptions=True)
    failures = [(j, r) for j, r in zip(jobs, results) if isinstance(r, Exception)]
    ok = [r for r in results if not isinstance(r, Exception)]

    duration = analysis["duration"]
    by_char: dict[str, list[dict]] = {}
    for r in ok:
        by_char.setdefault(r["character"], []).append(r)
    stems = []
    for ch, items in sorted(by_char.items()):
        conv_clips = [media.Clip(r["conv"], r["timeline_start"], fade_in=0.01, fade_out=0.02) for r in items]
        orig_clips = [media.Clip(r["orig"], r["timeline_start"]) for r in items]
        stem = await asyncio.to_thread(media.render_timeline, conv_clips, duration, p.dir("voice_stems") / f"{slugify(ch)}.wav")
        await asyncio.to_thread(media.render_timeline, orig_clips, duration, p.dir("voice_stems") / "original" / f"{slugify(ch)}.wav")
        stems.append(stem)
    all_stems = sorted(p.dir("voice_stems").glob("*.wav"))
    if all_stems:
        await asyncio.to_thread(media.sum_stems, [(s, 0.0) for s in all_stems], p.dir("voices") / "dialogue_full.wav")

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["character", "file", "turn", "timeline_in", "timeline_out", "text", "converted_file"])
    for r in sorted(ok, key=lambda r: r["timeline_start"]):
        t = r["turn"]
        w.writerow([r["character"], r["file"], t["index"], fmt_tc(r["timeline_start"]),
                    fmt_tc(r["timeline_start"] + t["end"] - t["start"]), t["text"], p.rel(r["conv"])])
    p.write_text("03_voices/dialogue_cue_sheet.csv", buf.getvalue())

    lines = [f"Converted {len(ok)}/{len(jobs)} dialogue turns."]
    lines += [f"  {p.rel(s)}" for s in stems]
    lines.append("  03_voices/dialogue_full.wav, 03_voices/dialogue_cue_sheet.csv")
    if failures:
        lines.append(f"{len(failures)} failed (re-run convert_dialogue to retry just those):")
        lines += [f"  {j['character']} turn {j['turn']['index']}: {str(e)[:200]}" for j, e in failures[:10]]
    return "\n".join(lines)


# =========================================================================
# 5. Soundscape
# =========================================================================
@mcp.tool()
def save_soundscape_plan(project_path: str, cues: list[dict[str, Any]], notes: str | None = None) -> str:
    """Save the sound-design plan you wrote for the whole video.

    Each cue: {"id"?, "layer": "ambience"|"room_tone"|"sfx"|"foley"|"transition",
    "start": sec, "duration": sec (or "end"), "prompt": str, "scene"?, "loop"?,
    "prompt_influence"? (0-1), "gain_db"?, "fade_in"?, "fade_out"?, "notes"?}.

    Design it like a sound designer: a continuous ambience/room-tone bed per
    location (loops, may span many seconds), plus spot SFX and foley synced to
    on-screen action (footsteps, doors, cloth, impacts), and transitions/whooshes
    on cuts. Prompts should be concrete and acoustic: source, material, distance,
    space/reverb, intensity, e.g. "close-up leather boots on wet cobblestone,
    slow pace, light reverb of a narrow alley at night". One-shot SFX are limited
    to 30s; longer ambiences are generated as seamless loops and repeated.
    """
    p = Project.open(project_path)
    plan = normalize_sound_cues(cues, _analysis(p)["duration"])
    p.write_json("04_soundscape/soundscape_plan.json", {"notes": notes or "", "updated_at": now_iso(), "cues": plan})
    md = ["# Soundscape plan\n", notes or "", "\n| id | layer | in | out | prompt |", "|---|---|---|---|---|"]
    for c in plan:
        md.append(f"| {c['id']} | {c['layer']} | {fmt_tc(c['start'])} | {fmt_tc(c['start'] + c['duration'])} | {c['prompt']} |")
    p.write_text("04_soundscape/soundscape_plan.md", "\n".join(md))
    layers: dict[str, int] = {}
    for c in plan:
        layers[c["layer"]] = layers.get(c["layer"], 0) + 1
    return f"Saved {len(plan)} cues {layers} to 04_soundscape/soundscape_plan.json (+ .md). Next: generate_soundscape."


@mcp.tool()
async def generate_soundscape(
    project_path: str, cue_ids: list[str] | None = None, regenerate: bool = False, ctx: Context | None = None
) -> str:
    """Generate every soundscape cue with ElevenLabs Sound Effects, download them to
    04_soundscape/cues/, and lay them on the timeline:
      04_soundscape/stems/<layer>.wav, 04_soundscape/soundscape_full.wav,
      04_soundscape/cue_sheet.csv
    Existing cue files are reused unless `regenerate` (optionally limited to `cue_ids`).
    """
    p = Project.open(project_path)
    plan = p.read_json("04_soundscape/soundscape_plan.json")
    if not plan:
        raise RuntimeError("No soundscape plan. Call save_soundscape_plan first.")
    cues = plan["cues"]
    ext = _ext_for_output()
    done = 0

    def cue_file(c: dict) -> Path:
        return p.dir("soundscape_cues") / f"{c['id']}_{c['layer']}_{slugify(c['prompt'])[:40]}{ext}"

    async def make(c: dict) -> Path:
        nonlocal done
        f = cue_file(c)
        force = regenerate and (not cue_ids or c["id"] in cue_ids)
        if force or not f.exists():
            gen_len = min(max(c["duration"], 0.5), 30.0)
            data = await _limited(
                el().sound_effect(c["prompt"], duration_seconds=gen_len, loop=c["loop"], prompt_influence=c["prompt_influence"])
            )
            _write_audio(f, data)
        done += 1
        await _progress(ctx, done, len(cues), f"{c['id']} {c['prompt'][:40]}")
        return f

    results = await asyncio.gather(*(make(c) for c in cues), return_exceptions=True)
    duration = _analysis(p)["duration"]
    by_layer: dict[str, list[media.Clip]] = {}
    failures = []
    for c, r in zip(cues, results):
        if isinstance(r, Exception):
            failures.append(f"{c['id']}: {str(r)[:200]}")
            continue
        by_layer.setdefault(c["layer"], []).append(
            media.Clip(r, c["start"], gain_db=c["gain_db"], fade_in=c["fade_in"], fade_out=c["fade_out"], length=c["duration"], loop=c["loop"])
        )
    stems = []
    for layer, clips in sorted(by_layer.items()):
        stems.append(await asyncio.to_thread(media.render_timeline, clips, duration, p.dir("soundscape_stems") / f"{layer}.wav"))
    if stems:
        await asyncio.to_thread(media.sum_stems, [(s, 0.0) for s in stems], p.dir("soundscape") / "soundscape_full.wav")

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "layer", "timeline_in", "timeline_out", "loop", "gain_db", "prompt", "file"])
    for c in cues:
        w.writerow([c["id"], c["layer"], fmt_tc(c["start"]), fmt_tc(c["start"] + c["duration"]), c["loop"], c["gain_db"], c["prompt"], p.rel(cue_file(c))])
    p.write_text("04_soundscape/cue_sheet.csv", buf.getvalue())

    lines = [f"Generated {len(cues) - len(failures)}/{len(cues)} soundscape cues."]
    lines += [f"  stem: {p.rel(s)}" for s in stems]
    lines.append("  04_soundscape/soundscape_full.wav, 04_soundscape/cue_sheet.csv")
    if failures:
        lines.append("Failed (re-run to retry):")
        lines += [f"  {f}" for f in failures]
    return "\n".join(lines)


# =========================================================================
# 6. Music
# =========================================================================
@mcp.tool()
def save_music_plan(project_path: str, cues: list[dict[str, Any]], notes: str | None = None) -> str:
    """Save the score plan you wrote, cut to the edit.

    Each cue: {"id"?, "start": sec, "end": sec (or "duration"), "prompt": str,
    "force_instrumental"? (default true), "gain_db"? (default -8), "fade_in"?,
    "fade_out"?, "sync_points"?: [{"time": sec, "event": "..."}], "notes"?,
    "composition_plan"?: ElevenLabs composition plan object for section-level control}.

    Spot it like a composer: start/end cues on scene cuts and emotional turns (use
    the scene list and audio energy curve from analyze_project), and leave space
    under dialogue. Prompts should state genre, instrumentation, tempo (BPM), key/
    mood, dynamics and structure tied to the cue's length, e.g. "tense minimal
    synth pulse, 90 BPM, D minor, low strings swell into a hit at 0:24, ends on a
    sustained drone". Each cue: 3s - 600s.
    """
    p = Project.open(project_path)
    plan = normalize_music_cues(cues, _analysis(p)["duration"])
    p.write_json("05_music/music_plan.json", {"notes": notes or "", "updated_at": now_iso(), "cues": plan})
    md = ["# Music plan\n", notes or "", "\n| id | in | out | prompt |", "|---|---|---|---|"]
    for c in plan:
        md.append(f"| {c['id']} | {fmt_tc(c['start'])} | {fmt_tc(c['end'])} | {c['prompt'] or '(composition plan)'} |")
    p.write_text("05_music/music_plan.md", "\n".join(md))
    return f"Saved {len(plan)} music cue(s) to 05_music/music_plan.json (+ .md). Next: generate_music."


@mcp.tool()
async def generate_music(
    project_path: str, cue_ids: list[str] | None = None, regenerate: bool = False, ctx: Context | None = None
) -> str:
    """Compose each music cue with ElevenLabs Music at the cue's exact length,
    download to 05_music/cues/, then lay them on the timeline with fades:
      05_music/music_full.wav, 05_music/cue_sheet.csv
    """
    p = Project.open(project_path)
    plan = p.read_json("05_music/music_plan.json")
    if not plan:
        raise RuntimeError("No music plan. Call save_music_plan first.")
    cues = plan["cues"]
    done = 0

    def cue_file(c: dict) -> Path:
        return p.dir("music_cues") / f"{c['id']}.mp3"

    async def make(c: dict) -> Path:
        nonlocal done
        f = cue_file(c)
        force = regenerate and (not cue_ids or c["id"] in cue_ids)
        if force or not f.exists():
            length_ms = int(min(max(c["duration"], 3.0), 600.0) * 1000)
            data = await _limited(
                el().compose_music(
                    prompt=c["prompt"],
                    music_length_ms=length_ms,
                    composition_plan=c.get("composition_plan"),
                    force_instrumental=c["force_instrumental"],
                )
            )
            f.write_bytes(data)
        done += 1
        await _progress(ctx, done, len(cues), f"{c['id']}")
        return f

    results = await asyncio.gather(*(make(c) for c in cues), return_exceptions=True)
    clips, failures = [], []
    for c, r in zip(cues, results):
        if isinstance(r, Exception):
            failures.append(f"{c['id']}: {str(r)[:200]}")
            continue
        clips.append(media.Clip(r, c["start"], gain_db=c["gain_db"], fade_in=c["fade_in"], fade_out=c["fade_out"], length=c["duration"]))
    if clips:
        await asyncio.to_thread(media.render_timeline, clips, _analysis(p)["duration"], p.dir("music") / "music_full.wav")

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "timeline_in", "timeline_out", "gain_db", "prompt", "file"])
    for c in cues:
        w.writerow([c["id"], fmt_tc(c["start"]), fmt_tc(c["end"]), c["gain_db"], c["prompt"], p.rel(cue_file(c))])
    p.write_text("05_music/cue_sheet.csv", buf.getvalue())

    lines = [f"Generated {len(clips)}/{len(cues)} music cues -> 05_music/cues/, 05_music/music_full.wav"]
    if failures:
        lines.append("Failed (re-run to retry):")
        lines += [f"  {f}" for f in failures]
    return "\n".join(lines)


# =========================================================================
# 7. Final preview
# =========================================================================
@mcp.tool()
async def render_preview(
    project_path: str,
    dialogue_gain_db: float = 0.0,
    soundscape_gain_db: float = -3.0,
    music_gain_db: float = -2.0,
    make_video: bool = True,
) -> str:
    """Combine dialogue + soundscape + music into 06_final/final_mix.wav and (optionally)
    put it on the video as 06_final/preview.mp4 for review.
    """
    p = Project.open(project_path)
    parts = [
        (p.dir("voices") / "dialogue_full.wav", dialogue_gain_db),
        (p.dir("soundscape") / "soundscape_full.wav", soundscape_gain_db),
        (p.dir("music") / "music_full.wav", music_gain_db),
    ]
    present = [(f, g) for f, g in parts if f.exists()]
    if not present:
        return "Nothing to mix yet."
    mix = await asyncio.to_thread(media.sum_stems, present, p.dir("final") / "final_mix.wav")
    lines = [f"Mixed {[p.rel(f) for f, _ in present]} -> {p.rel(mix)}"]
    if make_video:
        out = await asyncio.to_thread(media.mux_preview, p.video(), mix, p.dir("final") / "preview.mp4")
        lines.append(f"Preview video: {out}")
    return "\n".join(lines)


# =========================================================================
# Prompts
# =========================================================================
@mcp.prompt()
def audio_post_workflow(project_path: str) -> str:
    """Full step-by-step: voices, soundscape and music for one video."""
    return f"""You are the dialogue editor, sound designer and composer for the project at {project_path}.
Work through these stages with the elevenlabs-dubbing-studio tools. Keep the user in the loop.

0. project_status (list_projects shows existing ones; a bare name like "MyFilm" lands in the
   configured projects folder). If it doesn't exist, create_project, then open_in_finder on
   01_input so the user can drag in the video, script and dubbing; stop until they confirm.
1. analyze_project. Read the whole script (read_script if it was truncated). Then view_frames
   across ALL scenes (several calls) so you know locations, time of day, characters' looks,
   action beats and the mood of each scene. Take notes per scene.
2. transcribe_dubbing (pass the script's character names as keyterms, num_speakers if known).
   Match each speaker_id to a script character by comparing sample lines to the script.
   Confirm the mapping with the user if anything is ambiguous, then assign_characters.
3. Write one voice prompt per character (age, gender, accent, timbre, pace, energy, attitude,
   grounded in frames + script) and save_voice_prompts. Show them to the user.
4. For each character: design_voice_options (custom voices) and find_library_voices (existing
   voices). Then list_voice_options and ASK THE USER TO LISTEN AND CHOOSE (offer open_in_finder on
   the character's voice_options folder or a specific preview file). Do not choose for them.
   Offer more options (new prompt/seed/filters) until they are happy.
5. approve_voice for each choice, then convert_dialogue. Report the stems produced.
6. Write the soundscape plan scene by scene: an ambience/room-tone bed for every location,
   spot SFX/foley for every visible action, transitions on hard cuts. Use exact timecodes from
   the scene list and frames (view_frames at specific timestamps to pin actions).
   save_soundscape_plan, show the table, then generate_soundscape.
7. Spot the music to the edit: cue in/out on scene changes and emotional turns, using the audio
   energy curve and cut rhythm; describe genre, instrumentation, BPM, key, dynamics and hits
   at sync points. save_music_plan, show it, then generate_music.
8. render_preview and give the user the list of output folders and files.
Re-generate any single cue or voice the user dislikes (regenerate=True with cue_ids).
"""


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
