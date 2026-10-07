# ElevenLabs Dubbing Studio — MCP server

An MCP connector that turns Claude into your dialogue editor, sound designer and
composer, with ElevenLabs doing the generation.

You upload a **finished video render**, the **script** and the **dubbing audio**.
Claude watches the video (key frames), reads the script and the dubbing transcript, and writes
every prompt. The server:

1. **Voices**: splits the dubbing by character (speaker diarization), writes a voice prompt per
   character, gives you a **list of voice options to listen to and approve** (custom-designed
   voices plus matches from the ElevenLabs Voice Library), then re-voices each character's lines
   with the voice you approved (speech-to-speech, so the original timing and emotion are kept).
2. **Soundscape**: designs the full sound bed: ambience and room tone per location, foley,
   spot SFX and transitions, all timed to the edit.
3. **Music**: spots and composes a score cue by cue to the cuts and emotional beats of the edit.

Everything is downloaded into the project folder you choose. Every file is full-length and in
sync with the video, so you can drop it straight onto your NLE timeline.

---

## Project folder

`create_project` builds this structure at any path you give it:

```
MyFilm/
├── 01_input/
│   ├── video/       ← put the full video render here
│   ├── script/      ← script: .txt .md .fountain .fdx .pdf .docx .srt
│   └── dubbing/     ← the dubbing: one full dialogue mix OR one file per character
├── 02_analysis/     scene cuts, key frames, audio energy curve, diarized transcripts
├── 03_voices/
│   ├── voice_prompts.md / .json      the voice prompt Claude wrote per character
│   ├── VOICE_OPTIONS.md              approval sheet (option id + preview file per character)
│   ├── voice_options/<character>/    preview audio for every candidate voice
│   ├── approved_voices.json          your picks
│   ├── segments/<character>/         each line: original + converted
│   ├── stems/<character>.wav         converted voice, full-length, in sync
│   ├── stems/original/<character>.wav  original performance split per character
│   ├── dialogue_full.wav
│   └── dialogue_cue_sheet.csv
├── 04_soundscape/
│   ├── soundscape_plan.md / .json    the sound design plan (every cue + prompt + timecode)
│   ├── cues/                         every generated ambience / SFX / foley file
│   ├── stems/<layer>.wav             ambience, room_tone, foley, sfx, transition stems
│   ├── soundscape_full.wav
│   └── cue_sheet.csv
├── 05_music/
│   ├── music_plan.md / .json         the score plan
│   ├── cues/                         every generated music cue
│   ├── music_full.wav                cues laid out to the edit with fades
│   └── cue_sheet.csv
└── 06_final/
    ├── final_mix.wav                 dialogue + soundscape + music
    └── preview.mp4                   your video with the new mix, for review
```

## Install (no terminal)

1. Download **[`dist/elevenlabs-dubbing-studio.mcpb`](dist/elevenlabs-dubbing-studio.mcpb)**
   (on GitHub: open the file → **Download raw file**).
2. Open **Claude Desktop** → **Settings → Extensions**, and drag the `.mcpb` file in (or just
   double-click it). Click **Install**.
3. Fill in the settings form:
   - **ElevenLabs API key**: from elevenlabs.io → Profile → API keys
   - **Projects folder**: where your projects will live, e.g. `Documents/AudioProjects`
   - **Audio output format**: leave `mp3_44100_128` unless you're on Pro (`pcm_44100` = WAV)
4. Enable the extension. The first start takes a minute, because Claude Desktop downloads Python,
   the dependencies and a bundled ffmpeg automatically. After that it starts instantly.

You don't need Python, ffmpeg or the terminal. To update, install a newer `.mcpb` the same way.

<details>
<summary>Manual install (developers)</summary>

Requirements: Python 3.10+ (ffmpeg is bundled via `imageio-ffmpeg`; a system ffmpeg is used if found).

```bash
git clone https://github.com/Peekay1601/audio.git && cd audio
uv venv && uv pip install -e .
```

Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "elevenlabs-dubbing-studio": {
      "command": "/ABSOLUTE/PATH/TO/audio/.venv/bin/elevenlabs-dubbing-studio",
      "env": { "ELEVENLABS_API_KEY": "sk_...", "AUDIO_PROJECTS_ROOT": "/Users/you/AudioProjects" }
    }
  }
}
```

Claude Code: `claude mcp add elevenlabs-dubbing-studio -e ELEVENLABS_API_KEY=sk_... -e AUDIO_PROJECTS_ROOT=~/AudioProjects -- /ABSOLUTE/PATH/TO/audio/.venv/bin/elevenlabs-dubbing-studio`

Rebuild the extension: `npx @anthropic-ai/mcpb pack . dist/elevenlabs-dubbing-studio.mcpb`
</details>

## How to use it

1. In a Claude Desktop chat, say: *"Create an audio project called MyFilm and open the input folder"*.
   It's created inside your projects folder, and Finder / File Explorer opens on `01_input/`.
2. Drag your video, script and dubbing into the `video/`, `script/` and `dubbing/` folders.
3. Run the **`audio_post_workflow`** prompt (in Claude Desktop: the **+** / attach menu → the
   server's prompts), or just say *"Do the full audio post for ~/AudioProjects/MyFilm"*.
4. Claude will analyse the video, match speakers to characters, write voice prompts and
   generate voice options. Say *"play MAYA's options"* and Claude opens the previews for you
   (or see `03_voices/VOICE_OPTIONS.md`), then tell Claude your picks (e.g. *"MAYA → D02, ARJUN → L03"*). Nothing is converted until you
   approve.
5. Claude converts the dialogue, then writes and generates the soundscape and the music.
   Review the plans (`soundscape_plan.md`, `music_plan.md`) and ask for changes. Any single
   cue can be regenerated.

### Tools

| Stage | Tool | What it does |
|---|---|---|
| Setup | `create_project`, `list_projects`, `project_status` | Make the folder tree; list projects; show progress and next step |
| | `open_in_finder` | Opens a project folder or file (e.g. a voice preview) on your computer |
| Analyse | `analyze_project` | Scene cuts, key frames, audio energy curve, script + character guess |
| | `read_script`, `view_frames` | Claude reads the script and *looks at* the video |
| Dialogue | `transcribe_dubbing` | ElevenLabs Scribe: diarization + word timestamps → speaker turns |
| | `assign_characters` | speaker → character (or whole file → character) |
| Voices | `save_voice_prompts` | Saves Claude's voice prompt per character |
| | `design_voice_options` | ElevenLabs Voice Design: custom candidates from the prompt |
| | `find_library_voices` | Your voices + the public Voice Library, previews downloaded |
| | `list_voice_options` | Approval sheet for you |
| | `approve_voice` | Records **your** choice (saves it to your ElevenLabs account) |
| | `convert_dialogue` | Speech-to-speech per line, rebuilt as synced stems per character |
| Soundscape | `save_soundscape_plan`, `generate_soundscape` | Sound design plan → ElevenLabs SFX → layered stems |
| Music | `save_music_plan`, `generate_music` | Score plan → ElevenLabs Music → cues laid to the edit |
| Final | `render_preview` | Final mix WAV + preview MP4 |

## Notes and limits

- **Dubbing timing.** Dubbing files are assumed to start at 00:00 of the video. If one doesn't,
  tell Claude the offset (`assign_characters` takes `offset_seconds`).
- **Overlapping dialogue.** Diarization gives each word to one speaker, so lines where two
  characters talk over each other land in one character's stem. With one file per character,
  this problem goes away.
- **Long scenes.** ElevenLabs sound effects are at most 30s, so longer ambiences are generated
  as seamless loops and repeated to fill the cue. Each music cue can be 3s to 10 min long.
- **Plan tiers.** The default output is `mp3_44100_128`, which works on every plan. Set
  `ELEVENLABS_OUTPUT_FORMAT=pcm_44100` (Pro+) to get uncompressed audio. Voice Library voices
  and Voice Design count against your account's voice slots.
- **Re-running.** Every generate/convert step skips files that already exist, so if something
  fails or is interrupted, just run it again. Pass `regenerate=true` (with `cue_ids`) to redo
  specific cues.
- **Costs.** Transcription, voice design, speech-to-speech, SFX and music all use ElevenLabs
  credits.

## Development

```bash
uv pip install -e '.[dev]'
pytest
```

The tests use a fake ElevenLabs client and synthetic media, so they make no API calls.
