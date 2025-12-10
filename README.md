# Vibe Montage Maker

Eye + Brain MVP for collecting high-quality screenshots from online videos (YouTube/TikTok/Instagram/others via yt-dlp) and curating them by “vibe.” Outputs CapCut-ready kits, quick MP4 montages, and (optionally) prebuilt drafts via CapCutAPI.

## What it does
- **Eye**: Download video (720p for speed), detect scene cuts (PySceneDetect), extract one mid-scene frame per cut to avoid blur/fades.
- **Brain**: Embed frames with Google SigLIP (via Transformers) and store vectors in ChromaDB for semantic (“vibe”) search.
- **Exports**: Ranked and sequential assets, metadata, click-track audio, script stub, manifest, shareable zip, and optional MP4 montage. Optional automation to build a CapCut draft via CapCutAPI.

## Project structure
```
.
├── requirements.txt
├── run.py               # One-shot pipeline runner (ingest -> export -> mp4 -> optional draft)
├── src
│   ├── ingest.py        # Eye – download, scene detect, embed into vector DB (supports cookies file)
│   ├── search.py        # Brain UI – text search, export kits
│   ├── capcut_draft.py  # Optional – send kits to CapCutAPI, build draft
│   └── make_movie.py    # Optional – build quick MP4 montage from frames
├── downloads/           # auto-created; raw video files (git-ignored)
├── frames/              # auto-created; extracted frames (git-ignored)
├── chroma_db/           # auto-created; vector database (git-ignored)
├── organized_screenshots/ # ranked exports per vibe (git-ignored)
├── capcut_ready/        # CapCut kits (sequential assets, CSV, click track, zip; git-ignored)
└── renders/             # optional MP4 outputs (git-ignored)
```

## Requirements
- Python 3.10–3.13 (tested on macOS).
- `pip install -r requirements.txt` (yt-dlp, scenedetect[opencv], transformers, torch, chromadb, requests, moviepy, etc.).
- ffmpeg/avlib available for moviepy (install via `brew install ffmpeg` on macOS if missing).
- CapCut desktop/mobile for manual import; CapCutAPI (optional) for draft automation.

## One-shot (recommended)
```bash
git clone https://github.com/0xBased-lang/vibe-montage-maker
cd vibe-montage-maker
pip install -r requirements.txt

python run.py \
  --url "<VIDEO_URL>" \
  --vibe "neon city" \
  --count 8 \
  --bpm 120 \
  [--cookies /path/to/cookies.txt] \
  [--draft --api-base http://127.0.0.1:3000]
```
- Works with YouTube/TikTok/Instagram URLs (yt-dlp). Use `--cookies` for IG or gated content.
- Outputs:
  - `organized_screenshots/<vibe>/` (ranked)
  - `capcut_ready/<vibe>/` (sequential kit + zip)
  - `renders/<vibe>.mp4` (montage) unless `--no-mp4`
  - Optional CapCut draft via CapCutAPI if `--draft` is set

## Manual flow (if you prefer prompts)
```bash
python src/ingest.py      # paste URL (supports cookies via COOKIES_FILE env)
python src/search.py      # type vibe, count, BPM
python src/make_movie.py capcut_ready/<vibe>   # optional MP4
```

## What you get per vibe
In `organized_screenshots/<vibe>/`:
- Ranked copies: `01_*.jpg`, `02_*.jpg`, …

In `capcut_ready/<vibe>/`:
- Sequential frames: `0001.jpg`, `0002.jpg`, …
- `shots.csv`: rank, similarity score, source video ID, timestamp, original path.
- `script_stub.md`: notes for narration/captions per shot.
- `capcut_manifest.json`: BPM, beat duration, suggested clip durations, source metadata.
- `click_track.wav`: metronome audio (BPM you chose) for beat alignment.
- `<vibe>_capcut.zip`: archive of the entire kit for sharing.

Optional MP4 preview:
- `python src/make_movie.py capcut_ready/<vibe>` → H.264 MP4 at 24 fps (default per-frame duration from manifest beat duration if present; otherwise 1.0s). Override with `--duration`; attach audio with `--audio`.

## Importing into CapCut (manual, safest)
1) Open CapCut.
2) Drag `capcut_ready/<vibe>/` (or unzip and drag) into the media bin.
3) Drop `0001.jpg`, `0002.jpg`, … into the timeline; optionally drop `click_track.wav` to align beats.
4) Use `shots.csv` / `script_stub.md` for pacing and voiceover planning.

## Optional: auto-build a CapCut draft (CapCutAPI)
Prereq: run CapCutAPI (https://github.com/sun-guannan/CapCutAPI) locally; default base `http://127.0.0.1:3000`.
Command:
```bash
python src/capcut_draft.py <vibe> --api-base http://127.0.0.1:3000 [--audio file] [--ratio 9:16] [--fps 30]
```
What it does:
- Calls CapCutAPI endpoints (`create_draft`, `add_video`, `add_audio`, `save_draft`).
- Places each sequential frame at beat-aligned duration from `capcut_manifest.json`.
- Adds `click_track.wav` across the full timeline (if present).
- Outputs a `dfd_*` draft folder; copy it into your CapCut/Jianying drafts directory and open in CapCut.
See `docs/CAPCUT_API_USAGE.md` for endpoint assumptions and troubleshooting.

## Cookies (Instagram/TikTok gated content)
- Use `--cookies /path/to/cookies.txt` with `run.py`, or set `COOKIES_FILE` env before running `src/ingest.py`.
- Format: a standard Netscape cookies file exported from your browser.

## Defaults and behavior
- Ingestion: 720p; PySceneDetect threshold ~27; mid-scene frames.
- BPM: default 120 (prompt overridable).
- MP4: H.264, 24 fps, per-frame duration from manifest or 1.0s fallback.
- Data stays local: heavy folders are .gitignored (`downloads/`, `frames/`, `chroma_db/`, `organized_screenshots/`, `capcut_ready/`, `renders/`).

## Troubleshooting
- No matches or few results: broaden vibe text or request more results.
- Slow downloads: YouTube/TikTok throttling; try again or shorter video.
- CapCutAPI errors: ensure server reachable at `--api-base`; see `docs/CAPCUT_API_USAGE.md`.
- ffmpeg missing: install (`brew install ffmpeg` on macOS) for moviepy.
- Paths: when using CapCutAPI, prefer absolute/simple paths; pass cookies for IG if required.

## Roadmap
- Add “Flow” phase (color/brightness sorting, beat-synced montage assembly).
- Batch ingestion for playlists.
- Auto-generate CapCut `.capcutproj` templates (see `docs/CAPCUT_TEMPLATE_PLAN.md`).
- Optional: swap click-track with chosen soundtrack automatically.
