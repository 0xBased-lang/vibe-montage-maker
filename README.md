# Vibe Montage Maker

Eye + Brain MVP for collecting high-quality screenshots from YouTube videos and curating them by “vibe.” Outputs CapCut-ready kits and (optionally) prebuilt drafts via CapCutAPI.

## What it does
- **Eye**: Download video (720p for speed), detect scene cuts (PySceneDetect), extract one mid-scene frame per cut to avoid blur/fades.
- **Brain**: Embed frames with Google SigLIP (via Transformers) and store vectors in ChromaDB for semantic (“vibe”) search.
- **Exports**: Ranked and sequential assets, metadata, click-track audio, script stub, manifest, and a shareable zip. Optional automation to build a CapCut draft via CapCutAPI.

## Project structure
```
.
├── requirements.txt
├── src
│   ├── ingest.py          # Eye – download, scene detect, embed into vector DB
│   ├── search.py          # Brain UI – text search, export kits
│   └── capcut_draft.py    # Optional – send kits to CapCutAPI, build draft
├── downloads/             # auto-created; raw video files (git-ignored)
├── frames/                # auto-created; extracted frames (git-ignored)
├── chroma_db/             # auto-created; vector database (git-ignored)
├── organized_screenshots/ # ranked exports per vibe (git-ignored)
└── capcut_ready/          # CapCut kits (sequential assets, CSV, click track, zip; git-ignored)
```

## Requirements
- Python 3.10–3.13 (tested on macOS).
- `pip install -r requirements.txt` (yt-dlp, scenedetect[opencv], transformers, torch, chromadb, requests, etc.).
- CapCut desktop/mobile for manual import; CapCutAPI (optional) for draft automation.

## Quick start (non-technical friendly)
```bash
git clone https://github.com/0xBased-lang/vibe-montage-maker
cd vibe-montage-maker
pip install -r requirements.txt

# 1) Ingest a video
python src/ingest.py
# paste a YouTube URL when prompted

# 2) Export a vibe
python src/search.py
# describe a vibe (e.g., "neon city"), choose how many results, then BPM (default 120)
# outputs kits under organized_screenshots/<vibe>/ and capcut_ready/<vibe>/
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

## Importing into CapCut (manual, safest)
1) Open CapCut.
2) Drag `capcut_ready/<vibe>/` (or unzip and drag) into the media bin.
3) Drop `0001.jpg`, `0002.jpg`, … into the timeline; optionally drop `click_track.wav` to align beats.
4) Use `shots.csv` / `script_stub.md` for pacing and voiceover planning.

## Optional: auto-build a CapCut draft (CapCutAPI)
Prereq: run CapCutAPI (https://github.com/sun-guannan/CapCutAPI) locally; default base `http://127.0.0.1:3000`.
Command:
```bash
# after ingest + search
python src/capcut_draft.py <vibe> --api-base http://127.0.0.1:3000
```
What it does:
- Calls CapCutAPI endpoints (`create_draft`, `add_video`, `add_audio`, `save_draft`).
- Places each sequential frame at beat-aligned duration from `capcut_manifest.json`.
- Adds `click_track.wav` across the full timeline (if present).
- Outputs a `dfd_*` draft folder; copy it into your CapCut/Jianying drafts directory and open in CapCut.
See `docs/CAPCUT_API_USAGE.md` for endpoint assumptions and troubleshooting.

## Defaults and behavior
- Ingestion resolution: 720p (fast, sufficient for frame selection).
- Scene detection: PySceneDetect, threshold ~27, mid-scene frame extraction for stability.
- BPM default: 120 (adjust at export prompt).
- Data stays local: heavy folders are .gitignored (`downloads/`, `frames/`, `chroma_db/`, `organized_screenshots/`, `capcut_ready/`).

## Troubleshooting
- Missing matches: broaden the vibe text or request more results.
- Slow downloads: network throttling from YouTube; try again or use a shorter video.
- CapCutAPI errors: ensure server is running and `--api-base` matches; see `docs/CAPCUT_API_USAGE.md`.
- Paths: prefer absolute or simple paths when using CapCutAPI to avoid resolution issues.

## Roadmap
- Add “Flow” phase (color/brightness sorting, beat-synced montage assembly).
- Batch ingestion for playlists.
- Auto-generate CapCut `.capcutproj` templates (see `docs/CAPCUT_TEMPLATE_PLAN.md`).
- Optional: swap click-track with chosen soundtrack automatically.
