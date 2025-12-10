# Vibe Montage Maker

Eye + Brain MVP for collecting high-quality screenshots from YouTube videos and curating them by "vibe".

## Features
- **Scene-aware Screenshot Extraction**: downloads a video, finds every hard scene cut with PySceneDetect, and saves one stable frame per scene.
- **Semantic Indexing**: each frame is embedded with Google's SigLIP model, stored in a local ChromaDB vector database.
- **Natural-Language Retrieval**: describe any vibe ("neon city rain", "serene countryside"), and the system exports matching frames into an organized folder, ready for creative use.
- **CapCut-Ready Packages**: each vibe query now outputs sequential assets, metadata, click-track audio, and a ready-to-share ZIP so editors can drop everything directly into CapCut.

## Project Structure
```
.
├── requirements.txt
├── src
│   ├── ingest.py        # "The Eye" – download, scene detect, embed
│   └── search.py        # "The Brain UI" – text search & export
├── downloads/           # auto-created; raw video files (git-ignored)
├── frames/              # auto-created; extracted frames (git-ignored)
├── chroma_db/           # auto-created; vector database (git-ignored)
├── organized_screenshots/ # ranked exports per vibe (git-ignored)
└── capcut_ready/        # CapCut kits (sequential assets, CSV, click track, zip)
```

## Quickstart
1. **Install deps**
   ```bash
   git clone https://github.com/0xBased-lang/vibe-montage-maker
   cd vibe-montage-maker
   pip install -r requirements.txt
   ```
2. **Ingest a video**
   ```bash
   python src/ingest.py
   # paste a YouTube URL when prompted
   ```
   This populates `downloads/`, `frames/`, and `chroma_db/`.
3. **Search & export curated frames**
   ```bash
   python src/search.py
   # describe any vibe (e.g., "neon city"), choose how many results + BPM
   ```
   Outputs land in:
   - `organized_screenshots/<vibe>/` for reviewing picks.
   - `capcut_ready/<vibe>/` with sequential assets, metadata, script stub, click-track, manifest, plus `<vibe>_capcut.zip`.

## CapCut deliverables
Every vibe export now includes:
- `0001.jpg`, `0002.jpg`, ... sequential frames for immediate timeline drop.
- `shots.csv` with rank, similarity score, source video ID, and timestamps.
- `script_stub.md` for narration/caption notes per shot.
- `capcut_manifest.json` containing BPM, beat duration, suggested clip durations, and metadata for each asset.
- `click_track.wav` metronome audio (configurable BPM) for beat syncing.
- `<vibe>_capcut.zip` archive (stored in `capcut_ready/`) to share with collaborators who only need CapCut assets.

## Notes
- Only 720p video streams are downloaded for speed; original scenes can be redownloaded later if needed.
- Frame extraction grabs mid-scene frames to avoid blurry cuts/fades.
- All heavy artifacts are .gitignored; each user maintains their own local cache.

## Roadmap
- Add "Flow" phase (color/brightness sorting, beat-synced montage assembly)
- Support batch ingestion for playlists
- Auto-generate CapCut `.capcutproj` templates (see `docs/CAPCUT_TEMPLATE_PLAN.md`)
