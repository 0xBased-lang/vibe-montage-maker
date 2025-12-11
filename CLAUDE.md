# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### One-shot pipeline (recommended)
```bash
python run.py --url "<VIDEO_URL>" --vibe "neon city" --count 8 --bpm 120
```

Options:
- `--cookies /path/to/cookies.txt` - For Instagram/TikTok gated content
- `--no-mp4` - Skip MP4 montage generation
- `--highlights` - Extract highlight video clips (auto-detect interesting moments)
- `--highlight-count N` - Number of highlight clips (default 5, auto-scales for long videos)
- `--highlight-duration N` - Target clip duration in seconds (default 20)
- `--highlight-max N` - Maximum clip duration in seconds (default 30)
- `--no-expand` - Disable segment expansion (use single scenes)
- `--dynamic-count` - Force auto-scale highlight count based on video length
- `--min-score N` - Filter highlights below quality threshold (0-1)
- `--categories "action,funny,dramatic"` - Custom highlight categories
- `--no-categorize` - Disable multi-category scoring
- `--no-spread` - Disable temporal diversity optimization
- `--no-clustering` - Disable KMeans embedding clustering
- `--draft --api-base http://127.0.0.1:3000` - Build CapCut draft via CapCutAPI

### Manual flow
```bash
python src/ingest.py              # Paste URL, downloads + scene detect + embed
python src/search.py              # Type vibe, count, BPM → exports assets
python src/make_movie.py capcut_ready/<vibe>  # Optional MP4 from frames
python src/highlight.py <video> <query> <count>  # Optional highlight extraction
```

### CapCutAPI draft (requires CapCutAPI server running)
```bash
python src/capcut_draft.py <vibe> --api-base http://127.0.0.1:3000 --ratio 9:16 --fps 30
```

## Architecture

```
Eye → Brain → Exports → (Optional) Draft
```

**Eye (ingest.py)**: Downloads video via yt-dlp (720p), detects scene cuts with PySceneDetect (threshold ~27), extracts one mid-scene frame per cut to avoid blur/fades.

**Brain (search.py + ChromaDB)**: Embeds frames with Google SigLIP (`google/siglip-base-patch16-224`) and stores vectors in ChromaDB. Text queries are embedded and matched via cosine similarity for semantic "vibe" search.

**Exports**: Per-vibe outputs include:
- `organized_screenshots/<vibe>/` - Ranked copies (01_*.jpg, 02_*.jpg, ...)
- `capcut_ready/<vibe>/` - Sequential frames (0001.jpg, ...), shots.csv, capcut_manifest.json, click_track.wav, script_stub.md, zip archive

**Highlight Detection (highlight.py)**: Multimodal scoring combining audio peaks (40%), scene changes (30%), and semantic similarity (30%). Uses ffmpeg+numpy for audio analysis (no librosa dependency).

Key features:
- **Smart expansion**: Merges adjacent scenes into 10-30s clips while audio stays active
- **Multi-category scoring**: Scores each segment against 5 categories (action, funny, dramatic, key_moment, reaction) and assigns dominant category
- **Dynamic count**: Auto-scales highlight count based on video duration (~1/min for 5-30min, ~1/1.5min for 30-60min, ~1/2min for 1hr+)
- **Temporal diversity**: Ensures highlights spread across video timeline (auto-enabled for >5 min videos)
- **Clustering**: Groups visually similar segments via KMeans to discover emergent themes
- **Tagged output**: Filenames include category tag (e.g., `highlight_001_action.mp4`)

## Key Classes

- `VibeEngine` (src/ingest.py) - SigLIP model loading, image embedding, ChromaDB storage
- `VibeSearch` (src/search.py) - Text embedding, vector search, export generation
- `HighlightDetector` (src/highlight.py) - Multimodal highlight detection and clip extraction
- `CapCutAPIClient` (src/capcut_draft.py) - HTTP client for CapCutAPI endpoints

## External Dependencies

- **ffmpeg**: Required for moviepy video generation and audio analysis. Install via `brew install ffmpeg` on macOS.
- **CapCutAPI**: Optional, for automated draft building. See https://github.com/sun-guannan/CapCutAPI

## Data Directories (git-ignored)

All heavy folders are local-only: `downloads/`, `frames/`, `chroma_db/`, `organized_screenshots/`, `capcut_ready/`, `renders/`, `highlights/`
