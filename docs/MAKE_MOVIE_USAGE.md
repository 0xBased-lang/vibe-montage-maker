# Movie Assembly (Quick MP4 Preview)

`src/make_movie.py` assembles frames into an MP4 montage. It works with either:
- `capcut_ready/<vibe>/` (uses `beat_duration_seconds` from `capcut_manifest.json` if present)
- `organized_screenshots/<vibe>/` (falls back to fixed per-frame duration)

## Usage
```bash
python src/make_movie.py <folder> [--out renders/myvideo.mp4] [--duration 1.0] [--audio path/to/audio.mp3]
```

- `folder`: folder containing JPG frames (e.g., `capcut_ready/neon_city`).
- `--out`: optional output path (default: `renders/<foldername>.mp4`).
- `--duration`: per-frame duration in seconds; if omitted, the script uses `beat_duration_seconds` from the manifest (if present), else defaults to 1.0s.
- `--audio`: optional audio file; if omitted, script tries `click_track.wav` inside the folder.

## Notes
- Output is H.264 MP4 (24 fps) with AAC audio if provided.
- The script concatenates frames in sorted order (`*.jpg`).
- If audio is longer than video, it’s trimmed; if shorter, it ends earlier.
- Requires `moviepy` and ffmpeg/avlib in PATH.
