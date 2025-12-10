# CapCutAPI Integration (sun-guannan/CapCutAPI)

This project can now send your CapCut-ready assets to CapCutAPI to auto-build a draft timeline (video clips + click-track audio). CapCutAPI produces a `dfd_*` draft folder you can drop into CapCut/Jianying drafts and open directly.

## Prerequisites
- Install and run CapCutAPI (https://github.com/sun-guannan/CapCutAPI).
- Ensure FFmpeg is available if CapCutAPI needs it.
- Start the API server (default assumed base: `http://127.0.0.1:3000`).

## Generate assets (already in this repo)
1) Ingest a video: `python src/ingest.py`
2) Export a vibe: `python src/search.py` (choose vibe + BPM) → assets in `capcut_ready/<vibe>/`.

## Build a draft automatically
Run:
```bash
python src/capcut_draft.py <vibe> \
  --api-base http://127.0.0.1:3000 \
  --fps 30 \
  --ratio 9:16 \
  --audio click_track.wav
```
- Reads `capcut_ready/<vibe>/capcut_manifest.json`
- Adds sequential frames (`0001.jpg`...) as clips with beat-aligned durations
- Adds audio (click_track.wav by default) across the full timeline
- Saves draft via CapCutAPI

The script prints the saved draft path. Copy the `dfd_*` folder into your CapCut/Jianying drafts directory, then open it in CapCut. 

## Endpoint assumptions
The script calls these CapCutAPI endpoints:
- `POST /create_draft` {name, fps, ratio}
- `POST /add_video` {draft_id, video_path, start, end, volume}
- `POST /add_audio` {draft_id, audio_path, start, end, volume}
- `POST /save_draft` {draft_id}

If your CapCutAPI is configured differently, adjust `src/capcut_draft.py` or set `CAPCUT_API_BASE` and match paths accordingly.

## Troubleshooting
- If requests fail, confirm CapCutAPI is running and reachable at `--api-base`.
- If CapCutAPI schema differs, adapt `capcut_draft.py` to the correct endpoint names/payloads.
- Ensure paths you send to CapCutAPI are absolute or accessible to the server process.
