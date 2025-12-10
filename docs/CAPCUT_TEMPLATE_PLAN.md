# CapCut Template Automation Plan

Goal: emit partially configured CapCut projects so editors can open `.capcutproj` files with pre-arranged shots, durations, and audio. The format is a ZIP containing JSON manifests; reverse engineering shows the following minimum requirements:

1. **Project manifest (`project.json`)**
   - Global metadata: project name, fps, canvas aspect ratio.
   - Track list: video track(s), audio track(s), effect layers.
   - Each media reference includes an internal ID and relative path.

2. **Media table (`assets/*`)**
   - CapCut stores imported media inside the project folder (`assets/imported/`).
   - For our pipeline we will place sequential JPEGs + click-track audio there.

3. **Timeline definition**
   - Clip order, in/out durations, simple effects (Ken Burns) are described via JSON nodes referencing the media IDs.
   - We can map `capcut_manifest.json` entries to timeline clips by assigning `duration = beat_duration` and `start = cumulative_duration`.

## Implementation outline

1. **Serializer module (`capcut_template.py`)**
   - Takes `capcut_ready/<vibe>/` contents + manifest JSON.
   - Generates the CapCut JSON skeleton using recorded BPM + beat duration.
   - Copies assets into `template/<vibe>.capcutproj/assets/imported/`.

2. **Ken Burns / Motion presets**
   - Provide optional per-shot metadata (e.g., `pan: left-to-right`) in `manifest` for more dynamic templates.

3. **Configuration knobs**
   - Canvas (9:16, 16:9), base FPS, default transition style, optional text layers for captions loaded from `script_stub.md`.

4. **Packaging**
   - `.capcutproj` is just a directory; for sharing we can zip it similarly to existing CapCut kit.

5. **Safety / Compatibility**
   - Document CapCut version used for reverse-engineering to avoid compatibility surprises.
   - Provide fallback instructions if CapCut updates its schema.

## Next steps
- Inspect a sample CapCut project to capture exact JSON schema.
- Map our manifest fields to required CapCut properties.
- Implement serializer + CLI flag (e.g., `--capcut-project`).
- Add automated test that validates generated JSON structure before release.
