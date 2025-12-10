import argparse
import os
from pathlib import Path
from typing import List, Optional

from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
import json

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "renders"


def load_manifest(folder: Path) -> Optional[dict]:
    manifest_path = folder / "capcut_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            return json.load(f)
    return None


def gather_frames(folder: Path) -> List[Path]:
    frames = sorted(folder.glob("*.jpg"))
    if not frames:
        raise SystemExit(f"No JPG frames found in {folder}")
    return frames


def build_movie(frames: List[Path], duration: float, audio_path: Optional[Path], output: Path):
    clips = [ImageClip(str(f)).set_duration(duration) for f in frames]
    video = concatenate_videoclips(clips, method="compose")

    if audio_path and audio_path.exists():
        audio = AudioFileClip(str(audio_path))
        # trim or extend to video duration
        if audio.duration < video.duration:
            audio = audio.set_duration(audio.duration)
        else:
            audio = audio.set_duration(video.duration)
        video = video.set_audio(audio)
    else:
        print("[INFO] No audio attached (file missing or not provided)")

    output.parent.mkdir(parents=True, exist_ok=True)
    video.write_videofile(
        str(output),
        codec="libx264",
        audio_codec="aac",
        fps=24,
        preset="medium",
        threads=4,
    )


def main():
    parser = argparse.ArgumentParser(description="Assemble frames into an MP4 montage.")
    parser.add_argument("folder", help="Folder with frames (e.g., capcut_ready/<vibe> or organized_screenshots/<vibe>)")
    parser.add_argument("--out", default=None, help="Output mp4 path (default renders/<folder>.mp4)")
    parser.add_argument("--duration", type=float, default=None, help="Per-frame duration (seconds). If omitted, uses manifest beat duration if available, else 1.0")
    parser.add_argument("--audio", default=None, help="Optional audio file to attach (default: click_track.wav in folder if present)")
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")

    manifest = load_manifest(folder)
    beat_duration = None
    if manifest:
        beat_duration = manifest.get("beat_duration_seconds")

    per_frame = args.duration or beat_duration or 1.0
    frames = gather_frames(folder)

    audio_path = None
    if args.audio:
        audio_path = Path(args.audio).resolve()
    else:
        default_audio = folder / "click_track.wav"
        if default_audio.exists():
            audio_path = default_audio

    out_path = Path(args.out) if args.out else DEFAULT_OUTPUT_DIR / f"{folder.name}.mp4"
    build_movie(frames, per_frame, audio_path, out_path)
    print(f"Saved montage: {out_path}")


if __name__ == "__main__":
    main()
