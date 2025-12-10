import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
CAPCUT_DIR = BASE_DIR / "capcut_ready"
DEFAULT_API_BASE = os.environ.get("CAPCUT_API_BASE", "http://127.0.0.1:3000")


class CapCutAPIClient:
    """Lightweight client for CapCutAPI (sun-guannan/CapCutAPI) HTTP endpoints."""

    def __init__(self, base_url: str = DEFAULT_API_BASE):
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, payload: dict):
        url = f"{self.base_url}/{path.lstrip('/')}"
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def create_draft(self, name: str, fps: int = 30, aspect_ratio: str = "9:16"):
        return self._post("create_draft", {"name": name, "fps": fps, "ratio": aspect_ratio})

    def add_video(self, draft_id: str, video_path: str, start: float, end: float, volume: float = 1.0):
        return self._post(
            "add_video",
            {
                "draft_id": draft_id,
                "video_path": video_path,
                "start": start,
                "end": end,
                "volume": volume,
            },
        )

    def add_audio(self, draft_id: str, audio_path: str, start: float, end: float, volume: float = 1.0):
        return self._post(
            "add_audio",
            {
                "draft_id": draft_id,
                "audio_path": audio_path,
                "start": start,
                "end": end,
                "volume": volume,
            },
        )

    def save_draft(self, draft_id: str):
        return self._post("save_draft", {"draft_id": draft_id})


def load_manifest(vibe_dir: Path):
    manifest_path = vibe_dir / "capcut_manifest.json"
    if not manifest_path.exists():
        sys.exit(f"Manifest not found: {manifest_path}")
    with open(manifest_path, "r") as f:
        return json.load(f)


def gather_frames(vibe_dir: Path) -> List[Path]:
    frames = sorted(vibe_dir.glob("*.jpg"))
    if not frames:
        sys.exit(f"No frames found in {vibe_dir}")
    return frames


def main():
    parser = argparse.ArgumentParser(description="Create a CapCut draft via CapCutAPI using exported assets.")
    parser.add_argument("vibe", help="Vibe name (folder under capcut_ready)")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="CapCutAPI base URL (default env CAPCUT_API_BASE or http://127.0.0.1:3000)")
    parser.add_argument("--fps", type=int, default=30, help="Timeline FPS")
    parser.add_argument("--ratio", default="9:16", help="Canvas ratio, e.g., 9:16 or 16:9")
    parser.add_argument("--audio", default="click_track.wav", help="Audio file name within vibe folder to use (default: click_track.wav)")
    args = parser.parse_args()

    vibe_dir = CAPCUT_DIR / args.vibe
    if not vibe_dir.exists():
        sys.exit(f"Vibe folder not found: {vibe_dir}")

    manifest = load_manifest(vibe_dir)
    clips = manifest.get("clips", [])
    beat_duration = manifest.get("beat_duration_seconds", 1.0)

    frames = gather_frames(vibe_dir)
    audio_path = vibe_dir / args.audio
    if not audio_path.exists():
        print(f"[WARN] audio file not found: {audio_path}, continuing without audio")
        audio_path = None

    client = CapCutAPIClient(args.api_base)

    print(f"Creating draft via CapCutAPI at {args.api_base}...")
    draft_resp = client.create_draft(name=args.vibe, fps=args.fps, aspect_ratio=args.ratio)
    draft_id = draft_resp.get("result", {}).get("draft_id") or draft_resp.get("draft_id")
    if not draft_id:
        sys.exit(f"Failed to obtain draft_id from response: {draft_resp}")
    print(f"Draft created: {draft_id}")

    # Place each frame sequentially with beat-aligned durations
    timeline_pos = 0.0
    for idx, frame_path in enumerate(frames, start=1):
        start = timeline_pos
        end = timeline_pos + beat_duration
        client.add_video(draft_id, str(frame_path), start=start, end=end)
        timeline_pos = end
        if idx % 25 == 0:
            print(f"Added {idx} frames...")

    total_duration = timeline_pos

    # Add audio if available
    if audio_path:
        client.add_audio(draft_id, str(audio_path), start=0.0, end=total_duration)
        print(f"Added audio {audio_path} from 0.0 to {total_duration:.2f}s")

    save_resp = client.save_draft(draft_id)
    draft_url = save_resp.get("result", {}).get("draft_url") or save_resp.get("draft_url")
    print("Draft saved.")
    if draft_url:
        print(f"Draft path: {draft_url}")
    else:
        print(f"Save response: {save_resp}")

    print("\nNext: copy the generated draft folder (dfd_*) into your CapCut/Jianying drafts directory and open it in CapCut.")


if __name__ == "__main__":
    main()
