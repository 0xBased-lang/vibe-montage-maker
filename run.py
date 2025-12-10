import argparse
import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

# Defaults
DEFAULT_BPM = 120
DEFAULT_COUNT = 10


def run_cmd(cmd, env=None):
    result = subprocess.run(cmd, shell=True, env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)


def ensure_dirs():
    for p in [BASE_DIR / "downloads", BASE_DIR / "frames", BASE_DIR / "chroma_db", BASE_DIR / "organized_screenshots", BASE_DIR / "capcut_ready", BASE_DIR / "renders"]:
        p.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="One-shot pipeline: ingest -> search -> (optional) MP4 -> (optional) CapCut draft.")
    parser.add_argument("--url", required=True, help="YouTube URL to ingest")
    parser.add_argument("--vibe", required=True, help="Vibe text for search")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="How many screenshots to export (default 10)")
    parser.add_argument("--bpm", type=int, default=DEFAULT_BPM, help="BPM for click track and durations (default 120)")
    parser.add_argument("--no-mp4", action="store_true", help="Skip MP4 montage")
    parser.add_argument("--draft", action="store_true", help="Also build a CapCut draft via CapCutAPI")
    parser.add_argument("--api-base", default=os.environ.get("CAPCUT_API_BASE", "http://127.0.0.1:3000"), help="CapCutAPI base URL")
    parser.add_argument("--ratio", default="9:16", help="CapCutAPI canvas ratio (default 9:16)")
    parser.add_argument("--fps", type=int, default=30, help="CapCutAPI FPS (default 30)")
    parser.add_argument("--audio", default=None, help="Audio file for CapCut draft (default click_track.wav if present)")
    args = parser.parse_args()

    ensure_dirs()

    # 1) Ingest
    print("[1/4] Ingesting video...")
    run_cmd(f"cd {BASE_DIR} && printf '{args.url}\n' | python src/ingest.py")

    # 2) Search/export
    print("[2/4] Exporting vibe frames...")
    run_cmd(
        f"cd {BASE_DIR} && printf '{args.vibe}\n{args.count}\n{args.bpm}\nq\n' | python src/search.py"
    )

    # 3) MP4 montage
    mp4_path = None
    if not args.no_mp4:
        print("[3/4] Building MP4 montage...")
        mp4_path = BASE_DIR / "renders" / f"{args.vibe.replace(' ', '_')}.mp4"
        run_cmd(f"cd {BASE_DIR} && python src/make_movie.py capcut_ready/{args.vibe.replace(' ', '_')} --out {mp4_path}")

    # 4) CapCut draft (optional)
    draft_info = None
    if args.draft:
        print("[4/4] Building CapCut draft via CapCutAPI...")
        draft_cmd = (
            f"cd {BASE_DIR} && python src/capcut_draft.py {args.vibe.replace(' ', '_')} "
            f"--api-base {args.api_base} --fps {args.fps} --ratio {args.ratio}"
        )
        if args.audio:
            draft_cmd += f" --audio {args.audio}"
        run_cmd(draft_cmd)
        draft_info = "(see capcut_draft.py output for draft path)"

    print("\n=== Summary ===")
    vibe_safe = args.vibe.replace(" ", "_")
    print(f"Vibe: {args.vibe} (folder: {vibe_safe})")
    print(f"Ranked: organized_screenshots/{vibe_safe}/")
    print(f"CapCut kit: capcut_ready/{vibe_safe}/ (zip inside)")
    if mp4_path:
        print(f"MP4: {mp4_path}")
    if args.draft:
        print(f"Draft: {draft_info or 'draft built'}")
    print("Done.")


if __name__ == "__main__":
    main()
