import argparse
import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Defaults
DEFAULT_BPM = 120
DEFAULT_COUNT = 10
DEFAULT_HIGHLIGHTS = 5


def run_cmd(cmd, env=None):
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    result = subprocess.run(cmd, shell=True, env=merged_env)
    if result.returncode != 0:
        sys.exit(result.returncode)


def ensure_dirs():
    for p in [BASE_DIR / "downloads", BASE_DIR / "frames", BASE_DIR / "chroma_db", BASE_DIR / "organized_screenshots", BASE_DIR / "capcut_ready", BASE_DIR / "renders", BASE_DIR / "highlights"]:
        p.mkdir(parents=True, exist_ok=True)


def safe_name(text: str) -> str:
    return text.strip().replace(" ", "_").lower()


def main():
    parser = argparse.ArgumentParser(description="One-shot pipeline: ingest -> search -> (optional) MP4 -> (optional) CapCut draft.")
    parser.add_argument("--url", required=True, help="Video URL to ingest (YouTube/TikTok/Instagram/etc. via yt-dlp)")
    parser.add_argument("--vibe", required=True, help="Vibe text for search")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="How many screenshots to export (default 10)")
    parser.add_argument("--bpm", type=int, default=DEFAULT_BPM, help="BPM for click track and durations (default 120)")
    parser.add_argument("--cookies", default=None, help="Cookies file for yt-dlp (useful for Instagram/TikTok gated content)")
    parser.add_argument("--no-mp4", action="store_true", help="Skip MP4 montage")
    parser.add_argument("--draft", action="store_true", help="Also build a CapCut draft via CapCutAPI")
    parser.add_argument("--api-base", default=os.environ.get("CAPCUT_API_BASE", "http://127.0.0.1:3000"), help="CapCutAPI base URL")
    parser.add_argument("--ratio", default="9:16", help="CapCutAPI canvas ratio (default 9:16)")
    parser.add_argument("--fps", type=int, default=30, help="CapCutAPI FPS (default 30)")
    parser.add_argument("--audio", default=None, help="Audio file for CapCut draft (default click_track.wav if present)")
    # Highlight detection options
    parser.add_argument("--highlights", action="store_true", help="Extract highlight video clips (auto-detect interesting moments)")
    parser.add_argument("--highlight-count", type=int, default=DEFAULT_HIGHLIGHTS, help="Number of highlight clips to extract (default 5)")
    parser.add_argument("--highlight-query", default=None, help="Custom query for highlight detection (default: uses --vibe)")
    args = parser.parse_args()

    ensure_dirs()
    vibe_safe = safe_name(args.vibe)

    # Determine number of steps (count only steps that will actually run)
    total_steps = 2  # ingest + search always run
    if args.highlights:
        total_steps += 1
    if not args.no_mp4:
        total_steps += 1
    if args.draft:
        total_steps += 1
    step = 0

    # 1) Ingest
    step += 1
    print(f"[{step}/{total_steps}] Ingesting video...")
    ingest_env = {}
    if args.cookies:
        ingest_env["COOKIES_FILE"] = str(Path(args.cookies).resolve())
        print(f"Using cookies file for yt-dlp: {ingest_env['COOKIES_FILE']}")
    run_cmd(f"cd {BASE_DIR} && printf '{args.url}\n' | python src/ingest.py", env=ingest_env)

    # Find the downloaded video file for highlight extraction
    downloaded_video = None
    if args.highlights:
        # Look for the most recently downloaded video
        downloads_dir = BASE_DIR / "downloads"
        video_files = list(downloads_dir.glob("*.*"))
        if video_files:
            downloaded_video = max(video_files, key=lambda p: p.stat().st_mtime)

    # 2) Highlight extraction (optional)
    if args.highlights:
        step += 1
        print(f"[{step}/{total_steps}] Extracting highlight clips...")
        if downloaded_video and downloaded_video.exists():
            highlight_query = args.highlight_query or args.vibe
            run_cmd(
                f"cd {BASE_DIR} && python src/highlight.py "
                f'"{downloaded_video}" "{highlight_query}" {args.highlight_count}'
            )
        else:
            print("[WARN] Could not find downloaded video for highlight extraction")

    # 3) Search/export
    step += 1
    print(f"[{step}/{total_steps}] Exporting vibe frames...")
    run_cmd(
        f"cd {BASE_DIR} && printf '{args.vibe}\n{args.count}\n{args.bpm}\nq\n' | python src/search.py"
    )

    # 4) MP4 montage
    mp4_path = None
    if not args.no_mp4:
        step += 1
        print(f"[{step}/{total_steps}] Building MP4 montage...")
        mp4_path = BASE_DIR / "renders" / f"{vibe_safe}.mp4"
        run_cmd(f"cd {BASE_DIR} && python src/make_movie.py capcut_ready/{vibe_safe} --out {mp4_path}")

    # 5) CapCut draft (optional)
    draft_info = None
    if args.draft:
        step += 1
        print(f"[{step}/{total_steps}] Building CapCut draft via CapCutAPI...")
        draft_cmd = (
            f"cd {BASE_DIR} && python src/capcut_draft.py {vibe_safe} "
            f"--api-base {args.api_base} --fps {args.fps} --ratio {args.ratio}"
        )
        if args.audio:
            draft_cmd += f" --audio {args.audio}"
        run_cmd(draft_cmd)
        draft_info = "(see capcut_draft.py output for draft path)"

    print("\n=== Summary ===")
    print(f"Vibe: {args.vibe} (folder: {vibe_safe})")
    print(f"Ranked: organized_screenshots/{vibe_safe}/")
    print(f"CapCut kit: capcut_ready/{vibe_safe}/ (zip inside)")
    if args.highlights and downloaded_video:
        video_id = downloaded_video.stem
        print(f"Highlights: highlights/{video_id}/ ({args.highlight_count} clips)")
    elif args.highlights:
        print(f"Highlights: (extraction skipped - no video found)")
    if mp4_path:
        print(f"MP4: {mp4_path}")
    if args.draft:
        print(f"Draft: {draft_info or 'draft built'}")
    print("Done.")


if __name__ == "__main__":
    main()
