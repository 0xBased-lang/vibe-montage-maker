"""
Highlight detection module.

Combines multiple signals for reliable highlight detection:
- Audio peaks (ffmpeg + numpy) - physics-based, highly reliable
- Scene changes (PySceneDetect) - math-based, highly reliable
- Semantic similarity (SigLIP) - AI-based, well-validated

Formula: highlight_score = (audio × 0.4) + (scene_change × 0.3) + (semantic × 0.3)
"""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional

import torch
from PIL import Image
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from transformers import AutoProcessor, AutoModel

# Handle imports when run directly or as module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_analysis import (
    analyze_audio,
    get_energy_at_timestamp,
    is_near_audio_peak
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIGHLIGHTS_DIR = os.path.join(BASE_DIR, "highlights")
os.makedirs(HIGHLIGHTS_DIR, exist_ok=True)


@dataclass
class Segment:
    """Represents a video segment with highlight scoring."""
    start_time: float
    end_time: float
    mid_time: float
    scene_index: int

    # Scores (0-1 normalized)
    audio_score: float = 0.0
    scene_score: float = 0.0
    semantic_score: float = 0.0

    # Composite
    highlight_score: float = 0.0

    # Metadata
    frame_path: Optional[str] = None


class HighlightDetector:
    """
    Detects highlight moments in videos using multimodal analysis.

    Combines:
    - Audio energy/peaks (loud = interesting)
    - Scene change magnitude (big changes = transitions/action)
    - Semantic similarity to query (matches user intent)
    """

    def __init__(self, weights=None):
        """
        Initialize the detector.

        Args:
            weights: Dict with keys 'audio', 'scene', 'semantic' (should sum to 1.0)
        """
        self.weights = weights or {'audio': 0.4, 'scene': 0.3, 'semantic': 0.3}

        print("Loading SigLIP model for semantic scoring...")
        self.model_name = "google/siglip-base-patch16-224"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

    def _detect_scenes(self, video_path, threshold=27.0):
        """Detect scene boundaries using PySceneDetect."""
        print(f"Detecting scenes (threshold={threshold})...")
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))

        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scenes = scene_manager.get_scene_list()

        video_manager.release()
        print(f"Found {len(scenes)} scenes")
        return scenes

    def _embed_image(self, image_path):
        """Get SigLIP embedding for an image."""
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            return outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        except Exception as e:
            print(f"[WARN] Failed to embed {image_path}: {e}")
            return None

    def _embed_text(self, query):
        """Get SigLIP embedding for text query."""
        inputs = self.processor(text=[query], padding="max_length", return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
        return outputs / outputs.norm(p=2, dim=-1, keepdim=True)

    def _compute_semantic_score(self, image_embedding, query_embedding):
        """Compute cosine similarity between image and query embeddings."""
        if image_embedding is None:
            return 0.0
        similarity = torch.nn.functional.cosine_similarity(
            image_embedding, query_embedding
        ).item()
        # Normalize to 0-1 (cosine similarity is -1 to 1)
        return (similarity + 1) / 2

    def _extract_frame(self, video_path, timestamp, output_path):
        """Extract a single frame at timestamp using ffmpeg."""
        cmd = [
            "ffmpeg", "-y", "-ss", str(timestamp),
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            output_path
        ]
        subprocess.run(cmd, capture_output=True)
        return output_path if os.path.exists(output_path) else None

    def detect_highlights(self, video_path, query="exciting dramatic moment",
                          top_n=5, min_duration=2.0, scene_threshold=27.0):
        """
        Detect highlight segments in a video.

        Args:
            video_path: Path to video file
            query: Text query describing desired highlights
            top_n: Number of highlights to return
            min_duration: Minimum segment duration in seconds
            scene_threshold: PySceneDetect threshold (lower = more sensitive)

        Returns:
            List of Segment objects sorted by highlight_score
        """
        video_id = os.path.splitext(os.path.basename(video_path))[0]

        # 1. Detect scenes
        scenes = self._detect_scenes(video_path, threshold=scene_threshold)
        if not scenes:
            print("[WARN] No scenes detected")
            return []

        # 2. Extract audio peaks and energy (single extraction)
        audio_data = analyze_audio(video_path)
        peak_times = audio_data['peak_times']
        energy_times = audio_data['energy_times']
        energy_values = audio_data['energy_values']

        # 3. Prepare query embedding
        query_embedding = self._embed_text(query)

        # 4. Create segments and compute scores
        segments = []
        temp_frame_dir = os.path.join(HIGHLIGHTS_DIR, f"temp_{video_id}")
        os.makedirs(temp_frame_dir, exist_ok=True)

        # Normalize scene change scores (use content detector's internal scores if available)
        scene_scores_raw = []
        for idx, (start, end) in enumerate(scenes):
            # Estimate scene change "magnitude" by duration inverse
            # (shorter scenes often indicate quick cuts = action)
            duration = end.get_seconds() - start.get_seconds()
            if duration >= min_duration:
                # Score inversely proportional to duration (quick cuts = higher)
                scene_scores_raw.append(1.0 / max(duration, 0.5))
            else:
                scene_scores_raw.append(0)

        max_scene_score = max(scene_scores_raw) if scene_scores_raw else 1.0

        print(f"Scoring {len(scenes)} segments...")
        for idx, (start, end) in enumerate(scenes):
            start_sec = start.get_seconds()
            end_sec = end.get_seconds()
            mid_sec = (start_sec + end_sec) / 2
            duration = end_sec - start_sec

            # Skip very short segments
            if duration < min_duration:
                continue

            segment = Segment(
                start_time=start_sec,
                end_time=end_sec,
                mid_time=mid_sec,
                scene_index=idx
            )

            # Audio score: energy at segment + bonus if near peak
            audio_energy = get_energy_at_timestamp(energy_times, energy_values, mid_sec)
            peak_bonus = 0.3 if is_near_audio_peak(mid_sec, peak_times, tolerance=1.5) else 0.0
            segment.audio_score = min(1.0, audio_energy + peak_bonus)

            # Scene score (normalized)
            segment.scene_score = scene_scores_raw[idx] / max_scene_score if max_scene_score > 0 else 0

            # Semantic score (extract frame, embed, compare)
            frame_path = os.path.join(temp_frame_dir, f"frame_{idx}.jpg")
            extracted = self._extract_frame(video_path, mid_sec, frame_path)
            if extracted:
                segment.frame_path = frame_path
                image_embedding = self._embed_image(frame_path)
                segment.semantic_score = self._compute_semantic_score(image_embedding, query_embedding)

            # Composite highlight score
            segment.highlight_score = (
                self.weights['audio'] * segment.audio_score +
                self.weights['scene'] * segment.scene_score +
                self.weights['semantic'] * segment.semantic_score
            )

            segments.append(segment)

        # Sort by highlight score and return top N
        segments.sort(key=lambda s: s.highlight_score, reverse=True)
        top_segments = segments[:top_n]

        print(f"\nTop {len(top_segments)} highlights:")
        for i, seg in enumerate(top_segments, 1):
            print(f"  [{i}] {seg.start_time:.1f}s-{seg.end_time:.1f}s "
                  f"(score={seg.highlight_score:.3f}, "
                  f"audio={seg.audio_score:.2f}, "
                  f"scene={seg.scene_score:.2f}, "
                  f"semantic={seg.semantic_score:.2f})")

        # Clean up temp frames (keep only frames used in top segments)
        if os.path.exists(temp_frame_dir):
            try:
                shutil.rmtree(temp_frame_dir)
                print(f"Cleaned up temp frames: {temp_frame_dir}")
            except Exception as e:
                print(f"[WARN] Failed to clean up temp frames: {e}")

        return top_segments

    def extract_clips(self, video_path, segments, output_dir=None, prefix="highlight"):
        """
        Extract video clips for the given segments.

        Args:
            video_path: Source video path
            segments: List of Segment objects
            output_dir: Output directory (default: highlights/<video_id>/)
            prefix: Filename prefix

        Returns:
            List of output clip paths
        """
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        if output_dir is None:
            output_dir = os.path.join(HIGHLIGHTS_DIR, video_id)
        os.makedirs(output_dir, exist_ok=True)

        clip_paths = []
        print(f"\nExtracting {len(segments)} highlight clips to {output_dir}...")

        for idx, segment in enumerate(segments, 1):
            output_path = os.path.join(output_dir, f"{prefix}_{idx:03d}.mp4")

            # Use ffmpeg for reliable extraction
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(segment.start_time),
                "-i", video_path,
                "-t", str(segment.end_time - segment.start_time),
                "-c:v", "libx264",
                "-c:a", "aac",
                "-preset", "fast",
                output_path
            ]

            result = subprocess.run(cmd, capture_output=True)
            if result.returncode == 0 and os.path.exists(output_path):
                clip_paths.append(output_path)
                print(f"  Extracted: {os.path.basename(output_path)} "
                      f"({segment.end_time - segment.start_time:.1f}s)")
            else:
                print(f"  [WARN] Failed to extract clip {idx}")

        return clip_paths

    def export_manifest(self, video_path, segments, clip_paths, output_dir, query):
        """
        Export a manifest JSON with highlight metadata.

        Args:
            video_path: Source video path
            segments: List of Segment objects
            clip_paths: List of extracted clip paths
            output_dir: Output directory
            query: Original query used for detection
        """
        video_id = os.path.splitext(os.path.basename(video_path))[0]

        manifest = {
            "source_video": video_path,
            "video_id": video_id,
            "query": query,
            "weights": self.weights,
            "highlights": []
        }

        for idx, (segment, clip_path) in enumerate(zip(segments, clip_paths), 1):
            manifest["highlights"].append({
                "rank": idx,
                "clip_file": os.path.basename(clip_path) if clip_path else None,
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "duration": segment.end_time - segment.start_time,
                "scores": {
                    "highlight_score": round(segment.highlight_score, 4),
                    "audio_score": round(segment.audio_score, 4),
                    "scene_score": round(segment.scene_score, 4),
                    "semantic_score": round(segment.semantic_score, 4)
                }
            })

        manifest_path = os.path.join(output_dir, "highlights_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Manifest saved: {manifest_path}")
        return manifest_path


def detect_and_extract(video_path, query="exciting dramatic moment",
                       top_n=5, output_dir=None):
    """
    Convenience function to detect highlights and extract clips.

    Args:
        video_path: Path to video file
        query: Text describing desired highlights
        top_n: Number of highlights to extract
        output_dir: Output directory (optional)

    Returns:
        Tuple of (segments, clip_paths, manifest_path)
    """
    detector = HighlightDetector()

    # Detect highlights
    segments = detector.detect_highlights(video_path, query=query, top_n=top_n)

    if not segments:
        print("No highlights detected.")
        return [], [], None

    # Extract clips
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    if output_dir is None:
        output_dir = os.path.join(HIGHLIGHTS_DIR, video_id)

    clip_paths = detector.extract_clips(video_path, segments, output_dir=output_dir)

    # Export manifest
    manifest_path = detector.export_manifest(
        video_path, segments, clip_paths, output_dir, query
    )

    return segments, clip_paths, manifest_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python highlight.py <video_path> [query] [top_n]")
        print("Example: python highlight.py video.mp4 'exciting action' 5")
        sys.exit(1)

    video_path = sys.argv[1]
    query = sys.argv[2] if len(sys.argv) > 2 else "exciting dramatic moment"
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    segments, clips, manifest = detect_and_extract(video_path, query=query, top_n=top_n)

    print(f"\n=== Results ===")
    print(f"Detected {len(segments)} highlights")
    print(f"Extracted {len(clips)} clips")
    if manifest:
        print(f"Manifest: {manifest}")
