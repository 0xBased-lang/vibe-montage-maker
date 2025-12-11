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
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np
import torch
from PIL import Image
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from transformers import AutoProcessor, AutoModel

# Default highlight categories for multi-category scoring
DEFAULT_CATEGORIES = {
    "action": "fast intense movement action exciting dynamic energy",
    "funny": "funny humor laugh comedic amusing entertaining joke",
    "dramatic": "emotional dramatic tension powerful climactic suspense",
    "key_moment": "important key moment highlight memorable significant peak",
    "reaction": "reaction response surprise expression face emotion shock"
}

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

    # For expanded segments (merged scenes)
    scene_indices: Optional[List[int]] = None
    is_expanded: bool = False

    # NEW: Category scoring
    category_scores: Optional[Dict[str, float]] = None
    dominant_category: Optional[str] = None

    # NEW: Clustering
    cluster_id: Optional[int] = None
    cluster_theme: Optional[str] = None
    frame_embedding: Optional[np.ndarray] = None

    # NEW: Temporal diversity
    diversity_score: float = 1.0
    temporal_position: float = 0.0  # 0-1 position in video


def calculate_dynamic_count(video_duration: float, base_count: int = 5) -> int:
    """
    Calculate highlight count based on video duration.

    Scaling:
    - <5 min: base_count (default 5)
    - 5-30 min: ~1 per minute
    - 30-60 min: ~1 per 1.5 minutes
    - >60 min: ~1 per 2 minutes
    """
    if video_duration < 300:  # <5 min
        return base_count
    elif video_duration < 1800:  # 5-30 min
        return max(base_count, int(video_duration / 60))
    elif video_duration < 3600:  # 30-60 min
        return max(10, int(video_duration / 90))
    else:  # >1 hour
        return max(15, int(video_duration / 120))


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

    def _embed_categories(self, categories: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """Embed all category queries upfront for multi-category scoring."""
        category_embeddings = {}
        for name, query in categories.items():
            category_embeddings[name] = self._embed_text(query)
        return category_embeddings

    def _compute_category_scores(self, image_embedding, category_embeddings: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute semantic similarity against all categories."""
        if image_embedding is None:
            return {name: 0.0 for name in category_embeddings.keys()}

        scores = {}
        for name, cat_embedding in category_embeddings.items():
            similarity = torch.nn.functional.cosine_similarity(
                image_embedding, cat_embedding
            ).item()
            # Normalize to 0-1
            scores[name] = (similarity + 1) / 2
        return scores

    def _calculate_diversity_penalty(self, candidate_time: float, selected_times: List[float],
                                     video_duration: float) -> float:
        """
        Penalize candidates too close to already-selected highlights.

        Returns a score between 0-1 where 1 means good spread, 0 means clustered.
        """
        if not selected_times:
            return 1.0

        min_gap = min(abs(candidate_time - t) for t in selected_times)
        ideal_gap = video_duration / (len(selected_times) + 2)
        return min(1.0, min_gap / ideal_gap)

    def _cluster_segments(self, segments: List[Segment], n_clusters: int,
                          category_embeddings: Dict[str, torch.Tensor]) -> List[Segment]:
        """
        Cluster segments by their frame embeddings and assign cluster themes.

        Uses KMeans clustering and finds nearest category for each cluster centroid.
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            print("[WARN] sklearn not available, skipping clustering")
            return segments

        # Collect embeddings
        embeddings = []
        valid_segments = []
        for seg in segments:
            if seg.frame_embedding is not None:
                embeddings.append(seg.frame_embedding.flatten())
                valid_segments.append(seg)

        if len(embeddings) < n_clusters:
            print(f"[WARN] Not enough segments ({len(embeddings)}) for {n_clusters} clusters")
            return segments

        # Cluster
        X = np.array(embeddings)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Map each cluster to nearest category
        cluster_themes = {}
        for cluster_id in range(n_clusters):
            centroid = kmeans.cluster_centers_[cluster_id]
            centroid_tensor = torch.tensor(centroid).unsqueeze(0)

            best_category = None
            best_similarity = -1
            for cat_name, cat_embedding in category_embeddings.items():
                similarity = torch.nn.functional.cosine_similarity(
                    centroid_tensor.float(), cat_embedding.float()
                ).item()
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_category = cat_name

            cluster_themes[cluster_id] = best_category

        # Assign to segments
        for seg, label in zip(valid_segments, labels):
            seg.cluster_id = int(label)
            seg.cluster_theme = cluster_themes.get(label, "unknown")

        print(f"Clustered {len(valid_segments)} segments into {n_clusters} groups: "
              f"{dict((k, list(cluster_themes.values()).count(k)) for k in set(cluster_themes.values()))}")

        return segments

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds using ffprobe."""
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 0.0

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

    def _is_audio_silence(self, audio_data, start_time, end_time, threshold=0.15):
        """
        Check if a time range is mostly silent (conversation gap).

        Args:
            audio_data: Dict with 'energy_times' and 'energy_values'
            start_time: Start of range in seconds
            end_time: End of range in seconds
            threshold: Energy threshold below which is considered silence

        Returns:
            Boolean indicating if the range is mostly silent
        """
        mid_time = (start_time + end_time) / 2
        energy = get_energy_at_timestamp(
            audio_data['energy_times'],
            audio_data['energy_values'],
            mid_time,
            window=(end_time - start_time) / 2
        )
        return energy < threshold

    def _expand_segment(self, seed_idx, scenes, scene_scores, audio_data,
                        target_duration=20, min_duration=10, max_duration=30,
                        expand_threshold=0.4, silence_threshold=0.15):
        """
        Expand a seed scene into a longer highlight by merging adjacent scenes.

        Uses audio energy (70%) and scene scores (30%) to decide when to stop expanding.

        Args:
            seed_idx: Index of the seed scene
            scenes: List of (start_frame, end_frame) tuples from PySceneDetect
            scene_scores: List of highlight scores for each scene
            audio_data: Dict with audio analysis data
            target_duration: Target clip length in seconds
            min_duration: Minimum clip length
            max_duration: Maximum clip length
            expand_threshold: Minimum score ratio to keep expanding
            silence_threshold: Energy threshold for silence detection

        Returns:
            Tuple of (start_time, end_time, avg_score, scene_indices)
        """
        start_idx = seed_idx
        end_idx = seed_idx
        seed_score = scene_scores[seed_idx]

        # Get scene times helper
        def get_scene_times(idx):
            return (scenes[idx][0].get_seconds(), scenes[idx][1].get_seconds())

        # Calculate current duration
        def current_duration():
            s_start, _ = get_scene_times(start_idx)
            _, s_end = get_scene_times(end_idx)
            return s_end - s_start

        # Check if we should expand to include a scene
        def should_expand_to(idx):
            if idx < 0 or idx >= len(scenes):
                return False

            score = scene_scores[idx]
            # Check score threshold (relative to seed)
            if score < seed_score * expand_threshold:
                return False

            # Check for silence (audio-driven, 70% weight)
            s_start, s_end = get_scene_times(idx)
            if self._is_audio_silence(audio_data, s_start, s_end, silence_threshold):
                return False

            return True

        # Expand forward while quality stays high and under max duration
        while end_idx < len(scenes) - 1:
            potential_end = end_idx + 1
            _, potential_end_time = get_scene_times(potential_end)
            s_start, _ = get_scene_times(start_idx)
            potential_duration = potential_end_time - s_start

            # Stop if we'd exceed max duration
            if potential_duration > max_duration:
                break

            # Stop if target reached and next scene doesn't meet threshold
            if current_duration() >= target_duration and not should_expand_to(potential_end):
                break

            # Check expansion criteria
            if not should_expand_to(potential_end):
                break

            end_idx = potential_end

        # Expand backward (same logic)
        while start_idx > 0:
            potential_start = start_idx - 1
            potential_start_time, _ = get_scene_times(potential_start)
            _, s_end = get_scene_times(end_idx)
            potential_duration = s_end - potential_start_time

            # Stop if we'd exceed max duration
            if potential_duration > max_duration:
                break

            # Stop if target reached and prev scene doesn't meet threshold
            if current_duration() >= target_duration and not should_expand_to(potential_start):
                break

            # Check expansion criteria
            if not should_expand_to(potential_start):
                break

            start_idx = potential_start

        # If still under min_duration, force expand to nearest good scenes
        while current_duration() < min_duration:
            # Try expanding forward first
            if end_idx < len(scenes) - 1:
                end_idx += 1
            elif start_idx > 0:
                start_idx -= 1
            else:
                break  # Can't expand further

        # Calculate final times and average score
        final_start, _ = get_scene_times(start_idx)
        _, final_end = get_scene_times(end_idx)
        scene_indices = list(range(start_idx, end_idx + 1))
        avg_score = sum(scene_scores[i] for i in scene_indices) / len(scene_indices)

        return (final_start, final_end, avg_score, scene_indices)

    def detect_highlights(self, video_path, query="exciting dramatic moment",
                          top_n=5, min_duration=2.0, scene_threshold=27.0,
                          expand_segments=True, target_duration=20,
                          max_duration=30, expand_threshold=0.4,
                          # NEW: Category and diversity options
                          categorize=True, categories=None,
                          enable_diversity=None, min_score=0.0,
                          dynamic_count=None, enable_clustering=True):
        """
        Detect highlight segments in a video.

        Args:
            video_path: Path to video file
            query: Text query describing desired highlights (also used as primary semantic query)
            top_n: Number of highlights to return (may be overridden by dynamic_count)
            min_duration: Minimum segment duration in seconds
            scene_threshold: PySceneDetect threshold (lower = more sensitive)
            expand_segments: If True, merge adjacent scenes into longer highlights
            target_duration: Target clip length when expanding (seconds)
            max_duration: Maximum clip length when expanding (seconds)
            expand_threshold: Minimum score ratio to keep expanding (0-1)
            categorize: If True, score against multiple categories and assign dominant category
            categories: Dict of category_name -> query (default: DEFAULT_CATEGORIES)
            enable_diversity: If True, apply temporal diversity penalty (auto-enabled for >5 min videos)
            min_score: Filter out highlights below this score threshold (0-1)
            dynamic_count: If True, auto-scale highlight count based on video duration
            enable_clustering: If True, cluster segments by visual similarity

        Returns:
            List of Segment objects sorted by highlight_score
        """
        video_id = os.path.splitext(os.path.basename(video_path))[0]

        # 0. Get video duration for dynamic features
        video_duration = self._get_video_duration(video_path)
        print(f"Video duration: {video_duration:.1f}s ({video_duration/60:.1f} minutes)")

        # Auto-enable features for long videos (>5 min)
        is_long_video = video_duration > 300
        if enable_diversity is None:
            enable_diversity = is_long_video
        if dynamic_count is None:
            dynamic_count = is_long_video

        # Calculate actual highlight count
        actual_top_n = top_n
        if dynamic_count and video_duration > 0:
            actual_top_n = calculate_dynamic_count(video_duration, base_count=top_n)
            print(f"Dynamic count: {actual_top_n} highlights (from {top_n} base)")

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

        # 3. Prepare embeddings
        query_embedding = self._embed_text(query)

        # 3b. Prepare category embeddings if categorizing
        category_embeddings = None
        if categorize:
            cats = categories or DEFAULT_CATEGORIES
            print(f"Embedding {len(cats)} categories: {list(cats.keys())}")
            category_embeddings = self._embed_categories(cats)

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
            image_embedding = None
            if extracted:
                segment.frame_path = frame_path
                image_embedding = self._embed_image(frame_path)
                segment.semantic_score = self._compute_semantic_score(image_embedding, query_embedding)

                # Store embedding for clustering
                if image_embedding is not None:
                    segment.frame_embedding = image_embedding.numpy()

                # Category scores
                if categorize and category_embeddings:
                    segment.category_scores = self._compute_category_scores(image_embedding, category_embeddings)
                    # Dominant category = highest scoring
                    segment.dominant_category = max(segment.category_scores, key=segment.category_scores.get)

            # Temporal position (0-1 in video)
            if video_duration > 0:
                segment.temporal_position = mid_sec / video_duration

            # Composite highlight score
            segment.highlight_score = (
                self.weights['audio'] * segment.audio_score +
                self.weights['scene'] * segment.scene_score +
                self.weights['semantic'] * segment.semantic_score
            )

            segments.append(segment)

        # Build scene_scores list for expansion (indexed by scene index)
        scene_scores_list = [0.0] * len(scenes)
        for seg in segments:
            scene_scores_list[seg.scene_index] = seg.highlight_score

        # Apply min_score filter
        if min_score > 0:
            original_count = len(segments)
            segments = [s for s in segments if s.highlight_score >= min_score]
            print(f"Filtered by min_score ({min_score}): {original_count} -> {len(segments)} segments")

        # Clustering (if enabled and we have enough segments)
        if enable_clustering and categorize and category_embeddings and len(segments) >= 4:
            n_clusters = min(5, len(segments) // 4)
            if n_clusters >= 2:
                segments = self._cluster_segments(segments, n_clusters, category_embeddings)

        # Sort by highlight score to find best seeds
        segments.sort(key=lambda s: s.highlight_score, reverse=True)

        if expand_segments and len(segments) > 0:
            # Expand segments into longer highlights
            print(f"\nExpanding top segments (target={target_duration}s, max={max_duration}s)...")
            expanded_segments = []
            used_scenes = set()
            selected_times = []  # For diversity calculation

            for seed_segment in segments:
                seed_idx = seed_segment.scene_index

                # Skip if this scene is already used in another highlight
                if seed_idx in used_scenes:
                    continue

                # Expand this seed
                start_time, end_time, avg_score, scene_indices = self._expand_segment(
                    seed_idx, scenes, scene_scores_list, audio_data,
                    target_duration=target_duration,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    expand_threshold=expand_threshold
                )

                # Check for overlap with already-used scenes
                if any(idx in used_scenes for idx in scene_indices):
                    continue

                # Calculate diversity score for this candidate
                mid_time = (start_time + end_time) / 2
                diversity = 1.0
                if enable_diversity and video_duration > 0:
                    diversity = self._calculate_diversity_penalty(mid_time, selected_times, video_duration)

                # Mark scenes as used
                used_scenes.update(scene_indices)
                selected_times.append(mid_time)

                # Create expanded segment with all metadata from seed
                expanded = Segment(
                    start_time=start_time,
                    end_time=end_time,
                    mid_time=mid_time,
                    scene_index=seed_idx,
                    audio_score=seed_segment.audio_score,
                    scene_score=seed_segment.scene_score,
                    semantic_score=seed_segment.semantic_score,
                    highlight_score=avg_score,
                    scene_indices=scene_indices,
                    is_expanded=True,
                    # Copy category info from seed
                    category_scores=seed_segment.category_scores,
                    dominant_category=seed_segment.dominant_category,
                    cluster_id=seed_segment.cluster_id,
                    cluster_theme=seed_segment.cluster_theme,
                    diversity_score=diversity,
                    temporal_position=mid_time / video_duration if video_duration > 0 else 0
                )
                expanded_segments.append(expanded)

                if len(expanded_segments) >= actual_top_n:
                    break

            top_segments = expanded_segments
            print(f"Created {len(top_segments)} expanded highlights")
        else:
            # Original behavior: return individual scenes (with diversity if enabled)
            if enable_diversity and video_duration > 0:
                # Re-sort with diversity penalty
                selected = []
                selected_times = []
                remaining = segments.copy()

                while len(selected) < actual_top_n and remaining:
                    # Score each remaining segment with diversity
                    best_idx = 0
                    best_score = -1
                    for i, seg in enumerate(remaining):
                        div = self._calculate_diversity_penalty(seg.mid_time, selected_times, video_duration)
                        combined = seg.highlight_score * 0.75 + div * 0.25
                        if combined > best_score:
                            best_score = combined
                            best_idx = i

                    chosen = remaining.pop(best_idx)
                    chosen.diversity_score = self._calculate_diversity_penalty(
                        chosen.mid_time, selected_times, video_duration
                    )
                    selected.append(chosen)
                    selected_times.append(chosen.mid_time)

                top_segments = selected
            else:
                top_segments = segments[:actual_top_n]

        # Print summary with category info
        print(f"\nTop {len(top_segments)} highlights:")
        for i, seg in enumerate(top_segments, 1):
            duration = seg.end_time - seg.start_time
            scene_count = len(seg.scene_indices) if seg.scene_indices else 1
            cat_str = f", {seg.dominant_category}" if seg.dominant_category else ""
            div_str = f", spread={seg.diversity_score:.2f}" if enable_diversity else ""
            print(f"  [{i}] {seg.start_time:.1f}s-{seg.end_time:.1f}s "
                  f"({duration:.1f}s, {scene_count} scenes, "
                  f"score={seg.highlight_score:.3f}{cat_str}{div_str})")

        # Clean up temp frames
        if os.path.exists(temp_frame_dir):
            try:
                shutil.rmtree(temp_frame_dir)
                print(f"Cleaned up temp frames: {temp_frame_dir}")
            except Exception as e:
                print(f"[WARN] Failed to clean up temp frames: {e}")

        return top_segments

    def extract_clips(self, video_path, segments, output_dir=None, prefix="highlight",
                      include_category_tag=True):
        """
        Extract video clips for the given segments.

        Args:
            video_path: Source video path
            segments: List of Segment objects
            output_dir: Output directory (default: highlights/<video_id>/)
            prefix: Filename prefix
            include_category_tag: If True, include category in filename (highlight_001_action.mp4)

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
            # Build filename with optional category tag
            if include_category_tag and segment.dominant_category:
                filename = f"{prefix}_{idx:03d}_{segment.dominant_category}.mp4"
            else:
                filename = f"{prefix}_{idx:03d}.mp4"
            output_path = os.path.join(output_dir, filename)

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

    def export_manifest(self, video_path, segments, clip_paths, output_dir, query,
                        video_duration=None, categories_used=None):
        """
        Export a manifest JSON with highlight metadata.

        Args:
            video_path: Source video path
            segments: List of Segment objects
            clip_paths: List of extracted clip paths
            output_dir: Output directory
            query: Original query used for detection
            video_duration: Video duration in seconds (optional)
            categories_used: List of category names used (optional)
        """
        video_id = os.path.splitext(os.path.basename(video_path))[0]

        manifest = {
            "source_video": video_path,
            "video_id": video_id,
            "query": query,
            "video_duration": video_duration,
            "weights": self.weights,
            "extraction_settings": {
                "categories": categories_used or list(DEFAULT_CATEGORIES.keys()),
                "categorize_enabled": any(s.dominant_category for s in segments),
                "diversity_enabled": any(s.diversity_score < 1.0 for s in segments),
                "clustering_enabled": any(s.cluster_id is not None for s in segments)
            },
            "highlights": []
        }

        # Track category counts for summary
        category_counts = {}
        category_scores = {}
        cluster_counts = {}

        for idx, (segment, clip_path) in enumerate(zip(segments, clip_paths), 1):
            highlight_data = {
                "rank": idx,
                "clip_file": os.path.basename(clip_path) if clip_path else None,
                "start_time": round(segment.start_time, 2),
                "end_time": round(segment.end_time, 2),
                "duration": round(segment.end_time - segment.start_time, 2),
                "temporal_position": round(segment.temporal_position, 3),
                "scores": {
                    "highlight_score": round(segment.highlight_score, 4),
                    "audio_score": round(segment.audio_score, 4),
                    "scene_score": round(segment.scene_score, 4),
                    "semantic_score": round(segment.semantic_score, 4)
                }
            }

            # Add category info
            if segment.dominant_category:
                highlight_data["dominant_category"] = segment.dominant_category
                category_counts[segment.dominant_category] = category_counts.get(segment.dominant_category, 0) + 1
                if segment.dominant_category not in category_scores:
                    category_scores[segment.dominant_category] = []
                category_scores[segment.dominant_category].append(segment.highlight_score)

            if segment.category_scores:
                highlight_data["category_scores"] = {
                    k: round(v, 4) for k, v in segment.category_scores.items()
                }

            # Add clustering info
            if segment.cluster_id is not None:
                highlight_data["cluster_id"] = segment.cluster_id
                highlight_data["cluster_theme"] = segment.cluster_theme
                cluster_counts[segment.cluster_id] = cluster_counts.get(segment.cluster_id, 0) + 1

            # Add diversity info
            if segment.diversity_score < 1.0:
                highlight_data["diversity_score"] = round(segment.diversity_score, 3)

            # Add expansion info if present
            if segment.is_expanded and segment.scene_indices:
                highlight_data["merged_scenes"] = len(segment.scene_indices)
                highlight_data["scene_indices"] = segment.scene_indices

            manifest["highlights"].append(highlight_data)

        # Build summary
        manifest["summary"] = {
            "total_highlights": len(segments),
            "by_category": {
                cat: {
                    "count": count,
                    "avg_score": round(sum(category_scores.get(cat, [0])) / max(count, 1), 4)
                }
                for cat, count in category_counts.items()
            },
            "clusters_found": len(cluster_counts),
            "temporal_coverage": round(
                sum((s.end_time - s.start_time) for s in segments) / max(video_duration or 1, 1),
                3
            ) if video_duration else None
        }

        manifest_path = os.path.join(output_dir, "highlights_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Manifest saved: {manifest_path}")

        # Print category summary
        if category_counts:
            print(f"Category breakdown: {', '.join(f'{k}={v}' for k, v in sorted(category_counts.items(), key=lambda x: -x[1]))}")

        return manifest_path


def detect_and_extract(video_path, query="exciting dramatic moment",
                       top_n=5, output_dir=None, expand_segments=True,
                       target_duration=20, max_duration=30,
                       # New parameters
                       categorize=True, categories=None,
                       enable_diversity=None, min_score=0.0,
                       dynamic_count=None, enable_clustering=True):
    """
    Convenience function to detect highlights and extract clips.

    Args:
        video_path: Path to video file
        query: Text describing desired highlights
        top_n: Number of highlights to extract (may be overridden by dynamic_count)
        output_dir: Output directory (optional)
        expand_segments: If True, merge adjacent scenes into longer highlights
        target_duration: Target clip length when expanding (seconds)
        max_duration: Maximum clip length when expanding (seconds)
        categorize: If True, score against multiple categories
        categories: Dict of category_name -> query (default: DEFAULT_CATEGORIES)
        enable_diversity: If True, apply temporal diversity (auto for >5min videos)
        min_score: Filter highlights below this threshold
        dynamic_count: If True, auto-scale count based on video duration
        enable_clustering: If True, cluster segments by visual similarity

    Returns:
        Tuple of (segments, clip_paths, manifest_path)
    """
    detector = HighlightDetector()

    # Get video duration for manifest
    video_duration = detector._get_video_duration(video_path)

    # Detect highlights
    segments = detector.detect_highlights(
        video_path, query=query, top_n=top_n,
        expand_segments=expand_segments,
        target_duration=target_duration,
        max_duration=max_duration,
        categorize=categorize,
        categories=categories,
        enable_diversity=enable_diversity,
        min_score=min_score,
        dynamic_count=dynamic_count,
        enable_clustering=enable_clustering
    )

    if not segments:
        print("No highlights detected.")
        return [], [], None

    # Extract clips
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    if output_dir is None:
        output_dir = os.path.join(HIGHLIGHTS_DIR, video_id)

    clip_paths = detector.extract_clips(
        video_path, segments, output_dir=output_dir,
        include_category_tag=categorize
    )

    # Export manifest
    categories_used = list((categories or DEFAULT_CATEGORIES).keys()) if categorize else None
    manifest_path = detector.export_manifest(
        video_path, segments, clip_paths, output_dir, query,
        video_duration=video_duration,
        categories_used=categories_used
    )

    return segments, clip_paths, manifest_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract highlight clips from video")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("query", nargs="?", default="exciting dramatic moment",
                        help="Text query describing desired highlights")
    parser.add_argument("top_n", nargs="?", type=int, default=5,
                        help="Number of highlights to extract (base count for dynamic)")

    # Duration/expansion options
    parser.add_argument("--duration", type=int, default=20,
                        help="Target highlight duration in seconds (default 20)")
    parser.add_argument("--max-duration", type=int, default=30,
                        help="Maximum highlight duration in seconds (default 30)")
    parser.add_argument("--no-expand", action="store_true",
                        help="Disable segment expansion (use single scenes)")

    # NEW: Category and diversity options
    parser.add_argument("--dynamic-count", action="store_true",
                        help="Auto-scale highlight count based on video length (auto for >5min)")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Filter highlights below this quality threshold (0-1)")
    parser.add_argument("--no-categorize", action="store_true",
                        help="Disable multi-category scoring")
    parser.add_argument("--categories", type=str, default=None,
                        help="Custom categories as comma-separated list (e.g., 'action,funny,dramatic')")
    parser.add_argument("--no-spread", action="store_true",
                        help="Disable temporal diversity (auto for >5min videos)")
    parser.add_argument("--no-clustering", action="store_true",
                        help="Disable embedding clustering")

    args = parser.parse_args()

    # Parse custom categories if provided
    custom_categories = None
    if args.categories:
        custom_categories = {
            cat.strip(): f"{cat.strip()} highlight moment interesting"
            for cat in args.categories.split(",")
        }

    segments, clips, manifest = detect_and_extract(
        args.video_path,
        query=args.query,
        top_n=args.top_n,
        expand_segments=not args.no_expand,
        target_duration=args.duration,
        max_duration=args.max_duration,
        # New options
        categorize=not args.no_categorize,
        categories=custom_categories,
        enable_diversity=False if args.no_spread else None,  # None = auto
        min_score=args.min_score,
        dynamic_count=True if args.dynamic_count else None,  # None = auto
        enable_clustering=not args.no_clustering
    )

    print(f"\n=== Results ===")
    print(f"Detected {len(segments)} highlights")
    print(f"Extracted {len(clips)} clips")
    if manifest:
        print(f"Manifest: {manifest}")
