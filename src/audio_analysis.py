"""
Audio analysis module for highlight detection.

Uses ffmpeg + numpy for reliable audio peak detection.
This approach avoids librosa's numba dependency which has
compatibility issues with Python 3.13.

FFmpeg is the gold standard for audio/video processing,
used by YouTube, Netflix, and thousands of production systems.
"""

import os
import subprocess
import tempfile
import wave
import numpy as np


def _extract_audio_to_wav(video_path, sample_rate=22050):
    """
    Extract audio from video to a temporary WAV file using ffmpeg.

    Args:
        video_path: Path to video file
        sample_rate: Target sample rate

    Returns:
        Path to temporary WAV file
    """
    # Create temp file
    fd, wav_path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)

    # Use ffmpeg to extract audio
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # PCM 16-bit
        '-ar', str(sample_rate),  # Sample rate
        '-ac', '1',  # Mono
        wav_path
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")

    return wav_path


def _read_wav(wav_path):
    """
    Read audio samples from a WAV file.

    Returns:
        Tuple of (samples as numpy array, sample_rate)
    """
    with wave.open(wav_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)

    # Convert to numpy array
    samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    # Normalize to -1 to 1
    samples = samples / 32768.0

    return samples, sample_rate


def _compute_rms_energy(samples, hop_length=512):
    """
    Compute RMS energy for each frame.

    Args:
        samples: Audio samples array
        hop_length: Samples between frames

    Returns:
        Array of RMS energy values
    """
    n_frames = len(samples) // hop_length
    rms = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop_length
        end = start + hop_length
        frame = samples[start:end]
        rms[i] = np.sqrt(np.mean(frame ** 2))

    return rms


def _detect_peaks(signal, pre_max=3, post_max=3, pre_avg=3, post_avg=5,
                  delta=0.0, wait=10):
    """
    Detect peaks in a signal (similar to librosa.util.peak_pick).

    Args:
        signal: Input signal array
        pre_max, post_max: Window for local max detection
        pre_avg, post_avg: Window for averaging
        delta: Threshold offset
        wait: Minimum frames between peaks

    Returns:
        Array of peak indices
    """
    peaks = []
    last_peak = -wait

    # Normalize signal
    signal_norm = signal / (np.max(signal) + 1e-10)

    for i in range(pre_max, len(signal) - post_max):
        # Check if local maximum
        window = signal_norm[i - pre_max:i + post_max + 1]
        if signal_norm[i] != np.max(window):
            continue

        # Check against local average
        avg_start = max(0, i - pre_avg)
        avg_end = min(len(signal_norm), i + post_avg + 1)
        local_avg = np.mean(signal_norm[avg_start:avg_end])

        if signal_norm[i] >= local_avg + delta and i - last_peak >= wait:
            peaks.append(i)
            last_peak = i

    return np.array(peaks)


def extract_audio_peaks(video_path, sr=22050, hop_length=512,
                        pre_max=3, post_max=3, pre_avg=3, post_avg=5,
                        delta=0.05, wait=10):
    """
    Extract audio peak timestamps from a video file.

    Uses ffmpeg for audio extraction and numpy for analysis.
    This identifies moments where audio energy increases significantly.

    Args:
        video_path: Path to video file
        sr: Sample rate for audio analysis (22050 is standard)
        hop_length: Samples between analysis frames
        pre_max, post_max: Window for local max detection
        pre_avg, post_avg: Window for averaging
        delta: Threshold for peak picking
        wait: Minimum frames between peaks

    Returns:
        List of peak timestamps in seconds
    """
    print(f"Extracting audio peaks from {os.path.basename(video_path)}...")

    wav_path = None
    try:
        # Extract audio to WAV
        wav_path = _extract_audio_to_wav(video_path, sample_rate=sr)

        # Read audio samples
        samples, actual_sr = _read_wav(wav_path)

        # Compute RMS energy (onset strength approximation)
        rms = _compute_rms_energy(samples, hop_length=hop_length)

        # Compute onset envelope (differential of RMS)
        onset_env = np.diff(rms)
        onset_env = np.maximum(0, onset_env)  # Only increases
        onset_env = np.insert(onset_env, 0, 0)  # Pad to match length

        # Detect peaks
        peaks = _detect_peaks(
            onset_env,
            pre_max=pre_max,
            post_max=post_max,
            pre_avg=pre_avg,
            post_avg=post_avg,
            delta=delta,
            wait=wait
        )

        # Convert frame indices to time
        peak_times = peaks * hop_length / actual_sr

        print(f"Found {len(peak_times)} audio peaks")
        return peak_times.tolist()

    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


def get_audio_energy_profile(video_path, sr=22050, hop_length=512):
    """
    Get the audio energy profile over time.

    Returns an array of RMS energy values and corresponding timestamps.
    Useful for finding generally "loud" vs "quiet" sections.

    Args:
        video_path: Path to video file
        sr: Sample rate
        hop_length: Samples between analysis frames

    Returns:
        Tuple of (times, energy) arrays as lists
    """
    print(f"Computing audio energy profile for {os.path.basename(video_path)}...")

    wav_path = None
    try:
        # Extract audio to WAV
        wav_path = _extract_audio_to_wav(video_path, sample_rate=sr)

        # Read audio samples
        samples, actual_sr = _read_wav(wav_path)

        # Compute RMS energy
        rms = _compute_rms_energy(samples, hop_length=hop_length)

        # Compute timestamps
        times = np.arange(len(rms)) * hop_length / actual_sr

        return times.tolist(), rms.tolist()

    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


def get_energy_at_timestamp(energy_times, energy_values, timestamp, window=0.5):
    """
    Get the average energy around a specific timestamp.

    Args:
        energy_times: List of timestamps from get_audio_energy_profile
        energy_values: List of energy values
        timestamp: Target timestamp in seconds
        window: Time window around timestamp to average (seconds)

    Returns:
        Average energy value (0-1 normalized)
    """
    energy_times = np.array(energy_times)
    energy_values = np.array(energy_values)

    # Find indices within window
    mask = (energy_times >= timestamp - window) & (energy_times <= timestamp + window)

    if not np.any(mask):
        return 0.0

    # Normalize to 0-1 range
    max_energy = np.max(energy_values) if np.max(energy_values) > 0 else 1.0
    return float(np.mean(energy_values[mask]) / max_energy)


def is_near_audio_peak(timestamp, peak_times, tolerance=1.0):
    """
    Check if a timestamp is near any audio peak.

    Args:
        timestamp: Target timestamp in seconds
        peak_times: List of peak timestamps
        tolerance: How close is "near" (in seconds)

    Returns:
        Boolean indicating if timestamp is near a peak
    """
    for peak in peak_times:
        if abs(timestamp - peak) <= tolerance:
            return True
    return False


def get_loudest_segments(video_path, segment_duration=5.0, top_n=10, sr=22050):
    """
    Find the N loudest segments in a video.

    Args:
        video_path: Path to video file
        segment_duration: Duration of each segment in seconds
        top_n: Number of segments to return
        sr: Sample rate

    Returns:
        List of (start_time, end_time, avg_energy) tuples
    """
    print(f"Finding {top_n} loudest {segment_duration}s segments...")

    times, energy = get_audio_energy_profile(video_path, sr=sr)
    times = np.array(times)
    energy = np.array(energy)

    if len(times) == 0:
        return []

    # Calculate segments
    total_duration = times[-1]
    segments = []

    current_time = 0.0
    while current_time < total_duration:
        end_time = min(current_time + segment_duration, total_duration)

        # Get energy for this segment
        mask = (times >= current_time) & (times < end_time)
        if np.any(mask):
            avg_energy = np.mean(energy[mask])
            segments.append((current_time, end_time, avg_energy))

        current_time = end_time

    # Sort by energy and return top N
    segments.sort(key=lambda x: x[2], reverse=True)
    return segments[:top_n]


def analyze_audio(video_path, sr=22050, hop_length=512,
                   pre_max=3, post_max=3, pre_avg=3, post_avg=5,
                   delta=0.05, wait=10):
    """
    Combined audio analysis - extracts audio ONCE and returns both peaks and energy.

    This is more efficient than calling extract_audio_peaks() and
    get_audio_energy_profile() separately, as it only extracts audio once.

    Args:
        video_path: Path to video file
        sr: Sample rate for audio analysis
        hop_length: Samples between analysis frames
        Other args: Peak detection parameters

    Returns:
        Dict with 'peak_times', 'energy_times', 'energy_values'
    """
    print(f"Analyzing audio from {os.path.basename(video_path)}...")

    wav_path = None
    try:
        # Extract audio ONCE
        wav_path = _extract_audio_to_wav(video_path, sample_rate=sr)

        # Read audio samples
        samples, actual_sr = _read_wav(wav_path)

        # Compute RMS energy
        rms = _compute_rms_energy(samples, hop_length=hop_length)

        # Compute timestamps for energy
        energy_times = np.arange(len(rms)) * hop_length / actual_sr

        # Compute onset envelope for peak detection
        onset_env = np.diff(rms)
        onset_env = np.maximum(0, onset_env)
        onset_env = np.insert(onset_env, 0, 0)

        # Detect peaks
        peaks = _detect_peaks(
            onset_env,
            pre_max=pre_max,
            post_max=post_max,
            pre_avg=pre_avg,
            post_avg=post_avg,
            delta=delta,
            wait=wait
        )

        # Convert peak indices to time
        peak_times = peaks * hop_length / actual_sr

        print(f"Found {len(peak_times)} audio peaks")

        return {
            'peak_times': peak_times.tolist(),
            'energy_times': energy_times.tolist(),
            'energy_values': rms.tolist()
        }

    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        peaks = extract_audio_peaks(video_path)
        print(f"\nPeak timestamps: {peaks[:10]}..." if len(peaks) > 10 else f"\nPeak timestamps: {peaks}")

        loudest = get_loudest_segments(video_path, top_n=5)
        print(f"\nLoudest segments:")
        for start, end, energy in loudest:
            print(f"  {start:.1f}s - {end:.1f}s (energy: {energy:.4f})")
    else:
        print("Usage: python audio_analysis.py <video_path>")
