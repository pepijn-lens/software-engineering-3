"""
Measure average crash video duration for Hill Climbing vs Random Search.
Video length ≈ time until crash (shorter = crash earlier in episode).

Run: python measure_crash_video_durations.py
"""

import logging
from pathlib import Path

import numpy as np
from moviepy import VideoFileClip

# Reduce moviepy/imageio verbosity
logging.getLogger("moviepy").setLevel(logging.WARNING)


def get_durations(video_dir: str) -> list[float]:
    """Collect durations (seconds) of all .mp4 files under video_dir."""
    video_dir = Path(video_dir)
    if not video_dir.exists():
        return []
    durations = []
    for mp4 in video_dir.rglob("*.mp4"):
        try:
            with VideoFileClip(str(mp4)) as clip:
                durations.append(clip.duration)
        except Exception as e:
            print(f"  Skip {mp4}: {e}")
    return durations


def main():
    base = Path(__file__).parent / "videos"
    hc_dir = base / "hill_climbing"
    rs_dir = base / "random_search"

    hc_durations = get_durations(hc_dir)
    rs_durations = get_durations(rs_dir)

    def stats(name: str, d: list[float]) -> None:
        if not d:
            print(f"{name}: no videos found")
            return
        a = np.array(d)
        print(f"{name}:")
        print(f"  count   = {len(a)}")
        print(f"  mean    = {a.mean():.2f} s")
        print(f"  std     = {a.std():.2f} s")
        print(f"  min     = {a.min():.2f} s")
        print(f"  max     = {a.max():.2f} s")
        print()

    print("Average crash video length (≈ time until crash)\n")
    stats("Hill Climbing", hc_durations)
    stats("Random Search", rs_durations)

    if hc_durations and rs_durations:
        hc_mean = np.mean(hc_durations)
        rs_mean = np.mean(rs_durations)
        print(f"Difference: HC mean - RS mean = {hc_mean - rs_mean:.2f} s")
        print("(Shorter mean = crashes tend to happen earlier in the episode.)")


if __name__ == "__main__":
    main()
