"""Persons from frame to frame: each frame's detections matched to the tracks by their boxes' overlap (greedily, best
first, at least ``min_iou``); a track keeps its id while it is matched, and ends after ``max_missing`` frames without a
match. Ids are 1, 2, ... (as the Azure Kinect's body ids, never reused in a run)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Overlap of boxes (N, 4) and (M, 4) x0, y0, x1, y1 -> (N, M)."""
    a, b = a[:, None], b[None]
    w = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = w * h
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])  # noqa: E731
    return inter / np.maximum(area(a) + area(b) - inter, 1e-9)


@dataclass
class Track:
    id: int
    box: np.ndarray
    missing: int = 0


class Tracker:
    def __init__(self, min_iou: float = 0.3, max_missing: int = 10):
        self.min_iou, self.max_missing = min_iou, max_missing
        self.tracks: list[Track] = []
        self.next_id = 1

    def update(self, boxes: np.ndarray) -> list[int]:
        """The frame's boxes (N, 4) -> their ids, in the boxes' order."""
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        ids = [0] * len(boxes)
        if self.tracks and len(boxes):
            overlap = iou(np.stack([t.box for t in self.tracks]), boxes)
            while True:
                t, d = np.unravel_index(int(np.argmax(overlap)), overlap.shape)
                if overlap[t, d] < self.min_iou:
                    break
                track = self.tracks[t]
                track.box, track.missing = boxes[d], 0
                ids[d] = track.id
                overlap[t, :], overlap[:, d] = -1, -1
        matched = set(ids)
        for track in self.tracks:
            if track.id not in matched:
                track.missing += 1
        self.tracks = [t for t in self.tracks if t.missing <= self.max_missing]
        for d, i in enumerate(ids):
            if i == 0:
                ids[d] = self.next_id
                self.tracks.append(Track(self.next_id, boxes[d]))
                self.next_id += 1
        return ids
