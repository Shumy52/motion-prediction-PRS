"""SDD annotation preprocessing utilities."""

from __future__ import annotations

import json
import logging
import math
import random
import shlex
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Constants for the pipeline."""

    raw_annotations_dir: Path | str = Path("data/raw/archive/annotations")
    output_dir: Path | str = Path("data/processed/sdd_gru")
    past_length: int = 8
    future_length: int = 12
    frame_stride: int = 1
    drop_lost: bool = True
    drop_generated: bool = True
    drop_occluded: bool = True
    min_track_length: int | None = None
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 13
    dataset_name: str = "sdd_gru"

    def __post_init__(self) -> None:
        self.raw_annotations_dir = Path(self.raw_annotations_dir)
        self.output_dir = Path(self.output_dir)
        if self.past_length <= 0 or self.future_length <= 0:
            raise ValueError("past_length and future_length must be positive")
        if self.frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        if not (0.0 < self.train_ratio < 1.0):
            raise ValueError("train_ratio must lie in (0, 1)")
        if not (0.0 <= self.val_ratio < 1.0):
            raise ValueError("val_ratio must lie in [0, 1)")
        if self.train_ratio + self.val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio must be < 1")
        if self.min_track_length is None:
            self.min_track_length = self.past_length + self.future_length
        if self.min_track_length < self.past_length + self.future_length:
            raise ValueError(
                "min_track_length must be at least past_length + future_length"
            )

    def to_serializable(self) -> Dict[str, object]:
        data = asdict(self)
        data["raw_annotations_dir"] = str(self.raw_annotations_dir)
        data["output_dir"] = str(self.output_dir)
        return data


@dataclass
class TrackPoint:
    frame: int
    x: float
    y: float


@dataclass
class SequenceRecord:
    past: np.ndarray
    future: np.ndarray
    scene: str
    video: str
    track_key: str
    start_frame: int


class SDDPreprocessor:
    """Turns raw annotation text files into training-ready numpy arrays."""

    def __init__(self, config: PreprocessConfig) -> None:
        self.config = config

    def run(self) -> Dict[str, object]:
        annotation_files = sorted(
            self.config.raw_annotations_dir.glob("*/*/annotations.txt")
        )
        if not annotation_files:
            raise FileNotFoundError(
                f"No annotation files found under {self.config.raw_annotations_dir}"
            )

        logger.info("Found %d annotation files", len(annotation_files))
        records: List[SequenceRecord] = []

        for annotation_path in annotation_files:
            scene = annotation_path.parent.parent.name
            video = annotation_path.parent.name
            logger.info("Processing %s/%s", scene, video)
            track_points = self._load_track_points(annotation_path)
            for track_id, points in track_points.items():
                segments = self._split_segments(points)
                for segment in segments:
                    for past, future, start_frame in self._segment_to_sequences(segment):
                        track_key = f"{scene}/{video}/{track_id}"
                        records.append(
                            SequenceRecord(
                                past=past,
                                future=future,
                                scene=scene,
                                video=video,
                                track_key=track_key,
                                start_frame=start_frame,
                            )
                        )

        if not records:
            raise RuntimeError("No sequences produced. Check filtering parameters.")

        logger.info("Built %d sequences", len(records))
        split_lookup = self._assign_splits(records)
        split_payload = self._bucket_by_split(records, split_lookup)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        metadata = self._save_outputs(split_payload)
        return metadata

    def _load_track_points(self, annotation_path: Path) -> Dict[int, List[TrackPoint]]:
        track_points: Dict[int, List[TrackPoint]] = defaultdict(list)
        with annotation_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                fields = shlex.split(line)
                if len(fields) < 10:
                    continue
                track_id = int(fields[0])
                xmin, ymin, xmax, ymax = map(float, fields[1:5])
                frame = int(fields[5])
                lost = int(fields[6])
                occluded = int(fields[7])
                generated = int(fields[8])
                if self.config.drop_lost and lost:
                    continue
                if self.config.drop_occluded and occluded:
                    continue
                if self.config.drop_generated and generated:
                    continue
                x_center = (xmin + xmax) / 2.0
                y_center = (ymin + ymax) / 2.0
                track_points[track_id].append(TrackPoint(frame, x_center, y_center))
        for points in track_points.values():
            points.sort(key=lambda item: item.frame)
        return track_points

    def _split_segments(self, points: Sequence[TrackPoint]) -> List[List[TrackPoint]]:
        segments: List[List[TrackPoint]] = []
        if not points:
            return segments
        stride = self.config.frame_stride
        segment: List[TrackPoint] = [points[0]]
        for prev, curr in zip(points, points[1:]):
            if curr.frame - prev.frame == stride:
                segment.append(curr)
            else:
                if len(segment) >= self.config.min_track_length:
                    segments.append(segment)
                segment = [curr]
        if len(segment) >= self.config.min_track_length:
            segments.append(segment)
        return segments

    def _segment_to_sequences(
        self, segment: Sequence[TrackPoint]
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, int]]:
        past_len = self.config.past_length
        future_len = self.config.future_length
        window = past_len + future_len
        coords = np.array([(p.x, p.y) for p in segment], dtype=np.float32)
        frames = [p.frame for p in segment]
        for start_idx in range(0, len(segment) - window + 1):
            history = coords[start_idx : start_idx + past_len]
            target = coords[start_idx + past_len : start_idx + window]
            yield history, target, frames[start_idx]

    def _assign_splits(self, records: Sequence[SequenceRecord]) -> Dict[str, str]:
        track_ids = sorted({record.track_key for record in records})
        rng = random.Random(self.config.seed)
        rng.shuffle(track_ids)
        n_tracks = len(track_ids)
        n_train = max(1, int(math.floor(n_tracks * self.config.train_ratio)))
        n_val = int(math.floor(n_tracks * self.config.val_ratio))
        remaining = n_tracks - n_train - n_val
        if remaining <= 0:
            n_val = max(0, n_val - 1)
            remaining = n_tracks - n_train - n_val
        train_keys = set(track_ids[:n_train])
        val_keys = set(track_ids[n_train : n_train + n_val])
        test_keys = set(track_ids[n_train + n_val :])
        if not test_keys:
            test_keys = set(track_ids) - train_keys - val_keys
        lookup: Dict[str, str] = {}
        for key in train_keys:
            lookup[key] = "train"
        for key in val_keys:
            lookup[key] = "val"
        for key in test_keys:
            lookup[key] = "test"
        return lookup

    def _bucket_by_split(
        self,
        records: Sequence[SequenceRecord],
        split_lookup: Dict[str, str],
    ) -> Dict[str, Dict[str, List[object]]]:
        payload: Dict[str, Dict[str, List[object]]] = {
            split: {
                "past": [],
                "future": [],
                "scene": [],
                "video": [],
                "track_key": [],
                "start_frame": [],
            }
            for split in ("train", "val", "test")
        }
        for record in records:
            split = split_lookup.get(record.track_key, "train")
            bucket = payload[split]
            bucket["past"].append(record.past)
            bucket["future"].append(record.future)
            bucket["scene"].append(record.scene)
            bucket["video"].append(record.video)
            bucket["track_key"].append(record.track_key)
            bucket["start_frame"].append(record.start_frame)
        return payload

    def _save_outputs(
        self, payload: Dict[str, Dict[str, List[object]]]
    ) -> Dict[str, object]:
        stats = {}
        for split_name, data in payload.items():
            past = self._stack_array(data["past"], (0, self.config.past_length, 2))
            future = self._stack_array(data["future"], (0, self.config.future_length, 2))
            scene = np.array(data["scene"], dtype=np.str_)
            video = np.array(data["video"], dtype=np.str_)
            track_key = np.array(data["track_key"], dtype=np.str_)
            start_frame = np.array(data["start_frame"], dtype=np.int32)
            out_path = self.config.output_dir / f"{split_name}.npz"
            np.savez_compressed(
                out_path,
                past=past,
                future=future,
                scene=scene,
                video=video,
                track_key=track_key,
                start_frame=start_frame,
            )
            stats[split_name] = int(past.shape[0])
            logger.info("Saved %s (%d sequences)", out_path, stats[split_name])

        train_positions = None
        if payload["train"]["past"]:
            train_positions = np.concatenate(
                [
                    np.stack(payload["train"]["past"]).reshape(-1, 2),
                    np.stack(payload["train"]["future"]).reshape(-1, 2),
                ],
                axis=0,
            )
        normalization = None
        if train_positions is not None and len(train_positions):
            normalization = {
                "mean": train_positions.mean(axis=0).tolist(),
                "std": train_positions.std(axis=0).tolist(),
            }

        metadata = {
            "dataset_name": self.config.dataset_name,
            "config": self.config.to_serializable(),
            "num_sequences": stats,
            "normalization": normalization,
        }
        metadata_path = self.config.output_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        logger.info("Wrote %s", metadata_path)
        return metadata

    def _stack_array(self, tensors: List[np.ndarray], shape_hint: Tuple[int, ...]) -> np.ndarray:
        if not tensors:
            return np.zeros(shape_hint, dtype=np.float32)
        return np.stack(tensors).astype(np.float32)


__all__ = ["PreprocessConfig", "SDDPreprocessor"]
