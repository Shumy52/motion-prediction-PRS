#!/usr/bin/env python3
"""Command-line entrypoint for preprocessing the Stanford Drone Dataset."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.data_processing.preprocess import PreprocessConfig, SDDPreprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw SDD annotations into trajectory sequences"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/archive/annotations"),
        help="Directory that contains <scene>/<video>/annotations.txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/sdd_gru"),
        help="Destination directory for the processed numpy archives.",
    )
    parser.add_argument("--past-length", type=int, default=8, help="History length.")
    parser.add_argument("--future-length", type=int, default=12, help="Forecast length.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Required frame step.")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of tracks assigned to the training split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of tracks assigned to the validation split.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed for splits.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sdd_gru",
        help="Name stored inside metadata.json.",
    )
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=None,
        help="Optional override for the minimum contiguous frames per segment.",
    )
    parser.add_argument(
        "--keep-lost",
        action="store_true",
        help="Keep annotations flagged as lost (default drops them).",
    )
    parser.add_argument(
        "--keep-occluded",
        action="store_true",
        help="Keep annotations flagged as occluded (default drops them).",
    )
    parser.add_argument(
        "--keep-generated",
        action="store_true",
        help="Keep annotations flagged as generated (default drops them).",
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Verbosity for the logger.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    config = PreprocessConfig(
        raw_annotations_dir=args.raw_dir,
        output_dir=args.output_dir,
        past_length=args.past_length,
        future_length=args.future_length,
        frame_stride=args.frame_stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        dataset_name=args.dataset_name,
        min_track_length=args.min_track_length,
        drop_lost=not args.keep_lost,
        drop_occluded=not args.keep_occluded,
        drop_generated=not args.keep_generated,
    )
    preprocessor = SDDPreprocessor(config)
    metadata = preprocessor.run()
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
