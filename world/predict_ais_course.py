"""Predict 5/10/15-minute AIS vessel courses from a trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ais_course_transformer import AISCoursePredictor, history_features_for_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict future AIS vessel course.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ais-csv", type=Path, required=True)
    parser.add_argument("--mmsi", type=int, required=True)
    parser.add_argument("--time", required=True, help="UTC query time.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    arguments = parser.parse_args()

    predictor = AISCoursePredictor(arguments.checkpoint, device=arguments.device)
    features, history_end = history_features_for_prediction(
        arguments.ais_csv,
        arguments.mmsi,
        arguments.time,
        history_minutes=predictor.history_minutes,
        sample_interval_seconds=predictor.sample_interval_seconds,
    )
    directions, courses = predictor.predict(features)
    result = {
        "mmsi": arguments.mmsi,
        "history_end_utc": history_end.isoformat(),
        "predictions": [
            {
                "horizon_minutes": minutes,
                "course_degrees_true": float(course),
                "east": float(direction[0]),
                "north": float(direction[1]),
            }
            for minutes, direction, course in zip(
                predictor.horizons_minutes, directions, courses
            )
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
