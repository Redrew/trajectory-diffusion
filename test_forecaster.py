import json
import os
from argparse import ArgumentParser

from data.utils import *
from forecasting.evaluate import (
    av2_velocity,
    convert_forecast_labels,
    evaluate,
    nuscenes_velocity,
)
from forecasting.linear_forecaster import generate_linear_forecasts
from forecasting.lstm import generate_forecasts_from_model

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--dataset", default="av2", choices=["av2", "nuscenes"])
    argparser.add_argument("--split", default="val", choices=["val", "test"])
    argparser.add_argument(
        "--tracker",
        default="greedy_tracker",
        choices=["greedy_tracker", "ab3dmot_tracker"],
    )
    argparser.add_argument(
        "--forecaster",
        default="linear_forecaster",
        choices=["linear_forecaster", "lstm_forecaster", "transformer_forecaster"],
    )
    argparser.add_argument(
        "--device",
        default="cuda",
    )
    config = argparser.parse_args()
    config.time_delta = 0.5
    config.num_timesteps = 6
    config.K = 5
    config.ego_distance_threshold = 50

    if config.dataset == "av2":
        class_names = AV2_CLASS_NAMES
        class_velocity = av2_velocity
    elif config.dataset == "nuscenes":
        class_names = NUSCENES_CLASS_NAMES
        class_velocity = nuscenes_velocity
    _dataset_dir = os.path.join("results", f"{config.dataset}-{config.split}")
    results_dir = os.path.join(_dataset_dir, f"{config.tracker}_{config.forecaster}")
    track_predictions = load(
        os.path.join(_dataset_dir, config.tracker, "outputs", "track_predictions.pkl")
    )

    if config.forecaster == "linear_forecaster":
        VELOCITY_SCALAR = [1.0, 1.2, 1.4, 0.8, 0.6]
        forecasts = generate_linear_forecasts(
            track_predictions,
            velocity_scalar=VELOCITY_SCALAR,
            num_timesteps=config.num_timesteps,
            time_delta=config.time_delta,
        )
    elif config.forecaster == "lstm_forecaster":
        import torch

        from forecasting.lstm import *

        model = torch.load(os.path.join("models", config.dataset, "lstm.pt"))
        # run inference
        forecasts = generate_forecasts_from_model(
            model,
            load("results/av2-val/greedy_tracker/outputs/track_predictions.pkl"),
            config.num_timesteps,
            config.K,
            config.device,
        )
    elif config.forecaster == "transformer_forecaster":
        import torch

        from forecasting.transformer import *

        model = torch.load(os.path.join("models", config.dataset, "transformer.pt"))
        # run inference
        forecasts = generate_forecasts_from_model(
            model,
            load("results/av2-val/greedy_tracker/outputs/track_predictions.pkl"),
            config.num_timesteps,
            config.K,
            config.device,
        )
    else:
        raise Exception(f"Forecaster {config.forecaster} not supported")
    save(forecasts, os.path.join(results_dir, "forecast_predictions.pkl"))

    # evaluate
    # add ego_translation field to predictions
    if config.split == "test":
        print("Skipping evaluation")
    else:
        raw_labels = load(
            os.path.join("dataset", f"{config.dataset}-{config.split}", "labels.pkl")
        )
        for seq_id, frames in raw_labels.items():
            for frame in frames:
                if seq_id in forecasts and frame["timestamp_ns"] in forecasts[seq_id]:
                    for agent in forecasts[seq_id][frame["timestamp_ns"]]:
                        agent["ego_translation"] = frame["ego_translation"][:2]

        labels = convert_forecast_labels(
            raw_labels,
            num_timesteps=config.num_timesteps,
            ego_distance_threshold=config.ego_distance_threshold,
        )
        metrics = evaluate(
            forecasts,
            labels,
            config.K,
            class_names,
            class_velocity,
            config.num_timesteps,
            ego_distance_threshold=config.ego_distance_threshold,
        )
        print(metrics)
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
