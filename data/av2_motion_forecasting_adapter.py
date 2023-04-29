import json
import os
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from av2.map.map_api import ArgoverseStaticMap

from data.utils import *
from paths import PATHS

class_name_to_type = {
    "REGULAR_VEHICLE": "vehicle",
    "PEDESTRIAN": "pedestrian",
    "BICYCLIST": "cyclist",
    "MOTORCYCLIST": "motorcyclist",
    "WHEELED_RIDER": "cyclist",
    "BOLLARD": "unknown",
    "CONSTRUCTION_CONE": "unknown",
    "SIGN": "unknown",
    "CONSTRUCTION_BARREL": "unknown",
    "STOP_SIGN": "unknown",
    "MOBILE_PEDESTRIAN_CROSSING_SIGN": "unknown",
    "LARGE_VEHICLE": "vehicle",
    "BUS": "bus",
    "BOX_TRUCK": "vehicle",
    "TRUCK": "vehicle",
    "VEHICULAR_TRAILER": "vehicle",
    "TRUCK_CAB": "vehicle",
    "SCHOOL_BUS": "vehicle",
    "ARTICULATED_BUS": "vehicle",
    "MESSAGE_BOARD_TRAILER": "unknown",
    "BICYCLE": "riderless_bicycle",
    "MOTORCYCLE": "motorcyclist",
    "WHEELED_DEVICE": "riderless_bicycle",
    "WHEELCHAIR": "pedestrian",
    "STROLLER": "pedestrian",
    "DOG": "pedestrian",
}


def interpolate(df, factor=5):
    # upsample from 2Hz to 10Hz by linear interpolating states
    df.timestep *= factor
    df = df.set_index(df.timestep)
    df = df.reindex(range(df.index[0], df.index[-1] + 1))
    df = df.interpolate()
    df = df.fillna(method="ffill")
    df.timestep = df.timestep.astype(int)
    df.track_id = df.track_id.astype(int)
    df.object_category = df.object_category.astype(int)
    return df


def assign_timestamp_info(df):
    return df.assign(
        start_timestamp=int(df.timestamp.min()),
        end_timestamp=int(df.timestamp.max()),
        num_timestamps=len(df),
    )


def export_track_prediction(track_predictions, data_root: str, output_dir: str):
    data_root = Path(data_root)
    output_dir = Path(output_dir)

    for seq_id in progressbar(track_predictions.keys(), desc="processing sequences"):
        states = []
        frames = track_predictions[seq_id]
        for timestep, frame in enumerate(frames):
            for inst_i in range(len(frame["translation"])):
                states.append(
                    {
                        "observed": True,
                        "track_id": frame["track_id"][inst_i],
                        "object_type": class_name_to_type[frame["name"][inst_i]],
                        "object_category": 2,  # everything is SCORED_TRACK
                        "timestep": timestep,
                        "timestamp": str(frame["timestamp_ns"]),
                        "position_x": frame["translation"][inst_i, 0],
                        "position_y": frame["translation"][inst_i, 1],
                        "heading": frame["yaw"][inst_i],
                        "velocity_x": frame["velocity"][inst_i, 0],
                        "velocity_y": frame["velocity"][inst_i, 1],
                        "scenario_id": seq_id,
                        "focal_track_id": None,
                        "city": "",  # ok to be none
                    }
                )
        seq_df_2Hz = assign_timestamp_info(pd.DataFrame(states))
        seq_df_10Hz = (
            seq_df_2Hz.groupby("track_id").apply(interpolate).reset_index(drop=True)
        )
        seq_df_10Hz.timestep += 4

        # save map
        map_path = sorted((data_root / seq_id / "map").glob("log_map_archive_*.json"))[
            0
        ]
        avm = ArgoverseStaticMap.from_json(map_path)
        with open(map_path, "r") as f:
            static_map_elements = json.load(f)
        for id, lane_segment in static_map_elements["lane_segments"].items():
            lane_segment["centerline"] = [
                {"x": x, "y": y, "z": z}
                for x, y, z in avm.get_lane_segment_centerline(int(id))
            ]

        os.makedirs(Path("dataset") / "av2-map" / seq_id, exist_ok=True)
        augmented_map_path = (
            Path("dataset") / "av2-map" / seq_id / f"log_map_archive_{seq_id}.json"
        )
        with open(augmented_map_path, "w") as f:
            json.dump(static_map_elements, f)
        for offset in range(len(frames)):
            offset_seq_id = f"{seq_id}_{offset}"
            offset_seq_df = seq_df_10Hz.assign(
                timestep=45 + seq_df_10Hz.timestep - offset * 5
            )
            offset_seq_df = offset_seq_df[offset_seq_df.timestep >= 0]
            # create output dir
            os.makedirs(output_dir / offset_seq_id, exist_ok=True)
            df_path = output_dir / offset_seq_id / f"scenario_{offset_seq_id}.parquet"
            offset_seq_df.to_parquet(df_path)
            dest_map_path = (
                output_dir / offset_seq_id / f"log_map_archive_{offset_seq_id}.json"
            )
            if dest_map_path.exists():
                dest_map_path.unlink()
            os.symlink(augmented_map_path, dest_map_path)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument(
        "--split", default="val", type=str, choices=["train", "val", "test"]
    )
    argparser.add_argument("--tracker", default="greedy_tracker", type=str)
    config = argparser.parse_args()

    track_path = (
        f"results/av2-{config.split}/{config.tracker}/outputs/track_predictions.pkl"
    )
    data_root = PATHS["av2"][config.split]["dataset_dir"]
    output_dir = f"dataset/av2-{config.split}-mf"

    track_predictions = load(track_path)
    thresholds_by_class = {n: 0.4 for n in AV2_CLASS_NAMES}
    track_predictions = filter_by_class_thresholds(
        track_predictions, thresholds_by_class
    )
    export_track_prediction(track_predictions, data_root, output_dir)
