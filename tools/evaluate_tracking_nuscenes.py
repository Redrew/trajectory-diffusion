# evaluate using nuscenes evaluation
import warnings

warnings.filterwarnings(
    action="ignore",
    category=FutureWarning,
)

import json
import os
from argparse import ArgumentParser

import numpy as np
import pyquaternion
from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory as track_configs
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.utils import splits

from data.utils import load
from paths import PATHS

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument(
        "--tracker",
        default="greedy_tracker",
        choices=["greedy_tracker", "ab3dmot_tracker"],
    )
    argparser.add_argument("--split", default="val", choices=["val", "test"])
    config = argparser.parse_args()
    config.dataset = "nuscenes"
    config.nusc_version = "v1.0-trainval"

    results_dir = os.path.abspath(
        os.path.join("results", f"{config.dataset}-{config.split}", config.tracker)
    )
    outputs_dir = os.path.join(results_dir, "outputs")
    dataset_dir = PATHS["nuscenes"]["val"]["dataset_dir"]
    nuscenes_out_dir = os.path.join(results_dir, "nuscenes_eval")
    nuscenes_track_predictions_path = os.path.join(
        nuscenes_out_dir, "nuscenes_track_predictions.json"
    )
    nusc = NuScenes(config.nusc_version, dataroot=dataset_dir)

    print(f"Evaluating {config.tracker} tracker on {config.split} set")
    print(f"Loading labels and track predictions from directory {outputs_dir}")
    labels = load(
        os.path.join("results", f"{config.dataset}-{config.split}", "labels.pkl")
    )
    track_predictions = load(os.path.join(outputs_dir, "track_predictions.pkl"))

    nuscenes_track_predictions = {
        "results": {},
        "meta": {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        },
    }
    scene_name2token = {scene["name"]: scene["token"] for scene in nusc.scene}
    scene_names = getattr(splits, config.split)
    seq_ids_in_set = [scene_name2token[n] for n in scene_names]
    assert track_predictions.keys() >= set(seq_ids_in_set)

    for seq_id, frames in track_predictions.items():
        if seq_id not in seq_ids_in_set:
            continue
        for frame, label_frame in zip(frames, labels[seq_id]):
            assert frame["timestamp_ns"] == label_frame["timestamp_ns"]
            rotation = np.array(
                [
                    pyquaternion.Quaternion(
                        axis=[0, 0, 1], radians=yaw
                    ).elements.tolist()
                    for yaw in frame["yaw"]
                ]
            )
            frame_id = label_frame["frame_id"]
            nuscenes_frame = {
                "sample_token": frame_id,
                "translation": frame["translation"],
                "size": frame["size"],
                "velocity": frame["velocity"][:, :2],
                "tracking_id": frame["track_id"],
                "tracking_name": frame["name"],
                "tracking_score": frame["score"],
                "rotation": rotation,
            }
            nuscenes_instances = [
                {
                    k: v[instance_i].tolist() if isinstance(v, np.ndarray) else v
                    for k, v in nuscenes_frame.items()
                }
                for instance_i in range(len(frame["translation"]))
            ]
            nuscenes_track_predictions["results"][frame_id] = nuscenes_instances

    os.makedirs(nuscenes_out_dir, exist_ok=True)
    print(
        f"Saving nuscenes format track predictions to {nuscenes_track_predictions_path}"
    )
    with open(nuscenes_track_predictions_path, "w") as f:
        json.dump(nuscenes_track_predictions, f)

    nusc_eval = TrackingEval(
        config=track_configs("tracking_lt3d"),
        result_path=nuscenes_track_predictions_path,
        eval_set=config.split,
        output_dir=nuscenes_out_dir,
        verbose=True,
        nusc_version=config.nusc_version,
        nusc_dataroot=dataset_dir,
    )
    metrics_summary = nusc_eval.main()
