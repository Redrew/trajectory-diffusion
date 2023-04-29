import os
from argparse import ArgumentParser

import torch
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes

from data.utils import *
from paths import PATHS

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument(
        "--tracker",
        default="greedy_tracker",
        choices=["greedy_tracker", "ab3dmot_tracker"],
    )
    argparser.add_argument("--dataset", default="av2", choices=["av2", "nuscenes"])
    argparser.add_argument("--split", default="val", choices=["val", "test"])
    config = argparser.parse_args()

    outputs_dir = os.path.join(
        "results", f"{config.dataset}-{config.split}", config.tracker, "outputs"
    )
    track_predictions = load(os.path.join(outputs_dir, "track_predictions.pkl"))

    packed_prediction_lookup = {}
    paths = PATHS[config.dataset][config.split]
    if config.dataset == "av2":
        from data.av2_adapter import (
            read_city_SE3_ego_by_seq_id,
            transform_to_ego_reference,
            unpack_labels,
        )

        raw_labels = load(paths["infos"])
        label_list = unpack_labels(raw_labels)
        labels = group_frames(label_list)
        city_SE3_by_seq_id = read_city_SE3_ego_by_seq_id(
            paths["dataset_dir"], seq_ids=labels.keys()
        )
        ego_track_predictions = transform_to_ego_reference(
            track_predictions, city_SE3_by_seq_id
        )
    elif config.dataset == "nuscenes":
        from nuscenes import NuScenes

        from data.nuscenes_adapter import (
            transform_to_ego_reference,
            unpack_and_annotate_labels,
        )

        raw_labels_dict = load(paths["infos"])
        raw_labels, nusc_version = (
            raw_labels_dict["infos"],
            raw_labels_dict["metadata"]["version"],
        )
        raw_labels = list(sorted(raw_labels, key=lambda e: e["timestamp"]))
        nusc = NuScenes(nusc_version, dataroot=paths["dataset_dir"])
        label_list = unpack_and_annotate_labels(raw_labels, nusc, classes=[])
        ego_track_predictions = transform_to_ego_reference(track_predictions)

    for frame in ungroup_frames(ego_track_predictions):
        data = np.concatenate(
            [
                frame["translation"],
                frame["size"],
                frame["yaw"].reshape((-1, 1)),
                frame["velocity"],
            ],
            axis=-1,
        )
        packed_prediction_lookup[(frame["seq_id"], frame["timestamp_ns"])] = {
            "pts_bbox": {
                "boxes_3d": LiDARInstance3DBoxes(torch.tensor(data), box_dim=9),
                "scores_3d": torch.tensor(frame["score"]),
                "labels_3d": torch.tensor(frame["label"], dtype=int),
            }
        }

    packed_predictions = [
        packed_prediction_lookup[(frame["seq_id"], frame["timestamp_ns"])]
        for frame in label_list
    ]
    save(packed_predictions, os.path.join(outputs_dir, "mmdet3d_predictions.pkl"))
