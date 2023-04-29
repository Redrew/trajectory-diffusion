import itertools
from collections import defaultdict

import numpy as np

from data.utils import filter_by_ego_xy_distance, progressbar

time_delta = 0.5
dist_th = [0.5, 1, 2, 4]
velocity_profile = ["static", "linear", "non-linear"]

nuscenes_velocity = {
    "car": 1.79,
    "truck": 1.38,
    "trailer": 0.68,
    "bus": 2.68,
    "construction_vehicle": 0.16,
    "bicycle": 0.72,
    "motorcycle": 1.88,
    "emergency_vehicle": 1.44,
    "adult": 0.89,
    "child": 0.39,
    "police_officer": 0.81,
    "construction_worker": 0.34,
    "stroller": 0.70,
    "personal_mobility": 0.04,
    "pushable_pullable": 0.10,
    "debris": 0.05,
    "traffic_cone": 0.05,
    "barrier": 0.06,
}

av2_velocity = {
    "REGULAR_VEHICLE": 2.36,
    "PEDESTRIAN": 0.80,
    "BICYCLIST": 3.61,
    "MOTORCYCLIST": 4.08,
    "WHEELED_RIDER": 2.03,
    "BOLLARD": 0.02,
    "CONSTRUCTION_CONE": 0.02,
    "SIGN": 0.05,
    "CONSTRUCTION_BARREL": 0.03,
    "STOP_SIGN": 0.09,
    "MOBILE_PEDESTRIAN_CROSSING_SIGN": 0.03,
    "LARGE_VEHICLE": 1.56,
    "BUS": 3.10,
    "BOX_TRUCK": 2.59,
    "TRUCK": 2.76,
    "VEHICULAR_TRAILER": 1.72,
    "TRUCK_CAB": 2.36,
    "SCHOOL_BUS": 4.44,
    "ARTICULATED_BUS": 4.58,
    "MESSAGE_BOARD_TRAILER": 0.41,
    "BICYCLE": 0.97,
    "MOTORCYCLE": 1.58,
    "WHEELED_DEVICE": 0.37,
    "WHEELCHAIR": 1.50,
    "STROLLER": 0.91,
    "DOG": 0.72,
}


def agent_velocity(agent):
    if "future_translation" in agent:  # ground_truth
        return (
            agent["future_translation"][0][:2] - agent["current_translation"][:2]
        ) / time_delta

    else:  # predictions
        res = []
        for i in range(agent["prediction"].shape[0]):
            res.append(
                (agent["prediction"][i][0][:2] - agent["current_translation"][:2])
                / time_delta
            )

        return res


def trajectory_type(agent, class_velocity, num_timesteps):
    forecast_scalar = np.linspace(0, 1, num_timesteps + 1)
    if "future_translation" in agent:  # ground_truth
        time = agent["future_translation"].shape[0] * time_delta
        static_target = agent["current_translation"][:2]
        linear_target = agent["current_translation"][:2] + time * agent["velocity"][:2]

        final_position = agent["future_translation"][-1][:2]

        threshold = 1 + forecast_scalar[
            len(agent["future_translation"])
        ] * class_velocity.get(agent["name"], 0)
        if np.linalg.norm(final_position - static_target) < threshold:
            return "static"
        elif np.linalg.norm(final_position - linear_target) < threshold:
            return "linear"
        else:
            return "non-linear"

    else:  # predictions
        res = []
        time = agent["prediction"].shape[1] * time_delta

        threshold = 1 + forecast_scalar[len(agent["prediction"])] * class_velocity.get(
            agent["name"], 0
        )
        for i in range(agent["prediction"].shape[0]):
            static_target = agent["current_translation"][:2]
            linear_target = (
                agent["current_translation"][:2] + time * agent["velocity"][i][:2]
            )

            final_position = agent["prediction"][i][-1][:2]

            if np.linalg.norm(final_position - static_target) < threshold:
                res.append("static")
            elif np.linalg.norm(final_position - linear_target) < threshold:
                res.append("linear")
            else:
                res.append("non-linear")

        return res


def center_distance(pred_box, gt_box):
    return np.linalg.norm(pred_box - gt_box)


def calc_ap(precision, min_recall=0, min_precision=0):
    """Calculated average precision."""

    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1

    prec = np.copy(precision)
    prec = prec[
        round(100 * min_recall) + 1 :
    ]  # Clip low recalls. +1 to exclude the min recall bin.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)


def evaluate(
    predictions,
    ground_truth,
    K,
    class_names,
    class_velocity,
    num_timesteps,
    ego_distance_threshold,
):
    res = {}

    for profile in velocity_profile:
        res[profile] = {}
        for cname in class_names:
            res[profile][cname] = {"mAP_F": [], "ADE": [], "FDE": []}

    gt_agents = []
    pred_agents = []
    for seq_id in progressbar(
        ground_truth.keys(), desc="categorizing trajectory types"
    ):
        for timestamp in ground_truth[seq_id].keys():
            gt = [agent for agent in ground_truth[seq_id][timestamp]]

            if seq_id in predictions and timestamp in predictions[seq_id]:
                pred = [
                    agent
                    for agent in predictions[seq_id][timestamp]
                    if np.linalg.norm(
                        agent["current_translation"][:2] - agent["ego_translation"][:2]
                    )
                    <= ego_distance_threshold
                ]
            else:
                pred = []

            for agent in gt:
                if agent["future_translation"].shape[0] < 1:
                    continue

                agent["seq_id"] = seq_id
                agent["timestamp"] = timestamp
                agent["velocity"] = agent_velocity(agent)
                agent["trajectory_type"] = trajectory_type(
                    agent, class_velocity, num_timesteps
                )

                gt_agents.append(agent)

            for agent in pred:
                agent["seq_id"] = seq_id
                agent["timestamp"] = timestamp
                agent["velocity"] = agent_velocity(agent)
                agent["trajectory_type"] = trajectory_type(
                    agent, class_velocity, num_timesteps
                )

                pred_agents.append(agent)

    outputs = []
    for cname, profile, th in progressbar(
        list(itertools.product(class_names, velocity_profile, dist_th)),
        desc="evaluating forecasts",
    ):
        cvel = class_velocity[cname]
        outputs.append(
            accumulate(
                pred_agents, gt_agents, K, cname, profile, cvel, th, num_timesteps
            )
        )

    for apf, ade, fde, cname, profile in outputs:
        res[profile][cname]["mAP_F"].append(apf)
        res[profile][cname]["ADE"].append(ade)
        res[profile][cname]["FDE"].append(fde)

    for cname in class_names:
        for profile in velocity_profile:
            res[profile][cname]["mAP_F"] = round(
                np.mean(res[profile][cname]["mAP_F"]), 3
            )
            res[profile][cname]["ADE"] = round(np.mean(res[profile][cname]["ADE"]), 3)
            res[profile][cname]["FDE"] = round(np.mean(res[profile][cname]["FDE"]), 3)

    mAP_F = np.nanmean(
        [
            metrics["mAP_F"]
            for traj_metrics in res.values()
            for metrics in traj_metrics.values()
        ]
    )
    ADE = np.nanmean(
        [
            metrics["ADE"]
            for traj_metrics in res.values()
            for metrics in traj_metrics.values()
        ]
    )
    FDE = np.nanmean(
        [
            metrics["FDE"]
            for traj_metrics in res.values()
            for metrics in traj_metrics.values()
        ]
    )
    res["mean_mAP_F"] = mAP_F
    res["mean_ADE"] = ADE
    res["mean_FDE"] = FDE

    return res


def accumulate(
    pred_agents, gt_agents, K, class_name, profile, velocity, threshold, num_timesteps
):
    def match(gt, pred, profile):
        if gt == profile:
            return True

        if gt == "ignore" and pred == profile:
            return True

        return False

    pred = [agent for agent in pred_agents if agent["name"] == class_name]
    gt = [
        agent
        for agent in gt_agents
        if agent["name"] == class_name and agent["trajectory_type"] == profile
    ]
    conf = [agent["detection_score"] for agent in pred]
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(conf))][::-1]
    gt_agents_by_frame = defaultdict(list)
    for agent in gt:
        gt_agents_by_frame[(agent["seq_id"], agent["timestamp"])].append(agent)

    npos = len(gt)
    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------

    gt_profiles, pred_profiles, agent_ade, agent_fde, tp, fp = [], [], [], [], [], []
    taken = set()  # Initially no gt bounding box is matched.
    for ind in sortind:
        pred_agent = pred[ind]
        min_dist = np.inf
        match_gt_idx = None

        gt_agents_in_frame = gt_agents_by_frame[
            (pred_agent["seq_id"], pred_agent["timestamp"])
        ]
        for gt_idx, gt_agent in enumerate(gt_agents_in_frame):
            if not (pred_agent["seq_id"], pred_agent["timestamp"], gt_idx) in taken:
                # Find closest match among ground truth boxes
                this_distance = center_distance(
                    gt_agent["current_translation"], pred_agent["current_translation"]
                )
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < threshold

        forecast_scalar = np.linspace(0, 1, num_timesteps + 1)
        if is_match:
            taken.add((pred_agent["seq_id"], pred_agent["timestamp"], match_gt_idx))
            gt_match_agent = gt_agents_in_frame[match_gt_idx]

            gt_len = gt_match_agent["future_translation"].shape[0]

            forecast_match_th = [
                threshold + forecast_scalar[i] * velocity for i in range(gt_len + 1)
            ]

            if K == 1:
                ind = np.argmax(pred_agent["score"])
                forecast_dist = [
                    center_distance(
                        gt_match_agent["future_translation"][i],
                        pred_agent["prediction"][ind][i],
                    )
                    for i in range(gt_len)
                ]
                forecast_match = [
                    dist < th for dist, th in zip(forecast_dist, forecast_match_th[1:])
                ]

                ade = np.mean(forecast_dist)
                fde = forecast_dist[-1]

            elif K == 5:
                forecast_dist, forecast_match = None, None
                ade, fde = np.inf, np.inf

                for ind in range(K):
                    curr_forecast_dist = [
                        center_distance(
                            gt_match_agent["future_translation"][i],
                            pred_agent["prediction"][ind][i],
                        )
                        for i in range(gt_len)
                    ]
                    curr_forecast_match = [
                        dist < th
                        for dist, th in zip(curr_forecast_dist, forecast_match_th[1:])
                    ]

                    curr_ade = np.mean(curr_forecast_dist)
                    curr_fde = curr_forecast_dist[-1]

                    if curr_ade < ade:
                        forecast_dist = curr_forecast_dist
                        forecast_match = curr_forecast_match
                        ade = curr_ade
                        fde = curr_fde

                agent_ade.append(ade)
                agent_fde.append(fde)
                tp.append(forecast_match[-1])
                fp.append(not forecast_match[-1])

            gt_profiles.append(profile)
            pred_profiles.append("gent_apf.append(ignore")

        else:
            tp.append(False)
            fp.append(True)

            ind = np.argmax(pred_agent["score"])
            gt_profiles.append("ignore")
            pred_profiles.append(pred_agent["trajectory_type"][ind])

    select = [match(gt, pred, profile) for gt, pred in zip(gt_profiles, pred_profiles)]
    tp = np.array(tp)[select]
    fp = np.array(fp)[select]

    if len(tp) == 0:
        return np.nan, np.nan, np.nan, class_name, profile

    np.sum(tp)
    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)

    prec = tp / (fp + tp)
    rec = tp / float(npos)

    rec_interp = np.linspace(0, 1, 101)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)

    apf = calc_ap(prec)

    return apf, np.mean(agent_ade), np.mean(agent_fde), class_name, profile


def convert_forecast_labels(labels, num_timesteps: int, ego_distance_threshold: float):
    def index_array_values(array_dict, index):
        return {
            k: v[index] if isinstance(v, np.ndarray) else v
            for k, v in array_dict.items()
        }

    def array_dict_iterator(array_dict, length):
        return (index_array_values(array_dict, i) for i in range(length))

    labels = filter_by_ego_xy_distance(labels, ego_distance_threshold)
    forecast_labels = {}
    for seq_id, frames in labels.items():
        frame_dict = {}
        for frame_idx, frame in enumerate(frames):
            forecast_instances = []
            for instance in array_dict_iterator(frame, len(frame["translation"])):
                future_translations = []
                for future_frame in frames[
                    frame_idx + 1 : frame_idx + 1 + num_timesteps
                ]:
                    if instance["track_id"] not in future_frame["track_id"]:
                        break
                    future_translations.append(
                        future_frame["translation"][
                            future_frame["track_id"] == instance["track_id"]
                        ][0]
                    )
                if len(future_translations) == 0:
                    continue

                forecast_instances.append(
                    {
                        "current_translation": instance["translation"][:2],
                        "future_translation": np.array(future_translations)[:, :2],
                        "name": instance["name"],
                        "size": instance["size"],
                        "yaw": instance["yaw"],
                        "velocity": instance["velocity"][:2],
                        "label": instance["label"],
                        "ego_translation": np.array(frame["ego_translation"]),
                        "instance_id": instance["track_id"],
                    }
                )
            if forecast_instances:
                frame_dict[frame["timestamp_ns"]] = forecast_instances

        forecast_labels[seq_id] = frame_dict

    return forecast_labels
