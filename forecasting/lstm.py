import os
from argparse import ArgumentParser
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data.utils import *

FEATURE_SCALING = np.array([20, 20, 5, 5, 1])


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_dim,
        prediction_len,
        k,
        num_layers=2,
        embedding_dim=32,
        dropout_p=0.1,
    ):
        super().__init__()
        self.prediction_len = prediction_len
        self.k = k
        self.embedding_dim = embedding_dim
        self.class_embeddings = nn.Embedding(
            len(AV2_CLASS_NAMES), embedding_dim=embedding_dim
        )
        self.input_proj = nn.Linear(input_dim, embedding_dim)
        nn.init.kaiming_uniform_(self.input_proj.weight)
        self.output_proj = nn.Linear(embedding_dim, 2 * prediction_len * k)
        self.lstm_layers = nn.ModuleList(
            [nn.LSTM(embedding_dim, embedding_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input, class_cond=None):
        B, L, D = input.shape
        embedding = self.input_proj(input.reshape(B * L, -1)).reshape(B, L, -1)
        # add class embedding
        if class_cond is not None:
            embedding += self.class_embeddings(class_cond.reshape(-1, 1))
        x = F.relu(embedding)
        for lstm_layer in self.lstm_layers:
            x_out, state = lstm_layer(x)
            x = x + self.dropout(x_out)
        output = self.output_proj(x.reshape(B * L, -1)).reshape(
            B, L, self.k, self.prediction_len, -1
        )
        return output


class MotionPredictionDataset(Dataset):
    def __init__(self, data, prediction_length):
        self.data = data
        self.keys = list(data.keys())
        self.input_dim = next(iter(data.values()))["feature"].shape[1]
        self.output_dim = 2
        self.prediction_len = prediction_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        feature: state tensor,
            if the instance is missing, the state is set to zeros
        class_label: class label
        target: regression target
        mask: 0 if the regression target at timestep is missing
        """

        instance_data = self.data[self.keys[idx]]
        feature = torch.from_numpy(instance_data["feature"]).float()
        class_cond = torch.LongTensor([instance_data["class_label"]])
        target = torch.from_numpy(instance_data["target"]).float()
        target_mask = torch.from_numpy(instance_data["target_mask"]).bool()
        return feature, class_cond, target, target_mask


def preprocess_data(labels, max_len=20, prediction_len=6):
    # group tracks
    grouped_tracks = defaultdict(dict)
    for seq_id, frames in labels.items():
        for timestep, frame in enumerate(frames):
            for instance in array_dict_iterator(frame, len(frame["translation"])):
                grouped_tracks[f"{seq_id}:{instance['track_id']}"][timestep] = instance

    data = {}
    for id, instance_by_ts in progressbar(
        grouped_tracks.items(), desc="preprocessing track data"
    ):
        class_label = next(iter(instance_by_ts.values()))["label"]
        features, targets, masks = [], [], []

        # change all sequences to start at timestep=0
        min_ts = min(instance_by_ts.keys())
        instance_by_ts = {
            ts - min_ts: instance for ts, instance in instance_by_ts.items()
        }
        sorted_ts = list(sorted(instance_by_ts.keys()))
        last_timestep = sorted_ts[-1]

        # transform target to deltas
        deltas = {}
        for timestep in range(last_timestep):
            prev_timestep = next(ts for ts in reversed(sorted_ts) if ts <= timestep)
            next_timestep = next(ts for ts in sorted_ts if ts > timestep)
            delta = (
                instance_by_ts[next_timestep]["translation"]
                - instance_by_ts[prev_timestep]["translation"]
            )[:2] / (next_timestep - prev_timestep)
            deltas[timestep] = delta

        for timestep in range(max_len):
            # if features are missing, default to zeros
            feature = np.zeros(5)
            target = np.zeros((prediction_len, 2))
            mask = np.zeros(prediction_len)
            if timestep in instance_by_ts:
                instance = instance_by_ts[timestep]
                # format features (translation, velocity)
                translation_delta = (
                    instance["translation"] - instance_by_ts[0]["translation"]
                )
                # normalize inputs
                feature = (
                    np.concatenate(
                        [
                            translation_delta[:2],
                            instance["velocity"][:2],
                            np.array([instance["yaw"]]),
                        ]
                    )
                    / FEATURE_SCALING
                )

                # transform target to deltas
                future_timesteps = range(
                    timestep, min(timestep + prediction_len, last_timestep)
                )
                for i, future_ts in enumerate(future_timesteps):
                    target[i] = deltas[future_ts]
                    mask[i] = 1
            features.append(feature)
            targets.append(target)
            masks.append(mask)

        data[f"{id}:{min_ts}"] = {
            "feature": np.stack(features, axis=0),
            "target": np.stack(targets, axis=0),
            "target_mask": np.stack(masks, axis=0),
            "class_label": class_label,
        }

    return data


def k_best_loss(prediction, target, mask):
    """K best loss"""
    K = prediction.shape[2]
    loss = F.mse_loss(
        prediction,
        target.unsqueeze(2).repeat(1, 1, K, 1, 1),
        reduce=False,
    ).mean(axis=-1)
    # mask loss for invalid timesteps
    loss = loss.masked_fill(~mask.unsqueeze(dim=2), 0).mean(axis=-1)
    loss = loss.min(axis=-1).values.mean()
    return loss


def generate_forecasts_from_model(
    model,
    tracks,
    prediction_length,
    num_modes,  # k
    device: str = "cuda",
):
    # inference
    data = preprocess_data(tracks, prediction_len=prediction_length)
    dataset = MotionPredictionDataset(data, prediction_length)
    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
    )
    model = model.to(device).eval()

    forecasts = defaultdict(lambda: defaultdict(list))

    start_idx = 0
    for input, class_cond, target, mask in progressbar(dataloader, "running inference"):
        input, class_cond = input.to(device), class_cond.to(device)
        with torch.no_grad():
            prediction = model(input, class_cond)
        prediction = prediction.cpu().numpy()

        ids = dataset.keys[start_idx : start_idx + len(input)]
        start_idx += len(input)
        for id, prediction_ot in zip(ids, prediction):
            seq_id, track_id, start_ts = id.split(":")
            track_id, start_ts = int(track_id), int(start_ts)
            for i, delta_at_t in enumerate(prediction_ot):
                timestep = start_ts + i
                if timestep >= len(tracks[seq_id]):
                    continue
                detection_frame = tracks[seq_id][timestep]
                if track_id not in detection_frame["track_id"]:
                    continue
                detection = index_array_values(
                    detection_frame, list(detection_frame["track_id"]).index(track_id)
                )
                current_translation = detection["translation"][:2]
                prediction_at_t = current_translation + np.cumsum(delta_at_t, axis=1)

                timestamp = tracks[seq_id][timestep]["timestamp_ns"]
                forecast_elem = {
                    "current_translation": current_translation,
                    "detection_score": detection.get("score", 1),
                    "size": detection["size"],
                    "yaw": detection["yaw"],
                    "label": detection["label"],
                    "name": detection["name"],
                    "track_id": detection["track_id"],
                    "prediction": prediction_at_t,
                    "score": np.ones(num_modes) * detection.get("score", 1),
                    "instance_id": track_id,
                }
                forecasts[seq_id][timestamp].append(forecast_elem)
    forecasts = {seq_id: dict(preds_by_ts) for seq_id, preds_by_ts in forecasts.items()}
    return forecasts


def train_model(config, model_constructor, model_name):
    train_labels = load(f"dataset/{config.dataset}-train/labels.pkl")
    train_data = preprocess_data(train_labels, prediction_len=config.prediction_length)
    train_dataset = MotionPredictionDataset(train_data, config.prediction_length)
    dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )
    model = model_constructor(
        input_dim=train_dataset.input_dim,
        prediction_len=train_dataset.prediction_len,
        k=config.K,
        num_layers=config.num_layers,
    )

    # train
    model = model.to(config.device).train()
    optim = torch.optim.Adam(model.parameters(), config.learning_rate)

    for epoch in range(config.epochs):
        epoch_loss = []
        for input, class_cond, target, mask in dataloader:
            input, class_cond, target, mask = map(
                lambda x: x.to(config.device), (input, class_cond, target, mask)
            )
            prediction = model(input, class_cond)

            optim.zero_grad()
            loss = k_best_loss(prediction, target, mask)
            loss.backward()
            optim.step()

            epoch_loss.append(loss.detach().cpu().item())

        print(f"Epoch: {epoch}, Loss: {np.mean(epoch_loss)}")

    # save model
    model = model.cpu()
    model_path = f"models/{config.dataset}/{model_name}.pt"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model, model_path)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--dataset", default="av2", choices=["av2", "nuscenes"])
    argparser.add_argument("--learning_rate", default=1e-3, type=float)
    argparser.add_argument("--device", default="cuda")
    config = argparser.parse_args()
    config.prediction_length = 6
    config.K = 5
    config.epochs = 10
    config.num_layers = 4
    train_model(config, LSTMModel, "lstm")
