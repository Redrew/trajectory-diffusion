from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.utils import AV2_CLASS_NAMES
from forecasting.lstm import train_model


class AttentionAndLinearBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout_p=0.1):
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout_p, batch_first=True
        )
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        nn.init.kaiming_uniform_(self.linear.weight)
        self.dropout = nn.Dropout(dropout_p)
        self.layernorm1 = nn.LayerNorm(embedding_dim)
        self.layernorm2 = nn.LayerNorm(embedding_dim)

    def forward(self, query, seq, attn_mask):
        out, _ = self.multihead_attention(
            query, seq, seq, attn_mask=attn_mask, need_weights=False
        )
        out = self.layernorm1(seq + self.dropout(out))
        out = self.layernorm2(out + self.dropout(F.relu(self.linear(out))))
        return out


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        prediction_len,
        k,
        num_layers=2,
        embedding_dim=128,
        num_heads=4,
        dropout_p=0.1,
    ):
        super().__init__()
        self.prediction_len = prediction_len
        self.k = k
        self.num_heads = num_heads
        self.class_embeddings = nn.Embedding(
            len(AV2_CLASS_NAMES), embedding_dim=embedding_dim
        )
        self.input_proj = nn.Linear(input_dim, embedding_dim)
        self.output_proj = nn.Linear(embedding_dim, 2 * prediction_len * k)
        self.encoder_layers = nn.ModuleList(
            [
                AttentionAndLinearBlock(
                    embedding_dim, self.num_heads, dropout_p=dropout_p
                )
                for _ in range(num_layers)
            ]
        )
        self.loss = nn.MSELoss(reduction="none")

    def forward(self, input, class_cond=None):
        B, L, _ = input.shape
        embedding = self.input_proj(input.reshape(B * L, -1)).reshape(B, L, -1)
        _, _, D = embedding.shape
        # add class embedding
        x = embedding + sinusoidal_time_embedding(L, D, device=input.device)
        # add class conditioning
        if class_cond is not None:
            x += self.class_embeddings(class_cond.reshape(-1, 1))
        causal_mask = torch.tril(torch.ones((L, L), device=input.device)) == 0
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, x, attn_mask=causal_mask)
        output = self.output_proj(x.reshape(B * L, -1)).reshape(
            B, L, self.k, self.prediction_len, -1
        )
        return output


def sinusoidal_time_embedding(max_time: int, dim: int, device=None, time=None):
    if time is None:
        time = torch.arange(max_time, device=device).float().unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device).float() * -(np.log(max_time) / dim)
    )
    sin_embedding = torch.sin(time * div_term)
    cos_embedding = torch.cos(time * div_term)
    embedding = torch.cat((sin_embedding, cos_embedding), dim=-1)
    return embedding


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
    train_model(config, TransformerModel, "transformer")
