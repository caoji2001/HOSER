import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, max_len):
        super(CausalSelfAttention, self).__init__()

        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.relative_position_emb_k = nn.Linear(2, hidden_dim // num_heads)
        self.relative_position_emb_v = nn.Linear(2, hidden_dim // num_heads)

        self.c_attn = nn.Linear(hidden_dim, hidden_dim * 3)
        self.c_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer('bias', torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len))

    def forward(self, x, trace_distance_mat, trace_time_interval_mat, trace_len):
        B, T = x.size(0), x.size(1)

        q, k, v = self.c_attn(x).split(self.hidden_dim, dim=2)
        q = q.view(B, T, self.num_heads, self.hidden_dim // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.hidden_dim // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.hidden_dim // self.num_heads).transpose(1, 2)

        q1 = q
        k1 = k
        attn1 = q1 @ k1.transpose(-2, -1)

        q2 = q
        k2 = self.relative_position_emb_k(torch.cat([
            trace_distance_mat.unsqueeze(-1),
            trace_time_interval_mat.unsqueeze(-1)
        ], dim=-1))
        attn2 = (q2.permute(0, 2, 1, 3) @ k2.transpose(-2, -1)).permute(0, 2, 1, 3)

        attn = (attn1 + attn2) * (1.0 / math.sqrt(self.hidden_dim // self.num_heads))

        mask1 = (self.bias[:, :, :T, :T] == 0).expand(B, 1, T, T)
        mask2 = (torch.arange(T, dtype=torch.int64, device=x.device).unsqueeze(0) >= trace_len.unsqueeze(1))
        mask2 = mask2.unsqueeze(1).unsqueeze(2)
        mask2 = mask2.expand(B, 1, T, T)
        mask = mask1 | mask2

        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        v1 = v
        weight1 = attn @ v1

        v2 = self.relative_position_emb_v(torch.cat([
            trace_distance_mat.unsqueeze(-1),
            trace_time_interval_mat.unsqueeze(-1)
        ], dim=-1))
        weight2 = (attn.permute(0, 2, 1, 3) @ v2).permute(0, 2, 1, 3)

        x = (weight1 + weight2).permute(0, 2, 1, 3).contiguous().view(B, T, -1)
        x = self.resid_dropout(self.c_proj(x))
        return x


class Block(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, max_len):
        super(Block, self).__init__()

        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, num_heads, dropout, max_len)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, trace_distance_mat, trace_time_interval_mat, trace_len):
        x = x + self.attn(self.ln_1(x), trace_distance_mat, trace_time_interval_mat, trace_len)
        x = x + self.mlp(self.ln_2(x))
        return x


class TemporalEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalEncoder, self).__init__()

        self.time_emb = nn.Linear(1, hidden_dim)
     
    def forward(self, temporal_info):
        return torch.cos(self.time_emb(temporal_info.unsqueeze(-1)))


class TrajectoryEncoder(nn.Module):
    def __init__(self, config):
        super(TrajectoryEncoder, self).__init__()

        self.road_zone_fusion_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim*2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.temporal_encoder = TemporalEncoder(config.hidden_dim)

        self.transformer = nn.ModuleDict(dict(
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config.hidden_dim*2, config.num_heads, config.dropout, config.max_len) for _ in range(config.num_layers)])
        ))

    def forward(self, road_embedding, zone_embedding, temporal_info, trace_distance_mat, trace_time_interval_mat, trace_len):
        spatial_embedding = road_embedding + torch.sigmoid(self.road_zone_fusion_mlp(torch.cat([road_embedding, zone_embedding], dim=-1))) * zone_embedding
        temporal_embedding = self.temporal_encoder(temporal_info)
        x = torch.cat([spatial_embedding, temporal_embedding], dim=-1)

        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x, trace_distance_mat, trace_time_interval_mat, trace_len)
        return x
