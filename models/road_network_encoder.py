import torch
import torch.nn as nn
from torch_geometric.nn.models import GAT, GCN


class RoadNetworkEncoder(nn.Module):
    def __init__(self, config, feature):
        super(RoadNetworkEncoder, self).__init__()

        assert config.road_id_emb_dim+config.len_emb_dim+config.type_emb_dim+config.lon_emb_dim+config.lat_emb_dim == config.intersection_emb_dim == config.zone_id_emb_dim

        self.hidden_dim = config.road_id_emb_dim+config.len_emb_dim+config.type_emb_dim+config.lon_emb_dim+config.lat_emb_dim

        self.road_id_emb = nn.Embedding(config.road_id_num_embeddings, config.road_id_emb_dim)
        self.len_emb = nn.Linear(1, config.len_emb_dim)
        self.type_emb = nn.Embedding(config.type_num_embeddings, config.type_emb_dim)
        self.lon_emb = nn.Linear(1, config.lon_emb_dim)
        self.lat_emb = nn.Linear(1, config.lat_emb_dim)
        self.zone_id_emb = nn.Embedding(config.zone_id_num_embeddings, config.zone_id_emb_dim)

        self.road_gat = GAT(
            in_channels = config.road_id_emb_dim+config.len_emb_dim+config.type_emb_dim+config.lon_emb_dim+config.lat_emb_dim,
            hidden_channels = self.hidden_dim,
            out_channels = self.hidden_dim,

            num_layers = 2,
            edge_dim = 2,
            v2 = True,
        )
        self.register_buffer('road_attr_len', torch.from_numpy(feature.road_attr.len))
        self.register_buffer('road_attr_type', torch.from_numpy(feature.road_attr.type))
        self.register_buffer('road_attr_lon', torch.from_numpy(feature.road_attr.lon))
        self.register_buffer('road_attr_lat', torch.from_numpy(feature.road_attr.lat))
        self.register_buffer('road_edge_index', torch.from_numpy(feature.road_edge_index))
        self.register_buffer('intersection_attr', torch.from_numpy(feature.intersection_attr))

        self.zone_gcn = GCN(
            in_channels = config.zone_id_emb_dim,
            hidden_channels = self.hidden_dim,
            out_channels = self.hidden_dim,

            num_layers = 2,
            cached = True,
        )
        self.register_buffer('zone_edge_index', torch.from_numpy(feature.zone_edge_index))
        self.register_buffer('zone_edge_weight', torch.from_numpy(feature.zone_edge_weight))

    def get_road_embedding(self):
        all_road_id_embedding = self.road_id_emb.weight
        all_road_attr_embedding = torch.cat([
            self.len_emb(self.road_attr_len.unsqueeze(-1)),
            self.type_emb(self.road_attr_type),
            self.lon_emb(self.road_attr_lon.unsqueeze(-1)),
            self.lat_emb(self.road_attr_lat.unsqueeze(-1)),
        ], dim=1)

        all_road_embedding = torch.cat([all_road_id_embedding, all_road_attr_embedding], dim=1)
        all_road_embedding_after_gnn = self.road_gat(all_road_embedding, self.road_edge_index, edge_attr=self.intersection_attr)
        return all_road_embedding_after_gnn
    
    def get_zone_embedding(self):
        all_zone_embedding = self.zone_id_emb.weight
        all_zone_embedding_after_gnn = self.zone_gcn(all_zone_embedding, self.zone_edge_index, edge_weight=self.zone_edge_weight)
        return all_zone_embedding_after_gnn
