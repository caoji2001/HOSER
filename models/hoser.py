import torch
import torch.nn as nn
from .road_network_encoder import RoadNetworkEncoder
from .trajectory_encoder import TrajectoryEncoder
from .navigator import Navigator


class HOSER(nn.Module):
    def __init__(self, road_network_encoder_config, road_network_encoder_feature, trajectory_encoder_config, navigator_config, road2zone):
        super(HOSER, self).__init__()
        self.road_network_encoder = RoadNetworkEncoder(road_network_encoder_config, road_network_encoder_feature)
        self.trajectory_encoder = TrajectoryEncoder(trajectory_encoder_config)
        self.navigator = Navigator(navigator_config)

        self.register_buffer('road2zone', torch.from_numpy(road2zone))

        self.all_road_embedding_after_gnn = None
        self.all_zone_embedding_after_gnn = None

    def setup_road_network_features(self):
        self.all_road_embedding_after_gnn = self.road_network_encoder.get_road_embedding()
        self.all_zone_embedding_after_gnn = self.road_network_encoder.get_zone_embedding()

    def forward(self, trace_road_id, temporal_info, trace_distance_mat, trace_time_interval_mat, trace_len, destination_road_id, candidate_road_id, metric_distance, metric_angle):
        road_embedding = self.all_road_embedding_after_gnn[trace_road_id]
        zone_embedding = self.all_zone_embedding_after_gnn[self.road2zone[trace_road_id]]

        trajectory_embedding = self.trajectory_encoder(road_embedding, zone_embedding, temporal_info, trace_distance_mat, trace_time_interval_mat, trace_len)

        destination_zone_embedding = self.all_zone_embedding_after_gnn[self.road2zone[destination_road_id]]
        candidate_road_embedding = self.all_road_embedding_after_gnn[candidate_road_id]

        logits, time_pred = self.navigator(trajectory_embedding, destination_zone_embedding, candidate_road_embedding, metric_distance, metric_angle)

        return logits, time_pred

    # Compared to `forward()`, the batch size of the input here is 1, and only the last next spatio-temporal point is predicted
    @torch.no_grad()
    def infer(self, trace_road_id, temporal_info, trace_distance_mat, trace_time_interval_mat, trace_len, destination_road_id, candidate_road_id, metric_distance, metric_angle):

        road_embedding = self.all_road_embedding_after_gnn[trace_road_id]
        zone_embedding = self.all_zone_embedding_after_gnn[self.road2zone[trace_road_id]]

        trajectory_embedding = self.trajectory_encoder(road_embedding, zone_embedding, temporal_info, trace_distance_mat, trace_time_interval_mat, trace_len)

        destination_zone_embedding = self.all_zone_embedding_after_gnn[self.road2zone[destination_road_id]]
        candidate_road_embedding = self.all_road_embedding_after_gnn[candidate_road_id]

        logits, time_pred = self.navigator.infer(trajectory_embedding, destination_zone_embedding, candidate_road_embedding, metric_distance, metric_angle)
        return logits, time_pred








