import math
import multiprocessing
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
from haversine import haversine_vector
from shapely.geometry import LineString
import torch

from utils import get_angle


def init_shared_variables(reachable_road_id_dict, geo, road_center_gps):
    global global_reachable_road_id_dict, global_geo, global_road_center_gps
    global_reachable_road_id_dict = reachable_road_id_dict
    global_geo = geo
    global_road_center_gps = road_center_gps

def process_row(args):
    index, row = args

    rid_list = eval(row['rid_list'])
    time_list = row['time_list'].split(',')
    time_list = [datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ') for t in time_list]

    trace_road_id = np.array(rid_list[:-1])
    temporal_info = np.array([(t.hour * 60.0 + t.minute + t.second / 60.0) / 1440.0 for t in time_list[:-1]]).astype(np.float32)

    if time_list[0].weekday() >= 5:
        temporal_info *= -1.0

    trace_distance_mat = haversine_vector(global_road_center_gps[trace_road_id], global_road_center_gps[trace_road_id], 'm', comb=True).astype(np.float32)
    trace_distance_mat = np.clip(trace_distance_mat, 0.0, 1000.0) / 1000.0
    trace_time_interval_mat = np.abs(temporal_info[:, None] * 1440.0 - temporal_info * 1440.0)
    trace_time_interval_mat = np.clip(trace_time_interval_mat, 0.0, 5.0) / 5.0
    trace_len = len(trace_road_id)
    destination_road_id = rid_list[-1]

    candidate_road_id = np.empty(len(trace_road_id), dtype=object)
    for i, road_id in enumerate(trace_road_id):
        candidate_road_id[i] = np.array(global_reachable_road_id_dict[road_id])

    metric_dis = np.empty(len(trace_road_id), dtype=object)
    for i, candidate_road_id_list in enumerate(candidate_road_id):
        metric_dis[i] = haversine_vector(global_road_center_gps[candidate_road_id_list], global_road_center_gps[destination_road_id].reshape(1, -1), 'm', comb=True).reshape(-1).astype(np.float32)
        metric_dis[i] = np.log1p((metric_dis[i] - np.min(metric_dis[i])) / 100)

    metric_angle = np.empty(len(trace_road_id), dtype=object)
    for i, (road_id, candidate_road_id_list) in enumerate(zip(trace_road_id, candidate_road_id)):
        angle1 = np.vectorize(lambda candidate: get_angle(*(eval(global_geo.loc[road_id, 'coordinates'])[-1]), *(eval(global_geo.loc[candidate, 'coordinates'])[-1])))(candidate_road_id_list)
        angle2 = get_angle(*(eval(global_geo.loc[road_id, 'coordinates'])[-1]), *(eval(global_geo.loc[destination_road_id, 'coordinates'])[-1]))
        angle = np.abs(angle1 - angle2).astype(np.float32)
        angle = np.where(angle > math.pi, 2 * math.pi - angle, angle) / math.pi
        metric_angle[i] = angle

    candidate_len = np.array([len(candidate_road_id_list) for candidate_road_id_list in candidate_road_id])

    road_label = np.array([global_reachable_road_id_dict[rid_list[i]].index(rid_list[i + 1]) for i in range(len(trace_road_id))])
    timestamp_label = np.array([(time_list[i+1] - time_list[i]).total_seconds() for i in range(len(trace_road_id))]).astype(np.float32)

    return (
        index,
        {
            'trace_road_id': trace_road_id,
            'temporal_info': temporal_info,
            'trace_distance_mat': trace_distance_mat,
            'trace_time_interval_mat': trace_time_interval_mat,
            'trace_len': trace_len,
            'destination_road_id': destination_road_id,
            'candidate_road_id': candidate_road_id,
            'metric_dis': metric_dis,
            'metric_angle': metric_angle,
            'candidate_len': candidate_len,
            'road_label': road_label,
            'timestamp_label': timestamp_label,
        }
    )


class Dataset(torch.utils.data.Dataset):
    def __init__(self, geo_file, rel_file, traj_file):
        geo = pd.read_csv(geo_file)
        rel = pd.read_csv(rel_file)
        traj = pd.read_csv(traj_file)

        road_center_gps = []
        for _, row in geo.iterrows():
            coordinates = eval(row['coordinates'])
            road_line = LineString(coordinates=coordinates)
            center_coord = road_line.centroid
            road_center_gps.append((center_coord.y, center_coord.x))
        road_center_gps = np.array(road_center_gps)

        reachable_road_id_dict = dict()
        num_roads = len(geo)
        for i in range(num_roads):
            reachable_road_id_dict[i] = []
        for _, row in rel.iterrows():
            origin_id = row['origin_id']
            destination_id = row['destination_id']
            reachable_road_id_dict[origin_id].append(destination_id)

        with multiprocessing.Pool(processes=multiprocessing.cpu_count(), initializer=init_shared_variables, initargs=(reachable_road_id_dict, geo, road_center_gps)) as pool:
            results = list(tqdm(pool.imap_unordered(process_row, traj.iterrows()), total=len(traj), desc='load dataset'))

        self.trace_road_id = [None] * len(traj)
        self.temporal_info = [None] * len(traj)
        self.trace_distance_mat = [None] * len(traj)
        self.trace_time_interval_mat = [None] * len(traj)
        self.trace_len = [None] * len(traj)
        self.destination_road_id = [None] * len(traj)
        self.candidate_road_id = [None] * len(traj)
        self.metric_dis = [None] * len(traj)
        self.metric_angle = [None] * len(traj)
        self.candidate_len = [None] * len(traj)
        self.road_label = [None] * len(traj)
        self.timestamp_label = [None] * len(traj)

        for i, data in results:
            self.trace_road_id[i] = data['trace_road_id']
            self.temporal_info[i] = data['temporal_info']
            self.trace_distance_mat[i] = data['trace_distance_mat']
            self.trace_time_interval_mat[i] = data['trace_time_interval_mat']
            self.trace_len[i] = data['trace_len']
            self.destination_road_id[i] = data['destination_road_id']
            self.candidate_road_id[i] = data['candidate_road_id']
            self.metric_dis[i] = data['metric_dis']
            self.metric_angle[i] = data['metric_angle']
            self.candidate_len[i] = data['candidate_len']
            self.road_label[i] = data['road_label']
            self.timestamp_label[i] = data['timestamp_label']

    def __len__(self):
        return len(self.trace_road_id)
    
    def __getitem__(self, i):
        return (
            self.trace_road_id[i],
            self.temporal_info[i],
            self.trace_distance_mat[i],
            self.trace_time_interval_mat[i],
            self.trace_len[i],
            self.destination_road_id[i],
            self.candidate_road_id[i],
            self.metric_dis[i],
            self.metric_angle[i],
            self.candidate_len[i],
            self.road_label[i],
            self.timestamp_label[i],
        )