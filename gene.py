import os
import argparse
from queue import PriorityQueue
import json
import math
from datetime import datetime, timedelta
import random
import yaml
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from shapely.geometry import LineString
from haversine import haversine, haversine_vector
import torch
import torch.nn.functional as F

from utils import set_seed, create_nested_namespace, get_angle
from models.hoser import HOSER


class SearchNode:
    def __init__(self, trace_road_id, trace_datetime, log_prob):
        self.trace_road_id = trace_road_id
        self.trace_datetime = trace_datetime
        self.log_prob = log_prob

    def __ge__(self, other):
        return self.log_prob >= other.log_prob
    
    def __le__(self, other):
        return self.log_prob <= other.log_prob

    def __gt__(self, other):
        return self.log_prob > other.log_prob
    
    def __lt__(self, other):
        return self.log_prob < other.log_prob


class Searcher:
    def __init__(self, model, reachable_road_id_dict, geo, road_center_gps, timestamp_label_array_log1p_mean, timestamp_label_array_log1p_std, device):
        self.model = model.to(device)
        self.model.eval()

        self.reachable_road_id_dict = reachable_road_id_dict
        self.geo = geo
        self.road_center_gps = road_center_gps
        self.timestamp_label_array_log1p_mean = timestamp_label_array_log1p_mean
        self.timestamp_label_array_log1p_std = timestamp_label_array_log1p_std
        self.device = device

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                model.setup_road_network_features()

    def search(self, origin_road_id, origin_datetime, destination_road_id, max_search_step=5000):
        vis_set = set()
        pq = PriorityQueue()
        road_id2log_prob = dict()

        best_trace = None
        min_dis = float('inf')

        origin_node = SearchNode(trace_road_id=[origin_road_id], trace_datetime=[origin_datetime], log_prob=0)

        road_id2log_prob[origin_road_id] = 0
        pq.put((-origin_node.log_prob, origin_node))

        search_step = 0
        while (not pq.empty()) and (search_step < max_search_step):
            neg_log_prob, cur_node = pq.get()
            cur_road_id = cur_node.trace_road_id[-1]

            if cur_road_id in vis_set:
                continue
            vis_set.add(cur_road_id)

            if cur_road_id == destination_road_id:
                best_trace = cur_node.trace_road_id, cur_node.trace_datetime
                break

            dis = haversine(self.road_center_gps[cur_road_id], self.road_center_gps[destination_road_id], unit='m')
            if dis < min_dis:
                min_dis = dis
                best_trace = cur_node.trace_road_id, cur_node.trace_datetime

            reachable_road_id_list = self.reachable_road_id_dict[cur_road_id]
            assert len(reachable_road_id_list) > 0

            # Predicts the next spatio-temporal point based on the current state
            trace_road_id = np.array(cur_node.trace_road_id)
            temporal_info = np.array([(t.hour * 60.0 + t.minute + t.second / 60.0) / 1440.0 for t in cur_node.trace_datetime]).astype(np.float32)

            if cur_node.trace_datetime[0].weekday() >= 5:
                temporal_info *= -1.0

            trace_distance_mat = haversine_vector(self.road_center_gps[trace_road_id], self.road_center_gps[trace_road_id], 'm', comb=True).astype(np.float32)
            trace_distance_mat = np.clip(trace_distance_mat, 0.0, 1000.0) / 1000.0
            trace_time_interval_mat = np.abs(temporal_info[:, None] * 1440.0 - temporal_info * 1440.0)
            trace_time_interval_mat = np.clip(trace_time_interval_mat, 0.0, 5.0) / 5.0
            trace_len = len(trace_road_id)

            candidate_road_id = np.array(reachable_road_id_list)

            metric_dis = haversine_vector(self.road_center_gps[candidate_road_id], self.road_center_gps[destination_road_id].reshape(1, -1), 'm', comb=True).reshape(-1).astype(np.float32)
            metric_dis = np.log1p((metric_dis - np.min(metric_dis)) / 100)

            angle1 = np.vectorize(lambda candidate: get_angle(*(eval(self.geo.loc[cur_road_id, 'coordinates'])[-1]), *(eval(self.geo.loc[candidate, 'coordinates'])[-1])))(candidate_road_id)
            angle2 = get_angle(*(eval(self.geo.loc[cur_road_id, 'coordinates'])[-1]), *(eval(self.geo.loc[destination_road_id, 'coordinates'])[-1]))
            angle = np.abs(angle1 - angle2).astype(np.float32)
            angle = np.where(angle > math.pi, 2 * math.pi - angle, angle) / math.pi
            metric_angle = angle

            batch_trace_road_id = torch.from_numpy(np.array([trace_road_id])).to(self.device)
            batch_temporal_info = torch.from_numpy(np.array([temporal_info])).to(self.device)
            batch_trace_distance_mat = torch.from_numpy(np.array([trace_distance_mat])).to(self.device)
            batch_trace_time_interval_mat = torch.from_numpy(np.array([trace_time_interval_mat])).to(self.device)
            batch_trace_len = torch.from_numpy(np.array([trace_len])).to(self.device)
            batch_destination_road_id = torch.from_numpy(np.array([destination_road_id])).to(self.device)
            batch_candidate_road_id = torch.from_numpy(np.array([candidate_road_id])).to(self.device)
            batch_metric_dis = torch.from_numpy(np.array([metric_dis])).to(self.device)
            batch_metric_angle = torch.from_numpy(np.array([metric_angle])).to(self.device)

            with torch.cuda.amp.autocast():
                logits, time_pred = self.model.infer(batch_trace_road_id, batch_temporal_info, batch_trace_distance_mat, batch_trace_time_interval_mat, batch_trace_len, batch_destination_road_id, batch_candidate_road_id, batch_metric_dis, batch_metric_angle)

            logits = logits[0]
            output = F.softmax(logits, dim=-1)
            log_output = torch.log(output)
            log_output += cur_node.log_prob

            time_pred = time_pred[0]
            time_pred = time_pred * self.timestamp_label_array_log1p_std + self.timestamp_label_array_log1p_mean
            time_pred = torch.expm1(time_pred)
            time_pred = torch.clamp(time_pred, min=0.0)

            for index, candidate_road_id in enumerate(reachable_road_id_list):
                candidate_log_prob = log_output[index].item()
                next_datatime = cur_node.trace_datetime[-1] + timedelta(seconds=round(time_pred[index].item()))

                if candidate_road_id not in road_id2log_prob or candidate_log_prob > road_id2log_prob[candidate_road_id]:
                    new_node = SearchNode(
                        trace_road_id=cur_node.trace_road_id+[candidate_road_id],
                        trace_datetime=cur_node.trace_datetime+[next_datatime],
                        log_prob=candidate_log_prob
                    )
                    pq.put((-candidate_log_prob, new_node))
                    road_id2log_prob[candidate_road_id] = candidate_log_prob

            search_step += 1

        assert best_trace is not None
        return best_trace[0], best_trace[1]


def init_searcher(model, reachable_road_id_dict, geo, road_center_gps, timestamp_label_array_log1p_mean, timestamp_label_array_log1p_std, device):
    global searcher
    searcher = Searcher(model, reachable_road_id_dict, geo, road_center_gps, timestamp_label_array_log1p_mean, timestamp_label_array_log1p_std, device)

def process_task(args):
    (origin_road_id, destination_road_id), origin_datetime = args
    trace_road_id, trace_datetime = searcher.search(origin_road_id, origin_datetime, destination_road_id)
    trace_datetime = [t.strftime('%Y-%m-%dT%H:%M:%SZ') for t in trace_datetime]
    return trace_road_id, trace_datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--num_gene', type=int, default=5000)
    parser.add_argument('--processes', type=int, default=8)
    args = parser.parse_args()

    set_seed(args.seed)
    device = f'cuda:{args.cuda}'

    # Prepare model config and related features

    geo_file = f'./data/{args.dataset}/roadmap.geo'
    rel_file = f'./data/{args.dataset}/roadmap.rel'
    train_traj_file = f'./data/{args.dataset}/train.csv'
    val_traj_file = f'./data/{args.dataset}/val.csv'
    test_traj_file = f'./data/{args.dataset}/test.csv'
    road_network_partition_file = f'./data/{args.dataset}/road_network_partition'
    zone_trans_mat_file = f'./data/{args.dataset}/zone_trans_mat.npy'

    save_dir = f'./save/{args.dataset}/seed{args.seed}'
    tensorboard_log_dir = f'./tensorboard_log/{args.dataset}/seed{args.seed}'
    loguru_log_dir = f'./log/{args.dataset}/seed{args.seed}'
    gene_dir = f'./gene/{args.dataset}/seed{args.seed}'

    with open(f'./config/{args.dataset}.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config = create_nested_namespace(config)

    geo = pd.read_csv(geo_file)
    rel = pd.read_csv(rel_file)
    num_roads = len(geo)

    road_center_gps = []
    for _, row in geo.iterrows():
        coordinates = eval(row['coordinates'])
        road_line = LineString(coordinates=coordinates)
        center_coord = road_line.centroid
        road_center_gps.append((center_coord.y, center_coord.x))
    road_center_gps = np.array(road_center_gps)

    road_attr_len = geo['length'].to_numpy().astype(np.float32)
    road_attr_len = np.log1p(road_attr_len)
    road_attr_len = (road_attr_len - np.mean(road_attr_len)) / np.std(road_attr_len)

    road_attr_type = geo['highway'].values.tolist()
    if args.dataset in ['Beijing', 'San_Francisco']:
        for i in range(len(road_attr_type)):
            if road_attr_type[i].startswith('[') and road_attr_type[i].endswith(']'):
                info = eval(road_attr_type[i])
                road_attr_type[i] = info[0] if info[0] != 'unclassified' else info[1]
    le = LabelEncoder()
    road_attr_type = le.fit_transform(road_attr_type)

    road_attr_lon = np.array([LineString(coordinates=eval(row['coordinates'])).centroid.x for _, row in geo.iterrows()]).astype(np.float32)
    road_attr_lon = (road_attr_lon - np.mean(road_attr_lon)) / np.std(road_attr_lon)
    road_attr_lat = np.array([LineString(coordinates=eval(row['coordinates'])).centroid.y for _, row in geo.iterrows()]).astype(np.float32)
    road_attr_lat = (road_attr_lat - np.mean(road_attr_lat)) / np.std(road_attr_lat)

    adj_row = []
    adj_col = []
    adj_angle = []
    adj_reachability = []

    reachable_road_id_dict = dict()
    for i in range(num_roads):
        reachable_road_id_dict[i] = []
    for _, row in rel.iterrows():
        origin_id = row['origin_id']
        destination_id = row['destination_id']
        reachable_road_id_dict[origin_id].append(destination_id)

    coord2road_id = dict()
    for road_id, row in geo.iterrows():
        coord = json.loads(row['coordinates'], parse_float=str)
        start_coord = tuple(coord[0])
        end_coord = tuple(coord[-1])
        if start_coord not in coord2road_id:
            coord2road_id[start_coord] = [road_id]
        else:
            coord2road_id[start_coord].append(road_id)
        if end_coord not in coord2road_id:
            coord2road_id[end_coord] = [road_id]
        else:
            coord2road_id[end_coord].append(road_id)

    road_adj = np.zeros((num_roads, num_roads), dtype=bool)
    for k, v in coord2road_id.items():
        for road_id1 in v:
            for road_id2 in v:
                if road_id1 != road_id2:
                    road_adj[road_id1, road_id2] = True

    for road_id in range(num_roads):
        adj_road_id_list = np.where(road_adj[road_id])[0]
        for adj_road_id in adj_road_id_list:
            adj_row.append(road_id)
            adj_col.append(adj_road_id)

            road_id_coord = eval(geo.loc[road_id, 'coordinates'])
            adj_road_id_coord = eval(geo.loc[adj_road_id, 'coordinates'])
            
            road_id_angle = get_angle(road_id_coord[0][1], road_id_coord[0][0], road_id_coord[-1][1], road_id_coord[-1][0])
            adj_road_id_angle = get_angle(adj_road_id_coord[0][1], adj_road_id_coord[0][0], adj_road_id_coord[-1][1], adj_road_id_coord[-1][0])
            angle = abs(road_id_angle - adj_road_id_angle)
            if angle > math.pi:
                angle = math.pi * 2 - angle
            angle /= math.pi
            adj_angle.append(angle)

            if adj_road_id in reachable_road_id_dict[road_id]:
                adj_reachability.append(1.0)
            else:
                adj_reachability.append(0.0)

    road_edge_index = np.stack([
        np.array(adj_row).astype(np.int64),
        np.array(adj_col).astype(np.int64),
    ], axis=0)
    intersection_attr = np.stack([
        np.array(adj_angle).astype(np.float32),
        np.array(adj_reachability).astype(np.float32),
    ], axis=1)

    zone_trans_mat = np.load(zone_trans_mat_file)
    zone_edge_index = np.stack(zone_trans_mat.nonzero())

    zone_trans_mat = zone_trans_mat.astype(np.float32)
    D_inv_sqrt = 1.0 / np.sqrt(np.maximum(np.sum(zone_trans_mat, axis=1), 1.0))
    zone_trans_mat_norm = zone_trans_mat * D_inv_sqrt[:, np.newaxis] * D_inv_sqrt[np.newaxis, :]
    zone_edge_weight = zone_trans_mat_norm[zone_edge_index[0], zone_edge_index[1]]

    config.road_network_encoder_config.road_id_num_embeddings = num_roads
    config.road_network_encoder_config.type_num_embeddings = len(np.unique(road_attr_type))
    config.road_network_encoder_feature.road_attr.len = road_attr_len
    config.road_network_encoder_feature.road_attr.type = road_attr_type
    config.road_network_encoder_feature.road_attr.lon = road_attr_lon
    config.road_network_encoder_feature.road_attr.lat = road_attr_lat
    config.road_network_encoder_feature.road_edge_index = road_edge_index
    config.road_network_encoder_feature.intersection_attr = intersection_attr
    config.road_network_encoder_feature.zone_edge_index = zone_edge_index
    config.road_network_encoder_feature.zone_edge_weight = zone_edge_weight

    road2zone = []
    with open(road_network_partition_file, 'r') as file:
        for line in file:
            road2zone.append(int(line.strip()))
    road2zone = np.array(road2zone)

    # Prepare OD matrix

    test_traj = pd.read_csv(test_traj_file)

    od_mat = np.zeros((num_roads, num_roads), dtype=np.float32)
    for _, row in test_traj.iterrows():
        rid_list = eval(row['rid_list'])
        origin_id = rid_list[0]
        destination_id = rid_list[-1]
        od_mat[origin_id][destination_id] += 1.0

    non_zero_indices = np.flatnonzero(od_mat)
    od_flat = od_mat.ravel()[non_zero_indices]
    od_probabilities = od_flat / np.sum(od_flat)

    # Generating trajectories

    train_traj = pd.read_csv(train_traj_file)

    timestamp_label_array = []
    for _, row in train_traj.iterrows():
        time_list = row['time_list'].split(',')
        time_list = [datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ') for t in time_list]

        timestamp_label = np.array([(time_list[i+1] - time_list[i]).total_seconds() for i in range(len(time_list)-1)]).astype(np.float32)
        timestamp_label_array.extend(timestamp_label)
    timestamp_label_array = np.array(timestamp_label_array)
    timestamp_label_array_log1p_mean = np.log1p(timestamp_label_array).mean()
    timestamp_label_array_log1p_std = np.log1p(timestamp_label_array).std()

    print(f'timestamp_label_array_log1p_mean {timestamp_label_array_log1p_mean:.3f}')
    print(f'timestamp_label_array_log1p_std {timestamp_label_array_log1p_std:.3f}')

    od_indices = np.random.choice(non_zero_indices, size=args.num_gene, p=od_probabilities)
    od_coords = np.column_stack(np.unravel_index(od_indices, od_mat.shape))

    origin_datetime_list = [datetime.strptime(row.split(',')[0], '%Y-%m-%dT%H:%M:%SZ') for row in random.sample(list(train_traj['time_list']), args.num_gene)]

    model = HOSER(
        config.road_network_encoder_config,
        config.road_network_encoder_feature,
        config.trajectory_encoder_config,
        config.navigator_config,
        road2zone,
    )

    model_state_dict = torch.load(os.path.join(save_dir, f'best.pth'), map_location='cpu')
    model.load_state_dict(model_state_dict)

    gene_trace_road_id = [None] * args.num_gene
    gene_trace_datetime = [None] * args.num_gene

    torch.multiprocessing.set_start_method('spawn', force=True)
    initargs = (model, reachable_road_id_dict, geo, road_center_gps, timestamp_label_array_log1p_mean, timestamp_label_array_log1p_std, device)
    with torch.multiprocessing.Pool(processes=args.processes, initializer=init_searcher, initargs=initargs) as pool:
        tasks = zip(od_coords, origin_datetime_list)
        results = list(tqdm(pool.imap(process_task, tasks), total=len(od_coords), desc='Generating trajectories'))

        for i, (trace_road_id, trace_datetime_str) in enumerate(results):
            gene_trace_road_id[i] = trace_road_id
            gene_trace_datetime[i] = trace_datetime_str

    res_df = pd.DataFrame({
        'gene_trace_road_id': gene_trace_road_id,
        'gene_trace_datetime': gene_trace_datetime,
    })
    os.makedirs(gene_dir, exist_ok=True)
    now = datetime.now()
    res_df.to_csv(os.path.join(gene_dir, f'{now.strftime("%Y-%m-%d_%H-%M-%S")}.csv'), index=False)
