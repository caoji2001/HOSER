from tqdm import tqdm
import numpy as np
import pandas as pd


if __name__ == '__main__':
    for dataset in ['Beijing', 'Porto', 'San_Francisco']:
        print(f'Processing {dataset} dataset')

        road2zone = []
        with open(f'../{dataset}/road_network_partition', 'r') as file:
            for line in file:
                    road2zone.append(int(line.strip()))

        zone_cnt = max(road2zone) + 1
        zone_trans_mat = np.zeros((zone_cnt, zone_cnt), dtype=np.int64)

        traj = pd.read_csv(f'../{dataset}/train.csv')
        for _, row in tqdm(traj.iterrows(), total=len(traj)):
                rid_list = eval(row['rid_list'])
                zone_list = [road2zone[rid] for rid in rid_list]
                for prev_zone, next_zone in zip(zone_list[:-1], zone_list[1:]):
                    if prev_zone != next_zone:
                        zone_trans_mat[prev_zone, next_zone] += 1

        np.save(f'../{dataset}/zone_trans_mat.npy', zone_trans_mat)
