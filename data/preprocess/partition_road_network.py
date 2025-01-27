import os
import subprocess
import numpy as np
import pandas as pd


if __name__ == '__main__':
    current_directory = os.getcwd()
    for dataset in ['Beijing', 'Porto', 'San_Francisco']:
        print(f'Processing {dataset} dataset')

        geo = pd.read_csv(f'../{dataset}/roadmap.geo')
        rel = pd.read_csv(f'../{dataset}/roadmap.rel')

        num_roads = len(geo)
        undirected_matrix = np.zeros((num_roads, num_roads), dtype=bool)
        for _, row in rel.iterrows():
            origin_id = row['origin_id']
            destination_id = row['destination_id']
            undirected_matrix[origin_id, destination_id] = True
            undirected_matrix[destination_id, origin_id] = True

        num_edges = np.sum(undirected_matrix)
        assert num_edges % 2 == 0
        num_edges //= 2

        with open(f'../{dataset}/graph_input.tmp', 'w') as file:
            file.write(f'{num_roads} {num_edges}\n')
            for rid in range(num_roads):
                adj_rid_list = (np.where(undirected_matrix[rid]==True)[0]).tolist()
                file.write(' '.join([str(adj_rid + 1) for adj_rid in adj_rid_list]))
                file.write('\n')

        os.chdir(f'../{dataset}')
        result = subprocess.run(f'../../KaHIP/build/kaffpa ./graph_input.tmp --k 300 --seed 0 --preconfiguration=strong --output_filename=road_network_partition', shell=True, capture_output=True, text=True)
        print(result.stdout)
        os.chdir(current_directory)

        os.remove(f'../{dataset}/graph_input.tmp')
