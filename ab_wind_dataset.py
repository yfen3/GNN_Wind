import torch
from torch_geometric.data import Data, InMemoryDataset
import pandas as pd
import numpy as np
from Utils import data_utils 

"""
Custom AB wind data set class

Modified from source: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html

Returns:
    _type_: _description_
"""
class AbWindDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        self.number_of_neighbours = 9
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['process_ab_wind_data.pt']

    @property
    def processed_file_names(self):
        return ['process_ab_wind_data.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = self.process_from_raw_data()

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    def process_from_raw_data(self):
        data_list = []
        df_train = pd.read_csv('Data/processed_ab_wind_train.txt')
        unique_dates = df_train['date'].unique()
        subsampled_dates = unique_dates[0:100]
        subsampled_df_train = df_train[df_train['date'].isin(subsampled_dates)]

        # This contains processed data in format of 
        # Time 1 station A B C -> Time 2 station A B C ...  
        node_features, node_targets, edge_row, edge_col = data_utils.generate_data(subsampled_df_train, self.number_of_neighbours)
        
        # In total 20 stations, iterate through each group of stations
        for i in range(len(node_features)//20):
            data = Data(
                x = torch.tensor(np.array(node_features[i*20:i*20+20])[:, 1:].tolist(), dtype=torch.float32), 
                edge_index = torch.tensor([edge_row, edge_col], dtype=torch.long),
                y = torch.tensor(np.array(node_targets)[i*20:i*20+20].tolist(), dtype=torch.float32),
            )
            data_list.append(data)

        return data_list   

