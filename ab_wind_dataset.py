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
    def __init__(self, root, number_of_neighbours, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        self.number_of_neighbours = number_of_neighbours
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        raise NotImplementedError('Super class does not implement this method')

    @property
    def processed_file_names(self):
        raise NotImplementedError('Super class does not implement this method')

    def download(self):
        raise Exception('Missing raw data, data download not supported')
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
        df_raw_data = pd.read_csv(self.raw_paths[0])

        # This contains processed data in format of 
        # Time 1 station A B C -> Time 2 station A B C ...  
        node_features, node_targets, target_node_index, edge_row, edge_col = data_utils.generate_data(df_raw_data, self.number_of_neighbours)
        
        # In total 20 stations, iterate through each group of stations
        for i in range(len(node_features)//20):
            data = Data(
                x = torch.tensor(np.array(node_features[i*20:i*20+20])[:, 1:].tolist(), dtype=torch.float32), 
                edge_index = torch.tensor([edge_row, edge_col], dtype=torch.long),
                y = torch.tensor(np.array(node_targets)[i*20:i*20+20].tolist(), dtype=torch.float32),
            )
            data.target_node_index = torch.tensor(target_node_index[i*20:i*20+20], dtype=torch.long)
            data_list.append(data)

        return data_list   


class AbWindDatasetTrain(AbWindDataset):
    def __init__(self, root, number_of_neighbours, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        super().__init__(root, number_of_neighbours, transform, pre_transform, pre_filter, force_reload=force_reload)

    @property
    def raw_file_names(self):
        return ['processed_ab_wind_train.txt']

    @property
    def processed_file_names(self):
        return ['processed_ab_wind_data_train.pt']

class AbWindDatasetTest(AbWindDataset):
    def __init__(self, root, number_of_neighbours, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        super().__init__(root, number_of_neighbours, transform, pre_transform, pre_filter, force_reload=force_reload)

    @property
    def raw_file_names(self):
        return ['processed_ab_wind_test.txt']

    @property
    def processed_file_names(self):
        return ['processed_ab_wind_data_test.pt']
