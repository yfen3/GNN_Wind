# Get the data for expeirment
import pandas as pd 
import numpy as np
import time
from Utils import data_utils 
from sklearn.preprocessing import StandardScaler
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import networkx as nx
import torch
import torch.nn.functional as F
import torch_geometric.nn as nn
from sklearn.metrics import r2_score, root_mean_squared_error
import ab_wind_dataset
import matplotlib.pyplot as plt
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers, models

device = torch.device("cuda:0")
print(f"Keras version is {keras.__version__}")
print(f"Torch version is {torch.__version__}")
print(f"Num GPUs Available: {torch.cuda.device_count()}")

# Modified from source: https://github.com/zamirmehdi/GNN-Node-Regression/blob/main/src/main.py

"""
Configurations 
"""
dataset_file_location = 'data'
number_of_neighbours = 9

"""
Simple GNN model to test the code
Modified from source: https://towardsdatascience.com/structure-and-relationships-graph-neural-networks-and-a-pytorch-implementation-c9d83b71c041
"""
class GNNModel(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GNNModel, self).__init__()
        self.conv1 = nn.GATConv(num_node_features, 16)
        self.conv2 = nn.GATConv(16, 8)
        self.fc = nn.Linear(8, 1)  # Outputting a single value per node

    def forward(self, data, target_node_idx=None):
        x, edge_index = data.x, data.edge_index
        x = x.clone()

        # Mask the target node's feature with a value of zero! 
        # Aim is to predict this value from the features of the neighbours
        if target_node_idx is not None:
            x[target_node_idx] = torch.zeros_like(x[target_node_idx])

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc(x)

        return x

def RMSE_loss(yhat, y):
    return torch.sqrt(torch.mean((yhat - y) ** 2))

def run_model(model, dataset, num_epochs, batch_size = 32):
    optimizer = torch.optim.Adam(model.parameters())
    mse_loss = torch.nn.MSELoss()
    model = model.to(device)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 

    model.train()
    # Training loop
    for epoch in range(num_epochs):
        t0 = time.time()
        accumulated_loss = 0 
        optimizer.zero_grad()
        loss = 0  
        for data in train_loader:
            data = data.to(device)
            target_nodes = data.target_node_index
            output = model(data, target_nodes)  # get predictions with the target node masked
                                        # check the feed forward part of the model
            target = data.y[target_nodes] 
            prediction = output
            loss += mse_loss(prediction, target)
            #Update parameters at the end of each set of batches
            loss.backward() 
            optimizer.step()
            optimizer.zero_grad()
            accumulated_loss += loss.item()
            loss = 0

        duration = time.time() - t0
        average_loss = accumulated_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Average Loss: {average_loss}, Time: {duration}')

    return model