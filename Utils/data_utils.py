"""This is the data utils file, it mainly handles the embedding of the data"""

import numpy as np
import pandas as pd
import geopy.distance as distance

"""
A function to find the closest n neighbours to a target station,
results are returned in ascending order of distance 

For computational efficiency, it is recommanded only one intance of stations are passed in 
unique_stations 
"""
def find_clostest_n_neighbours(target, unique_stations, number_of_neighbours, min_distance=0, max_distance=99999):
    station_with_locations = unique_stations.copy()

    distances = station_with_locations.apply(
        lambda row: distance.distance(
            [row['latitude'], row['longitude']], [target[0], target[1]]).km,
        axis=1
    )
    station_with_locations['distance'] = distances

    station_in_range = station_with_locations.loc[(station_with_locations['distance'] >= min_distance)
                                                  & (station_with_locations['distance'] <= max_distance)]

    station_to_use = station_in_range.nsmallest(number_of_neighbours, 'distance')
    
    return station_to_use


"""
Given the nodes, extract the features and target for the graph 
"""
def extract_node_feature_target(nodes, features_to_use=None, target_features_to_use=None):
    
    if features_to_use is None:
        features_to_use = ['name','latitude', 'longitude', 'temp', 'wind_direction']
    if target_features_to_use is None:
        target_features_to_use = ['wind_speed']
    
    node_features = nodes[features_to_use].to_numpy()
    targets = nodes[target_features_to_use].to_numpy()

    return node_features, targets

"""
Given the target station name, find the nearest neighbours within the distance

This function should be the main entry point for the data generation.

nodes = {x,y, features} where x and y are all wind speeds,
"""
def generate_data(raw_data, number_of_neighbours):
    node_features = []
    node_targets = []
    edges_attributes = []
    edges_row = []
    edges_col = []
    
    # All stations in the dataset
    # TODO: Maybe find a way to move this to a config file
    stations_to_dict = {
        'CAMROSE' : 0, 'CORONATION CLIMATE' : 1, 'EDMONTON INTL A' : 2, 'LLOYDMINSTER' : 3,
       'ROCKY MTN HOUSE (AUT)' : 4, 'VEGREVILLE' : 5, 'EDMONTON STONY PLAIN CS' : 6,
       'DRUMHELLER EAST' : 7, 'LACOMBE CDA 2' : 8, 'BROOKS' : 9, 'CALGARY INTL A' : 10,
       'CLARESHOLM' : 11, 'LETHBRIDGE' : 12, 'LETHBRIDGE CDA' : 13, 'MEDICINE HAT' : 14,
       'MEDICINE HAT RCS' : 15, 'STRATHMORE AGDM' : 16, 'MILK RIVER' : 17, 'ONEFOUR CDA' : 18,
       'BANFF CS' : 19}

    for timestamp in raw_data['date'].unique():
        #This gives all station data per timestamp
        temp_nodes = raw_data.loc[raw_data['date'] == timestamp]
        temp_node_features, temp_targets = extract_node_feature_target(temp_nodes)
        node_features.extend(temp_node_features)
        node_targets.extend(temp_targets)

    # From the nodes, we can generate edges
    # Those nodes are stationary, so the connection only needs to be computed once
    # Currently, those are undirected unweighted edges
    if len(node_features) != 0:
        for station, idx in stations_to_dict.items():
            # Set one station as the target
            target = temp_nodes.loc[temp_nodes['name'] == station]
            target_latitude = target.iloc[0]['latitude']
            target_longitude = target.iloc[0]['longitude']
            # select all other stations
            rest_stations = temp_nodes.loc[temp_nodes['name'] != station]
            # Find the n closest neighbours, distanct to the target is also included
            neighbour_stations = find_clostest_n_neighbours([target_latitude, target_longitude], rest_stations, number_of_neighbours)
            neighbour_index = [stations_to_dict[name] for name in neighbour_stations['name'].unique()]

            edges_row.extend(np.repeat(idx, len(neighbour_index)))
            edges_col.extend(neighbour_index)
    else:
        raise Exception("No data found to generate edges")
    
    

    return node_features, node_targets, edges_row, edges_col
