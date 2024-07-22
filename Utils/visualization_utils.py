"""
This is the utils for viusalization.
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import torch_geometric


def get_error_plot(actuals, predictions):

    axis_limit = np.max(np.array(actuals).flatten())
    fig, axis = plt.subplots(1, 1, figsize=(8,8)) 
    axis.set_ylim(0,axis_limit)
    axis.set_xlim(0,axis_limit)
    axis.plot([0, axis_limit], [0, axis_limit], 'k--')
    axis.plot(actuals, predictions, 'ro', alpha=.2)
    fig.show()

def get_graph_embedding_visualization(graph_data, to_undirected=True):
    G = nx.Graph() 
    g = torch_geometric.utils.to_networkx(graph_data, to_undirected=to_undirected)
    nx.draw(g)