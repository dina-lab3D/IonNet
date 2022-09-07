"""
Utility functions for dataset creation
"""

from torch.utils.data import WeightedRandomSampler
import numpy as np
import torch
import torch_geometric.loader as geom_loader
from config import base_config as config

def create_random_sampler(train_dataset):
    """
    WeightedRandomSampler gets the weights for each class per sample and makes sure to sample batches that
    are roughly equal with number of positive and negative samples.
    https://www.maskaravivek.com/post/pytorch-weighted-random-sampler/ for more information.
    @param train_dataset:
    @return:
    """

    # using loader for the workers to access the data much faster, shuffle is false to keep maintain the order
    train_graph_loader = geom_loader.DataLoader(train_dataset, config['train_dict']['batch_size'], num_workers=config['train_dict']['num_workers'], shuffle=False)
    y_train = []
    for batch in train_graph_loader:
        y_train += batch.y.int().tolist()
    y_train = np.array(y_train)
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in [0, 1]])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    return sampler


def evaluate_dataset(dataset, dataset_name):
    graph_loader = geom_loader.DataLoader(dataset, config['train_dict']['batch_size'], num_workers=config['train_dict']['num_workers'], shuffle=False)
    y_train = []
    for batch in graph_loader:
        y_train += batch.y.int().tolist()
    print(f"{dataset_name} has a total of {len(dataset)} items")
    print(f"{dataset_name} has pos: {sum(y_train)} neg: {len(y_train)- sum(y_train)}")