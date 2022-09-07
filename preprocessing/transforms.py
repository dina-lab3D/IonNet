import random
from scipy.spatial import distance
import torch
from torch_geometric.transforms import BaseTransform
import numpy as np
from config import base_config
from collections.abc import Iterable


class ChangeLabelAccordingToDistance(BaseTransform):
    """
    change label according to the distance from nearest MG
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, data):
        """
        change the label according to the threshold
        @param data:
        @return:
        """

        label = (data.mg_dist <= self.threshold)
        label = torch.tensor([label])
        data.y = label
        return data


class NodeRepresentationAblation(BaseTransform):
    """
    Changes each node embedding in the graph to have 0 in a given list of indices.
    """

    def __init__(self, index_to_zero: [int]):
        self.index_to_zero = index_to_zero

    def __call__(self, data):
        """
        make each node representation in the data at the given index_to_zero, zero
        @param data:
        @return: data after the transformation, charge is in 3 and ASA is in 15.
        """

        for index in self.index_to_zero: # the for is to allow index_to_zero to be any iterable and not just a list.
            data.x[:, index] = 0
        return data



class MGTranslate(BaseTransform):
    r"""Translate node position for MG atom, label of data must be 1.
    Randomly calls on translate with some probability 'p' passed during initialization
    When working, the translation moves each coordinate in some direction up to 'translate' distance


    Args:
        translate (sequence or float or int): Maximum translation in each
            dimension, defining the range
            :math:`(-\mathrm{translate}, +\mathrm{translate})` to sample from.
            If :obj:`translate` is a number instead of a sequence, the same
            range is used for each dimension.
    """
    def __init__(self, translate, p, thresh):
        self.translate = translate
        self.p = p
        self.thresh = thresh

    def __call__(self, data):
        """
        For each data point if it is a positive sample at some probability p it will translate the position
        of the mg atom while also looking for conflicts (moving the atom too close to some other atom). The call will
        run until there are no conflicts.
        @param data:
        @return:
        """
        if random.uniform(0, 1) <= self.p:
            conflicts = True
            n, dim = data.coordinates.size()
            mg_coords = data.coordinates[0]
            mg_dist_check = dist_check(mg_coords, self.thresh)
            ts = []
            iter = 0  # make sure if the call gets stuck, it ends after 100 iterations at the very most.
            while conflicts:
                ts.clear()
                for i in range(dim):
                    ts.append(torch.FloatTensor(1).uniform_(-abs(self.translate[i]), abs(self.translate[i])))
                conflicts = any(map(mg_dist_check, data.coordinates[1:], torch.stack(ts, dim=-1).squeeze()))
                iter += 1
                if iter > 100:
                    print("could not find proper conformation for data, returning data with no changes.")
                    return data

            data.coordinates[0] += torch.stack(ts, dim=-1).squeeze()
            self.__fix_graph(data)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.translate})'

    def __fix_graph(self, data):
        """
        Fixes entire graph after translation, this includes updating edges and then fixing distances.
        @param data:
        @return:
        """

        bad_indices = []

        for i, node_idx in enumerate(data.edge_index[0]):
            if node_idx != 0:
                break
            else:
                cur_dist = distance.euclidean(data.coordinates[0], data.coordinates[i])
                data.edge_attr[i] = cur_dist
                if cur_dist > base_config['radius']:
                    bad_indices.append(i)
        # to fix the graph make sure to delete indices with a distance above the radius
        mask = np.full(len(data.edge_attr), False)
        mask[bad_indices] = True
        mask = ~mask
        data.edge_attr = data.edge_attr[mask]
        data.edge_index = data.edge_index[:, mask]


def dist_check(coords, thresh):
    return lambda other_coords, translation: distance.euclidean(coords + translation, other_coords) < thresh
