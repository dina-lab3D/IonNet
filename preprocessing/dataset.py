import argparse
import glob

import os
import torch
from typing import List, Sequence, Optional, Callable, Union, Tuple, Any
import os.path as osp
import multiprocessing as mp
import itertools
import numpy as np
import pickle
import h5py
from torch_geometric.data import Dataset, InMemoryDataset, Data
from torch_geometric.transforms import ToUndirected
import math
from tqdm import tqdm
from preprocessing.transforms import MGTranslate, ChangeLabelAccordingToDistance, NodeRepresentationAblation
from config import base_config

"""
List of supported datasets
"""
supported_datasets = ['graph', 'graph_translate', 'graph_change_label', 'graph_change_representation', 'inference_dataset']

"""
dataset factory
"""


def get_dataset(dataset_id, root=None, kfold_path=None):
    if dataset_id == 'graph':
        return GraphMGDataset(root, split_path=kfold_path)
    if dataset_id == 'graph_translate':
        return GraphMGDataset(root, split_path=kfold_path, transform=MGTranslate(base_config['preprocess_dict']['translate'],
                                                                                 base_config['preprocess_dict']['p'], base_config['preprocess_dict']['thresh']))
    if dataset_id == 'graph_change_label':
        return GraphMGDataset(root, split_path=kfold_path, transform=ChangeLabelAccordingToDistance(base_config['positive_label_threshold']))
    if dataset_id == 'graph_change_representation':
        return GraphMGDataset(root, split_path=kfold_path, transform=NodeRepresentationAblation(base_config['ablation_index_to_zero']))
    if dataset_id == 'inference_dataset':
        return GraphMGInMemoryDataset(root)
    raise Exception('unsupported dataset: %s' % dataset_id)


"""
Graph representation dataset. 
The database is composed of h5 files containing embeddings for each node in the graph, edges and a distance matrix
Guides used for this interface are:
https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs
"""


class GraphMGDataset(Dataset):

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:

        if os.path.isfile(self.raw_path_premade):
            with open (self.raw_path_premade, 'rb') as fp:
                my_files = pickle.load(fp)
                return my_files
        my_files = []
        for root, dirs, files in os.walk(self.raw_dir):
            base_raw_dir_name = 'raw'
            cur_root = root[root.index(base_raw_dir_name) + len(base_raw_dir_name) + 1:]
            my_files.extend([os.path.join(cur_root, file) for file in files if file.endswith('.h5')])
        with open(self.raw_path_premade, 'wb') as fp:
            pickle.dump(my_files, fp)
        return my_files


    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        my_files = []
        base_processed_dir_name = 'processed'
        for root, dirs, files in os.walk(self.processed_dir):
            # worst case will result in an empty string
            cur_root = root[root.index(base_processed_dir_name) + len(base_processed_dir_name) + 1:]
            my_files.extend([osp.join(cur_root, file) for file in files if file.startswith('data')])
        return my_files

    def process(self):
        self.__create_subdirs()
        if self.multiprocess:
            with mp.Manager() as manager:
                d = manager.dict()
                with mp.Pool(8) as pool:
                    pool.starmap(self.preprocess_helper, tqdm(zip(itertools.repeat(d), self.raw_paths, range(len(self.raw_paths))), total=len(self.raw_paths)))
                    pool.close()
                # `d` is a DictProxy object that can be converted to dict
                self.data_map = dict(d)
                with open(self.data_map_path, 'wb') as f:
                    pickle.dump(self.data_map, f)
        else:
            self.data_map = dict()
            for path, idx in zip(self.raw_paths, range(len(self.raw_paths))):
                self.preprocess_helper(self.data_map, path, idx)
            with open(self.data_map_path, 'wb') as f:
                pickle.dump(self.data_map, f)

    def __create_subdirs(self):
        num_of_subdirs = math.ceil(len(self.raw_paths) / 1000)
        for i in range(num_of_subdirs):
            os.mkdir(osp.join(self.processed_dir, f'processed_{i}'))

    def preprocess_helper(self, shared_dict, raw_path, idx):
        edge1_header = "edge1"
        edge2_header = "edge2"
        embedding_header = "repr"
        distance_header = 'distances'
        label_header = "label"
        mg_dist_header = "closest_mg"
        coordinate_header = "coordinates"
        num_of_features = 1
        corrupted_path = "/cs/usr/punims/Desktop/punims-dinaLab/Databases/MG_Graph_DB/corrupted.txt"
        # Read data from `raw_path`.
        with h5py.File(raw_path, 'r') as file:
            attributes = file['repr'].attrs
            try:
                edge1 = attributes.get(edge1_header)
                edge2 = attributes.get(edge2_header)
                embedding = attributes.get(embedding_header)
                distances = file[distance_header][()]  # syntax to get the numpy array
                dist_to_closest_mg = attributes.get(mg_dist_header)
                if "PB" in raw_path:
                    label = int(dist_to_closest_mg <= 3)
                else:
                    label = attributes.get(label_header)
                coordinates = file[coordinate_header][()]
                data = Data(x=torch.tensor(embedding), edge_index=torch.tensor(np.array([edge1, edge2]), dtype=torch.long),
                            edge_attr=self.create_edge_attributes(distances, edge1, edge2, num_of_features),
                            y=torch.tensor([label]),
                            name=raw_path, mg_dist=dist_to_closest_mg, coordinates=torch.tensor(coordinates)
                            )
                data = ToUndirected()(data)

                if self.pre_filter is not None and not self.pre_filter(data):
                    return

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_path = osp.join(osp.join(self.processed_dir, f'processed_{idx//1000}', f'data_{idx}.pt'))
                torch.save(data, data_path)
                shared_dict[raw_path] = idx

            except KeyError:
                print(f"corrupt file found {raw_path}, doesn't contain distances. writing it to {corrupted_path}")
                # TODO consider changing this because now with multiprocessing this can have concurrency issues
                with open(corrupted_path, 'a') as corrutped_file:
                    # open every time because it is a rare operation
                    corrutped_file.write(raw_path)


    @property
    def raw_paths(self) -> List[str]:
        files = to_list(self.raw_file_names)
        return [osp.join(self.raw_dir, f) for f in files]

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        if osp.isfile(osp.join(self.processed_dir, f'data_{idx}.pt')):
            data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        else:
            data = torch.load(osp.join(self.processed_dir, f'processed_{idx//1000}/data_{idx}.pt'))
        return data

    def __init__(self, root: Optional[str] = None, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None,
                 split_path: Optional[str] = None, multiprocess: bool = False):
        self.multiprocess = multiprocess
        self.data_map = dict()
        self.data_map_path = os.path.join(os.path.join(root, 'processed'), 'raw_name_to_idx_map.pkl')
        self.raw_path_premade = os.path.join(root, 'raw_path_name_list.pkl')
        super().__init__(root, transform, pre_transform, pre_filter)
        if os.path.isfile(self.data_map_path):
            with open(self.data_map_path, 'rb') as f:
                self.data_map = pickle.load(f)
        self.split_path = split_path
        if split_path:
            self.train_mask, self.val_mask, self.test_mask = self.create_masks()

    def indices(self) -> Sequence:
        return super().indices()

    def create_edge_attributes(self, distance_matrix, edge1, edge2, num_of_features):
        edge_attributes = torch.empty(size=[len(edge1), num_of_features])
        torch_distances = torch.from_numpy(distance_matrix)
        for z, x, y in zip(range(len(edge1)), edge1.astype('int'), edge2.astype('int')):
            edge_attributes[z] = torch_distances[x, y]
        return edge_attributes

    def create_masks(self) -> Tuple:
        """
        Creates train val and test masks using split_path file
        """

        return self.split_according_to_file(self.split_path, self.processed_dir)


    def parse_partition_file(self, train_list, validation_list, test_list, partition_file):
        """
        helper function for split_according_to_file
        Note that the files contain more information about the chains, however we discard that information and just
        add all files from the same PDB file to the same set (training, validation or testing)
        The order of the partition files are train test val!
        :param train_list:
        :param validation_list:
        :param test_list:
        :param partition_file:
        :return:
        """
        with open(partition_file) as part_file:
            train_flag,  test_flag = False, False
            lines = part_file.readlines()
            line_iter = iter(lines)
            while "Example" not in "".join(next(line_iter).split()):
                # skip first two lines
                pass
            while not train_flag:
                line = "".join(next(line_iter).split())
                if "Example" in line:
                    train_flag = True
                else:
                    train_list.add(line)
            while not test_flag:
                line = "".join(next(line_iter).split())
                if "Example" in line:
                    test_flag = True
                else:
                    test_list.add(line)
            for line in list(line_iter):
                validation_list.add("".join(line.split()))


    def split_according_to_file(self, partition_file: str, positive_path: str):
        """
        helper function for 'get_ids'. receives a partition file and reads it. all samples in the positive
        directory will be split according to this text file.
        the file has the following format:
        "seed used xx
        Train Samples
        ...
        Validation Samples
        ...
        Test Samples
        ..."

        :param partition_file: kfold file according to which the split shall be done.
        :return:
        """

        train_set = set()
        validation_set = set()
        test_set = set()
        self.parse_partition_file(train_set, validation_set, test_set, partition_file)

        assert not train_set.intersection(validation_set)
        assert not train_set.intersection(test_set)
        assert not validation_set.intersection(test_set)

        # NOTE: This code will add all samples from the same PDB file into the same set
        # NOTE: if a PDB file was taken out of the samples it will just show up as 0 in all sets.

        train_return = [0] * len(self)
        validation_return = [0] * len(self)
        test_return = [0] * len(self)
        for raw_path in self.raw_paths:
            i = self.data_map[raw_path]
            filename_raw = raw_path
            filename_raw = os.path.basename(filename_raw) #TODO consider why am I working with raw files instead of processed?
            if "PB" in filename_raw:
                pdbfile_name = filename_raw.split('_')[1]
            else:
                pdbfile_name = filename_raw.split('_', 1)[0]
            if any([pdbfile_name in f for f in train_set]):
                train_return[i] = 1
                continue
            elif any([pdbfile_name in f for f in validation_set]):
                validation_return[i] = 1
                continue

            elif any([pdbfile_name in f for f in test_set]):
                test_return[i] = 1


        return np.argwhere(np.array(train_return)==1), np.argwhere(np.array(validation_return)==1), np.argwhere(np.array(test_return)==1)

    def train_val_test_split(self, mode: str):
        """
        receives a string to decide how to split
        DEFAULT - split according to dataset's default split that was made during preprocessing
        RANDOM - Splits completely at random with a default 0.7 train, 0.1 val and 0.2 test
        @param dataset: the dataset needed to be split
        @param mode: in what mode to split the dataset
        @return:
        """
        modes = ["DEFAULT", "RANDOM"]
        if mode not in modes:
            print(f'{mode} is not a valid mode string, please try again')
        if mode == "DEFAULT":
            return self[self.train_mask], self[self.val_mask], self[self.test_mask]
        elif mode == "RANDOM":
            torch.manual_seed(12345)
            self.shuffle()
            data_len = len(self)
            train_end = math.floor(0.7 * data_len)
            val_end = math.floor(0.1*data_len + train_end)
            return self[:train_end], self[train_end: val_end], self[val_end:]


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]



class GraphMGInMemoryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['raw_probes.h5']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        edge1_header = "edge1"
        edge2_header = "edge2"
        embedding_header = "repr"
        distance_header = 'distances'
        mg_dist_header = "closest_mg"
        coordinate_header = "coordinates"
        num_of_features = 1
        corrupted_path = "/cs/usr/punims/Desktop/punims-dinaLab/Databases/MG_Graph_DB/corrupted.txt"
        # Read data from `raw_path`.
        self._data_list = []
        with h5py.File(self.raw_paths[0], 'r') as file:
            for group_name in file.keys():
                attributes = file[group_name]['attributes'].attrs
                try:
                    edge1 = attributes.get(edge1_header)
                    edge2 = attributes.get(edge2_header)
                    embedding = attributes.get(embedding_header)
                    distances = file[group_name][distance_header][()]  # syntax to get the numpy array
                    dist_to_closest_mg = attributes.get(mg_dist_header)
                    label = int(dist_to_closest_mg <= 3)
                    coordinates = file[group_name][coordinate_header][()]
                    data = Data(x=torch.tensor(embedding), edge_index=torch.tensor(np.array([edge1, edge2]), dtype=torch.long),
                                edge_attr=self.create_edge_attributes(distances, edge1, edge2, num_of_features),
                                y=torch.tensor([label]),
                                name=group_name, mg_dist=dist_to_closest_mg, coordinates=torch.tensor(coordinates)
                                )
                    data = ToUndirected()(data)

                    if self.pre_filter is not None and not self.pre_filter(data):
                        return

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    self._data_list.append(data)


                except KeyError:
                    print(f"corrupt file found {self.raw_paths[0]}, doesn't contain distances. writing it to {corrupted_path}")
                    with open(corrupted_path, 'a') as corrutped_file:
                        # open every time because it is a rare operation
                        corrutped_file.write(self.raw_paths[0])
        self.data, self.slices = self.collate(self._data_list)
        torch.save(self.collate(self._data_list), self.processed_paths[0])

    def create_edge_attributes(self, distance_matrix, edge1, edge2, num_of_features):
        edge_attributes = torch.empty(size=[len(edge1), num_of_features])
        torch_distances = torch.from_numpy(distance_matrix)
        for z, x, y in zip(range(len(edge1)), edge1.astype('int'), edge2.astype('int')):
            edge_attributes[z] = torch_distances[x, y]
        return edge_attributes


