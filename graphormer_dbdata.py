import json
import logging
import os
import pickle
from os.path import exists, join

import lmdb
import torch
from fairseq.data import FairseqDataset

from graphormer_utils import build_pyg_graph
from utils.common_utils import get_number_of_total_chunks

logger = logging.getLogger(__name__)


class RetroGraphormerDBDataset(FairseqDataset):
    def __init__(self, db_paths, is_test=False) -> None:
        super(RetroGraphormerDBDataset).__init__()
        self.db_paths = db_paths
        self._init_db()
        self.len = self.offsets[-1]
        self.collater = None
        self.is_test = is_test

    def __getstate__(self):
        return (self.db_paths, self.len,  self.collater, self.is_test)

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.db_paths, self.len, self.collater, self.is_test = state
        self.envs = []

    def _init_db(self):
        logger.info(f'init in process {os.getpid()}')

        self.offsets = [0]
        self.envs = []
        self.txns = []

        for db_path in self.db_paths:
            env = lmdb.Environment(
                db_path,
                readonly=True,
                readahead=True,
                meminit=False,
                lock=False,
            )
            n_entries = env.stat()["entries"]
            self.offsets.append(n_entries + self.offsets[-1])
            self.envs.append(env)
            self.txns.append(env.begin())
            logger.info(
                f'loading {n_entries} records from {db_path} in process {os.getpid()}')
        logger.info(
            f'loaded total {self.offsets[-1]} records in process {os.getpid()}')

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if not self.envs:
            self._init_db()

        if index < 0 or index >= self.len:
            raise IndexError

        for i in range(len(self.txns)):
            if index >= self.offsets[i] and index < self.offsets[i+1]:

                data = pickle.loads(self.txns[i].get(
                    f"{index-self.offsets[i]}".encode()))
                reactants_g, goldens_g, products_g, products_golden_g = data
                if self.is_test:
                    reactants_g, products_g = build_pyg_graph(reactants_g, torch.tensor([0]), index),\
                        build_pyg_graph(products_g, torch.tensor([0]), index)
                    return reactants_g, products_g

                reactants_g, goldens_g, products_g, products_golden_g = \
                    build_pyg_graph(reactants_g, torch.tensor([0]), index),\
                    build_pyg_graph(goldens_g, torch.tensor([1]), index),\
                    build_pyg_graph(products_g, torch.tensor([0]), index),\
                    build_pyg_graph(products_golden_g,
                                    torch.tensor([1]), index)

                return reactants_g, goldens_g, products_g, products_golden_g


def build_dataset_file_list(input_dir, total_chunks, file_id):
    dataset_list = []
    for chunk_id in range(total_chunks):
        db_path = join(
            input_dir, f'{chunk_id}_{file_id}.db')
        dataset_list.append(db_path)
    return dataset_list


def build_split_files(args, dataset, split):
    input_dir = join(args.root, 'graph', dataset)
    total_chunks = get_number_of_total_chunks('gh', dataset)
    if split == 'train':
        train_files = []
        for file_id in range(8):
            train_files.extend(build_dataset_file_list(
                input_dir, total_chunks, file_id))
        return train_files
    elif split == 'valid':
        sample_data_files = min(3, total_chunks)
        return build_dataset_file_list(
            input_dir, total_chunks, 'test')[:sample_data_files]
    else:
        return build_dataset_file_list(
            input_dir, total_chunks, split)
