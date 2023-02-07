import json
import math
from os.path import join

import torch
from dgl.data.utils import load_graphs
from torch.utils.data import ChainDataset, DataLoader, IterableDataset

from utils.model_utils import collate_molgraphs


class RetroDataset(IterableDataset):
    def __init__(self, raw_file, reactants_g_file, goldens_g_file, products_g_file, products_golden_g_file, batchsize) -> None:
        super(RetroDataset).__init__()
        self.raw = json.load(open(raw_file))
        self.reactants_g_file = reactants_g_file
        self.goldens_g_file = goldens_g_file
        self.products_g_file = products_g_file
        self.products_golden_g_file = products_golden_g_file
        self.batchsize = batchsize
        self.start = 0
        self.end = len(self.raw)

    def __len__(self):
        return len(self.raw)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.raw)
        else:
            per_worker = int(
                math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        for i in range(iter_start, iter_end, self.batchsize):
            cur_end = i + self.batchsize
            if cur_end > iter_end:
                cur_end = iter_end
            idx = list(range(i, cur_end))
            yield self.raw[i:cur_end], \
                load_graphs(self.reactants_g_file, idx)[0], \
                load_graphs(self.goldens_g_file, idx)[0], \
                load_graphs(self.products_g_file, idx)[0], \
                load_graphs(self.products_golden_g_file, idx)[0]


def build_dataset(input_dir, total_chunks, file_id, batch_size):
    dataset_list = []
    for chunk_id in range(total_chunks):
        raw_file = join(
            input_dir, f'rxns_{chunk_id}_{file_id}.json')
        reactants_g_file = join(
            input_dir, f'reactants_{chunk_id}_{file_id}.bin')
        goldens_g_file = join(
            input_dir, f'goldens_{chunk_id}_{file_id}.bin')
        products_g_file = join(
            input_dir, f'products_{chunk_id}_{file_id}.bin')
        products_golden_g_file = join(
            input_dir, f'products_golden_{chunk_id}_{file_id}.bin')
        
        print(f'loading file {raw_file}')
        dataset_list.append(RetroDataset(raw_file, reactants_g_file,
                            goldens_g_file, products_g_file, products_golden_g_file, batch_size))
    return dataset_list


def build_dataloader(args, is_training, dataset):
    input_dir = join(args.root, 'data', dataset, '3_gengraph')
    batch_size = args.batch_size
    total_chunks = 5

    if is_training:
        print(f'number of workers: {args.num_workers}')
        all_datasets = []
        for file_id in range(8):
            all_datasets.extend(build_dataset(
                input_dir, total_chunks, file_id, batch_size))
        train_set = ChainDataset(all_datasets[:-total_chunks])
        val_set = ChainDataset(all_datasets[-total_chunks:])

        train_loader = DataLoader(dataset=train_set, collate_fn=collate_molgraphs,
                                    batch_size=None, shuffle=False, num_workers=args.num_workers)
        val_loader = DataLoader(dataset=val_set, collate_fn=collate_molgraphs,
                                batch_size=None, shuffle=False, num_workers=args.num_workers)
        return train_loader, val_loader
    else:
        test_set_list = build_dataset(
            input_dir, total_chunks, 'test', batch_size)
        test_loader = DataLoader(dataset=ChainDataset(test_set_list), collate_fn=collate_molgraphs,
                                    batch_size=None, shuffle=False)
        return test_loader, test_set_list
