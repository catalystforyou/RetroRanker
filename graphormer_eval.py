import json
import logging
from os.path import exists, join
from pathlib import Path

import torch
from fairseq import checkpoint_utils, options
from fairseq.data import FairseqDataset
from fairseq.tasks import FairseqTask
from tqdm import tqdm

from graphormer_fairseq import UnreusedEpochBatchIterator
from graphormer_rank import GraphormerRanker
from graphormer_task import GraphRankL2Loss, GraphRankTask
from graphormer_utils import (graphrank_base_architecture,
                              graphrank_large_architecture,
                              graphrank_slim_architecture,
                              rank_base_architecture)
from utils.model_utils import move_to_device

logger = logging.getLogger(__name__)

def main():
    parser = options.get_generation_parser(default_task='graph_rank')
    parser.add_argument('--chunk_id', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default='./')
    args = options.parse_args_and_arch(parser)
    if not exists(args.out_dir):
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    logger.info('| loading model from {}'.format(args.path))
    task: FairseqTask = None
    model: GraphormerRanker = None
    models, _model_args, task = checkpoint_utils.load_model_ensemble_and_task([
                                                                              args.path], arg_overrides={'root': args.root, 'dataset_name': args.dataset_name})
    model = models[0]
    model = move_to_device(model, device=device)
    model.eval()
    task.load_dataset('test', epoch=args.chunk_id)
    dataset: FairseqDataset = task.dataset('test')
    indices = dataset.ordered_indices()

    # create mini-batches with given size constraints
    batch_sampler = dataset.batch_by_size(
        indices,
        max_tokens=None,
        max_sentences=args.batch_size,
        required_batch_size_multiple=1,
    )
    
    data_iter = UnreusedEpochBatchIterator(
        dataset=dataset,
        collate_fn=dataset.collater,
        batch_sampler=batch_sampler,
        num_workers=3,
        buffer_size=10,
        disable_shuffling=True).next_epoch_itr(shuffle=False, fix_batches_to_gpus=False, set_dataset_epoch=True)
    scores = []
    ids = []
    for batch in tqdm(data_iter, mininterval=100):
        batch = move_to_device(batch, device)
        score = model(**batch["net_input"])
        ids.extend(batch["net_input"]['batched_data'][0]['idx'].tolist())
        scores.extend((score[:, 1] - score[:, 0]).tolist())
    result = dict(zip(ids, scores))
    json.dump(result, open(
        join(args.out_dir, f'output_{args.chunk_id}.json'), 'w'))


if __name__ == '__main__':
    main()
