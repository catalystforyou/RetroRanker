import contextlib
import logging
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.data import (FairseqDataset, NestedDictionaryDataset,
                          NumSamplesDataset, data_utils)
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import FairseqDataclass, FairseqTask, register_task
from omegaconf import II, OmegaConf, open_dict

# from graphormer_data import (RetroGraphormerDataset,build_split_files)
from graphormer_dbdata import RetroGraphormerDBDataset, build_split_files
from graphormer_fairseq import UnreusedEpochBatchIterator
from graphormer_rank import GraphormerRanker
from graphormer_utils import collator_gh

logger = logging.getLogger(__name__)


@dataclass
class GraphRankConfig(FairseqDataclass):
    root: str = field(
        default="../",
        metadata={"help": "root dir of the dataset"},
    )

    dataset_name: str = field(
        default="rsmiles",
        metadata={"help": "name of the dataset"},
    )

    batch_size: int = field(
        default=32,
        metadata={"help": "batch size"},
    )

    num_classes: int = field(
        default=1,
        metadata={"help": "number of classes or regression targets"},
    )

    max_nodes: int = field(
        default=128,
        metadata={"help": "max nodes per graph"},
    )

    num_atoms: int = field(
        default=128 * 512,  # actually is 106
        metadata={"help": "number of atom types in the graph"},
    )

    num_edges: int = field(
        default=16 * 512,  # actually is 13
        metadata={"help": "number of edge types in the graph"},
    )

    num_in_degree: int = field(
        default=512,
        metadata={"help": "number of in degree types in the graph"},
    )

    num_out_degree: int = field(
        default=512,
        metadata={"help": "number of out degree types in the graph"},
    )

    num_spatial: int = field(
        default=512,
        metadata={"help": "number of spatial types in the graph"},
    )

    num_edge_dis: int = field(
        default=128,
        metadata={"help": "number of edge dis types in the graph"},
    )

    multi_hop_max_dist: int = field(
        default=5,
        metadata={"help": "max distance of multi-hop edges"},
    )

    spatial_pos_max: int = field(
        default=1024,
        metadata={"help": "max distance of multi-hop edges"},
    )

    edge_type: str = field(
        default="multi_hop",
        metadata={"help": "edge type in the graph"},
    )

    seed: int = II("common.seed")

    pretrained_model_name: str = field(
        default="none",
        metadata={"help": "name of used pretrained model"},
    )

    load_pretrained_model_output_layer: bool = field(
        default=False,
        metadata={"help": "whether to load the output layer of pretrained model"},
    )

    user_data_dir: str = field(
        default="",
        metadata={"help": "path to the module of user-defined dataset"},
    )


@register_task("graph_rank", dataclass=GraphRankConfig)
class GraphRankTask(FairseqTask):
    def __init__(self, cfg):
        super().__init__(cfg)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        # """Load a given dataset split (e.g., train, valid, test)."""

        assert split in ["train", "valid", "test"]
        # if split in "test":
        #     raise
        paths = build_split_files(self.cfg, self.cfg.dataset_name, split)        
        if split == 'train':
            chunk_size = 30
            paths = [paths[i:i + chunk_size] for i in range(0, len(paths), chunk_size)]
            paths = paths[(epoch - 1) % len(paths)]
        if split == 'test':
            paths = paths[epoch:epoch+1]
            
        logger.info(f'loading {len(paths)} files for {split}')
        batched_data = RetroGraphormerDBDataset(paths, is_test= split == 'test')
        batched_data.collater = partial(collator_gh, max_node=self.cfg.max_nodes,
                                        multi_hop_max_dist=self.cfg.multi_hop_max_dist, spatial_pos_max=self.cfg.spatial_pos_max)

        data_sizes = np.array([self.max_nodes()] * len(batched_data))
        dataset = NestedDictionaryDataset(
            {
                "nsamples": NumSamplesDataset(),
                "net_input": {"batched_data": batched_data}
            },
            sizes=data_sizes,
        )
        self.datasets[split] = dataset

    def build_model(self, cfg):

        with open_dict(cfg) if OmegaConf.is_config(cfg) else contextlib.ExitStack():
            cfg.max_nodes = self.cfg.max_nodes

        model = GraphormerRanker.build_model(cfg, self)

        return model

    def max_nodes(self):
        return self.cfg.max_nodes

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None

    @property
    def label_dictionary(self):
        return None

    def has_sharded_data(self, split):
        if split == 'train':
            return True
        return False

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):
        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        if max_positions is not None:
            indices = self.filter_indices_by_size(
                indices, dataset, max_positions, ignore_invalid_inputs
            )

        # create mini-batches with given size constraints
        batch_sampler = dataset.batch_by_size(
            indices,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )
        
        # return an unreused, sharded iterator
        epoch_iter = UnreusedEpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
            skip_remainder_batch=skip_remainder_batch,
            grouped_shuffling=grouped_shuffling,
            reuse_dataloader=False,
        )
        return epoch_iter

@register_criterion("Gl2_loss", dataclass=FairseqDataclass)
class GraphRankL2Loss(FairseqCriterion):
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"][0]["x"].shape[1]

        logits = model(**sample["net_input"])
        targets = torch.ones(
            logits.shape[0], dtype=torch.long, device=logits.device)
        loss = nn.CrossEntropyLoss(reduction="sum")(logits, targets)
        acc_output = logits.detach().cpu().numpy()
        acc = len(acc_output[acc_output[:, 0] < acc_output[:, 1]])
        logging_output = {
            "loss": loss.data,
            "acc": acc,
            "sample_size": logits.size(0),
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        acc_sum = sum(log.get("acc", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("acc", acc_sum / sample_size,
                           sample_size, round=6)
        metrics.log_scalar("loss", loss_sum / sample_size,
                           sample_size, round=6)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
