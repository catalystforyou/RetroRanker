import logging
import os

import torch
from fairseq.data.iterators import (BufferedIterator, CountingIterator,
                                    EpochBatchIterator, FrozenBatchSampler)

logger = logging.getLogger(__name__)


class UnreusedEpochBatchIterator(EpochBatchIterator):

    def _get_iterator_for_epoch(
        self, epoch, shuffle, fix_batches_to_gpus=False, offset=0
    ):
        self.batch_sampler = FrozenBatchSampler(
            self.ordered_batches,
            epoch,
            fix_batches_to_gpus,
            shuffle,
            initial_offset=offset,
        )

        if offset > 0 and len(self.batch_sampler) == 0:
            return None

        if self.num_workers > 0:
            os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"

        # Create data loader
        itr = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_sampler=self.batch_sampler,
            num_workers=self.num_workers,
            timeout=self.timeout,
            pin_memory=False,
            persistent_workers=False,
        )

        # Wrap with a BufferedIterator if needed
        if self.buffer_size > 0:
            itr = BufferedIterator(self.buffer_size, itr)

        # Wrap with CountingIterator
        itr = CountingIterator(itr, start=offset)

        if self.skip_remainder_batch:
            # TODO: Below is a lazy implementation which discard the final batch regardless
            # of whether it is a full batch or not.
            total_num_itrs = len(self.batch_sampler) - 1
            itr.take(total_num_itrs)
            logger.info(
                f"skip final residual batch, total_num_itrs = {total_num_itrs}")

        return itr
