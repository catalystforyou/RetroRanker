from fairseq_cli.train import cli_main

from graphormer_rank import GraphormerRanker
from graphormer_task import GraphRankL2Loss, GraphRankTask
from graphormer_utils import (graphrank_base_architecture,
                              graphrank_large_architecture,
                              graphrank_slim_architecture,
                              rank_base_architecture)

if __name__ == '__main__':
    cli_main()
