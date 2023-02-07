import logging
from typing import Optional, Tuple

import numpy as np
import torch
from dgl import DGLGraph
from fairseq.models import register_model_architecture
from torch_geometric.data import Data as PYGGraph

from Graphormer.graphormer.data.collator import collator
from Graphormer.graphormer.data.wrapper import algos, convert_to_single_emb
from Graphormer.graphormer.models.graphormer import (
    base_architecture, graphormer_base_architecture,
    graphormer_large_architecture, graphormer_slim_architecture)
from graphormer_rank import GraphormerRanker

logger = logging.getLogger(__name__)

# modified from
# https://github.com/microsoft/Graphormer/blob/main/graphormer/data/dgl_datasets/dgl_dataset.py
def extract_edge_and_node_features(
        graph_data: DGLGraph
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
]:
    def extract_tensor_from_node_or_edge_data(
        feature_dict: dict, num_nodes_or_edges
    ):
        int_feature_list = []

        def extract_tensor_from_dict(feature: torch.Tensor):
            if feature.dtype == torch.int32 or feature.dtype == torch.long:
                int_feature_list.append(feature)
            elif feature.dtype == torch.float32 or feature.dtype == torch.float64:
                int_feature_list.append(feature.to(torch.int32))

        for feature_key in feature_dict:
            feature_or_dict = feature_dict[feature_key]
            if isinstance(feature_or_dict, torch.Tensor):
                extract_tensor_from_dict(feature_or_dict)
            elif isinstance(feature_or_dict, dict):
                for feature in feature_or_dict:
                    extract_tensor_from_dict(feature)
        int_feature_tensor = (
            torch.from_numpy(np.zeros(shape=[num_nodes_or_edges, 1])).long()
            if len(int_feature_list) == 0
            else torch.cat(int_feature_list)
        )
        return int_feature_tensor

    node_int_feature = extract_tensor_from_node_or_edge_data(
        graph_data.ndata, graph_data.num_nodes()
    )
    edge_int_feature = extract_tensor_from_node_or_edge_data(
        graph_data.edata, graph_data.num_edges()
    )
    return (
        node_int_feature,
        edge_int_feature
    )


def preprocess_dgl_graph(
    graph_data: DGLGraph, y: torch.Tensor, idx: int
) -> PYGGraph:
    if not graph_data.is_homogeneous:
        raise ValueError(
            "Heterogeneous DGLGraph is found. Only homogeneous graph is supported."
        )
    N = graph_data.num_nodes()

    node_int_feature, edge_int_feature = extract_edge_and_node_features(
        graph_data)
    edge_index = graph_data.edges()
    attn_edge_type = torch.zeros(
        [N, N, edge_int_feature.shape[1]], dtype=torch.long
    )
    attn_edge_type[
        edge_index[0].long(), edge_index[1].long()
    ] = convert_to_single_emb(edge_int_feature)
    dense_adj = graph_data.adj().to_dense().type(torch.int)
    shortest_path_result, path = algos.floyd_warshall(dense_adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros(
        [N + 1, N + 1], dtype=torch.float)  # with graph token

    pyg_graph = PYGGraph()
    pyg_graph.x = convert_to_single_emb(node_int_feature)
    pyg_graph.adj = dense_adj
    pyg_graph.attn_bias = attn_bias
    pyg_graph.attn_edge_type = attn_edge_type
    pyg_graph.spatial_pos = spatial_pos
    pyg_graph.in_degree = dense_adj.long().sum(dim=1).view(-1)
    pyg_graph.out_degree = pyg_graph.in_degree
    pyg_graph.edge_input = torch.from_numpy(edge_input).long()
    if y.dim() == 0:
        y = y.unsqueeze(-1)
    pyg_graph.y = y
    pyg_graph.idx = idx

    return pyg_graph


def preprocess_dgl_graph_simple(graph_data: DGLGraph):
    if not graph_data.is_homogeneous:
        raise ValueError(
            "Heterogeneous DGLGraph is found. Only homogeneous graph is supported."
        )
    N = graph_data.num_nodes()
    node_int_feature, edge_int_feature = extract_edge_and_node_features(
        graph_data)
    edge_index = graph_data.edges()
    dense_adj = graph_data.adj().to_dense().type(torch.int)
    shortest_path_result, path = algos.floyd_warshall(dense_adj.numpy())
    return (N, node_int_feature, edge_int_feature, edge_index, dense_adj, shortest_path_result, path)


def build_pyg_graph(inputs, y: torch.Tensor, idx: int) -> PYGGraph:
    N, node_int_feature, edge_int_feature, edge_index, dense_adj, shortest_path_result, path = inputs
    attn_edge_type = torch.zeros(
        [N, N, edge_int_feature.shape[1]], dtype=torch.long
    )
    attn_edge_type[
        edge_index[0].long(), edge_index[1].long()
    ] = convert_to_single_emb(edge_int_feature)
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros(
        [N + 1, N + 1], dtype=torch.float)  # with graph token
    pyg_graph = PYGGraph()
    pyg_graph.x = convert_to_single_emb(node_int_feature)
    pyg_graph.adj = dense_adj
    pyg_graph.attn_bias = attn_bias
    pyg_graph.attn_edge_type = attn_edge_type
    pyg_graph.spatial_pos = spatial_pos
    pyg_graph.in_degree = dense_adj.long().sum(dim=1).view(-1)
    pyg_graph.out_degree = pyg_graph.in_degree
    pyg_graph.edge_input = torch.from_numpy(edge_input).long()
    if y.dim() == 0:
        y = y.unsqueeze(-1)
    pyg_graph.y = y
    pyg_graph.idx = idx
    return pyg_graph


# modified from
# https://github.com/microsoft/Graphormer/blob/main/graphormer/data/collator.py
def collator_gh(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [item for item in items if all(
        [ele.x.size(0) <= max_node for ele in item])]
    return [collator(its, max_node=max_node+1, multi_hop_max_dist=multi_hop_max_dist, spatial_pos_max=spatial_pos_max) for its in list(map(list, zip(*items)))]


# modified from
# https://github.com/microsoft/Graphormer/blob/main/graphormer/models/graphormer.py
@register_model_architecture("graphranker", "graphranker")
def rank_base_architecture(args):
    base_architecture(args)


@register_model_architecture("graphranker", "graphranker_base")
def graphrank_base_architecture(args):
    graphormer_base_architecture(args)


@register_model_architecture("graphranker", "graphranker_large")
def graphrank_large_architecture(args):
    graphormer_large_architecture(args)


@register_model_architecture("graphranker", "graphranker_slim")
def graphrank_slim_architecture(args):
    graphormer_slim_architecture(args)
