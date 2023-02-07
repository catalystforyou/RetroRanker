import logging

import torch
import torch.nn as nn
from fairseq.models import BaseFairseqModel, register_model
from fairseq.utils import safe_hasattr

from Graphormer.graphormer.models.graphormer import (GraphormerGraphEncoder,
                                                     GraphormerModel,
                                                     base_architecture)

logger = logging.getLogger(__name__)


@register_model("graphranker")
class GraphormerRanker(BaseFairseqModel):
    def __init__(self, args, reactant_encoder: GraphormerGraphEncoder, product_encoder: GraphormerGraphEncoder):
        super().__init__()
        self.args = args
        self.reactant_encoder = reactant_encoder
        self.product_encoder = product_encoder
        self.scorer = nn.Linear(args.encoder_embed_dim * 2, 2)

    @staticmethod
    def add_args(parser):
        GraphormerModel.add_args(parser)

    def max_nodes(self):
        return self.reactant_encoder.max_nodes

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_architecture(args)
        if not safe_hasattr(args, "max_nodes"):
            args.max_nodes = args.tokens_per_sample

        logger.info(args)
        # TODO: fix the code
        reactant_encoder = GraphormerGraphEncoder(
            # < for graphormer
            num_atoms=args.num_atoms,
            num_in_degree=args.num_in_degree,
            num_out_degree=args.num_out_degree,
            num_edges=args.num_edges,
            num_spatial=args.num_spatial,
            num_edge_dis=args.num_edge_dis,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            # >
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            encoder_normalize_before=args.encoder_normalize_before,
            pre_layernorm=args.pre_layernorm,
            apply_graphormer_init=args.apply_graphormer_init,
            activation_fn=args.activation_fn,
        )
        product_encoder = GraphormerGraphEncoder(
            # < for graphormer
            num_atoms=args.num_atoms,
            num_in_degree=args.num_in_degree,
            num_out_degree=args.num_out_degree,
            num_edges=args.num_edges,
            num_spatial=args.num_spatial,
            num_edge_dis=args.num_edge_dis,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            # >
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            encoder_normalize_before=args.encoder_normalize_before,
            pre_layernorm=args.pre_layernorm,
            apply_graphormer_init=args.apply_graphormer_init,
            activation_fn=args.activation_fn,
        )
        # tie weights
        reactant_encoder.graph_attn_bias = product_encoder.graph_attn_bias
        reactant_encoder.graph_node_feature = product_encoder.graph_node_feature
        return cls(args, reactant_encoder, product_encoder)

    def calc_sore(self, reactants_g, products_g):
        _, reactants_rep = self.reactant_encoder(reactants_g)
        _, products_rep = self.product_encoder(products_g)
        feature = torch.cat([reactants_rep, products_rep], dim=1)
        return self.scorer(feature)

    def forward(self, batched_data, **kwargs):
        if len(batched_data) == 2:
            reactants_g, products_g = batched_data
            score_pred = self.calc_sore(reactants_g, products_g)
            return score_pred

        reactants_g, goldens_g, products_g, products_golden_g = batched_data
        score_pred = self.calc_sore(reactants_g, products_g)
        score_golden = self.calc_sore(goldens_g, products_golden_g)
        return score_golden - score_pred
