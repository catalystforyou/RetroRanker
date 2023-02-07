import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgllife.model.gnn.attentivefp import AttentiveFPGNN
from dgllife.model.gnn.pagtn import PAGTNGNN
from dgllife.model.gnn.wln import WLN
from dgllife.model.readout.attentivefp_readout import AttentiveFPReadout


class LabelSmoothingCrossEntropy(nn.Module):
    """ https://github.com/Guzpenha/transformer_rankers/blob/master/transformer_rankers/models/losses/label_smoothing.py
    """

    def __init__(self, smoothing=0.01):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, inputs, target):
        log_prob = F.log_softmax(inputs, dim=-1)
        weight = inputs.new_ones(inputs.size()) * \
            (self.smoothing / inputs.size(-1))
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing) + 
                        self.smoothing / (inputs.size(-1)))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss.mean()

class RetroEncoder(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, node_out_feats):
        super(RetroEncoder, self).__init__()
        self.input_node = nn.Linear(node_in_feats, node_out_feats)
        self.input_edge = nn.Linear(edge_in_feats, node_out_feats)
        self.input_node_p = nn.Linear(node_in_feats, node_out_feats)
        self.input_edge_p = nn.Linear(edge_in_feats, node_out_feats)
        self.gnn = AttentiveFPGNN(node_feat_size=node_out_feats,
                            edge_feat_size=node_out_feats, graph_feat_size=node_out_feats, num_layers=3, dropout=0.2)
        self.gnn_p = AttentiveFPGNN(node_feat_size=node_out_feats,
                            edge_feat_size=node_out_feats, graph_feat_size=node_out_feats, num_layers=3, dropout=0.2)
        # self.gnn = WLN(node_out_feats, node_out_feats, node_out_feats)
        # self.gnn_p = WLN(node_out_feats, node_out_feats, node_out_feats)
        # self.fuse = nn.Linear(node_out_feats*2, node_out_feats)
        self.readout_before = AttentiveFPReadout(node_out_feats, num_timesteps=3, dropout=0.2)
        self.readout_after = AttentiveFPReadout(node_out_feats, num_timesteps=3, dropout=0.2)
        self.readout_c = AttentiveFPReadout(node_out_feats, num_timesteps=3, dropout=0.2)
        self.readout_d = AttentiveFPReadout(node_out_feats, num_timesteps=3, dropout=0.2)
        self.readout_p = AttentiveFPReadout(node_out_feats, num_timesteps=3, dropout=0.2)
        self.linear =  nn.Sequential(
                                nn.Linear(node_out_feats*6, node_out_feats), 
                                nn.ReLU(), 
                                nn.Dropout(0.2),
                                # nn.Linear(node_out_feats*4, node_out_feats*2), 
                                # nn.ReLU(), 
                                # nn.Dropout(0.2),
                                nn.Linear(node_out_feats, 2))

    def forward(self, g, g_p, mask_node, mask_edge, mask_product):
        # mask_node = mask_node.cuda()
        # g, g_p, mask_node, mask_edge, mask_product = g.to('cuda:0'), g_p.to('cuda:0'), mask_node.to('cuda:0'), mask_edge.to('cuda:0'), mask_product.to('cuda:0')
        node_feats = g.ndata['h']
        edge_feats = g.edata['e']
        atom_feats = self.input_node(node_feats)
        bond_feats = self.input_edge(edge_feats)
        atom_feats = self.gnn(g, atom_feats, bond_feats)
        node_feats_p = g_p.ndata['h']
        edge_feats_p = g_p.edata['e']
        atom_feats_p = self.input_node_p(node_feats_p)
        bond_feats_p = self.input_edge_p(edge_feats_p)
        atom_feats_p = self.gnn_p(g_p, atom_feats_p, bond_feats_p)
        atom_feats_out = self.readout_before(g, atom_feats)
        atom_feats_out_p = self.readout_after(g_p, atom_feats_p)
        bond_feats = torch.mm(torch.diag(mask_edge).cuda(), bond_feats)
        g.edata['e'] = bond_feats
        bond_output = dgl.readout_edges(g, feat='e', op='sum')
        atom_feats_c = torch.mm(torch.diag(mask_node[0]).cuda(), atom_feats)  # node features for changed atoms
        atom_feats_d = torch.mm(torch.diag(mask_node[1]).cuda(), atom_feats)  # node features for changed atoms
        atom_feats_p = torch.mm(torch.diag(mask_product).cuda(), atom_feats_p)
        atom_feats_c = self.readout_c(g, atom_feats_c)
        atom_feats_d = self.readout_d(g, atom_feats_d)
        atom_feats_p = self.readout_p(g_p, atom_feats_p)
        '''g.ndata['h'] = atom_feats_c
        atom_feats_c = dgl.readout_nodes(g, feat='h', op='sum')
        g.ndata['h'] = atom_feats_d
        atom_feats_d = dgl.readout_nodes(g, feat='h', op='sum')
        g_p.ndata['h'] = atom_feats_p
        atom_feats_p = dgl.readout_nodes(g_p, feat='h', op='sum')'''
        g_feats = torch.cat((atom_feats_out, atom_feats_out_p, atom_feats_c, atom_feats_d, atom_feats_p, bond_output), dim=1)
        g_score = self.linear(g_feats)
        return g_score

class RetroRanker(nn.Module):
    def __init__(self, node_in_feats=106, edge_in_feats=13, node_out_feats=512):
        super(RetroRanker, self).__init__()
        self.encoder = RetroEncoder(node_in_feats=node_in_feats, edge_in_feats=edge_in_feats, node_out_feats=node_out_feats)
        # self.loss = nn.MarginRankingLoss(0.5)
        self.loss = LabelSmoothingCrossEntropy(0.01)

    def forward(self, mol1, mol2=None):
        g1, g_p1, mask_n1, mask_e1, mask_p1 = mol1
        pos_score = self.encoder(g1, g_p1, mask_n1, mask_e1, mask_p1)

        if mol2 is None:
            return pos_score

        g2, g_p2, mask_n2, mask_e2, mask_p2 = mol2
        neg_score = self.encoder(g2, g_p2, mask_n2, mask_e2, mask_p2)
        # loss = self.loss(pos_score, neg_score, torch.ones(pos_score.shape).cuda())
        output = pos_score - neg_score
        loss = self.loss(output, torch.ones(output.shape[0], dtype=torch.long, device=mask_p1.device).view(-1))
        return loss, output