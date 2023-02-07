from os.path import exists, join
from pathlib import Path

import dgl
import numpy as np
import torch
from dgllife.utils import EarlyStopping
from torch.optim import Adam, lr_scheduler


def move_to_device(maybe_tensor, device):
    if torch.is_tensor(maybe_tensor) or (hasattr(maybe_tensor, 'to') and callable(getattr(maybe_tensor, 'to'))):
        return maybe_tensor.to(device)
    elif isinstance(maybe_tensor, np.ndarray):
        return torch.from_numpy(maybe_tensor).to(device).contiguous()
    elif isinstance(maybe_tensor, dict):
        return {
            key: move_to_device(value, device)
            for key, value in maybe_tensor.items()
        }
    elif isinstance(maybe_tensor, list):
        return [move_to_device(x, device) for x in maybe_tensor]
    elif isinstance(maybe_tensor, tuple):
        return tuple([move_to_device(x, device) for x in maybe_tensor])
    return maybe_tensor


def gen_masks(raws, graphs, goldens, products):
    mask_node = [[], []]
    mask_edge = []
    mask_node_golden = [[], []]
    mask_edge_golden = []
    mask_node_product = [[], []]
    for raw, graph, golden, product in zip(raws, graphs, goldens, products):
        # print(graph)
        curr_node = np.zeros([2, graph.number_of_nodes()], dtype=np.uint8)
        curr_node[0][raw['changed_rs']] = 1
        mask_node[0] += curr_node[0].tolist()
        curr_node[1][raw['discard']] = 1
        mask_node[1] += curr_node[1].tolist()
        curr_edge = np.zeros(graph.number_of_edges(), dtype=np.uint8)
        curr_edge[raw['changed_bonds']] = 1
        mask_edge += curr_edge.tolist()
        curr_node_golden = np.zeros(
            [2, golden.number_of_nodes()], dtype=np.uint8)
        curr_node_golden[0][raw['changed_rs_golden']] = 1
        mask_node_golden[0] += curr_node_golden[0].tolist()
        curr_node_golden[1][raw['discard_golden']] = 1
        mask_node_golden[1] += curr_node_golden[1].tolist()
        curr_edge_golden = np.zeros(golden.number_of_edges(), dtype=np.uint8)
        curr_edge_golden[raw['changed_bonds_golden']] = 1
        mask_edge_golden += curr_edge_golden.tolist()
        curr_node_product = np.zeros(
            [2, product.number_of_nodes()], dtype=np.uint8)
        curr_node_product[0][raw['changed_p']] = 1
        mask_node_product[0] += curr_node_product[0].tolist()
        curr_node_product[1][raw['changed_p_golden']] = 1
        mask_node_product[1] += curr_node_product[1].tolist()
    return [torch.tensor(np.array(mask_node), dtype=torch.float), torch.tensor(np.array(mask_edge), dtype=torch.float), torch.tensor(np.array(mask_node_golden), dtype=torch.float), torch.tensor(np.array(mask_edge_golden), dtype=torch.float), torch.tensor(np.array(mask_node_product), dtype=torch.float)]


def collate_molgraphs(data):
    raws, graphs, goldens, products, products_golden = data
    masks = gen_masks(raws, graphs, goldens, products)
    # atom_labels, bond_labels = make_labels(graphs, labels, masks)
    bg = dgl.batch(graphs)
    bg_golden = dgl.batch(goldens)
    bg_product = dgl.batch(products)
    bg_product_golden = dgl.batch(products_golden)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    bg_golden.set_n_initializer(dgl.init.zero_initializer)
    bg_golden.set_e_initializer(dgl.init.zero_initializer)
    bg_product.set_n_initializer(dgl.init.zero_initializer)
    bg_product.set_e_initializer(dgl.init.zero_initializer)
    bg_product_golden.set_n_initializer(dgl.init.zero_initializer)
    bg_product_golden.set_e_initializer(dgl.init.zero_initializer)
    return bg, bg_golden, bg_product, bg_product_golden, masks


def build_model_path(args):
    file_path = join(args.root, 'model', args.dataset, args.dataset+'_AF.pt')
    Path(file_path).parent.resolve().mkdir(parents=True, exist_ok=True)
    return file_path


def build_predict_score_path(args, predict_set=None):
    if not predict_set:
        predict_set = args.dataset
    file_path = join(args.root, 'output', f'{args.dataset}_{predict_set}_scores.json')
    Path(file_path).parent.resolve().mkdir(parents=True, exist_ok=True)
    return file_path


def build_ckpt_path(args, epoch):
    file_path = join(args.root, 'model', args.dataset,
                     f'{epoch}_{args.dataset}_AF.pt')
    Path(file_path).parent.resolve().mkdir(parents=True, exist_ok=True)
    return file_path


def load_model(model, args, is_training):
    model = model.to(device=args.device)
    model_path = build_model_path(args)
    if is_training:
        optimizer = Adam(model.parameters(
        ), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=args.schedule_step)

        if exists(model_path):
            user_answer = input('%s exists, want to (a) overlap (b) continue from checkpoint (c) make a new model?' % model_path)
            if user_answer == 'a':
                stopper = EarlyStopping(patience=args.patience, filename=model_path)
                print('Overlap exsited model and training a new model...')
            elif user_answer == 'b':
                stopper = EarlyStopping(patience=args.patience, filename=model_path)
                model.load_state_dict(torch.load(
                    model_path)['model_state_dict'])
                print('Train from exsited model checkpoint...')
        else:
            stopper = EarlyStopping(patience=args.patience, filename=model_path)
        return model, optimizer, scheduler, stopper

    else:
        model.load_state_dict(torch.load(
            model_path)['model_state_dict'])
        model.eval()
        return model
