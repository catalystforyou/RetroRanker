import itertools
import json
import math
from collections import defaultdict
from os.path import exists

import numpy as np
import torch
from tqdm import tqdm

from retro_ranker import RetroRanker
from retro_dataloader import build_dataloader
from train_model import predict_score
from utils.common_utils import parse_training_config, get_number_of_total_test_records
from utils.model_utils import build_predict_score_path, load_model

def atomwise_tokenizer(smi, exclusive_tokens = None):
    """
    Tokenize a SMILES molecule at atom-level:
        (1) 'Br' and 'Cl' are two-character tokens
        (2) Symbols with bracket are considered as tokens
    exclusive_tokens: A list of specifical symbols with bracket you want to keep. e.g., ['[C@@H]', '[nH]'].
    Other symbols with bracket will be replaced by '[UNK]'. default is `None`.
    """
    import re
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]

    if exclusive_tokens:
        for i, tok in enumerate(tokens):
            if tok.startswith('['):
                if tok not in exclusive_tokens:
                    tokens[i] = '[UNK]'
    return ' '.join(tokens)

def excluding_filter(score, ratio, retain):
    num = len(score)
    new_rank = np.zeros(num, dtype=np.int32)
    new_rank[np.argsort(score)[:int(num*ratio)]] = 200
    if retain < 1 and retain != 0:
        retain = int(num*retain)
    if retain >= 1:
        retain = min(len(score), retain)
        new_rank[:retain] = [n for n in range(retain)]
    n = retain
    for i in range(n, num):
        if new_rank[i] == 0:
            new_rank[i] = n
            n += 1
    sorted = np.sort(-np.array(score)[new_rank == 200]).tolist()
    new_sort = [sorted.index(score) for score in -np.array(score)[new_rank == 200]]
    
    new_rank[new_rank == 200] = n + np.array(new_sort)
    return new_rank


def S1(raws, dataset, reranking=False, base_acc=None, total = 0):
    if total==0:
        total = get_number_of_total_test_records(dataset)
    accuracy = [0]*200
    max_ratio = []
    for raw in raws:
        if raw['pred'] == raw['tgt']:
            if not reranking:
                accuracy[raw['orig_rank']] += 1
                max_ratio.append(raw['orig_rank'])
            else:
                accuracy[raw['rerank']] += 1
                max_ratio.append(raw['rerank'])
            continue
    if not reranking:
        print('Results before RetroRanker:')
    else:
        print('Results after RetroRanker S1:')
    res = []
    show_topk = 50
    for i in range(show_topk):
        acc = sum(accuracy[:i+1])/total
        if not base_acc:
            print('Top %2d accuracy: %.4f' % (i+1, acc))
        else:
            print('Top %2d accuracy: %.4f, baseline: %.4f, change %.4f' % (i+1, acc, base_acc[i], acc-base_acc[i]))
        res.append(acc)
    return res


def S2(raws, dataset, base_acc=None, total = 0):
    if total==0:
        total = get_number_of_total_test_records(dataset)
    accuracy = [0]*200
    id2preds = defaultdict(list)
    for raw in raws:
        id2preds[raw['list_id']].append(raw)
    for _, preds in id2preds.items():
        preds = sorted(preds, key=lambda ele: (ele['orig_rank']+ele['rerank']))
        for r, pred in enumerate(preds):
            if pred['pred'] == pred['tgt']:
                    accuracy[r] += 1
            continue
    print('Results after RetroRanker S2:')
    res = []
    show_topk = 50
    for i in range(show_topk):
        acc = sum(accuracy[:i+1])/total
        if not base_acc:
            print('Top %2d accuracy: %.4f' % (i+1, acc))
        else:
            print('Top %2d accuracy: %.4f, baseline: %.4f, change %.4f' % (i+1, acc, base_acc[i], acc-base_acc[i]))
        res.append(acc)
    return res



def build_reranked_data(raw_data, scores, rerank_ratio=0.5, retain_topk=5):
    reranked_raw_data = []
    id2preds = defaultdict(list)
    for raw, score in zip(raw_data, scores):
        raw['rank_score'] = score
        id2preds[raw['list_id']].append(raw)
    for _, preds in tqdm(id2preds.items(), desc='build reranked list', mininterval=5):
        preds = sorted(preds, key=lambda ele: ele['orig_rank'])
        max_rank = preds[-1]['orig_rank']
        preds_with_place_holder = []
        cur_pos = 0
        for pos in range(max_rank+1):
            if preds[cur_pos]['orig_rank'] == pos:
                preds_with_place_holder.append(preds[cur_pos])
                cur_pos += 1
            else:
                preds_with_place_holder.append(None)

        cur_scores = []
        for p_id, pred in enumerate(preds_with_place_holder):
            if pred:
                cur_scores.append(pred['rank_score'])
            else:
                for q_id in range(p_id+1, len(preds_with_place_holder)):
                    if preds_with_place_holder[q_id]:
                        cur_scores.append(
                            preds_with_place_holder[q_id]['rank_score'])
                        break
        reranks = excluding_filter(cur_scores, rerank_ratio, retain_topk).tolist()
        for raw, rerank in zip(preds_with_place_holder, reranks):
            if raw:
                raw['rerank'] = rerank
                reranked_raw_data.append(raw)
    return reranked_raw_data

def main(args):
    test_loader, test_set_list = build_dataloader(args, False, args.testset)
    raw_data = list(itertools.chain.from_iterable(
        [test_set.raw for test_set in test_set_list]))
    score_file = build_predict_score_path(args, args.testset)
    if exists(score_file) and not args.regen:
        scores = json.load(open(score_file))
    else:
        model = RetroRanker(node_out_feats=args.out_node_feats)
        model = load_model(model, args, False)
        scores = []
        with torch.no_grad():
            print('calc ranking scores, estimated batches: ',
                  math.ceil(len(test_loader) / float(args.batch_size)))
            for b_id, batch_data in enumerate(test_loader):
                bg, bg_golden, bg_products, bg_products_golden, masks = batch_data
                score = predict_score(
                    model, bg, bg_golden, bg_products, bg_products_golden, masks, args.device)
                scores += (score[:, 1] - score[:, 0]).tolist()
                if (b_id +1) % 100 == 0:
                    print('.')
                else:
                    print('.', end='')
            print('done')

        json.dump(scores, open(score_file, 'w'))

    res = S1(raw_data, dataset=args.testset)
    reranked_raw_data = build_reranked_data(raw_data, scores, rerank_ratio=1, retain_topk=0)
    S2(reranked_raw_data, dataset=args.testset, base_acc=res)
    # S1(reranked_raw_data, dataset=args.testset, base_acc=res, reranking=True)


if __name__ == '__main__':
    args = parse_training_config(add_pred_conf=True)
    args.device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    main(args)
