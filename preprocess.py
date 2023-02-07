import json
from collections import defaultdict
from multiprocessing import Pool
from os.path import join

import numpy as np
from rdkit import Chem, RDLogger
from tqdm import tqdm
import pandas as pd

from utils.mol_utils import get_atom_count, is_valid_prediction
from utils.smiles_utils import canonicalize_smiles

RDLogger.DisableLog('rdApp.*')


def eval_instance(inputs): # sorting the augmented predictions and calculate the rank
    src_smi, tgt_smi, a_id2preds, tid = inputs
    product_mol = Chem.MolFromSmiles(src_smi)
    product_mol_atom_count = get_atom_count(product_mol)
    pred2score = defaultdict(int)
    highest = {}
    smi2mol = {}

    for a_id, preds in a_id2preds.items():
        existed_set = set()
        for rank, p in enumerate(preds):
            mol = Chem.MolFromSmiles(''.join(p.split(' ')))
            if mol is None:
                continue
            cur_mol_count = get_atom_count(mol)
            if not is_valid_prediction(product_mol_atom_count, cur_mol_count):
                continue
            canno_p = Chem.MolToSmiles(mol, isomericSmiles=True)
            if canno_p in existed_set:
                continue
            smi2mol[canno_p] = mol
            pred2score[canno_p] += 1/(1+rank)
            if canno_p not in highest:
                highest[canno_p] = rank
            else:
                highest[canno_p] = min(highest[canno_p], rank)
            existed_set.add(canno_p)
    for key in highest.keys():
        pred2score[key] += -10e8 * highest[key]
    sorted_preds = sorted(pred2score.items(), key=lambda x: x[1], reverse=True)
    rank = 10000
    for r, (p, score) in enumerate(sorted_preds):
        if p == tgt_smi:
            rank = r
            break
    return sorted_preds, rank, tid

def canonicalize_smiles_clear_map(smiles,return_max_frag=True): # canonicalize the smiles
    # This function is from R-SMILES (https://github.com/otori-bird/retrosynthesis)
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        [atom.ClearProp('molAtomMapNumber') for atom in mol.GetAtoms() if atom.HasProp('molAtomMapNumber')]
        try:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        except:
            if return_max_frag:
                return '',''
            else:
                return ''
        if return_max_frag:
            sub_smi = smi.split(".")
            sub_mol = [Chem.MolFromSmiles(smiles) for smiles in sub_smi]
            sub_mol_size = [(sub_smi[i], len(m.GetAtoms())) for i, m in enumerate(sub_mol) if m is not None]
            if len(sub_mol_size) > 0:
                return smi, canonicalize_smiles_clear_map(sorted(sub_mol_size,key=lambda x:x[1],reverse=True)[0][0],return_max_frag=False)
            else:
                return smi, ''
        else:
            return smi
    else:
        if return_max_frag:
            return '',''
        else:
            return ''

def compute_rank(prediction,raw=False,alpha=1.0): # compute the rank
    # This function is from R-SMILES (https://github.com/otori-bird/retrosynthesis)
    valid_score = [[k for k in range(len(prediction[j]))] for j in range(len(prediction))]
    invalid_rates = [0 for k in range(len(prediction[0]))]
    rank = {}
    max_frag_rank = {}
    highest = {}
    if raw:
        # no test augmentation
        assert len(prediction) == 1
        for j in range(len(prediction)):
            for k in range(len(prediction[j])):
                if prediction[j][k][0] == "":
                    invalid_rates[k] += 1
            # error detection
            prediction[j] = [i for i in prediction[j] if i[0] != ""]
            for k, data in enumerate(prediction[j]):
                rank[data] = 1 / (alpha * k + 1)
    else:

        for j in range(len(prediction)):
            for k in range(len(prediction[j])):
                # predictions[i][j][k] = canonicalize_smiles_clear_map(predictions[i][j][k])
                if prediction[j][k][0] == "":
                    valid_score[j][k] = 10 + 1
                    invalid_rates[k] += 1
            # error detection and deduplication
            de_error = [i[0] for i in sorted(list(zip(prediction[j], valid_score[j])), key=lambda x: x[1]) if i[0][0] != ""]
            prediction[j] = list(set(de_error))
            prediction[j].sort(key=de_error.index)
            for k, data in enumerate(prediction[j]):
                if data in rank:
                    rank[data] += 1 / (alpha * k + 1)
                else:
                    rank[data] = 1 / (alpha * k + 1)
                if data in highest:
                    highest[data] = min(k,highest[data])
                else:
                    highest[data] = k
        for key in rank.keys():
            rank[key] += highest[key] * -1e8
    return rank,invalid_rates

def load_predictions(n_best, total_aug, model_name, datasplit, output_dir, prediction_file=None, post_fix=''):
    if not prediction_file:
        prediction_file = join(output_dir, f'{model_name}/{model_name}_{datasplit}.txt')
    src_list = []
    tgt_list = []
    with open(join(output_dir, '../..', f'src-{datasplit}{post_fix}.txt')) as src_f, open(join(output_dir, '../..', f'tgt-{datasplit}{post_fix}.txt')) as tgt_f:
        src_lines = src_f.readlines()
        tgt_lines = tgt_f.readlines()
        cur_src_list = []
        cur_tgt_list = []
        for src, tgt in tqdm(zip(src_lines, tgt_lines), total=len(src_lines)):
            cur_src_list.append(src.strip())
            cur_tgt_list.append(tgt.strip())
            if len(cur_tgt_list) == total_aug:
                src_list.append(canonicalize_smiles(''.join(cur_src_list[0].split(' '))))
                tgt_list.append(canonicalize_smiles(''.join(cur_tgt_list[0].split(' '))))
                tgt_set_debug = set(cur_tgt_list)
                if len(tgt_set_debug) > 1:
                    tgt_set_debug = set([canonicalize_smiles(''.join(s.split(' ')))
                                        for s in tgt_set_debug])
                assert len(tgt_set_debug) == 1
                cur_tgt_list = []
                cur_src_list = []

    with open(prediction_file) as pred_f:
        pred_lines = pred_f.readlines()
        assert len(pred_lines) == len(tgt_list) * n_best * total_aug
        t_id2a_id2preds = defaultdict(lambda: defaultdict(list))
        for pid, pred in enumerate(pred_lines):
            pred = pred.strip()
            t_id = pid // (n_best * total_aug)
            a_id = (pid % (n_best * total_aug)) // n_best
            t_id2a_id2preds[t_id][a_id].append(pred)

    assert len(tgt_list) == len(t_id2a_id2preds)

    accuracies = np.zeros([len(tgt_list), 50], dtype=np.float32)
    results = []
    with Pool(24) as p:
        for res in tqdm(p.imap(eval_instance, ((src_list[t_id], tgt_list[t_id], a_id2preds, t_id) for t_id, a_id2preds in t_id2a_id2preds.items()))):
            sorted_preds, rank, t_id = res
            tgt_smi = tgt_list[t_id]
            accuracies[t_id, rank:] = 1
            results.append((src_list[t_id], tgt_smi, [ele[0]
                           for ele in sorted_preds]))
    mean_accuracies = np.mean(accuracies, axis=0)
    for n in range(50):
        print(f"Top {n+1} accuracy: {mean_accuracies[n] * 100: .2f} %")

    return results


def process_AT():
    output_dir = 'data/AT/1_preprocess'
    total_aug = 6  # 1 canonicalize + 5 augmentation
    n_best = 10
    
    # this is the test file
    datasplit = 'test'
    results = load_predictions(
        n_best, total_aug, 'AT', datasplit, output_dir)
    json.dump(results, open(join(output_dir, 'AT_test.json'), 'w'))

    # this is the train+dev file
    datasplit = 'train'
    results = load_predictions(
        n_best, total_aug, 'AT', datasplit, output_dir)
    json.dump(results, open(join(output_dir, 'AT_train.json'), 'w'))

    import random
    random.seed(0)
    random.shuffle(results)
    for i in range(8):
        chunked_data = results[i::8]
        print(f'chunk {i} for training: {len(chunked_data)}')
        json.dump(chunked_data, open(join(output_dir, f'AT_{i}.json'), 'w'))

def process_rsmiles():
    output_dir = 'data/R-SMILES/1_preprocess'
    total_aug = 5
    n_best = 10
    traindev = []

    # this is the train file
    datasplit = 'train'
    results = load_predictions(
        n_best, total_aug, 'R-SMILES', datasplit, output_dir)
    traindev += results
    json.dump(results, open(join(output_dir, 'R-SMILES_train.json'), 'w'))


    # this is the val file
    datasplit = 'val'
    results = load_predictions(
        n_best, total_aug, 'R-SMILES', datasplit, output_dir)
    traindev += results
    json.dump(results, open(join(output_dir, 'R-SMILES_val.json'), 'w'))

    # this is the test file
    datasplit = 'test'
    results = load_predictions(
        n_best, total_aug, 'R-SMILES', datasplit, output_dir)
    json.dump(results, open(join(output_dir, 'R-SMILES_test.json'), 'w'))

    import random
    random.seed(0)
    random.shuffle(traindev)
    for i in range(8):
        chunked_data = traindev[i::8]
        print(f'chunk {i} for training: {len(chunked_data)}')
        json.dump(chunked_data, open(join(output_dir, f'R-SMILES_{i}.json'), 'w'))


        
if __name__ == '__main__':
    process_AT()
    process_rsmiles()
