import argparse
import json
from os.path import join

from rdkit import Chem
from tqdm import tqdm


def load_all_rxns(pred_file): # Loading the predictions
    print(pred_file)
    preds = json.load(open(pred_file))
    processed = []
    all_rxns = []
    for idx, (s, t, pred) in tqdm(enumerate(preds)):
        if len(pred) == 1:
            continue
        t_mol = Chem.MolFromSmiles(t)
        s_mol = Chem.MolFromSmiles(s)
        if not t_mol or not s_mol:
            continue
        t = Chem.MolToSmiles(t_mol)
        s = Chem.MolToSmiles(s_mol)
        if t not in set(pred):
            continue

        all_rxns.append(f'{t}>>{s}')
        for rank, curr_p in enumerate(pred):
            p_mol = Chem.MolFromSmiles(curr_p)
            if not p_mol:
                continue
            curr_p = Chem.MolToSmiles(p_mol)
            if len(curr_p) > 0:
                all_rxns.append(f'{curr_p}>>{s}')
                processed.append({'src': s, 'tgt': t, 'pred': curr_p,
                                  'list_id': idx, 'orig_rank': rank})
    print(len(all_rxns))
    all_rxns = list(set(all_rxns))
    print(len(all_rxns))
    return all_rxns, processed


def add_mapping(all_rxns, chunk_size=10): # Mapping the reactions
    from rxnmapper import RXNMapper
    rxn_mapper = RXNMapper()
    mapped_rxns = {}
    curr_rxns = []
    for rxn in tqdm(all_rxns):
        curr_rxns.append(rxn)
        if len(curr_rxns) == chunk_size:
            try:
                results = rxn_mapper.get_attention_guided_atom_maps(curr_rxns)
                results = [r['mapped_rxn'] for r in results]
                for curr_rxn, result in zip(curr_rxns, results):
                    mapped_rxns[curr_rxn] = result
            except Exception as e:
                print(e)
            curr_rxns = []
    if len(curr_rxns) != 0:
        results = rxn_mapper.get_attention_guided_atom_maps(curr_rxns)
        results = [r['mapped_rxn'] for r in results]
        for curr_rxn, result in zip(curr_rxns, results):
            mapped_rxns[curr_rxn] = result
    return mapped_rxns


# This function is modified from Coley's former work
# https://github.com/yanfeiguan/reactivity_predictions_substitution/blob/master/GNN/graph_utils/mol_graph.py 
def get_reacting_core(rs, p, buffer):
    '''
    use molAtomMapNumber of molecules
    buffer: neighbor to be cosidered as reacting center
    return: atomidx of reacting core
    '''
    def _get_buffer(m, cores, buffer):
        neighbors = set(cores)

        for i in range(buffer):
            neighbors_temp = list(neighbors)
            for c in neighbors_temp:
                neighbors.update([n.GetIdx()
                                 for n in m.GetAtomWithIdx(c).GetNeighbors()])

        neighbors = [m.GetAtomWithIdx(x).GetAtomMapNum() for x in neighbors]

        return neighbors

    def _verify_changes(r_mols, p_mol, core_rs, core_p, discard_rs):
        core_rs = core_rs + discard_rs
        r_mols, p_mol = Chem.AddHs(r_mols), Chem.AddHs(p_mol)
        remove_rs_idx, remove_p_idx = [], []
        for atom in r_mols.GetAtoms():
            if atom.GetIdx() in core_rs or atom.GetAtomMapNum() == 0:
                remove_rs_idx.append(atom.GetIdx())
        for atom in p_mol.GetAtoms():
            if atom.GetIdx() in core_p or atom.GetAtomMapNum() == 0:
                remove_p_idx.append(atom.GetIdx())
        r_mols, p_mol = Chem.RWMol(r_mols), Chem.RWMol(p_mol)
        for idx in sorted(remove_rs_idx, reverse=True):
            r_mols.RemoveAtom(idx)
        for idx in sorted(remove_p_idx, reverse=True):
            p_mol.RemoveAtom(idx)
        return Chem.MolToSmiles(r_mols) == Chem.MolToSmiles(p_mol)

    r_mols = Chem.MolFromSmiles(rs)
    p_mol = Chem.MolFromSmiles(p)

    rs_dict = {a.GetAtomMapNum(): a for a in r_mols.GetAtoms()}
    rs_bond_dict = {'{}-{}'.format(*sorted([b.GetBeginAtom().GetAtomMapNum(),
                                            b.GetEndAtom().GetAtomMapNum()])): b
                    for b in r_mols.GetBonds()}

    p_dict = {a.GetAtomMapNum(): a for a in p_mol.GetAtoms()}
    p_bond_dict = {'{}-{}'.format(*sorted([b.GetBeginAtom().GetAtomMapNum(),
                                          b.GetEndAtom().GetAtomMapNum()])): b
                   for b in p_mol.GetBonds()}

    rs_reactants = []
    for r_smiles in rs.split('.'):
        for a in Chem.MolFromSmiles(r_smiles).GetAtoms():
            if a.GetAtomMapNum() in p_dict:
                rs_reactants.append(r_smiles)
                break
    rs_reactants = '.'.join(rs_reactants)

    core_mapnum = set()
    core_bond = set()
    for a_map in p_dict:

        a_neighbor_in_p = set([a.GetAtomMapNum()
                              for a in p_dict[a_map].GetNeighbors()])
        a_neighbor_in_rs = set([a.GetAtomMapNum()
                               for a in rs_dict[a_map].GetNeighbors()])
        if a_neighbor_in_p != a_neighbor_in_rs:
            core_mapnum.add(a_map)
        else:
            for a_neighbor in a_neighbor_in_p:
                b_in_p = p_mol.GetBondBetweenAtoms(
                    p_dict[a_neighbor].GetIdx(), p_dict[a_map].GetIdx())
                b_in_r = r_mols.GetBondBetweenAtoms(
                    rs_dict[a_neighbor].GetIdx(), rs_dict[a_map].GetIdx())
                if b_in_p.GetBondType() != b_in_r.GetBondType():
                    core_bond.add(b_in_r.GetIdx())
                    core_mapnum.add(a_map)

    for k, v in rs_bond_dict.items():
        if (k not in p_bond_dict.keys() and k != '0-0') or (k.split('_')[0] in core_mapnum and k.split('_')[1] in core_mapnum):
            # the marked bond changes here only contain those between heavy atoms
            core_bond.add(v.GetIdx())

    core_rs = _get_buffer(r_mols, [rs_dict[a].GetIdx()
                          for a in core_mapnum], buffer)
    core_p = _get_buffer(p_mol, [p_dict[a].GetIdx()
                         for a in core_mapnum], buffer)

    fatom_index_rs = \
        {a.GetAtomMapNum(): a.GetIdx() for a in r_mols.GetAtoms()}
    fatom_index_p = \
        {a.GetAtomMapNum(): a.GetIdx() for a in p_mol.GetAtoms()}

    core_rs = [fatom_index_rs[x] for x in core_rs]
    core_p = [fatom_index_p[x] for x in core_p]

    discard_rs = []

    for atom in r_mols.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            discard_rs.append(atom.GetIdx())

    if _verify_changes(r_mols, p_mol, core_rs, core_p, discard_rs):
        return core_rs, discard_rs, list(core_bond), core_p
    else:
        return [], [], [], []


def get_changes(key, mapped_rxns, cache={}):
    # TODO: updating the code to find the changes on bonds
    if key not in mapped_rxns:
        reactant, product = key.split('>')[0], key.split('>')[-1]
        return (reactant, product, [], [], [], [])        

    mapped_sml = mapped_rxns[key]
    if mapped_sml in cache:
        return cache[mapped_sml]
    reactant, product = mapped_sml.split('>')[0], mapped_sml.split('>')[-1]
    try:
        changed_rs, discard_rs, changed_bonds, changed_p = get_reacting_core(
            reactant, product, 0)
    except Exception as e:
        print(e)
        print(mapped_sml)
        changed_rs, discard_rs, changed_bonds, changed_p = [], [], [], []
    cache[mapped_sml] = (reactant, product, changed_rs,
                         changed_bonds, discard_rs, changed_p)
    return cache[mapped_sml]


def get_changes_all(mapped_rxns, toprocess_data):
    failed_cases = 0
    cache = {}
    for process in tqdm(toprocess_data):
        key_pre = process['pred'] + '>>' + process['src']
        rs, p, changed_rs, changed_bonds, discard, changed_p = get_changes(
            key_pre, mapped_rxns, cache=cache)
        if len(changed_rs + changed_bonds + discard + changed_p) == 0:
            failed_cases += 1
        key_golden = process['tgt'] + '>>' + process['src']
        golden_rs, golden_p, changed_rs_golden, changed_bonds_golden, discard_golden, changed_p_golden = get_changes(
            key_golden, mapped_rxns, cache=cache)
        process['product'] = p
        process['golden_product'] = golden_p
        process['rs'] = rs
        process['golden_rs'] = golden_rs
        process['changed_rs'] = changed_rs
        process['changed_bonds'] = changed_bonds
        process['discard'] = discard
        process['changed_p'] = changed_p
        process['changed_rs_golden'] = changed_rs_golden
        process['changed_bonds_golden'] = changed_bonds_golden
        process['discard_golden'] = discard_golden
        process['changed_p_golden'] = changed_p_golden
    print('failed_cases:', failed_cases)


def parse_mapping_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk_id', type=str, default='0')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='AT')
    args = parser.parse_args()
    return args


def process(args):
    data_dir = join(args.data_dir, args.dataset)
    pred_file = join(data_dir, '1_preprocess', f'{args.dataset}_{args.chunk_id}.json')
    all_rxns, toprocess_data = load_all_rxns(pred_file)
    mapped_rxns = add_mapping(all_rxns)
    json.dump(mapped_rxns, open(
        join(data_dir, '2_mapping', f'{args.dataset}_mapped_{args.chunk_id}.json'), 'w'))
    json.dump(toprocess_data, open(
        join(data_dir, '2_mapping', f'{args.dataset}_toprocess_{args.chunk_id}.json'), 'w'))
    # If you have already mapped the reactions, you can directly load the mapped reactions
    '''
    mapped_rxns = json.load(
        open(join(data_dir, f'{args.dataset}_mapped_{args.chunk_id}.json')))
    toprocess_data = json.load(
        open(join(data_dir, f'{args.dataset}_toprocess_{args.chunk_id}.json')))
    '''

    get_changes_all(mapped_rxns, toprocess_data)
    json.dump(toprocess_data, open(
        join(data_dir, '2_mapping', f'{args.dataset}_processed_{args.chunk_id}.json'), 'w'))
    


if __name__ == '__main__':
    args = parse_mapping_args()
    process(args)