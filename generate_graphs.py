import json
import os
import pickle
from functools import partial
from multiprocessing import Pool
from os.path import join

import lmdb
import numpy as np

import dgl.backend as F
import torch
from dgl.data.utils import save_graphs
from dgllife.utils import CanonicalBondFeaturizer, WeaveAtomFeaturizer
from dgllife.utils.featurizers import (ConcatFeaturizer, WeaveAtomFeaturizer,
                                       atom_chiral_tag_one_hot,
                                       atom_degree_one_hot,
                                       atom_explicit_valence_one_hot,
                                       atom_formal_charge_one_hot,
                                       atom_hybridization_one_hot,
                                       atom_implicit_valence_one_hot,
                                       atom_is_aromatic,
                                       atom_total_num_H_one_hot,
                                       atom_type_one_hot, one_hot_encoding)
from dgllife.utils.mol_to_graph import construct_bigraph_from_mol
from pympler import asizeof
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, rdmolfiles, rdmolops
from tqdm import tqdm

from graphormer_utils import preprocess_dgl_graph_simple
from utils.common_utils import parse_config


class RetroAtomFeaturizer(WeaveAtomFeaturizer):
    def __init__(self, atom_data_field='h', atom_types=None, chiral_types=None,
                 hybridization_types=None):
        super(RetroAtomFeaturizer, self).__init__()

        self._atom_data_field = atom_data_field

        if atom_types is None:
            atom_types = ['H', 'B', 'C', 'Si', 'N', 'O',
                          'F', 'As', 'Se', 'P', 'S', 'Cl', 'Br', 'I']
        self._atom_types = atom_types

        if chiral_types is None:
            chiral_types = [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW]
        self._chiral_types = chiral_types

        if hybridization_types is None:
            hybridization_types = [Chem.rdchem.HybridizationType.SP,
                                   Chem.rdchem.HybridizationType.SP2,
                                   Chem.rdchem.HybridizationType.SP3,
                                   Chem.rdchem.HybridizationType.SP3D,
                                   Chem.rdchem.HybridizationType.SP3D2]
        self._hybridization_types = hybridization_types

        self._featurizer = ConcatFeaturizer([
            partial(atom_type_one_hot, allowable_set=atom_types,
                    encode_unknown=True),
            partial(atom_chiral_tag_one_hot, allowable_set=chiral_types),
            partial(atom_degree_one_hot, allowable_set=list(range(6))),
            partial(atom_explicit_valence_one_hot, allowable_set=list(
                range(5)), encode_unknown=False),
            partial(atom_implicit_valence_one_hot, allowable_set=list(
                range(5)), encode_unknown=False),
            atom_formal_charge_one_hot, atom_is_aromatic, atom_total_num_H_one_hot,
            partial(atom_hybridization_one_hot,
                    allowable_set=hybridization_types)
        ])

    def __call__(self, mol, change, discard):
        """Featurizes the input molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            Mapping atom_data_field as specified in the input argument to the atom
            features, which is a float32 tensor of shape (N, M), N is the number of
            atoms and M is the feature size.
        """
        atom_features = []
        num_atoms = mol.GetNumAtoms()

        # Get information for donor and acceptor
        fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        mol_featurizer = ChemicalFeatures.BuildFeatureFactory(fdef_name)
        mol_feats = mol_featurizer.GetFeaturesForMol(mol)
        is_donor, is_acceptor = self.get_donor_acceptor_info(mol_feats)
        sml = Chem.MolToSmiles(mol)
        dscp_num = [0, 0, 0]
        for s in sml.split('.'):
            if len(Chem.MolFromSmiles(s).GetAtoms()) < 0.5 * num_atoms:
                dscp_num[0] += 1
            elif len(Chem.MolFromSmiles(s).GetAtoms()) < num_atoms:
                dscp_num[1] += 1
            else:
                dscp_num[2] += 1
        dscp_num_ohe = one_hot_encoding(dscp_num[0], [0, 1, 2, 3], encode_unknown=False) + one_hot_encoding(
            dscp_num[1], [0, 1, 2, 3], encode_unknown=False) + one_hot_encoding(dscp_num[2], [0, 1], encode_unknown=False)
        # Get a symmetrized smallest set of smallest rings
        # Following the practice from Chainer Chemistry (https://github.com/chainer/
        # chainer-chemistry/blob/da2507b38f903a8ee333e487d422ba6dcec49b05/chainer_chemistry/
        # dataset/preprocessors/weavenet_preprocessor.py)
        sssr = Chem.GetSymmSSSR(mol)

        for i in range(num_atoms):
            change_bond_dscp = [0, 0, 0, 0, 0]
            atom = mol.GetAtomWithIdx(i)
            neighbors = [i.GetIdx() for i in atom.GetNeighbors()]
            changed_neighbors = len(set(neighbors).intersection(set(change)))
            discard_neighbors = len(set(neighbors).intersection(set(discard)))
            if atom.GetIdx() in change and changed_neighbors + discard_neighbors > 0:
                for n in set(neighbors).intersection(set(change+discard)):
                    carbon_count = sum(
                        [mol.GetAtomWithIdx(n).GetSymbol() == 'C', atom.GetSymbol() == 'C'])
                    change_bond_dscp[carbon_count] += 1
                    if mol.GetBondBetweenAtoms(i, n).GetBondType() == Chem.rdchem.BondType.DOUBLE:
                        change_bond_dscp[3] += 1
                    elif mol.GetBondBetweenAtoms(i, n).GetBondType() == Chem.rdchem.BondType.AROMATIC:
                        change_bond_dscp[4] += 1
            elif atom.GetIdx() in discard and changed_neighbors > 0:
                for n in set(neighbors).intersection(set(change)):
                    carbon_count = sum(
                        [mol.GetAtomWithIdx(n).GetSymbol() == 'C', atom.GetSymbol() == 'C'])
                    change_bond_dscp[carbon_count] += 1
                    if mol.GetBondBetweenAtoms(i, n).GetBondType() == Chem.rdchem.BondType.DOUBLE:
                        change_bond_dscp[3] += 1
                    elif mol.GetBondBetweenAtoms(i, n).GetBondType() == Chem.rdchem.BondType.AROMATIC:
                        change_bond_dscp[4] += 1
            bond_ohe = []
            for num in change_bond_dscp:
                bond_ohe += one_hot_encoding(num,
                                             [0, 1, 2], encode_unknown=False)
            # Features that can be computed directly from RDKit atom instances, which is a list
            feats = self._featurizer(atom)
            # Donor/acceptor indicator
            feats.append(float(is_donor[i]))
            feats.append(float(is_acceptor[i]))
            # Count the number of rings the atom belongs to for ring size between 3 and 8
            count = [0 for _ in range(3, 9)]
            for ring in sssr:
                ring_size = len(ring)
                if i in ring and 3 <= ring_size <= 8:
                    count[ring_size - 3] += 1
            feats.extend(count)
            feats.extend([atom.GetIdx() in change, atom.GetIdx() in discard])
            feats.extend(one_hot_encoding(
                len(change), allowable_set=list(range(4)), encode_unknown=True))
            feats.extend(one_hot_encoding(
                len(discard), allowable_set=list(range(6)), encode_unknown=True))
            feats.extend(one_hot_encoding(changed_neighbors,
                         allowable_set=list(range(4)), encode_unknown=True))
            feats.extend(one_hot_encoding(discard_neighbors,
                         allowable_set=list(range(4)), encode_unknown=True))
            feats.extend(dscp_num_ohe)
            feats.extend(bond_ohe)
            atom_features.append(feats)
        atom_features = np.stack(atom_features)

        return {self._atom_data_field: F.zerocopy_from_numpy(atom_features.astype(np.float32))}


def gen_molgraph(sml, change, discard):
    # modified from https://github.com/kaist-amsg/LocalRetro
    node_featurizer = RetroAtomFeaturizer()  # atom_types = elem_list)
    edge_featurizer = CanonicalBondFeaturizer(self_loop=True)

    def mol_to_graph(mol, change, discard, graph_constructor, node_featurizer, edge_featurizer,
                     canonical_atom_order, explicit_hydrogens=False, num_virtual_nodes=0):
        if mol is None:
            print('Invalid mol found')
            return None

        # Whether to have hydrogen atoms as explicit nodes
        if explicit_hydrogens:
            mol = Chem.AddHs(mol)

        if canonical_atom_order:
            new_order = rdmolfiles.CanonicalRankAtoms(mol)
            mol = rdmolops.RenumberAtoms(mol, new_order)
        g = graph_constructor(mol)

        if node_featurizer is not None:
            g.ndata.update(node_featurizer(mol, change, discard))

        if edge_featurizer is not None:
            g.edata.update(edge_featurizer(mol))

        if num_virtual_nodes > 0:
            num_real_nodes = g.num_nodes()
            real_nodes = list(range(num_real_nodes))
            g.add_nodes(num_virtual_nodes)

            # Change Topology
            virtual_src = []
            virtual_dst = []
            for count in range(num_virtual_nodes):
                virtual_node = num_real_nodes + count
                virtual_node_copy = [virtual_node] * num_real_nodes
                virtual_src.extend(real_nodes)
                virtual_src.extend(virtual_node_copy)
                virtual_dst.extend(virtual_node_copy)
                virtual_dst.extend(real_nodes)
            g.add_edges(virtual_src, virtual_dst)

            for nk, nv in g.ndata.items():
                nv = torch.cat([nv, torch.zeros(g.num_nodes(), 1)], dim=1)
                nv[-num_virtual_nodes:, -1] = 1
                g.ndata[nk] = nv

            for ek, ev in g.edata.items():
                ev = torch.cat([ev, torch.zeros(g.num_edges(), 1)], dim=1)
                ev[-num_virtual_nodes * num_real_nodes * 2:, -1] = 1
                g.edata[ek] = ev

        return g

    def mol_to_bigraph(mol, change, discard, add_self_loop=True,
                       node_featurizer=node_featurizer,
                       edge_featurizer=edge_featurizer,
                       canonical_atom_order=False,
                       explicit_hydrogens=False,
                       num_virtual_nodes=0):
        return mol_to_graph(mol, change, discard, partial(construct_bigraph_from_mol, add_self_loop=add_self_loop),
                            node_featurizer, edge_featurizer,
                            canonical_atom_order, explicit_hydrogens, num_virtual_nodes)

    mol = Chem.MolFromSmiles(sml)
    return mol_to_bigraph(mol, change, discard)


def work(inputs): # multi-process
    rxn, is_test, save_type = inputs
    if rxn['pred'] == rxn['tgt'] and not is_test:
        return None
    reactants_graph = gen_molgraph(
        rxn['rs'], rxn['changed_rs'], rxn['discard'])
    golden_graph = gen_molgraph(
        rxn['golden_rs'], rxn['changed_rs_golden'], rxn['discard_golden'])
    product_graph = gen_molgraph(rxn['product'], rxn['changed_p'], [])
    product_golden_graph = gen_molgraph(
        rxn['golden_product'], rxn['changed_p_golden'], [])

    if save_type == 'pyg':
        if reactants_graph:
            reactants_graph = preprocess_dgl_graph_simple(
                reactants_graph)
        if golden_graph:
            golden_graph = preprocess_dgl_graph_simple(
                golden_graph)
        if product_graph:
            product_graph = preprocess_dgl_graph_simple(
                product_graph)
        if product_golden_graph:
            product_golden_graph = preprocess_dgl_graph_simple(
                product_golden_graph)

    return pickle.dumps((reactants_graph, golden_graph, product_graph, product_golden_graph, rxn))


def get_chunked_data(args):
    file_identifier = args.file_identifier
    cur_chunks_id = args.chunk_id
    input_dir = join('data', args.dataset, '2_mapping')
    file_path = join(
        input_dir, f'{args.dataset}_processed_{file_identifier}.json')
    data = json.load(open(file_path))
    print(file_path + ': ', len(data))
    chunked_data = data[cur_chunks_id::args.total_chunks]
    print(f'chunked: {cur_chunks_id}', len(chunked_data))
    return chunked_data


def main(args):
    chunk_id = args.chunk_id
    file_identifier = args.file_identifier
    is_test = file_identifier == 'test'
    output_dir = join('data', args.dataset, '3_gengraph')
    save_type = args.save_type
    print(f'is_test: {is_test}')
    data = get_chunked_data(args)
    reactants = []
    goldens = []
    products = []
    products_golden = []
    rxns = []
    num_graph_invalid = 0
    with Pool(args.nprocessors) as p:
        for res in tqdm(p.imap(work, ((item, is_test, save_type) for item in data)), total=len(data), desc='Build Graph', mininterval=20):
            if not res:
                continue
            reactants_graph, golden_graph, product_graph, products_golden_graph, rxn = pickle.loads(
                res)
            if reactants_graph and golden_graph and product_graph and products_golden_graph:
                reactants.append(reactants_graph)
                goldens.append(golden_graph)
                products.append(product_graph)
                products_golden.append(products_golden_graph)
                rxns.append(rxn)
            else:
                num_graph_invalid += 1
    print(f'num_graph_invalid: {num_graph_invalid}')
    print(f'collected: {len(rxns)}')
    if save_type == 'pyg':
        to_save = []
        for item_id, item in tqdm(enumerate(zip(reactants, goldens, products, products_golden)), total=len(reactants), mininterval=10):
            item = pickle.dumps(item)
            to_save.append((f"{item_id}".encode(), item))
        print('build lmdb file')
        map_size = int(asizeof.asizeof(to_save) * 1.2)
        env = lmdb.Environment(join(output_dir, f'{chunk_id}_{file_identifier}.db'), map_size=map_size)
        with env.begin(write=True) as txn:
            cursor = txn.cursor()
            cursor.putmulti(to_save)
        env.close()
        json.dump(rxns, open(
            join(output_dir, f'rxns_{chunk_id}_{file_identifier}.pt.json'), 'w'))
    else:
        save_graphs(
            join(output_dir, f'reactants_{chunk_id}_{file_identifier}.bin'), reactants)
        save_graphs(
            join(output_dir, f'goldens_{chunk_id}_{file_identifier}.bin'), goldens)
        save_graphs(
            join(output_dir, f'products_{chunk_id}_{file_identifier}.bin'), products)
        save_graphs(
            join(output_dir, f'products_golden_{chunk_id}_{file_identifier}.bin'), products_golden)
        json.dump(rxns, open(
            join(output_dir, f'rxns_{chunk_id}_{file_identifier}.json'), 'w'))

#  python generate_graphs.py --dataset AT
#           --total_chunks 10 --chunk_id 1 --file_identifier 0

if __name__ == "__main__":
    args = parse_config()
    main(args)
