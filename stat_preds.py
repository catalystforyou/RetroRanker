import json
from multiprocessing import Pool
from os.path import join

from rdkit import Chem
from tqdm import tqdm


def count_atom(smi):
    mol = Chem.MolFromSmiles(smi)  
    if mol:        
        return mol.GetNumAtoms()
    return -1


if __name__ == '__main__':
    root = '/home/leifa/teamdrive2/projects/retrorank/data'
    file_identifier = list(range(8)) + ['test']
    all_smiles = set()
    for file_id in file_identifier:
        for dataset in ['vanilla', 'rsmiles']:
            file_name = join(root, f'{dataset}_{file_id}.json')
            print(f'load {file_name}')
            output = json.load(open(file_name))
            for src, tgt, preds in output:
                all_smiles.add(src)
                all_smiles.add(tgt)
                all_smiles.update(preds)                           

    atoms = 0
    max_atoms = 0
    valid = 0
    with Pool(24) as p:
        for res in tqdm(p.imap(count_atom, all_smiles), total=len(all_smiles)):
            if res>0:
                valid+=1                
                atoms += res
                max_atoms = max(max_atoms, res)
    
    print(max_atoms)
    print(atoms/valid)
    # 973
    # 29.6

    