from collections import defaultdict


def get_atom_count(mol):
    atom_count = defaultdict(int)
    for atom in mol.GetAtoms():
        atom_count[atom.GetSymbol()] += 1
    return atom_count


def is_valid_prediction(product_atom_count, reactant_atom_count):
    for k,v in product_atom_count.items():
        if k not in reactant_atom_count:
            return False
        if reactant_atom_count[k] < v:
            return False
    return True