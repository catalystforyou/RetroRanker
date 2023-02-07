from rdkit import Chem

def canonicalize_smiles(smiles: str, isomeric=True) -> str:
    """Get SMILES strings in canonical form

    Args:
        smiles (str): a SMILES string of a molecule or reaction
        isomeric (bool, optional): with stereo information. Defaults to True.

    Returns:
        str: a canonical SMILES string of a molecule or reaction
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=isomeric)
    else:
        return None