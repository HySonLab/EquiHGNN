import numpy as np
import torch_geometric.transforms as T
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
from torch_geometric.data import Data


def compute_ring_features(ring: frozenset[int], molecule: Chem.Mol) -> tuple[float]:
    """https://github.com/NSAPH-Projects/topological-equivariant-networks/blob/main/etnn/qm9/lifts/ring.py"""
    ring_atoms = [molecule.GetAtomWithIdx(idx) for idx in ring]
    ring_size = float(len(ring))
    is_aromatic = float(all(atom.GetIsAromatic() for atom in ring_atoms))
    has_heteroatom = float(
        any(atom.GetSymbol() not in ("C", "H") for atom in ring_atoms)
    )
    is_saturated = float(
        all(
            atom.GetHybridization() == Chem.HybridizationType.SP3 for atom in ring_atoms
        )
    )
    return (ring_size, is_aromatic, has_heteroatom, is_saturated)


def extract_ring_info(mol) -> set[tuple[set, set]]:
    """Indentify rings in a graph"""
    cells = set()
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        node_idc = frozenset(ring)
        feature_vector = compute_ring_features(node_idc, mol)
        cells.add((node_idc, feature_vector))
    return cells


def he_conj(mol):
    """get node index and hyperedge index of conjugated structure in a molecule

    Args:
        mol (RDKit MOL): input molecule

    Returns:
        tuple: node index and hyperedge index
    """
    num_atom = mol.GetNumAtoms()
    reso = Chem.ResonanceMolSupplier(mol)
    num_he = reso.GetNumConjGrps()
    # assert num_he != 0
    n_idx, e_idx = [], []
    for i in range(num_atom):
        _conj = reso.GetAtomConjGrpIdx(i)
        if _conj > -1 and _conj < num_he:
            n_idx.append(i)
            e_idx.append(_conj)
    return n_idx, e_idx


def edge_order(e_idx):
    e_order = []
    for i in range(len(set(e_idx))):
        e_order.append(e_idx.count(i))
    return e_order


def smi2hgraph(smiles_string):
    """
    Converts a SMILES string to hypergraph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string)

    # atoms
    atom_fvs = []
    for atom in mol.GetAtoms():
        atom_fvs.append(atom_to_feature_vector(atom))

    # bonds
    num_bond_features = 1  # bond type (single, double, triple, conjugated)
    if len(mol.GetBonds()) > 0:  # mol has bonds
        n_idx, e_idx, bond_fvs = [], [], []
        for i, bond in enumerate(mol.GetBonds()):
            n_idx.append(bond.GetBeginAtomIdx())
            n_idx.append(bond.GetEndAtomIdx())
            e_idx.append(i)
            e_idx.append(i)
            bond_type = bond_to_feature_vector(bond)[0]
            bond_fvs.append([bond_type])

    else:  # mol has no bonds
        print("Invalid SMILES: {}".format(smiles_string))
        n_idx, e_idx = [], []
        bond_fvs = np.empty((0, num_bond_features), dtype=np.int64)
        return (atom_fvs, n_idx, e_idx, bond_fvs)

    # hyperedges for conjugated bonds
    he_n, he_e = he_conj(mol)
    num_bond = mol.GetNumBonds()
    if len(he_n) != 0:
        he_e = [_id + num_bond for _id in he_e]
        n_idx += he_n
        e_idx += he_e
        bond_fvs += len(set(he_e)) * [num_bond_features * [5]]

    return (atom_fvs, n_idx, e_idx, bond_fvs)


def mol2hgraph(mol):
    """
    Converts an RDKit Mol object to a hypergraph Data object.
    :input: RDKit Mol object
    :return: graph object
    """

    # atoms
    atom_fvs = []
    for atom in mol.GetAtoms():
        atom_fvs.append(atom_to_feature_vector(atom))

    # bonds
    num_bond_features = 1  # bond type (single, double, triple, conjugated)
    if len(mol.GetBonds()) > 0:  # mol has bonds
        n_idx, e_idx, bond_fvs = [], [], []
        for i, bond in enumerate(mol.GetBonds()):
            n_idx.append(bond.GetBeginAtomIdx())
            n_idx.append(bond.GetEndAtomIdx())
            e_idx.append(i)
            e_idx.append(i)
            bond_type = bond_to_feature_vector(bond)[0]
            bond_fvs.append([bond_type])

    else:  # mol has no bonds
        print("Invalid molecule: {}".format(Chem.MolToSmiles(mol)))
        n_idx, e_idx = [], []
        bond_fvs = np.empty((0, num_bond_features), dtype=np.int64)
        return (atom_fvs, n_idx, e_idx, bond_fvs)

    # hyperedges for conjugated bonds
    he_n, he_e = he_conj(mol)
    num_bond = mol.GetNumBonds()
    if len(he_n) != 0:
        he_e = [_id + num_bond for _id in he_e]
        n_idx += he_n
        e_idx += he_e
        bond_fvs += len(set(he_e)) * [num_bond_features * [5]]

    return (atom_fvs, n_idx, e_idx, bond_fvs)


class HData(Data):
    """PyG data class for molecular hypergraphs"""

    def __init__(
        self,
        x=None,
        edge_index=None,
        edge_attr=None,
        y=None,
        pos=None,
        edge_index0=None,
        edge_index1=None,
        n_e=None,
        smi=None,
        **kwargs
    ):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)
        self.edge_index0 = edge_index0
        self.edge_index1 = edge_index1
        self.n_e = n_e
        self.smi = smi

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index0":
            return self.x.size(0)
        if key == "edge_index1":
            return self.n_e
        else:
            return super().__inc__(key, value, *args, **kwargs)


class OneTarget(T.BaseTransform):
    def __init__(self, target=0):
        super().__init__()
        self.target = target

    def __call__(self, data):
        # Specify target.
        data.y = data.y[:, self.target]
        return data


def mol2graph(mol):
    """
    Converts molecule object to graph Data object
    :input: molecule object
    :return: graph object
    """

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = dict()
    graph["edge_index"] = edge_index
    graph["edge_feat"] = edge_attr
    graph["node_feat"] = x
    graph["num_nodes"] = len(x)

    return graph


if __name__ == "__main__":
    # test code
    pass
