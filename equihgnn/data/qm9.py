import os
import os.path as osp

import pandas as pd
import torch
import torch.hub
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset, extract_zip
from tqdm import tqdm

from equihgnn.common.registry import registry
from equihgnn.data.utils import HData, edge_order, mol2hgraph


def download_url(url, output_path):
    if not os.path.exists(output_path):
        print(f"Downloading {url} to {output_path}...")
        torch.hub.download_url_to_file(url, output_path, progress=True)
    else:
        print(f"File already exists at {output_path}, skipping download.")


# Reference: https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/qm9.py
class QM9Base(InMemoryDataset):
    r"""The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
    Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
    about 130,000 molecules with 19 regression targets.
    Each molecule includes complete spatial information for the single low
    energy conformation of the atoms in the molecule.
    In addition, we provide the atom features from the `"Neural Message
    Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper.

    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | Target | Property                         | Description                                                                       | Unit                                        |
    +========+==================================+===================================================================================+=============================================+
    | 0      | :math:`\mu`                      | Dipole moment                                                                     | :math:`\textrm{D}`                          |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 1      | :math:`\alpha`                   | Isotropic polarizability                                                          | :math:`{a_0}^3`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 2      | :math:`\epsilon_{\textrm{HOMO}}` | Highest occupied molecular orbital energy                                         | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 3      | :math:`\epsilon_{\textrm{LUMO}}` | Lowest unoccupied molecular orbital energy                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 4      | :math:`\Delta \epsilon`          | Gap between :math:`\epsilon_{\textrm{HOMO}}` and :math:`\epsilon_{\textrm{LUMO}}` | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 5      | :math:`\langle R^2 \rangle`      | Electronic spatial extent                                                         | :math:`{a_0}^2`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 6      | :math:`\textrm{ZPVE}`            | Zero point vibrational energy                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 7      | :math:`U_0`                      | Internal energy at 0K                                                             | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 8      | :math:`U`                        | Internal energy at 298.15K                                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 9      | :math:`H`                        | Enthalpy at 298.15K                                                               | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 10     | :math:`G`                        | Free energy at 298.15K                                                            | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 11     | :math:`c_{\textrm{v}}`           | Heat capavity at 298.15K                                                          | :math:`\frac{\textrm{cal}}{\textrm{mol K}}` |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 12     | :math:`U_0^{\textrm{ATOM}}`      | Atomization energy at 0K                                                          | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 13     | :math:`U^{\textrm{ATOM}}`        | Atomization energy at 298.15K                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 14     | :math:`H^{\textrm{ATOM}}`        | Atomization enthalpy at 298.15K                                                   | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 15     | :math:`G^{\textrm{ATOM}}`        | Atomization free energy at 298.15K                                                | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+

    """

    HAR2EV = 27.211386246
    KCALMOL2EV = 0.04336414

    conversion = torch.tensor(
        [
            1.0,
            1.0,
            HAR2EV,
            HAR2EV,
            HAR2EV,
            1.0,
            HAR2EV,
            HAR2EV,
            HAR2EV,
            HAR2EV,
            HAR2EV,
            1.0,
            KCALMOL2EV,
            KCALMOL2EV,
            KCALMOL2EV,
            KCALMOL2EV,
        ]
    )

    targets = [
        "mu",
        "alpha",
        "homo",
        "lumo",
        "gap",
        "r2",
        "zpve",
        "u0",
        "u298",
        "h298",
        "g298",
        "cv",
        "u0_atom",
        "u298_atom",
        "h298_atom",
        "g298_atom",
    ]

    raw_url = (
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
        "molnet_publish/qm9.zip"
    )
    raw_url2 = "https://ndownloader.figshare.com/files/3195404"

    def __init__(
        self,
        root,
        force_reload=False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        use_ring: bool = False,
    ):
        self.use_ring: bool = use_ring
        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload
        )

        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.use_ring:
            print("Use rings as the hyperedge also")

    def mean(self, target):
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return y[:, target].mean().item()

    def std(self, target):
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return y[:, target].std().item()

    def raw_file_names(self):
        return ["gdb9.sdf", "gdb9.sdf.csv", "uncharacterized.txt"]

    def download(self) -> None:

        file_path = osp.join(self.raw_dir, "qm9.zip")
        download_url(self.raw_url, file_path)
        extract_zip(file_path, self.raw_dir)
        os.unlink(file_path)

        download_url(self.raw_url2, osp.join(self.raw_dir, "3195404"))
        os.rename(
            osp.join(self.raw_dir, "3195404"),
            osp.join(self.raw_dir, "uncharacterized.txt"),
        )

    def extract_targets(self, df):
        targets = df[self.targets].values
        targets = torch.tensor(targets, dtype=torch.float)
        targets = targets * self.conversion.unsqueeze(0)
        return targets


@registry.register_data("qm9_hg_3d")
class QM9HGraph3D(QM9Base):

    __doc__ = QM9Base.__doc__

    @property
    def processed_file_names(self):
        suffix = "_use_ring" if self.use_ring else "_no_ring"
        return [f"3dhg_data_{suffix}.pt"]

    def compute_3dhgraph(self, sdf_path, csv_path):
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
        df = pd.read_csv(csv_path)
        targets = self.extract_targets(df)

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            if mol is None:
                continue

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)
            atomic_number = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            z = torch.tensor(atomic_number, dtype=torch.long)
            y = targets[i].unsqueeze(0)

            try:
                atom_fvs, n_idx, e_idx, bond_fvs = mol2hgraph(
                    mol, use_ring=self.use_ring
                )
            except Exception as e:
                print(e)
                continue

            x = torch.tensor(atom_fvs, dtype=torch.long)
            edge_index0 = torch.tensor(n_idx, dtype=torch.long)
            edge_index1 = torch.tensor(e_idx, dtype=torch.long)
            edge_attr = torch.tensor(bond_fvs, dtype=torch.long)
            n_e = len(edge_index1.unique())
            e_order = torch.tensor(edge_order(e_idx), dtype=torch.long)

            data = HData(
                x=x,
                z=z,
                pos=pos,
                y=y,
                idx=i,
                n_e=n_e,
                edge_index0=edge_index0,
                edge_index1=edge_index1,
                edge_attr=edge_attr,
                e_order=e_order,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        return data_list

    def process(self):

        data_list = self.compute_3dhgraph(self.raw_paths[0], self.raw_paths[1])
        torch.save(self.collate(data_list), self.processed_paths[0])


@registry.register_data("qm9_hg")
class QM9HGraph(QM9Base):

    __doc__ = QM9Base.__doc__

    @property
    def processed_file_names(self):
        suffix = "_use_ring" if self.use_ring else "_no_ring"
        return [f"hg_data_{suffix}.pt"]

    def compute_hgraph(self, sdf_path, csv_path):
        df = pd.read_csv(csv_path)
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
        targets = self.extract_targets(df)

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            atom_fvs, n_idx, e_idx, bond_fvs = mol2hgraph(mol)
            x = torch.tensor(atom_fvs, dtype=torch.long)
            edge_index0 = torch.tensor(n_idx, dtype=torch.long)
            edge_index1 = torch.tensor(e_idx, dtype=torch.long)
            edge_attr = torch.tensor(bond_fvs, dtype=torch.long)
            y = targets[i].unsqueeze(0)
            n_e = len(edge_index1.unique())
            e_order = torch.tensor(edge_order(e_idx), dtype=torch.long)

            data = HData(
                x=x,
                y=y,
                n_e=n_e,
                #  smi=smi,
                edge_index0=edge_index0,
                edge_index1=edge_index1,
                edge_attr=edge_attr,
                e_order=e_order,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        return data_list

    def process(self):

        data_list = self.compute_hgraph(self.raw_paths[0], self.raw_paths[1])
        torch.save(self.collate(data_list), self.processed_paths[0])


@registry.register_data("qm9_g")
class QM9Graph(QM9Base):

    __doc__ = QM9Base.__doc__

    @property
    def processed_file_names(self):
        suffix = "_use_ring" if self.use_ring else "_no_ring"
        return [f"g_data_{suffix}.pt"]

    def compute_graph(self, sdf_path, csv_path):
        df = pd.read_csv(csv_path)
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
        targets = self.extract_targets(df)

        with open(self.raw_paths[2]) as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split("\n")[9:-2]]

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):

            if i in skip:
                continue

            atom_fvs = [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]
            x = torch.tensor(atom_fvs, dtype=torch.long)

            rows, cols, bond_fvs = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                rows += [start, end]
                cols += [end, start]
                bond_type = bond_to_feature_vector(bond)[0]
                bond_fvs.append([bond_type])
                bond_fvs.append([bond_type])

            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            edge_attr = torch.tensor(bond_fvs, dtype=torch.long)

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=targets[i].unsqueeze(0),
                idx=i,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        return data_list

    def process(self):

        data_list = self.compute_graph(self.raw_paths[0], self.raw_paths[1])
        torch.save(self.collate(data_list), self.processed_paths[0])


@registry.register_data("qm9_g_3d")
class QM9Graph3D(QM9Base):

    __doc__ = QM9Base.__doc__

    @property
    def processed_file_names(self):
        suffix = "_use_ring" if self.use_ring else "_no_ring"
        return [f"3dg_data_{suffix}.pt"]

    def compute_graph(self, sdf_path, csv_path):
        df = pd.read_csv(csv_path)
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
        targets = self.extract_targets(df)

        with open(self.raw_paths[2]) as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split("\n")[9:-2]]

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):

            if i in skip:
                continue

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            atom_fvs = [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]
            x = torch.tensor(atom_fvs, dtype=torch.long)

            rows, cols, bond_fvs = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                rows += [start, end]
                cols += [end, start]
                bond_type = bond_to_feature_vector(bond)[0]
                bond_fvs.append([bond_type])
                bond_fvs.append([bond_type])

            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            edge_attr = torch.tensor(bond_fvs, dtype=torch.long)

            data = Data(
                x=x,
                pos=pos,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=targets[i].unsqueeze(0),
                idx=i,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        return data_list

    def process(self):

        data_list = self.compute_graph(self.raw_paths[0], self.raw_paths[1])
        torch.save(self.collate(data_list), self.processed_paths[0])
