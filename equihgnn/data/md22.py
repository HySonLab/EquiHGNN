import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from torch_geometric.data import Data, InMemoryDataset, download_url
from tqdm.auto import tqdm

from equihgnn.common.registry import registry
from equihgnn.data.utils import HData, edge_order, mol2hgraph
from equihgnn.data.xyz2mol import xyz2mol


class MD22Base(InMemoryDataset):
    raw_url = "http://www.quantum-machine.org/gdml/data/npz/"

    molecule_files = {
        "Ac_Ala3_NHMe": "md22_Ac-Ala3-NHMe.npz",
        "Docosahexaenoic_acid": "md22_DHA.npz",
        "Stachyose": "md22_stachyose.npz",
        "AT_AT": "md22_AT-AT.npz",
        "AT_AT_CG_CG": "md22_AT-AT-CG-CG.npz",
        "Buckyball_catcher": "md22_buckyball-catcher.npz",
        "Double_walled_nanotube": "md22_dw_nanotube.npz",
    }

    task_id = {
        0: "Ac_Ala3_NHMe",
        1: "Docosahexaenoic_acid",
        2: "Stachyose",
        3: "AT_AT",
        4: "AT_AT_CG_CG",
        5: "Buckyball_catcher",
        6: "Double_walled_nanotube",
    }

    available_molecules = list(molecule_files.keys())

    def __init__(
        self,
        root: str,
        target: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        name = self.task_id.get(target, 0)

        if name not in self.available_molecules:
            raise ValueError(
                f"Unknown dataset target '{name}'. Available: {', '.join(self.available_molecules)}"
            )

        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    def mean(self, target):
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return y[:, target].mean().item()

    def std(self, target):
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return y[:, target].std().item()

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> str:
        return self.molecule_files[self.name]

    def download(self) -> None:
        download_url(self.raw_url + self.raw_file_names, self.raw_dir)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)}, name='{self.name}')"


@registry.register_data("md22_g")
@registry.register_data("md22_g_3d")
class MD22Graph(MD22Base):
    @property
    def processed_file_names(self) -> List[str]:
        return ["g_data.pt"]

    def process(self) -> None:
        raw_data = np.load(self.raw_paths[0])

        z = torch.from_numpy(raw_data["z"]).long()
        positions = torch.from_numpy(raw_data["R"]).float()
        energies = torch.from_numpy(raw_data["E"]).float()
        # forces = torch.from_numpy(raw_data["F"]).float()  # If needed

        # Convert first conformation to RDKit mol for bond extraction
        rdkit_mol = xyz2mol(z.tolist(), positions[0].tolist(), charge=0)[0]

        atom_fvs = [atom_to_feature_vector(atom) for atom in rdkit_mol.GetAtoms()]
        x = torch.tensor(atom_fvs, dtype=torch.long)

        rows, cols, bond_fvs = [], [], []
        for bond in rdkit_mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            rows += [start, end]
            cols += [end, start]
            bond_type = bond_to_feature_vector(bond)[0]
            bond_fvs.append([bond_type])
            bond_fvs.append([bond_type])

        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr = torch.tensor(bond_fvs, dtype=torch.long)

        data_list = []
        for pos, y in tqdm(zip(positions, energies), total=positions.shape[0]):
            y = torch.tensor([y], dtype=torch.float)
            data = Data(
                x=x.clone(),
                z=z.clone(),
                pos=pos,
                edge_index=edge_index.clone(),
                edge_attr=edge_attr.clone(),
                y=y.squeeze(0),
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])


@registry.register_data("md22_hg")
@registry.register_data("md22_hg_3d")
class MD22HGraph(MD22Base):
    @property
    def processed_file_names(self) -> List[str]:
        return ["hg_data.pt"]

    def process(self) -> None:
        it = zip(self.raw_paths, self.processed_paths)
        for raw_path, processed_path in it:
            raw_data = np.load(raw_path)

            z = torch.from_numpy(raw_data["z"]).long()
            positions = torch.from_numpy(raw_data["R"]).float()
            energies = torch.from_numpy(raw_data["E"]).float()
            # forces = torch.from_numpy(raw_data["F"]).float()

            rdkit_mol = xyz2mol(z.tolist(), positions[0].tolist(), charge=0)[0]
            try:
                atom_fvs, n_idx, e_idx, bond_fvs = mol2hgraph(rdkit_mol)
            except Exception as e:
                print(e)
                continue

            x = torch.tensor(atom_fvs, dtype=torch.long)
            edge_index0 = torch.tensor(n_idx, dtype=torch.long)
            edge_index1 = torch.tensor(e_idx, dtype=torch.long)
            edge_attr = torch.tensor(bond_fvs, dtype=torch.long)
            n_e = len(edge_index1.unique())
            e_order_tensor = torch.tensor(edge_order(e_idx), dtype=torch.long)

            data_list = []
            for pos, y in tqdm(zip(positions, energies), total=positions.shape[0]):
                y = torch.tensor([y], dtype=torch.float)

                data = HData(
                    x=x.clone(),
                    y=y.squeeze(0),
                    n_e=n_e,
                    pos=pos,
                    z=z.clone(),
                    edge_index0=edge_index0.clone(),
                    edge_index1=edge_index1.clone(),
                    edge_attr=edge_attr.clone(),
                    e_order=e_order_tensor.clone(),
                )
                data.__num_nodes__ = len(x)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            self.save(data_list, processed_path)
