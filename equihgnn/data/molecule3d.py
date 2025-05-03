import os
import os.path as osp
import shutil

import gdown
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset, extract_zip
from tqdm import tqdm

from equihgnn.common.registry import registry
from equihgnn.data.utils import HData, edge_order, mol2graph, mol2hgraph


class MoleculeBase(InMemoryDataset):
    """
    A `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ data interface for
    datasets used in molecule generation.

    .. note::
        Some datasets may not come with any node labels, like :obj:`moses`.
        Since they don't have any properties in the original data file. The process of the
        dataset can only save the current input property and will load the same  property
        label when the processed dataset is used. You can change the augment :obj:`processed_filename`
        to re-process the dataset with intended property.

    Args:
        root (string, optional): Root directory where the dataset should be saved. (default: :obj:`./`)
        split (string, optional): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
        split_mode (string, optional): Mode of split chosen from :obj:`"random"` and :obj:`"scaffold"`.
            (default: :obj:`penalized_logp`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    targets = [
        "dipole x",
        "dipole y",
        "dipole z",
        "homo",
        "lumo",
        "homolumogap",
        "scf energy",
    ]

    gdrive_id = "1y-EyoDYMvWZwClc2uvXrM4_hQBtM85BI"
    dir_name = "molecule3d"

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):

        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [
            "combined_mols_0_to_1000000.sdf",
            "combined_mols_1000000_to_2000000.sdf",
            "combined_mols_2000000_to_3000000.sdf",
            "combined_mols_3000000_to_3899647.sdf",
            "properties.csv",
            "random_split_inds.json",
            "scaffold_split_inds.json",
            "random_test_split_inds.json",
            "scaffold_test_split_inds.json",
        ]

    def download(self):
        print("Downloading from Google Drive")

        file_path = osp.join(self.raw_dir, "data.zip")
        gdown.download_folder(id=self.gdrive_id, output=self.raw_dir)
        extract_zip(file_path, self.root)
        os.unlink(file_path)

        unziped_folder = osp.join(self.root, "data", "raw")

        print(f"Move data from {unziped_folder} to {self.raw_dir}")

        for item in os.listdir(unziped_folder):
            source_item = osp.join(unziped_folder, item)
            destination_item = osp.join(self.raw_dir, item)
            shutil.move(source_item, destination_item)

        shutil.rmtree(osp.dirname(unziped_folder))

        macosxfile = osp.join(self.root, "__MACOSX")
        if osp.exists(macosxfile):
            shutil.rmtree(macosxfile)

    def mean(self, target):
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return y[:, target].mean().item()

    def std(self, target):
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return y[:, target].std().item()

    def extract_targets(self, df):
        targets = df[self.targets].values
        targets = torch.tensor(targets, dtype=torch.float)
        return targets


@registry.register_data("molecule_g")
@registry.register_data("molecule_g_3d")
class MoleculeGraph(MoleculeBase):

    __doc__ = MoleculeBase.__doc__

    @property
    def processed_file_names(self):
        return ["g_data.pt"]

    def compute_graph(self):
        target_df = pd.read_csv(osp.join(self.raw_dir, "properties.csv"))
        targets = self.extract_targets(target_df)
        data_list = []
        sdf_paths = [
            osp.join(self.raw_dir, "combined_mols_0_to_1000000.sdf"),
            osp.join(self.raw_dir, "combined_mols_1000000_to_2000000.sdf"),
            osp.join(self.raw_dir, "combined_mols_2000000_to_3000000.sdf"),
            osp.join(self.raw_dir, "combined_mols_3000000_to_3899647.sdf"),
        ]
        suppl_list = [
            Chem.SDMolSupplier(p, removeHs=False, sanitize=True) for p in sdf_paths
        ]

        abs_idx = -1
        for i, suppl in enumerate(suppl_list):
            for j in tqdm(range(len(suppl)), desc=f"{i+1}/{len(sdf_paths)}"):
                abs_idx += 1
                mol = suppl[j]

                if mol is None:
                    continue

                # smiles = Chem.MolToSmiles(mol)
                coords = mol.GetConformer().GetPositions()
                z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

                try:
                    graph = mol2graph(mol)
                except Exception as e:
                    print(e)
                    continue

                data = Data()
                data.__num_nodes__ = int(graph["num_nodes"])
                data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                data.z = z
                data.y = targets[abs_idx].unsqueeze(0)
                data.pos = torch.tensor(coords, dtype=torch.float32)
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        return data_list

    def process(self):
        data_list = self.compute_graph()
        torch.save(self.collate(data_list), self.processed_paths[0])


@registry.register_data("molecule_hg")
@registry.register_data("molecule_hg_3d")
class MoleculeHGraph(MoleculeBase):

    __doc__ = MoleculeBase.__doc__

    @property
    def processed_file_names(self):
        return ["hg_data.pt"]

    def process(self):
        target_df = pd.read_csv(osp.join(self.raw_dir, "properties.csv"))
        targets = self.extract_targets(target_df)
        data_list = []
        sdf_paths = [
            osp.join(self.raw_dir, "combined_mols_0_to_1000000.sdf"),
            osp.join(self.raw_dir, "combined_mols_1000000_to_2000000.sdf"),
            osp.join(self.raw_dir, "combined_mols_2000000_to_3000000.sdf"),
            osp.join(self.raw_dir, "combined_mols_3000000_to_3899647.sdf"),
        ]
        suppl_list = [
            Chem.SDMolSupplier(p, removeHs=False, sanitize=True) for p in sdf_paths
        ]

        abs_idx = -1
        for i, suppl in enumerate(suppl_list):
            for j in tqdm(range(len(suppl)), desc=f"{i+1}/{len(sdf_paths)}"):
                abs_idx += 1
                mol = suppl[j]

                if mol is None:
                    continue

                # smiles = Chem.MolToSmiles(mol)
                coords = mol.GetConformer().GetPositions()
                z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

                try:
                    atom_fvs, n_idx, e_idx, bond_fvs = mol2hgraph(mol)
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
                    pos=torch.tensor(coords, dtype=torch.float32),
                    y=targets[abs_idx].unsqueeze(0),
                    idx=i,
                    n_e=n_e,
                    edge_index0=edge_index0,
                    edge_index1=edge_index1,
                    edge_attr=edge_attr,
                    e_order=e_order,
                )

                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])
