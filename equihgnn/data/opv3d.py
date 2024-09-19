import os
import os.path as osp

import pandas as pd
import torch
import torch.hub
from ogb.utils import smiles2graph
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset, extract_gz  # download_url,
from tqdm import tqdm

from equihgnn.common.registry import registry
from equihgnn.data.utils import HData, edge_order, mol2hgraph, smi2hgraph


def download_url(url, output_path):
    if not os.path.exists(output_path):
        print(f"Downloading {url} to {output_path}...")
        torch.hub.download_url_to_file(url, output_path, progress=True)
    else:
        print(f"File already exists at {output_path}, skipping download.")


class OPVBase(InMemoryDataset):
    r"""The 3D graph dataset class for the OPV dataset
    from the `"J. Chem. Phys., 2019, 150, 234111." <https://doi.org/10.1063/1.5099132>`_ paper,
    consisting of about 90,823 molecules with 8 regression targets.

    Args:
        root (string): Root directory where the dataset should be saved.
        polymer (bool): whether it's polymeric tasks
        partition (string): which dataset split
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

    # https://data.nrel.gov/submissions/236

    raw_url0 = "https://data.nrel.gov/system/files/236/1712697052-mol_train.csv.gz"
    raw_url1 = "https://data.nrel.gov/system/files/236/1712697052-mol_valid.csv.gz"
    raw_url2 = "https://data.nrel.gov/system/files/236/1712697052-mol_test.csv.gz"

    def __init__(
        self,
        root,
        polymer=False,
        partition="train",
        force_reload=False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        use_ring: bool = False,
    ):
        assert polymer in [True, False]
        self.polymer = polymer
        assert partition in ["train", "valid", "test"]
        self.partition = partition
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

    def extract_sdf_csv(self, df, fname_sdf, fname_csv):
        """Extract data from original CSV file to SDF file (3D coordinates)
        and CSV file (targets)"""
        mol_txt = df["mol"].values.tolist()
        with open(fname_sdf, "w") as f:
            for _txt in mol_txt:
                f.write(_txt)
                f.write("$$$$\n")
        df_target = df[
            [
                "smile",
                "gap",
                "homo",
                "lumo",
                "spectral_overlap",
                "homo_extrapolated",
                "lumo_extrapolated",
                "gap_extrapolated",
                "optical_lumo_extrapolated",
            ]
        ]
        df_target.to_csv(fname_csv)

    def download(self):

        raw_files = [
            "train.sdf",
            "train.csv",
            "polymer_train.sdf",
            "polymer_train.csv",
            "valid.sdf",
            "valid.csv",
            "test.sdf",
            "test.csv",
        ]

        raw_paths = [osp.join(self.raw_dir, raw_file) for raw_file in raw_files]

        print("Downloading OPV train dataset...")
        file_path = osp.join(self.raw_dir, "mol_train.csv.gz")
        download_url(self.raw_url0, file_path)
        extract_gz(file_path, self.raw_dir)
        os.unlink(file_path)
        df_train = pd.read_csv(osp.join(self.raw_dir, "mol_train.csv"))
        self.extract_sdf_csv(df_train, raw_paths[0], raw_paths[1])
        df_train = df_train.dropna(subset=["gap_extrapolated"])
        self.extract_sdf_csv(df_train, raw_paths[2], raw_paths[3])

        print("Downloading OPV valid dataset...")
        file_path = osp.join(self.raw_dir, "mol_valid.csv.gz")
        download_url(self.raw_url1, file_path)
        extract_gz(file_path, self.raw_dir)
        os.unlink(file_path)
        df_valid = pd.read_csv(osp.join(self.raw_dir, "mol_valid.csv"))
        self.extract_sdf_csv(df_valid, raw_paths[4], raw_paths[5])

        print("Downloading OPV test dataset...")
        file_path = osp.join(self.raw_dir, "mol_test.csv.gz")
        download_url(self.raw_url2, file_path)
        extract_gz(file_path, self.raw_dir)
        os.unlink(file_path)
        df_test = pd.read_csv(osp.join(self.raw_dir, "mol_test.csv"))
        self.extract_sdf_csv(df_test, raw_paths[6], raw_paths[7])


@registry.register_data("opv_hg_3d")
class OPVHGraph3D(OPVBase):

    __doc__ = OPVBase.__doc__

    @property
    def raw_file_names(self):
        return [
            "train.sdf",
            "train.csv",
            "polymer_train.sdf",
            "polymer_train.csv",
            "valid.sdf",
            "valid.csv",
            "test.sdf",
            "test.csv",
        ]

    @property
    def processed_file_names(self):
        suffix = "_use_ring" if self.use_ring else "_no_ring"

        if self.partition == "train" and not self.polymer:
            return [f"hg3d_train_{suffix}.pt"]
        elif self.partition == "train" and self.polymer:
            return [f"hg3d_polymer_train_{suffix}.pt"]
        elif self.partition == "valid":
            return [f"hg3d_valid_{suffix}.pt"]
        elif self.partition == "test":
            return [f"hg3d_test_{suffix}.pt"]

    def compute_3dhgraph(self, sdf_path, csv_path):
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
        df = pd.read_csv(csv_path)
        target = df.iloc[:, 2:].values.tolist()

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            if mol is None:
                continue

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)
            atomic_number = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            z = torch.tensor(atomic_number, dtype=torch.long)
            y = torch.tensor([target[i]], dtype=torch.float)

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
        if self.partition == "train" and self.polymer is False:
            data_list = self.compute_3dhgraph(self.raw_paths[0], self.raw_paths[1])
            torch.save(self.collate(data_list), self.processed_paths[0])

        elif self.partition == "train" and self.polymer is True:
            data_list = self.compute_3dhgraph(self.raw_paths[2], self.raw_paths[3])
            torch.save(self.collate(data_list), self.processed_paths[0])

        elif self.partition == "valid":
            data_list = self.compute_3dhgraph(self.raw_paths[4], self.raw_paths[5])
            torch.save(self.collate(data_list), self.processed_paths[0])

        elif self.partition == "test":
            data_list = self.compute_3dhgraph(self.raw_paths[6], self.raw_paths[7])
            torch.save(self.collate(data_list), self.processed_paths[0])


@registry.register_data("opv_g_3d")
class OPVGraph3D(OPVBase):

    __doc__ = OPVBase.__doc__

    @property
    def raw_file_names(self):
        return [
            "train.sdf",
            "train.csv",
            "polymer_train.sdf",
            "polymer_train.csv",
            "valid.sdf",
            "valid.csv",
            "test.sdf",
            "test.csv",
        ]

    @property
    def processed_file_names(self):
        suffix = "_use_ring" if self.use_ring else "_no_ring"

        if self.partition == "train" and not self.polymer:
            return [f"g3d_train_{suffix}.pt"]
        elif self.partition == "train" and self.polymer:
            return [f"g3d_polymer_train_{suffix}.pt"]
        elif self.partition == "valid":
            return [f"g3d_valid_{suffix}.pt"]
        elif self.partition == "test":
            return [f"g3d_test_{suffix}.pt"]

    def compute_3dgraph(self, sdf_path, csv_path):
        # create graph data list from SDF and CSV files
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
        df = pd.read_csv(csv_path)
        target = df.iloc[:, 2:].values.tolist()

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)
            atomic_number = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            z = torch.tensor(atomic_number, dtype=torch.long)
            y = torch.tensor([target[i]], dtype=torch.float)
            data = Data(z=z, pos=pos, y=y, idx=i)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        return data_list

    def process(self):
        if self.partition == "train" and self.polymer is False:
            data_list = self.compute_3dgraph(self.raw_paths[0], self.raw_paths[1])
            torch.save(self.collate(data_list), self.processed_paths[0])

        elif self.partition == "train" and self.polymer is True:
            data_list = self.compute_3dgraph(self.raw_paths[2], self.raw_paths[3])
            torch.save(self.collate(data_list), self.processed_paths[0])

        elif self.partition == "valid":
            data_list = self.compute_3dgraph(self.raw_paths[4], self.raw_paths[5])
            torch.save(self.collate(data_list), self.processed_paths[0])

        elif self.partition == "test":
            data_list = self.compute_3dgraph(self.raw_paths[6], self.raw_paths[7])
            torch.save(self.collate(data_list), self.processed_paths[0])


@registry.register_data("opv_hg")
class OPVHGraph(OPVBase):

    __doc__ = OPVBase.__doc__

    @property
    def raw_file_names(self):
        return ["train.csv", "polymer_train.csv", "valid.csv", "test.csv"]

    @property
    def processed_file_names(self):
        suffix = "_use_ring" if self.use_ring else "_no_ring"

        if self.partition == "train" and not self.polymer:
            return [f"hg_train_{suffix}.pt"]
        elif self.partition == "train" and self.polymer:
            return [f"hg_polymer_train_{suffix}.pt"]
        elif self.partition == "valid":
            return [f"hg_valid_{suffix}.pt"]
        elif self.partition == "test":
            return [f"hg_test_{suffix}.pt"]

    def compute_hgraph(self, csv_path):
        df = pd.read_csv(csv_path)
        smiles = df["smile"].values.tolist()

        target = df.iloc[:, 2:].values
        target = torch.tensor(target, dtype=torch.float)

        data_list = []
        for i, smi in enumerate(tqdm(smiles)):

            atom_fvs, n_idx, e_idx, bond_fvs = smi2hgraph(smi, use_ring=self.use_ring)
            x = torch.tensor(atom_fvs, dtype=torch.long)
            edge_index0 = torch.tensor(n_idx, dtype=torch.long)
            edge_index1 = torch.tensor(e_idx, dtype=torch.long)
            edge_attr = torch.tensor(bond_fvs, dtype=torch.long)
            y = target[i].unsqueeze(0)
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

        if self.partition == "train" and self.polymer is False:
            data_list = self.compute_hgraph(self.raw_paths[0])
            torch.save(self.collate(data_list), self.processed_paths[0])

        elif self.partition == "train" and self.polymer is True:
            data_list = self.compute_hgraph(self.raw_paths[1])
            torch.save(self.collate(data_list), self.processed_paths[0])

        elif self.partition == "valid":
            data_list = self.compute_hgraph(self.raw_paths[2])
            torch.save(self.collate(data_list), self.processed_paths[0])

        elif self.partition == "test":
            data_list = self.compute_hgraph(self.raw_paths[3])
            torch.save(self.collate(data_list), self.processed_paths[0])


@registry.register_data("opv_g")
class OPVGraph(OPVBase):

    __doc__ = OPVBase.__doc__

    @property
    def raw_file_names(self):
        return ["train.csv", "polymer_train.csv", "valid.csv", "test.csv"]

    @property
    def processed_file_names(self):
        suffix = "_use_ring" if self.use_ring else "_no_ring"

        if self.partition == "train" and not self.polymer:
            return [f"g_train_{suffix}.pt"]
        elif self.partition == "train" and self.polymer:
            return [f"g_polymer_train_{suffix}.pt"]
        elif self.partition == "valid":
            return [f"g_valid_{suffix}.pt"]
        elif self.partition == "test":
            return [f"g_test_{suffix}.pt"]

    def compute_graph(self, csv_path):
        df = pd.read_csv(csv_path)
        smiles = df["smile"].values.tolist()
        target = df.iloc[:, 2:].values.tolist()

        # Convert SMILES into graph data
        data_list = []
        for i, smi in enumerate(tqdm(smiles)):

            # get graph data from SMILES
            graph = smiles2graph(smi)
            x = torch.tensor(graph["node_feat"], dtype=torch.long)
            edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
            edge_attr = torch.tensor(graph["edge_feat"], dtype=torch.long)
            y = torch.tensor([target[i]], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        return data_list

    def process(self):

        if self.partition == "train" and self.polymer is False:
            data_list = self.compute_graph(self.raw_paths[0])
            torch.save(self.collate(data_list), self.processed_paths[0])

        elif self.partition == "train" and self.polymer is True:
            data_list = self.compute_graph(self.raw_paths[1])
            torch.save(self.collate(data_list), self.processed_paths[0])

        elif self.partition == "valid":
            data_list = self.compute_graph(self.raw_paths[2])
            torch.save(self.collate(data_list), self.processed_paths[0])

        elif self.partition == "test":
            data_list = self.compute_graph(self.raw_paths[3])
            torch.save(self.collate(data_list), self.processed_paths[0])
