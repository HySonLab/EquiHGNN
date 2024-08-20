import os
import os.path as osp
import pandas as pd
from tqdm import tqdm
import torch
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    # download_url,
    extract_zip,
)

import torch.hub

from rdkit import Chem
from ogb.utils import smiles2graph

from data.utils import smi2hgraph, edge_order, HData

def download_url(url, output_path):
    if not os.path.exists(output_path):
        print(f"Downloading {url} to {output_path}...")
        torch.hub.download_url_to_file(url, output_path, progress=True)
    else:
        print(f"File already exists at {output_path}, skipping download.")

class QM9Base(InMemoryDataset):
    r""" 
    """
    
    raw_url = (
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
        "molnet_publish/qm9.zip"
    )
    raw_url2 = "https://ndownloader.figshare.com/files/3195404"
    
    targets = [
        "gap",
        "homo",
        "lumo",
        # "mu",
        # "alpha",
        # "r2",
        # "zpve",
        # "U0",
        # "U",
        # "H",
        # "G",
        # "Cv",
        # "U0_atom",
        # "U_atom",
        # "H_atom",
        # "G_atom",
        # "A",
        # "B",
        # "C",
    ]

    def __init__(self, root, force_reload=False,
                 transform=None, pre_transform=None, pre_filter=None):
        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload)

        self.data, self.slices = torch.load(self.processed_paths[0])

    def mean(self, target):
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return y[:, target].mean().item()

    def std(self, target):
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return y[:, target].std().item()

    def raw_file_names(self):
        return ["gdb9.sdf", "gdb9.sdf.csv", "uncharacterized.txt"]

    def download(self) -> None:

        file_path = osp.join(self.raw_dir, 'qm9.zip')
        download_url(self.raw_url, file_path)
        extract_zip(file_path, self.raw_dir)
        os.unlink(file_path)

        download_url(self.raw_url2, osp.join(self.raw_dir, "3195404"))
        os.rename(
            osp.join(self.raw_dir, "3195404"),
            osp.join(self.raw_dir, "uncharacterized.txt"),
        )

class QM9HGraph3D(QM9Base):

    __doc__ = QM9Base.__doc__
    
    @property
    def processed_file_names(self) -> str:
        return ["3dhg_data.pt"] 
    
    def compute_3dhgraph(self, sdf_path, csv_path):
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
        df = pd.read_csv(csv_path)
        target = df[self.target].values

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)
            atomic_number = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            z = torch.tensor(atomic_number, dtype=torch.long)
            y = torch.tensor([target[i]], dtype=torch.float)
                
            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            atom_fvs, n_idx, e_idx, bond_fvs = mol2hgraph(mol)
            x = torch.tensor(atom_fvs, dtype=torch.long)
            edge_index0 = torch.tensor(n_idx, dtype=torch.long)
            edge_index1 = torch.tensor(e_idx, dtype=torch.long)
            edge_attr = torch.tensor(bond_fvs, dtype=torch.long)
            n_e = len(edge_index1.unique())
            e_order = torch.tensor(edge_order(e_idx), dtype=torch.long)
            
            data = Data(x=x, z=z, pos=pos, y=y, idx=i,
                        n_e=n_e, 
                        # smi=smi,
                        edge_index0=edge_index0,
                        edge_index1=edge_index1,
                        edge_attr=edge_attr,
                        e_order=e_order)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            data_list.append(data)
        
        return data_list

    def process(self):

        data_list = self.compute_3dhgraph(self.raw_paths[0], self.raw_paths[1])
        torch.save(self.collate(data_list), self.processed_paths[0])


class QM9HGraph(QM9Base):

    __doc__ = QM9Base.__doc__

    @property
    def processed_file_names(self):
        return ['hg_data.pt']

    def compute_hgraph(self, sdf_path, csv_path):
        df = pd.read_csv(csv_path)
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)

        target = df[self.target]
        target = torch.tensor(target.values, dtype=torch.float)

        data_list = []
        # for i, smi in enumerate(tqdm(smiles)):
        for i, mol in enumerate(tqdm(suppl)):
            # smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            # atom_fvs, n_idx, e_idx, bond_fvs = smi2hgraph(smi)
            atom_fvs, n_idx, e_idx, bond_fvs = mol2hgraph(mol)
            x = torch.tensor(atom_fvs, dtype=torch.long)
            edge_index0 = torch.tensor(n_idx, dtype=torch.long)
            edge_index1 = torch.tensor(e_idx, dtype=torch.long)
            edge_attr = torch.tensor(bond_fvs, dtype=torch.long)
            y = target[i].unsqueeze(0)
            n_e = len(edge_index1.unique())
            e_order = torch.tensor(edge_order(e_idx), dtype=torch.long)

            data = HData(x=x, y=y, n_e=n_e, 
                        #  smi=smi,
                         edge_index0=edge_index0,
                         edge_index1=edge_index1,
                         edge_attr=edge_attr,
                         e_order=e_order)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        
        return data_list

    def process(self):

        data_list = self.compute_hgraph(self.raw_paths[0], self.raw_paths[1])
        torch.save(self.collate(data_list), self.processed_paths[0])


