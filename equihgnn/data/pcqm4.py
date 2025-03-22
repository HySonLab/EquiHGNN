import os
import os.path as osp
import shutil

import pandas as pd
import torch
from ogb.utils.url import download_url, extract_zip
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset, extract_tar
from tqdm import tqdm

from equihgnn.common.registry import registry
from equihgnn.data.utils import HData, edge_order, mol2graph, mol2hgraph


class PCQM4Mv2Base(InMemoryDataset):
    # Old url hosted at Stanford
    # md5sum: 65b742bafca5670be4497499db7d361b
    # self.url = f'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2.zip'
    # New url hosted by DGL team at AWS--much faster to download
    url = "https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip"
    url_3d = "http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz"

    def __init__(self, root="dataset", transform=None, pre_transform=None):
        """
        The molecular hypergraph dataset class for PCQM4Mv2 dataset
        <https://ogb.stanford.edu/docs/lsc/pcqm4mv2/>
            - root (str): the dataset folder will be located at root/pcqm4m-v2
            - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                * The default smiles2hgraph requires rdkit to be installed
        """
        super(PCQM4Mv2Base, self).__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["data.csv.gz", "pcqm4m-v2-train.sdf"]

    def download(self):
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)

        for item in os.listdir(osp.join(self.root, "pcqm4m-v2", "raw")):
            source_item = osp.join(self.root, "pcqm4m-v2", "raw", item)
            destination_item = osp.join(self.raw_dir, item)
            shutil.move(source_item, destination_item)

        shutil.rmtree(osp.join(self.root, "pcqm4m-v2"))

        path = download_url(self.url_3d, self.raw_dir)
        extract_tar(path, self.raw_dir)
        os.unlink(path)


@registry.register_data("pcqm_hg")
@registry.register_data("pcqm_hg_3d")
class PCQM4Mv2HGraph(PCQM4Mv2Base):
    __doc__ == PCQM4Mv2Base.__doc__

    @property
    def processed_file_names(self):
        return ["hg_data.pt"]

    def process(self):
        data_df = pd.read_csv(self.raw_paths[0])
        suppl = Chem.SDMolSupplier(self.raw_paths[1], removeHs=False, sanitize=False)

        smiles_list = data_df["smiles"]
        homolumogap_list = data_df["homolumogap"]

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            smiles = smiles_list[i]
            homolumogap = homolumogap_list[i]

            try:
                atom_fvs, n_idx, e_idx, bond_fvs = mol2hgraph(mol)
            except Exception as e:
                print(e)
                continue

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)
            x = torch.tensor(atom_fvs, dtype=torch.long)
            edge_index0 = torch.tensor(n_idx, dtype=torch.long)
            edge_index1 = torch.tensor(e_idx, dtype=torch.long)
            edge_attr = torch.tensor(bond_fvs, dtype=torch.long)
            y = torch.tensor([homolumogap], dtype=torch.float)
            n_e = len(edge_index1.unique())
            e_order = torch.tensor(edge_order(e_idx), dtype=torch.long)

            data = HData(
                x=x,
                y=y,
                n_e=n_e,
                smi=smiles,
                pos=pos,
                edge_index0=edge_index0,
                edge_index1=edge_index1,
                edge_attr=edge_attr,
                e_order=e_order,
            )
            data.__num_nodes__ = len(x)

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print("Saving...")
        torch.save(self.collate(data_list), self.processed_paths[0])


@registry.register_data("pcqm_g")
@registry.register_data("pcqm_g_3d")
class PCQM4Mv2Graph(PCQM4Mv2Base):

    __doc__ == PCQM4Mv2Base.__doc__

    @property
    def processed_file_names(self):
        return ["g_data.pt"]

    def process(self):
        data_df = pd.read_csv(self.raw_paths[0])
        suppl = Chem.SDMolSupplier(self.raw_paths[1], removeHs=False, sanitize=False)

        smiles_list = data_df["smiles"]
        homolumogap_list = data_df["homolumogap"]

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            smiles = smiles_list[i]
            homolumogap = homolumogap_list[i]

            if homolumogap is None or pd.isna(homolumogap):
                print("Target is None")
                continue

            if mol is None:
                continue

            z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

            try:
                graph = mol2graph(mol)
            except Exception as e:
                print(e)
                continue

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)
            data = Data()
            data.__num_nodes__ = int(graph["num_nodes"])
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            data.z = z
            data.y = torch.tensor([homolumogap], dtype=torch.float)
            data.pos = pos
            data.smile = smiles
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print("Saving...")
        torch.save(self.collate(data_list), self.processed_paths[0])
