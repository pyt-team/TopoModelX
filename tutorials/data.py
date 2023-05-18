"""Retrieve datasets from the web and process into formats usable by torch_geometric."""
import os
import sys
import urllib
from typing import Optional

import numpy as np
import torch
import trimesh
from scipy.stats import special_ortho_group
from toponetx import CellComplex
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_gz,
    extract_tar,
    extract_zip,
)


def download_url_custom(
    url: str, folder: str, log: bool = True, filename: Optional[str] = None
):
    r"""Download the content of an URL to a specific folder.

    Parameters
    ----------
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)

    Returns
    -------
        string: The path to the downloaded file.
    """
    if filename is None:
        filename = url.rpartition("/")[2]
        filename = filename if filename[0] == "?" else filename.split("?")[0]

    path = os.path.join(folder, filename)

    if log:
        print(f"Downloading {url}", file=sys.stderr)

    os.makedirs(folder, exist_ok=True)

    data = urllib.request.urlopen(url)

    with open(path, "wb") as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path


class Shrec16AugDataset(InMemoryDataset):
    """The SHREC 16 dataset from the `"SHREC'16: Partial Matching of Deformable Shapes."""

    def __init__(
        self,
        root,
        name="shrec_16",
        split="train",
        num_rot=30,
        cat=False,
        num_features=50,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.name = name
        self.root = root
        self.url = "https://www.dropbox.com/s/c7leaw4nvnbk4rs/shrec_16.zip?dl=1"
        self.split = split
        self.train_idx = None
        self.test_idx = None
        self.labels = np.load(os.path.join(self.raw_dir, "labels.npy"))
        self.cat = cat
        self.num_feats = num_features
        self.num_rotations = num_rot
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_dir(self):
        """Path to the directory containing the raw data."""
        return os.path.join(self.root, self.name, "raw", self.name)

    @property
    def processed_dir(self):
        """Path to the directory containing the processed data."""
        return os.path.join(self.root, self.name, "processed")

    @property
    def num_classes(self) -> int:
        """The number of classes in the dataset."""
        return np.unique(self.labels).shape[0]

    @property
    def raw_file_names(self):
        """Names of the raw files."""
        with open(os.path.join(self.raw_dir, "file_names.txt"), "r") as f:
            l1 = []
            for line in f:
                line = line.strip("\n")
                l1.append(line)
        # l1 = [f'T{i}.obj' for i in range(600)]
        l2 = ["file_names.txt", "train_idx.npy", "test_idx.npy", "labels.npy"]
        self.train_idx = np.load(os.path.join(self.raw_dir, "train_idx.npy"))
        self.test_idx = np.load(os.path.join(self.raw_dir, "test_idx.npy"))
        return l1 + l2

    @property
    def processed_file_names(self):
        """Names of the processed files."""
        if self.split == "train":
            return [
                f"train_data_{ii}_aug_{k}.pt"
                for ii in range(self.train_idx.shape[0])
                for k in range(self.num_rotations)
            ]
        elif self.split == "test":
            return [f"test_data_{ii}.pt" for ii in range(self.test_idx.shape[0])]

    def download(self):
        """Download the dataset."""
        path = download_url_custom(self.url, self.raw_dir)
        # # path = self.raw_dir
        # # os.system('wget https://www.dropbox.com/s/rhk5q9uv7m4ry7g/shrec_16.zip')
        extract_zip(path, self.raw_dir)
        os.remove(path)

    @staticmethod
    def get_attr(sc: CellComplex, label, num_features, mode="inv_euclidean", mesh=None):
        """Get attributes from a cell complex."""
        if mesh is None:
            mesh = sc.to_trimesh()
        pos = mesh.vertices
        vn = mesh.vertex_normals

        A0 = sc.adjacency_matrix(0, signed=False).toarray()
        A1 = sc.adjacency_matrix(1, signed=False).toarray()
        A2 = sc.coadjacency_matrix(2, signed=False).toarray()
        B1 = sc.incidence_matrix(1, signed=False).toarray()
        B2 = sc.incidence_matrix(2, signed=False).toarray()

        B1T = B1.T
        B2T = B2.T

        num_ver = B1.shape[0]
        num_edges = B1.shape[1]
        num_faces = B2.shape[1]
        assert num_edges == B2.shape[0]

        x_v = np.zeros((num_ver, num_features))
        x_e = np.zeros((num_edges, num_features))
        x_f = np.zeros((num_faces, num_features))

        x_v = torch.tensor(x_v)
        x_e = torch.tensor(x_e)
        x_f = torch.tensor(x_f)

        A0 = torch.tensor(A0)
        A1 = torch.tensor(A1)
        A2 = torch.tensor(A2)
        B1 = torch.tensor(B1)
        B2 = torch.tensor(B2)
        B1T = torch.tensor(B1T)
        B2T = torch.tensor(B2T)

        y = torch.tensor(label, dtype=torch.long)
        pos = torch.tensor(pos)
        vn = torch.tensor(vn)
        data = Data(x=x_v, x_e=x_e, x_f=x_f, pos=pos, vn=vn, y=y)
        data_matrices = Data(A0=A0, A1=A1, A2=A2, B1=B1, B2=B2, B1T=B1T, B2T=B2T)
        return data, data_matrices

    def process(self):
        """Read data into large `Data` list."""
        fnames = []
        with open(os.path.join(self.raw_dir, "file_names.txt"), "r") as f:
            for line in f:
                line = line.strip("\n")
                fnames.append(line)
        labels = np.load(os.path.join(self.raw_dir, "labels.npy"))
        train_file_names = [fnames[idx] for idx in self.train_idx]
        test_file_names = [fnames[idx] for idx in self.test_idx]
        train_labels = labels[self.train_idx]
        test_labels = labels[self.test_idx]

        for ii, m in enumerate(train_file_names):
            print(f"Done {ii + 1}/{len(train_file_names)}", end="\r")
            if os.path.exists(os.path.join(self.processed_dir, f"train_data_{ii}.pt")):
                continue
            label = train_labels[ii]
            mesh = trimesh.load_mesh(os.path.join(self.raw_dir, m), process=False)
            sc = CellComplex.from_trimesh(mesh)
            data, data_matrices = self.get_attr(sc, label, self.num_feats, mesh=mesh)
            fname = os.path.join(self.processed_dir, f"train_data_{ii}_aug_{0}.pt")
            torch.save(data, fname)
            mtx_fname = os.path.join(self.processed_dir, f"train_data_{ii}_matrices.pt")
            torch.save(data_matrices, mtx_fname)
            for k in range(1, self.num_rotations):
                x = special_ortho_group.rvs(3)
                nodes = np.array(mesh.vertices) @ x
                mesh_r = trimesh.Trimesh(
                    vertices=nodes, faces=mesh.faces, process=False
                )
                sc_r = CellComplex.from_trimesh(mesh_r)
                data, _ = self.get_attr(sc_r, label, self.num_feats, mesh=mesh_r)
                fname = os.path.join(self.processed_dir, f"train_data_{ii}_aug_{k}.pt")
                torch.save(data, fname)
        for ii, m in enumerate(test_file_names):
            print(f"Done {ii + 1}/{len(test_file_names)}", end="\r")
            mesh = trimesh.load(os.path.join(self.raw_dir, m))
            sc = CellComplex.from_trimesh(mesh)
            label = test_labels[ii]
            data, data_matrices = self.get_attr(sc, label, self.num_feats, mesh=mesh)
            fname = os.path.join(self.processed_dir, f"test_data_{ii}.pt")
            torch.save(data, fname)
            mtx_fname = os.path.join(self.processed_dir, f"test_data_{ii}_matrices.pt")
            torch.save(data_matrices, mtx_fname)

    def len(self):
        """Return the number of files in the dataset."""
        return len(self.processed_file_names)

    def get(self, idx):
        """Get the `idx`-th example."""
        if self.split == "test":
            data = torch.load(
                os.path.join(self.processed_dir, f"{self.split}_data_{idx}.pt")
            )
            data_matrices = torch.load(
                os.path.join(self.processed_dir, f"{self.split}_data_{idx}_matrices.pt")
            )

        else:
            data = torch.load(
                os.path.join(
                    self.processed_dir,
                    f"{self.split}_data_{idx//self.num_rotations}_aug_{idx%self.num_rotations}.pt",
                )
            )
            data_matrices = torch.load(
                os.path.join(
                    self.processed_dir,
                    f"{self.split}_data_{idx//self.num_rotations}_matrices.pt",
                )
            )
        if isinstance(data, tuple):
            data = data[0]
        x_v = data.x[:, 0 : self.num_feats]
        x_e = data.x_e[:, 0 : self.num_feats]
        x_f = data.x_f[:, 0 : self.num_feats]
        pos = data.pos
        vn = data.vn

        if self.cat:
            x_v = torch.column_stack([pos, vn])
        data.x = x_v
        data.x_e = x_e
        data.x_f = x_f
        data.A0 = data_matrices.A0
        data.A1 = data_matrices.A1
        data.A2 = data_matrices.A2
        data.B1 = data_matrices.B1
        data.B2 = data_matrices.B2
        data.B1T = data_matrices.B1T
        data.B2T = data_matrices.B2T

        return data
