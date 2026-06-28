import torch

import os
import sys
import glob
from pathlib import Path
from tqdm import tqdm

from dataloaders.mesh_container import MeshContainer
from pyFM.mesh import TriMesh


def get_geodesic(mesh):
    """Full pairwise geodesic distance matrix (V x V) via the vendored pyFM TriMesh."""
    trimesh = TriMesh(mesh.vert, mesh.face)
    return trimesh.get_geodesic()


if __name__ == "__main__":
    # Usage: ./generate_geodesic.py <data_root> <split> <idx>
    # Processes shape pair <idx> under <data_root>/<split>/<idx>/, writing
    # {0,1}_geodesics.pt = a float V x V geodesic distance matrix.
    data_root = os.path.expanduser(sys.argv[1])
    split = sys.argv[2]
    for i in tqdm(range(int(sys.argv[3]), int(sys.argv[3]) + 1)):
        path = f"{data_root}/{split}/{i}"
        print(path)
        for index in ("0", "1"):
            shape_file = glob.glob(str(Path(path) / f"{index}_*.off"))
            mesh = MeshContainer().load_from_file(shape_file[0])
            geodesics = torch.from_numpy(get_geodesic(mesh)).float()
            torch.save(geodesics, path + f"/{index}_geodesics.pt")
