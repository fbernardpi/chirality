import torch
import numpy as np
import networkx as nx

import os
import sys
import glob
from pathlib import Path
from tqdm import tqdm

from dataloaders.mesh_container import MeshContainer

device = "cpu"


def get_cos_sim(mesh):
    """Build the line graph of the mesh and return, for every line-graph edge
    (a triple of consecutive vertices a-b-c), the cosine similarity between the
    two surface-tangent edge directions at the shared vertex b."""
    faces = torch.from_numpy(mesh.face)
    edges = torch.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = torch.unique(torch.sort(edges, dim = -1).values, dim = 0)

    graph = nx.Graph()
    graph.add_edges_from(edges.cpu().numpy())
    graph = graph.to_directed()
    line_graph = nx.line_graph(graph)

    # Per-vertex normals (area-weighted face normals)
    vertices = torch.from_numpy(mesh.vert).to(device = device)
    vertex_normals = torch.zeros_like(vertices, device = device)
    face_normals = torch.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
        dim = 1,
    )
    for i, face in enumerate(faces):
        for vertex in face:
            vertex_normals[vertex] += face_normals[i]
    vertex_normals = torch.nn.functional.normalize(vertex_normals, dim = 1)

    # Each line-graph edge is a triple (a, b, c) with shared middle vertex b
    line_graph_edges = torch.tensor([(*t[0], t[1][1]) for t in list(line_graph.edges())])
    for edge in line_graph.edges():
        assert edge[0][1] == edge[1][0]

    edge1 = vertices[line_graph_edges[:, 1]] - vertices[line_graph_edges[:, 0]]
    edge2 = vertices[line_graph_edges[:, 2]] - vertices[line_graph_edges[:, 1]]

    # Project both edges onto the tangent plane at the shared vertex b
    normal_b = vertex_normals[line_graph_edges[:, 1]]
    edge1_proj = edge1 - (edge1 * normal_b).sum(-1, keepdim = True) * normal_b
    edge2_proj = edge2 - (edge2 * normal_b).sum(-1, keepdim = True) * normal_b

    cos_sim = torch.nn.functional.cosine_similarity(edge1_proj, edge2_proj, dim = 1)
    return cos_sim.cpu(), line_graph_edges.cpu()


if __name__ == "__main__":
    # Usage: ./generate_conjugate.py <data_root> <split> <idx>
    # Processes shape pair <idx> under <data_root>/<split>/<idx>/, writing
    # {0,1}_line_graph.pt = {"cos_sim", "line_graph_edges"}.
    batch_size = 10
    data_root = os.path.expanduser(sys.argv[1])
    split = sys.argv[2]
    for i in tqdm(range(batch_size * int(sys.argv[3]), batch_size * (int(sys.argv[3]) + 1))):
        path = f"{data_root}/{split}/{i}"
        print(path)
        for index in ("0", "1"):
            shape_file = glob.glob(str(Path(path) / f"{index}_*.off"))
            mesh = MeshContainer().load_from_file(shape_file[0])
            cos_sim, line_graph_edges = get_cos_sim(mesh)
            assert torch.max(line_graph_edges) < len(mesh.vert)
            assert len(cos_sim) == len(line_graph_edges)
            torch.save({"cos_sim": cos_sim, "line_graph_edges": line_graph_edges}, path + f"/{index}_line_graph.pt")
