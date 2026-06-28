import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from dataclasses import dataclass

import os
import random
import glob
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
import trimesh

from dataloaders.mesh_container import MeshContainer
from prompt import get_prompt
from chirality import ChiralityDisentangler

device = "cuda"

parser = argparse.ArgumentParser()
parser.add_argument("--train", nargs = 2, help="Path to training data and training category")
parser.add_argument("--test", nargs = 2, help="Path to test data and test category")
parser.add_argument("--pretrained", help="Path to directory containing pretrained model.pt")
parser.add_argument("--save_path", help="Path where to save resulting model. Ignored if not --train")
args = parser.parse_args()


@dataclass
class Config:
    feature_type: tuple = ("textured_sd_features", "textured_dino_features")
    model_type: str = "mlp"
    num_layers: int = 2
    chirality_dim: int = 1
    normalization_type: str = "beforeAndAfter"
    force_orthogonal: bool = True
    skip_connection: bool = True
    enable_non_chiral: bool = True
    detach_consistency: bool = False
    lambda_dis: float = 0.5
    lambda_sim: float = 0.5
    lambda_inv: float = 0.1
    lambda_line: float = 5.0
    lambda_consistency: float = 1.0
    target_iterations: int = 5000
    val_interval: int = 2000
    seed: int = 0
config = Config()

torch.manual_seed(config.seed)
np.random.seed(config.seed)
random.seed(config.seed)
print(config)


feature_size_dict = {
    "textured_dino_features": 768,
    "textured_clip_features": 1024,
    "textured_sd_features": 3200,
    "untextured_dino_features": 768,
    "untextured_clip_features": 1024,
    "untextured_sd_features": 3200,
    "diff3f_features": 2048,
}
feature_size = sum(feature_size_dict[ft] for ft in config.feature_type)


def process_path(path, feature_type, category):
    """Collect the shape/feature/chirality/line-graph files for one shape pair."""
    shape_files, feature_files, chirality_files, graph_files = [], [], [], []
    for index in ("0", "1"):
        shape_file = glob.glob(str(Path(path) / f"{index}_*.off"))
        assert len(shape_file) == 1, shape_file
        prompt = get_prompt(shape_file[0])
        if (category == "human" and prompt != "human") or (category == "animal" and prompt == "human"):
            continue
        feature_file = tuple(glob.glob(str(Path(path) / f"{index}_features/{ft}.pt"))[0] for ft in feature_type)
        chirality_file = glob.glob(str(Path(path) / f"{index}_chirality.txt")) + glob.glob(str(Path(path) / f"{index}_annotation.npy"))
        assert len(chirality_file) <= 1, chirality_file
        chirality_file = chirality_file[0] if len(chirality_file) == 1 else None
        graph_file = glob.glob(str(Path(path) / f"{index}_line_graph.pt"))
        shape_files.append(shape_file[0])
        feature_files.append(feature_file)
        chirality_files.append(chirality_file)
        graph_files.append(graph_file[0] if len(graph_file) == 1 else None)
    return shape_files, feature_files, chirality_files, graph_files


class BecosSymmetryDataset(Dataset):
    def __init__(self, data_dir, feature_type, num = None, category = "all", paired = False, work_parallel = True):
        pairs = glob.glob(str(data_dir / "*"))
        pairs.sort(key = lambda path: int(path.split("/")[-1]))
        print(f"Searching {len(pairs)} pairs")

        shape_files, feature_files, chirality_files, graph_files = [], [], [], []
        with ThreadPoolExecutor(max_workers = None if work_parallel else 1) as executor:
            results = list(tqdm(executor.map(lambda p: process_path(p, feature_type, category), pairs),
                                total = len(pairs), desc = "Finding shape and feature files"))
        for s, f, c, g in results:
            shape_files += s
            feature_files += f
            chirality_files += c
            graph_files += g

        if category == "all":
            assert len(shape_files) == 2 * len(pairs)

        self.shape_files = shape_files[:num]
        self.feature_files = feature_files[:num]
        self.chirality_files = chirality_files[:num]
        self.graph_files = graph_files[:num]
        self.feature_type = feature_type
        self.paired = paired

    def __len__(self):
        assert len(self.shape_files) == len(self.feature_files)
        return len(self.shape_files) if not self.paired else len(self.shape_files) // 2

    def getitem(self, idx):
        shape = MeshContainer().load_from_file(self.shape_files[idx])
        features = []
        for x in self.feature_files[idx]:
            f = torch.load(x)
            if len(f.shape) == 2:
                f = f.unsqueeze(0).expand(2, *f.shape)
            features.append(torch.nn.functional.normalize(f, dim = -1) / len(self.feature_files[idx]))
        features = torch.cat(features, -1)

        if self.chirality_files[idx] is None:
            chirality_info = torch.zeros(len(shape.vert)).bool()
        elif self.chirality_files[idx].endswith(".txt"):
            chirality_info = torch.from_numpy(np.loadtxt(self.chirality_files[idx])).bool()
        else:
            chirality_info = torch.from_numpy(np.load(self.chirality_files[idx])).bool()

        graph_dict = torch.load(self.graph_files[idx])
        cos_sim, line_graph_edges = graph_dict["cos_sim"], graph_dict["line_graph_edges"]

        split_path = self.shape_files[idx].split("/")
        path = ("/".join(split_path[:-1]), int(split_path[-1][0]))
        return shape, features, chirality_info, cos_sim, line_graph_edges, path

    def __getitem__(self, idx):
        if not self.paired:
            return self.getitem(idx)
        return self.getitem(2 * idx), self.getitem(2 * idx + 1)


def collate_fn(batch):
    return batch[0]


def make_loaders(data_folder, category):
    base = Path(os.path.expanduser(data_folder))
    loaders, pair_loaders = {}, {}
    for split in ("train", "val", "test"):
        data = BecosSymmetryDataset(base / split, config.feature_type, category = category)
        print(f"Size of the {split} data:", len(data))
        loaders[split] = DataLoader(data, batch_size = 1, shuffle = (split == "train"), collate_fn = collate_fn)
        pair_data = BecosSymmetryDataset(base / split, config.feature_type, category = category, paired = True)
        pair_loaders[split] = DataLoader(pair_data, batch_size = 1, shuffle = False, collate_fn = collate_fn)
    return loaders, pair_loaders


total_iterations = 1
def run_epoch(model, loader, optim, do_evaluate = False, val_loaders = None):
    avg_loss = 0
    avg_dissimilarity_loss = 0
    avg_similarity_loss = 0
    avg_invertibility_loss = 0
    avg_line_loss = 0
    avg_consistency_loss = 0
    avg_accuracy = 0

    length = 0
    for mesh, both_features, chirality, cos_sim, line_graph_edges, path in tqdm(loader):
        chirality = chirality.to(device = device)
        both_features = both_features.to(device = device).float()
        cos_sim = cos_sim.to(device = device).float()
        line_graph_edges = line_graph_edges.to(device = device)
        both_features = torch.nn.functional.normalize(both_features, dim = -1)

        both_chirality_feature, both_non_chirality_feature, forward_features, backward_features = model(both_features, return_backward = True)
        backward_features = torch.nn.functional.normalize(backward_features, dim = -1)
        invertibility_loss = torch.linalg.norm(both_features - backward_features) / (both_features.shape[1])**0.5

        chirality_feature, chirality_feature_flip = both_chirality_feature
        dissimilarity_loss = - torch.linalg.norm(chirality_feature - chirality_feature_flip) / (len(chirality_feature)**0.5)

        non_chirality_feature, non_chirality_feature_flip = both_non_chirality_feature
        similarity_loss = torch.linalg.norm(non_chirality_feature - non_chirality_feature_flip) / (len(chirality_feature)**0.5)

        # Line-graph (conjugate) loss: chirality should be constant along surface lines
        line_loss = - cos_sim + torch.abs(chirality_feature[line_graph_edges[:, 0]] - chirality_feature[line_graph_edges[:, 1]])[:, 0]**2 + torch.abs(chirality_feature[line_graph_edges[:, 1]] - chirality_feature[line_graph_edges[:, 2]])[:, 0]**2
        line_loss_flip = - cos_sim + torch.abs(chirality_feature_flip[line_graph_edges[:, 0]] - chirality_feature_flip[line_graph_edges[:, 1]])[:, 0]**2 + torch.abs(chirality_feature_flip[line_graph_edges[:, 1]] - chirality_feature_flip[line_graph_edges[:, 2]])[:, 0]**2
        line_loss_per_vertex = torch.full((len(chirality_feature),), 1000., device = device)
        line_loss_per_vertex.scatter_reduce_(0, line_graph_edges[:, 1], line_loss, reduce = "amin", include_self = True)
        line_loss_per_vertex_flip = torch.full((len(chirality_feature),), 1000., device = device)
        line_loss_per_vertex_flip.scatter_reduce_(0, line_graph_edges[:, 1], line_loss_flip, reduce = "amin", include_self = True)
        line_loss = line_loss_per_vertex.mean() + line_loss_per_vertex_flip.mean()

        # Consistency loss: non-chirality similarity weighted by chirality agreement is orthogonal-like
        cosine_similarity = both_non_chirality_feature @ both_non_chirality_feature.swapaxes(1, 2)
        cosine_similarity = cosine_similarity - cosine_similarity.min()
        cosine_similarity = cosine_similarity / cosine_similarity.max()
        weighting = torch.stack([(chirality_feature.flatten()[:, None] - chirality_feature.flatten()[None, :])**2,
                                 (chirality_feature_flip.flatten()[:, None] - chirality_feature_flip.flatten()[None, :])**2], dim = 0)
        weighting = weighting - weighting.min()
        weighting = weighting / weighting.max()
        if config.detach_consistency:
            weighting = weighting.detach()
        cosine_similarity = cosine_similarity * weighting
        identity = torch.eye(cosine_similarity.shape[1], device = device)
        product = cosine_similarity @ cosine_similarity / cosine_similarity.shape[1]
        consistency_loss = torch.linalg.norm(torch.stack([identity, identity], dim = 0) - product, ord = "fro", dim = (1, 2)).sum() / (cosine_similarity.shape[1])

        loss = config.lambda_dis * dissimilarity_loss \
            + config.lambda_inv * invertibility_loss \
            + config.lambda_sim * similarity_loss \
            + config.lambda_line * line_loss \
            + config.lambda_consistency * consistency_loss

        assignment = (chirality_feature > 0).flatten()
        accuracy = torch.mean((assignment == chirality).float())

        if optim is not None:
            loss.backward()
            optim.step()
            optim.zero_grad()

        avg_loss += loss.item()
        avg_dissimilarity_loss += dissimilarity_loss.item()
        avg_similarity_loss += similarity_loss.item()
        avg_invertibility_loss += invertibility_loss.item()
        avg_line_loss += line_loss.item()
        avg_consistency_loss += consistency_loss.item()
        avg_accuracy += accuracy.item()
        length += 1

        global total_iterations
        if do_evaluate and val_loaders is not None and total_iterations % config.val_interval == 0:
            with torch.no_grad():
                val_result = run_epoch(model, val_loaders["val"], None)
                val_result = {f"val_{k}": v for k, v in val_result.items()}
                print(total_iterations, val_result)
        if optim is not None:
            total_iterations += 1
        if optim is not None and total_iterations > config.target_iterations:
            break

    length = max(length, 1)
    return {"loss": avg_loss / length,
            "dissimilarity_loss": avg_dissimilarity_loss / length,
            "similarity_loss": avg_similarity_loss / length,
            "invertibility_loss": avg_invertibility_loss / length,
            "line_loss": avg_line_loss / length,
            "consistency_loss": avg_consistency_loss / length,
            "accuracy": max(avg_accuracy / length, 1 - avg_accuracy / length)}


def full_matching(model, loader, ignore_non_chiral = False):
    mean_error = 0
    mean_acc = 0
    thresholds = torch.linspace(0., 1., 100, device = device)
    for dat1, dat2 in tqdm(loader):
        features = []
        paths = []
        for mesh, both_features, _, _, _, path in (dat1, dat2):
            both_features = both_features.to(device = device).float()
            both_features = torch.nn.functional.normalize(both_features, dim = -1)
            _, both_non_chirality_feature, _, _ = model(both_features, return_backward = True)
            if ignore_non_chiral:
                feature, _ = both_features
            else:
                feature, _ = both_non_chirality_feature
            features.append(torch.nn.functional.normalize(feature, dim = -1))
            paths.append(path)

        feature1, feature2 = features
        path1, path2 = paths
        cosine = feature1 @ feature2.T
        pred_corr = torch.argmax(cosine, -1)

        assert path1[0] == path2[0]
        if not os.path.exists(path1[0] + "/corres_01.npy"):
            return {"mean_matching_error_full": -1, "mean_matching_acc@1%_full": -1,
                    "mean_matching_acc@5%_full": -1, "mean_matching_acc@10%_full": -1,
                    "mean_matching_acc_auc_full": -1}
        gt_corr = torch.from_numpy(np.load(path1[0] + "/corres_01.npy")).to(device = device)
        if len(gt_corr.shape) > 1:
            faces = torch.from_numpy(mesh.face).to(device = device)
            gt_corr = faces[gt_corr[:, 0].long(), torch.argmax(gt_corr[:, 1:], -1)]
        mask = (gt_corr > 0)

        geodesics = torch.load(path2[0] + f"/{path2[1]}_geodesics.pt", map_location = device)
        error = geodesics[pred_corr, gt_corr][mask]
        normalisation_factor = (float(trimesh.Trimesh(mesh.vert, mesh.face).area))**0.5
        error = error / normalisation_factor

        mean_acc += (error[None, :] <= thresholds[:, None]).float().mean(-1)
        mean_error += error.mean().item()

    return {"mean_matching_error_full": mean_error / len(loader),
            "mean_matching_acc@1%_full": (mean_acc[1] / len(loader)).item(),
            "mean_matching_acc@5%_full": (mean_acc[5] / len(loader)).item(),
            "mean_matching_acc@10%_full": (mean_acc[10] / len(loader)).item(),
            "mean_matching_acc_auc_full": (mean_acc.mean() / len(loader)).item()}


def flip_matching(model, loader, use_gt = False):
    mean_error = 0
    mean_acc = 0
    thresholds = torch.linspace(0., 1., 100, device = device)
    for mesh, both_features, _, _, _, path1 in tqdm(loader):
        both_features = both_features.to(device = device).float()
        both_features = torch.nn.functional.normalize(both_features, dim = -1)
        both_chirality_feature, both_non_chirality_feature, _, _ = model(both_features, return_backward = True)
        chirality_feature, _ = both_chirality_feature
        non_chirality_feature, _ = both_non_chirality_feature

        feature1 = torch.nn.functional.normalize(non_chirality_feature, dim = -1)
        cosine = feature1 @ feature1.T

        if not os.path.exists(path1[0] + f"/{path1[1]}_chirality.txt"):
            return {"mean_matching_error_flip": -1, "mean_matching_acc@1%_flip": -1,
                    "mean_matching_acc@5%_flip": -1, "mean_matching_acc@10%_flip": -1,
                    "mean_matching_acc_auc_flip": -1}
        if use_gt:
            label = torch.from_numpy(np.loadtxt(path1[0] + f"/{path1[1]}_chirality.txt")).to(device = device)
        else:
            label = (chirality_feature > 0).flatten()
        mask = label[:, None] == label[None, :]
        cosine[mask] = -1
        pred_corr = torch.argmax(cosine, -1)
        assert (label[pred_corr] != label).all()

        if not os.path.exists(path1[0] + f"/{path1[1]}_orig_to_flip.npy"):
            return {"mean_matching_error_flip": -1, "mean_matching_acc@1%_flip": -1,
                    "mean_matching_acc@5%_flip": -1, "mean_matching_acc@10%_flip": -1,
                    "mean_matching_acc_auc_flip": -1}
        gt_corr = torch.from_numpy(np.load(path1[0] + f"/{path1[1]}_orig_to_flip.npy")).to(device = device)
        if len(gt_corr.shape) > 1:
            faces = torch.from_numpy(mesh.face).to(device = device)
            gt_corr = faces[gt_corr[:, 0].long(), torch.argmax(gt_corr[:, 1:], -1)]
        mask = (gt_corr > 0)

        geodesics = torch.load(path1[0] + f"/{path1[1]}_geodesics.pt", map_location = device)
        error = geodesics[pred_corr, gt_corr][mask]
        normalisation_factor = (float(trimesh.Trimesh(mesh.vert, mesh.face).area))**0.5
        error = error / normalisation_factor

        mean_acc += (error[None, :] <= thresholds[:, None]).float().mean(-1)
        mean_error += error.mean().item()

    return {"mean_matching_error_flip": mean_error / len(loader),
            "mean_matching_acc@1%_flip": (mean_acc[1] / len(loader)).item(),
            "mean_matching_acc@5%_flip": (mean_acc[5] / len(loader)).item(),
            "mean_matching_acc@10%_flip": (mean_acc[10] / len(loader)).item(),
            "mean_matching_acc_auc_flip": (mean_acc.mean() / len(loader)).item()}


def evaluate(model, loaders, pair_loaders, split):
    with torch.no_grad():
        result = run_epoch(model, loaders[split], None)
        result |= full_matching(model, pair_loaders[split])
        result |= flip_matching(model, loaders[split])
        result["objective"] = result["accuracy"] * result["mean_matching_acc_auc_full"] * result["mean_matching_acc_auc_flip"]
    result = {f"{split}_{k}": v for k, v in result.items()}
    print(result)
    return result


model = ChiralityDisentangler(feature_size, config.num_layers, config.model_type, config.normalization_type,
                              chirality_dim = config.chirality_dim, force_orthogonal = config.force_orthogonal,
                              skip_connection = config.skip_connection, enable_non_chiral = config.enable_non_chiral).to(device = device)
optim = torch.optim.Adam(model.parameters(), lr = 0.001)

if args.pretrained is not None:
    model.load_state_dict(torch.load(os.path.join(args.pretrained, "model.pt"), map_location = device))

train_loaders = train_pair_loaders = None
if args.train is not None:
    train_loaders, train_pair_loaders = make_loaders(args.train[0], args.train[1])

test_loaders = test_pair_loaders = None
if args.test is not None:
    test_loaders, test_pair_loaders = make_loaders(args.test[0], args.test[1])

if args.train is not None:
    while total_iterations < config.target_iterations:
        train_result = run_epoch(model, train_loaders["train"], optim, do_evaluate = True, val_loaders = train_loaders)
        train_result = {"train_" + k: v for k, v in train_result.items()}
        print(total_iterations, train_result)

if args.save_path is not None and args.train is not None:
    os.makedirs(args.save_path, exist_ok = True)
    torch.save(model.state_dict(), os.path.join(args.save_path, "model.pt"))

if args.test is not None:
    evaluate(model, test_loaders, test_pair_loaders, "val")
    evaluate(model, test_loaders, test_pair_loaders, "test")
