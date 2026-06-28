# SYMMETRY

This repository includes code for both "Chi: Symmetry Understanding of 3D Shapes via Chirality Disentanglement" and "Symmetry Informative and Agnostic Feature Disentanglement for 3D Shapes".

## Chi: Symmetry Understanding of 3D Shapes via Chirality Disentanglement [ICCV 2025]
<a href='https://wei-kang-wang.github.io/chirality/'><img src='https://img.shields.io/badge/Project-Page-green'></a>  [![ArXiv](https://img.shields.io/badge/arXiv-2508.05505-b31b1b.svg)](https://arxiv.org/pdf/2508.05505)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
![](assets/iccv_teaser.jpg)

[Project Webpage](https://diff3f.github.io/) | [Paper](https://arxiv.org/pdf/2508.05505)

## Symmetry Informative and Agnostic Feature Disentanglement for 3D Shapes [3DV 2026]
<a href='https://tweissberg.github.io/chirality/'><img src='https://img.shields.io/badge/Project-Page-green'></a>  [![ArXiv](https://img.shields.io/badge/arXiv-2601.14804-b31b1b.svg)](https://arxiv.org/pdf/2601.14804)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

![](assets/3dv_teaser.png)

[Project Webpage](https://tweissberg.github.io/chirality/) | [Paper](https://arxiv.org/pdf/2601.14804)

## Setup
```shell
conda create -n chi python=3.10
conda activate chi
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d::pytorch3d meshplot
pip3 install ./third_party/ODISE 
pip3 install ./third_party/Mask2Former 
pip3 install diffusers==0.21.4 huggingface_hub==0.17.3 transformers==4.34.1 opencv-python==4.6.0.66 scikit-learn matplotlib numpy==1.25.0 plyfile trimesh potpourri3d robust-laplacian open3d accelerate==0.20.3 pillow==9.5.0 timm==0.6.11 networkx
```

## Data download & preparation:
Download [BECOS](https://github.com/NafieAmrani/becos-code) and generate the benchmark dataset.
Then precompute all necessary features using (for example).
```shell
./generate_images.py <dataset_name> <split> <idx>
./generate_features.py <dataset_name> <split> <idx>
```
For the 3DV version, additionally precompute the line graph (required for training) and the
geodesic distances (required for the correspondence evaluation):
```shell
./generate_conjugate.py <data_root> <split> <idx>   # writes {0,1}_line_graph.pt
./generate_geodesic.py <data_root> <split> <idx>    # writes {0,1}_geodesics.pt
```

## Usage
### Chi: Symmetry Understanding of 3D Shapes via Chirality Disentanglement [ICCV 2025]

Training a model from scratch (with testing):
```shell
CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ./chi.py --train <data_folder> all --test <data_folder> all
```

Evaluating the provided model:
```shell
CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ./chi.py --test <data_folder> all --pretrained ./pretrained/iccv/
```


### Symmetry Informative and Agnostic Feature Disentanglement for 3D Shapes [3DV 2026]

Make sure the features (`generate_images.py` / `generate_features.py`), line graphs
(`generate_conjugate.py`) and geodesics (`generate_geodesic.py`) have been precomputed for the
dataset, then:

Training a model from scratch (with testing):
```shell
python3 ./chi_3dv.py --train <data_folder> all --test <data_folder> all --save_path <out_dir>
```

Evaluating the provided model:
```shell
python3 ./chi_3dv.py --test <data_folder> all --pretrained ./pretrained/3dv/
```

This reports per-shape chirality accuracy together with the full-shape and left/right
correspondence matching metrics (`mean_matching_acc@{1,5,10}%`, AUC) and the combined objective.

## BibTeX

```shell
@inproceedings{wang2025symmetry,
  title     = {Symmetry Understanding of 3D Shapes via Chirality Disentanglement},
  author    = {Weikang Wang, Tobias Weißberg, Nafie El Amrani and Florian Bernard},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year      = {2025}
}
```

```shell
@inproceedings{weissberg2025symmetry,
  title     = {Symmetry Informative and Agnostic Feature Disentanglement for 3D Shapes},
  author    = {Tobias Wei{\ss}berg, Weikang Wang, Paul Roetzer, Nafie El Amrani and Florian Bernard},
  booktitle = {International Conference on 3D Vision (3DV)},
  year      = {2026},
  note      = {to appear}
}
```


