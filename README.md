# LoTS-Net

Hyperspectral pathology image segmentation with text-guided multimodal learning. Supports fully-supervised and semi-supervised training.

## Setup

```bash
conda create -n lotsnet python=3.10 -y
conda activate lotsnet
pip install -r requirements.txt
```

## Data

Download from [Baidu Netdisk](https://pan.baidu.com/s/1n7vRc5juMfV0Z1PZy8G-IQ?pwd=5hq5) (code: `5hq5`) and extract into `data/`:

```
data/
└── MDC_Resized_256_320_to_256_256_overlap_0_0/
    ├── images/
    ├── masks/
    ├── texts/
    ├── split_sup.csv
    └── split_semi.csv
```

## Training

```bash
bash scripts/train_sup.sh    # fully-supervised
bash scripts/train_semi.sh     # semi-supervised
```

Hyperparameters are set at the top of each script. Logs and checkpoints go to `records/`.
