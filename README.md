<h1 align="center">Last-Meter Precision Navigation for UAVs</h1>


<p align="center">
  <!-- Dataset -->
  <a href="https://huggingface.co/datasets/YaxuanLi/pairUAV">
    <img src="https://img.shields.io/badge/Dataset-HF%20Data-d8b04c?style=for-the-badge" alt="Dataset">
  </a>

  <!-- Paper -->
  <a href="https://your-paper-link-here">
    <img src="https://img.shields.io/badge/Paper-arXiv-d46a5a?style=for-the-badge" alt="Paper">
  </a>

  <!-- Email -->
  <a href="yaxuanli.cn@gmail.com">
    <img src="https://img.shields.io/badge/Email-YaxuanLi-6b6b6b?style=for-the-badge" alt="Email">
  </a>
</p>

<hr />

## Project Structure

```
dreamNav/
├── models/   
├── pairUAV/
│   ├── data_process.sh 
│   └── University-Release.zip
├── step1/                  
│   ├── SuperGlue/   
│   ├── step1.py      
│   ├── test.py    
│   └── run.sh           
└── step2/                 
```

---

## 1. Environment Setup

Create a unified conda environment for both Step1 and Step2:

```bash
conda create -n dreamnav python=3.9
conda activate dreamnav

# Install PyTorch with CUDA support
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Install all dependencies
pip install -r requirements.txt

# Download pretrained models
huggingface-cli download Ramos-Ramos/dino-resnet-50 --local-dir models/dino_resnet
huggingface-cli download Boese0601/MagicDance control_sd15_ini.ckpt --local-dir models/controlnet
huggingface-cli download openai/clip-vit-large-patch14 --local-dir models/controlnet/clip-vit-large-patch14
```

---

## 2. Data Preparation

### 2.1 Download University-1652 Dataset

Download [University-1652](https://github.com/layumi/University1652-Baseline) upon request (Usually I will reply you in 5 minutes). You may use the [request template](https://github.com/layumi/University1652-Baseline/blob/master/Request.md).

### 2.2 Download and Process PairUAV Dataset

Download and process the PairUAV dataset:

```bash
cd pairUAV/
bash data_process.sh
```

This script downloads the dataset from HuggingFace and extracts train/test/tours data to the `pairUAV/` directory.

## 3. Stage 1: Coarse Estimation

### 3.1 Run SuperGlue Feature Matching

First, perform feature matching on image pairs:

```bash
cd ../step1/SuperGlue

# Run feature matching
bash run.sh
cd ..

```

This generates matching results in `matches_data/`.

### 3.2 Train Stage 1 Model

```bash
cd step1
bash run.sh
cd ..
```

**Key Parameters:**
- `--lr`: Backbone learning rate, default 1e-6
- `--lr_regressor`: Regressor learning rate, default 5e-3
- `--epochs`: Number of training epochs, default 5
- `--batch_size`: 256

**Outputs:**
- `output.log`: Training log
- `step1_seen.json`: Best prediction results (used in Stage 2)

---

## Appendix: Baseline

### A.1 DINOv3 (ViT-7B)

This baseline extracts image embeddings using a pretrained [DINOv3-ViT-7B](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m) model and trains a linear regressor to predict heading and range.

#### A.1.1 Environment Setup

```bash
cd baseline/dinov3

conda create -n dinov3 python=3.9
conda activate dinov3

# Install PyTorch with CUDA support
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements.txt
```

#### A.1.2 Download Pretrained Model

Login to HuggingFace and download the DINOv3-ViT-7B checkpoint:

```bash
huggingface-cli login
huggingface-cli download facebook/dinov3-vit7b16-pretrain-lvd1689m --local-dir ../../models/dinov3_7b
```

#### A.1.3 Extract Embeddings

Extract DINOv3 embeddings for all tour images. The script reads images from `pairUAV/tours/` and saves `.pkl` embedding files to `baseline/dinov3/embedding/`:

```bash
python extract_embeddings.py
```

#### A.1.4 Train & Evaluate

Train the linear regressor and evaluate on the test set:

```bash
bash run.sh
```

**Key Parameters (in `run.sh`):**
- `--lr_regressor`: Regressor learning rate, default `1e-3`
- `--epochs`: Number of training epochs, default `4`
- `--warmup_epochs`: Warmup epochs, default `1`

**Outputs:**
- `test_results.log`: Per-epoch evaluation results (Range MAE, Heading MAE, Success Rate)

---