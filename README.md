# DreamNav

DreamNav is a drone vision-based navigation prediction system trained in two stages.

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
- `--epochs`: Number of training epochs, default 2
- `--batch_size`: 256

**Outputs:**
- `output.log`: Training log
- `step1_seen.json`: Best prediction results (used in Stage 2)

---