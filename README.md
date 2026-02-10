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
│   └── step1.py           
└── step2/                 
```

---


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
huggingface-cli download lllyasviel/ControlNet --local-dir models/controlnet
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
