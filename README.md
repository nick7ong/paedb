# PAEDB Dataset Generation Scripts

This repository contains code for generating the Primary-Ambient Extraction Dataset (PAEDB), including:

---

## 1. Environment Setup

### **Create a conda environment**

```bash
conda create -n paedb-env python=3.11 -y
conda activate paedb-env
```

### **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 2. Download Base Datasets

### Download MUSDB18HQ and MOISESDB

```bash
chmod +x download_musdb18hq.sh
./download_musdb18hq.sh dataset/
```

Will have to manually download MOISESDB from their [website](https://music.ai/research/) due to licensing agreements of the data.

---

## 3. Generate the PAEDB Dataset

### Example Usage

```bash
python generate_paedb.py \
    --dirs dataset/musdb18hq dataset/moisesdb \
    --output dataset/paedb \
    --type elae \
    --split-ratios 0.8 0.1 0.1
```

### Directory Structure
```
output_root/
└── train/
│    └── <song-title>_<stem>/
│          ├── mixture.wav
│          ├── primary.wav
│          └── ambience.wav
└── valid/
└── test/
```

---


## Citation (*Placeholder*)

```bibtex
@misc{tong2026paedb,
  author       = {Nicholas N Tong},
  title        = {PAEDB: Primary-Ambient Extraction Dataset},
  year         = {2026},
  howpublished = {\url{https://github.com/nick7ong/paedb}}
}
```

