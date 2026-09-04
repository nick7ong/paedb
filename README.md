# PAEDB Dataset Generation Scripts

This repository contains code for generating the Primary-Ambient Extraction Dataset (PAEDB).

### To-Dos:
- Add audio examples

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
@inproceedings{tong2026paedb,
  title        = {{PAEDB}: A Synthetic Primary-Ambient Dataset Generation Pipeline for Automatic Upmixing Using Deep Neural Networks},
  author       = {Tong, Nicholas N. and Collins, Tom},
  booktitle    = {Proceedings of the 29th International Conference on Digital Audio Effects (DAFx26)},
  year         = {2026},
  address      = {Cambridge, MA, USA},
}
```

