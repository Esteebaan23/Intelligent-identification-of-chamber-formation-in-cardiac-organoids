# Hybrid ResNet + ViT Organoid Classifier

Binary classifier for organoid microscopy images using a ResNet50 + DeiT3 hybrid model.  
Classes: **Chamber Forming** (label 0) vs **Chamber Nonforming** (label 1).

---

## Directory structure

```
BASE_DIR/
├── Chamber Forming/
│   ├── CH1/
│   │   └── *.tif
│   ├── CH2/
│   │   └── *.tif
│   └── CH3/
│       └── *.tif
└── Chamber Nonforming/
    ├── CH1/
    │   └── *.tif
    ├── CH2/
    │   └── *.tif
    └── CH3/
        └── *.tif
```

`BASE_DIR` defaults to `/Documents/Organoids`. Change it in the configuration block at the bottom of `train.py` and the constants block at the top of `evaluate.py`.

---

## Image specifications

| Property  | Requirement                                       |
|-----------|---------------------------------------------------|
| Format    | `.tif` (TIFF)                                     |
| Color     | RGB or grayscale (grayscale is stacked to 3ch)    |
| Size      | Any — resized to **224 × 224** before inference   |
| Bit depth | 8-bit recommended; 16-bit will be read by OpenCV  |
| Naming    | No constraints — any filename with `.tif` extension is loaded |

---

## Channels and preprocessing

CLAHE is applied to all channels during both training and validation.  
`RemoveBrightBorderFlood` is applied only for CH3 (bright border artifact common in that channel).

| Channel | CLAHE | `RemoveBrightBorderFlood` | Notes                                        |
|---------|-------|--------------------------|----------------------------------------------|
| CH3     | Yes   | Yes                      | Border artifact removal + contrast enhancement |
| CH2     | Yes   | No                       | Contrast enhancement only                    |
| CH1     | Yes   | No                       | Contrast enhancement only                    |

**CLAHE** (Contrast Limited Adaptive Histogram Equalization, clip=2.0, tile=8×8) improves local contrast before the model sees the image. Applied consistently in train and validation.  
**RemoveBrightBorderFlood** detects and fills bright rectangular border artifacts via a flood-fill from the image edges. Triggered when the flooded region covers ≥ 500 px.

---

## Switching channels

### In `train.py`

Find the configuration block near the bottom of the file:

```python
CHANNEL  = 'CH3'                   # Options: 'CH1', 'CH2', 'CH3'
BASE_DIR = "/Documents/Organoids"
```

Change `CHANNEL` to `'CH1'` or `'CH2'`. The preprocessing pipeline and checkpoint name (`best_model_{CHANNEL}.pth`) update automatically.

### In `evaluate.py`

Find the constants block near the top:

```python
CHANNEL    = "CH3"
MODEL_PATH = "best_model_CH3.pth"
BASE_DIR   = "/Documents/Organoids"
```

Change `CHANNEL` and `MODEL_PATH` to match the trained checkpoint, e.g.:

```python
CHANNEL    = "CH2"
MODEL_PATH = "best_model_CH2.pth"
```

`APPLY_CLAHE` and the border cleaner are set automatically based on `CHANNEL` — no manual changes needed.

---

## Train / validation split

- Split: **80 % train / 20 % validation**, stratified by class.
- Fixed seed `42` everywhere — the split is always identical across runs.
- `evaluate.py` uses the same seed and split ratio so it always evaluates on the exact 20 % held-out set used during training.

---

## Running training

```bash
python train.py
```

Outputs:
- `best_model_{CHANNEL}.pth` — best checkpoint (saved when val accuracy **and** F1 both improve)
- `best_model_{CHANNEL}_history.xlsx` — per-epoch metrics (loss, accuracy, precision, recall, F1)

---

## Running evaluation

```bash
python evaluate.py
```

Outputs (inside `val_hybrid_reports/`):
- `results.xlsx` — per-image predictions, probabilities, and correctness flag
- `classification_report.txt` — precision / recall / F1 per class
- `confusion_matrix.csv` / `confusion_matrix.png` — confusion matrix
- `split_train.csv` / `split_val.csv` — exact file paths in each split
- `misclassified/misclf_NNNN_predP_trueT.png` — 5-panel figures for every misclassified image (Original · Cleaned · Grad-CAM · ViT Attention · Combined)

---

## Model architecture

```
Input (3 × 224 × 224)
       │
       ├─► ResNet50 (pretrained ImageNet) → 2048-dim features
       │
       └─► DeiT3-Base patch16/224 (pretrained) → 768-dim features
                      │
               Concatenate → 2816-dim
                      │
               Linear(2816 → 1) + Sigmoid
                      │
                   [0, 1]  (probability of Chamber Nonforming)
```

Threshold for classification: **0.5** (`THRESH` in `evaluate.py`).

---

## Hyperparameters

| Parameter      | Value  | Location                                              |
|----------------|--------|-------------------------------------------------------|
| Epochs         | 50     | `train.py` — `train_model(..., num_epochs=50)`        |
| Batch size     | 16     | `train.py` — `DataLoader(..., batch_size=16)`         |
| Learning rate  | 1e-5   | `train.py` — `Adam(lr=1e-5)`                          |
| Loss           | BCE    | `train.py` — `nn.BCELoss()`                           |
| Val batch size | 1      | `evaluate.py` — `BATCH_SIZE = 1`                      |
| Threshold      | 0.5    | `evaluate.py` — `THRESH = 0.5`                        |
| CLAHE clip     | 2.0    | `train.py` / `evaluate.py` — `OrganoidDataset`        |
| CLAHE tile     | 8 × 8  | `train.py` / `evaluate.py` — `OrganoidDataset`        |
| Border band    | 23 %   | `RemoveBrightBorderFlood` — `band_frac=0.23`          |
| Min border px  | 500    | `RemoveBrightBorderFlood` — `min_remove_px=500`       |

---

## Dependencies

```
torch
torchvision
timm
opencv-python
Pillow
numpy
pandas
scikit-learn
tqdm
matplotlib
seaborn
openpyxl
```
