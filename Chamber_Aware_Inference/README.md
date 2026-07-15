# Chamber-Aware Framework

Two-stage inference system for organoid microscopy images:

1. **Classification** (`Classification.py`) — classifies each CH3 image as *Chamber Forming* or *Chamber Nonforming* using a hybrid ResNet50 + DeiT3 model.
2. **Virtual Staining** (`Staining.py`) — takes the classified images and synthesizes the corresponding CH1 and CH2 channels from CH3 using a ResUNet model per class.

Run both stages in sequence with `Chamber_Aware_Framework.py`.

---

## Directory structure

```
System/
├── Chamber_Aware_Framework.py            # entry point — runs both stages
├── Classification.py
├── Staining.py
├── README.md
├── input/               # place CH3 images here before running
│   ├── *.tif  (files whose name contains "CH3")
│   └── CH3/   (optional subfolder — also scanned automatically)
│       └── *.tif
├── Files/               # model weights — must exist before running
│   ├── best_model_CH3.pth
│   ├── CH1_ResUNet_forming_model.keras
│   ├── CH1_ResUNet_nonforming_model.keras
│   ├── CH2_ResUNet_forming_model.keras
│   └── CH2_ResUNet_nonforming_model.keras
└── results/             # created automatically on first run
```

---

## Input requirements

| Property     | Requirement                                                    |
|--------------|----------------------------------------------------------------|
| Channel      | CH3 only — filename must contain `CH3` (case-insensitive)     |
| Format       | `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`                       |
| Size         | Any — classification resizes to 224×224; colorization to 256×256 internally, then upscaled back |
| Location     | `input/` root or `input/CH3/` subfolder                       |

---

## Model files required (`Files/`)

| File                                    | Used by            |
|-----------------------------------------|--------------------|
| `best_model_CH3.pth`                    | Classification     |
| `CH1_ResUNet_forming_model.keras`       | Staining       |
| `CH1_ResUNet_nonforming_model.keras`    | Staining       |
| `CH2_ResUNet_forming_model.keras`       | Staining       |
| `CH2_ResUNet_nonforming_model.keras`    | Staining       |

---

## Running the pipeline

```bash
python Chamber_Aware_Framework.py
```

This runs Classification first, then Colorization. Both stages must complete without error for the pipeline to finish.

To run a single stage independently:

```bash
python Classification.py
python Staining.py
```

> **Note:** `Staining.py` reads `results/results.xlsx` produced by `Classification.py`, so Classification must run first.

---

## Stage 1 — Classification outputs

Per image, inside `results/{Chamber Forming | Chamber Nonforming}/{image_name}/`:

| File                         | Description                                              |
|------------------------------|----------------------------------------------------------|
| `original.png`               | Original input image (lossless copy)                     |
| `cleaned.png`                | After CLAHE + RemoveBrightBorderFlood preprocessing      |
| `gradcam.png`                | Grad-CAM heatmap overlaid on cleaned image               |
| `vit.png`                    | ViT attention map overlaid on cleaned image              |
| `combined.png`               | Average of Grad-CAM + ViT attention overlaid             |
| `composite5_pred.png`        | 5-panel figure: Original · Cleaned · Grad-CAM · ViT · Combined |

Global summary: `results/results.xlsx`

| Column              | Description                              |
|---------------------|------------------------------------------|
| `file_name`         | Original filename                        |
| `input_path`        | Full path to the input image             |
| `pred_id`           | 0 = Chamber Forming, 1 = Nonforming      |
| `pred_name`         | Human-readable class label               |
| `prob_nonforming`   | Sigmoid probability of class 1           |
| `threshold`         | Classification threshold (default 0.5)   |
| `output_folder`     | Path to this image's result subfolder    |
| `*_png`             | Paths to each output image               |

---

## Stage 2 — Colorization outputs

Per image, appended into the same subfolder created by Classification:

| File                               | Description                                       |
|------------------------------------|---------------------------------------------------|
| `{name}_ResUNet_PRED_CH1.png`      | Synthesized CH1 channel at original input size    |
| `{name}_ResUNet_PRED_CH2.png`      | Synthesized CH2 channel at original input size    |
| `{name}_ResUNet_OVERLAY.png`       | CH3 base with CH1 (green) and CH2 (red) overlaid  |
| `{name}_PANEL.png`                 | Side-by-side panel: Original · CH1 · CH2 · Overlay |

Global summary: `results/colorization_results.xlsx`

> All colorization outputs are saved as **PNG (lossless)** at the **exact pixel dimensions of the input image**. The model runs internally at 256×256 and the result is upscaled back using LANCZOS resampling.

---

## Preprocessing (Classification stage)

Both CLAHE and `RemoveBrightBorderFlood` are applied to every CH3 image before classification, matching the training pipeline exactly.

| Step                      | Detail                                          |
|---------------------------|-------------------------------------------------|
| CLAHE                     | clipLimit=2.0, tileGridSize=8×8                 |
| RemoveBrightBorderFlood   | band=23% of min side, fill=mean, dilate 1 iter  |

---

## Configuration

### `Classification.py`

```python
INPUT_DIR   = "input"                     # folder to scan for CH3 images
MODEL_PATH  = "Files/best_model_CH3.pth"  # classification checkpoint
OUT_ROOT    = "results"                   # output root
THRESH      = 0.5                         # classification threshold
```

### `Staining.py`

```python
IMG_SIZE    = 256                          # model internal inference resolution
EXCEL_PATH  = "results/results.xlsx"       # produced by Classification.py
OUTPUT_ROOT = "results"
```

---

## Dependencies

```
# Classification
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

# Colorization
tensorflow
scipy
openpyxl
```
