# Cardiac Organoid Chamber-Aware Pipeline

Deep learning pipeline for the analysis of cardiac organoid microscopy images. The project has three components that are trained/used separately but combine into a single inference flow:

1. **Classification** — hybrid ResNet50 + DeiT3 (ViT) model that classifies CH3 images as *Chamber Forming* or *Chamber Nonforming*.
2. **Virtual staining (GAN)** — U-Net with CBAM attention + PatchGAN discriminator that synthesizes the CH1/CH2 fluorescence signals from the CH3 brightfield image.
3. **Inference (Chamber-Aware Framework)** — combines the two models above into a two-stage pipeline: classifies each image, then synthesizes the fluorescence channels according to the predicted class.

---

## Pipeline overview

<p align="center">
  <img src="assets/pipeline_overview.png" alt="Pipeline overview" width="80%">
</p>

---

## Repository structure

```
.
├── assets/
│   └── pipeline_overview.png  # diagram referenced in this README
├── classification/          # Training/evaluation of the hybrid ResNet+ViT classifier
│   ├── train.py
│   └── evaluate.py
├── staining/                 # Training/evaluation of the virtual staining GAN
│   ├── train.py
│   └── evaluate.py
├── Chamber_Aware_Inference/                 # Chamber-Aware Framework (two-stage pipeline)
│   ├── Chamber_Aware_Framework.py
│   ├── Classification.py
│   ├── Staining.py
│   ├── input/                 # CH3 images to process
│   ├── Files/                  # trained model weights
│   └── results/                # generated automatically
├── requirements.txt
└── README.md
```


---

## Installation

Clone the repository:

```bash
git clone https://github.com/Esteebaan23/Intelligent-identification-of-chamber-formation-in-cardiac-organoids.git
cd Intelligent-identification-of-chamber-formation-in-cardiac-organoids
```

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt` combines the dependencies of all three components (PyTorch/timm for classification, TensorFlow for the staining GAN, plus shared utilities).

---

## Data

All three training components expect the same input folder structure:

```
BASE_DIR/
├── Chamber Forming/
│   ├── CH1/*.tif
│   ├── CH2/*.tif
│   └── CH3/*.tif
└── Chamber Nonforming/
    ├── CH1/*.tif
    ├── CH2/*.tif
    └── CH3/*.tif
```

`BASE_DIR` is configured inside each `train.py`/`evaluate.py` (defaults to `/Documents/Organoids`).

---

## Training

### 1. Classifier (`classification/`)

```bash
cd Classification
python train.py      # trains best_model_{CHANNEL}.pth
python evaluate.py   # evaluates on the 20% held-out split
```

- Configurable input channel (`CH1`, `CH2`, `CH3`) in the configuration block of `train.py`/`evaluate.py`.
- Stratified 80/20 split, fixed seed `42`.
- Outputs: `.pth` checkpoint, per-epoch metrics history, classification reports, and Grad-CAM/ViT figures for misclassified images.

### 2. Virtual staining / GAN (`staining/`)

```bash
cd Staining
python train.py      # trains best_model_forming.keras or best_model_nonforming.keras
python evaluate.py
```

- Trained **once per class** (`CLASS_TO_TRAIN = "forming"` or `"nonforming"`) and **once per target channel** (`TARGET_TAG = "CH1"` or `"CH2"`) — up to 4 runs total to cover both classes and both fluorescence channels.
- Input is always CH3 (brightfield); target is CH1 (green) or CH2 (red).
- Outputs: `.keras` checkpoint, PSNR/SSIM metrics, and input/target/generated images from the test set.

---

## Inference (`inference/`)

The Chamber-Aware Framework runs both stages in sequence on new images.

**Before running it**, place the already-trained checkpoints in `inference/Files/`:

```
Files/
├── best_model_CH3.pth
├── CH1_ResUNet_forming_model.keras
├── CH1_ResUNet_nonforming_model.keras
├── CH2_ResUNet_forming_model.keras
└── CH2_ResUNet_nonforming_model.keras
```

And place the CH3 images to process in `inference/input/` (or `inference/input/CH3/`). Accepted formats: `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`; the filename must contain `CH3`.

Run the full pipeline:

```bash
cd Chamber_Aware_Inference
python Chamber_Aware_Framework.py
```

This runs `Classification.py` first (classifies each image and produces `results/results.xlsx`), then `Staining.py` (reads that Excel file and synthesizes CH1/CH2 based on the predicted class).

Each stage can also be run independently, but **Staining.py depends on the Excel file produced by Classification.py**, so Classification must run first:

```bash
python Classification.py
python Staining.py
```

### Inference outputs

For each image, under `results/{Chamber Forming | Chamber Nonforming}/{image_name}/`:

- `original.png`, `cleaned.png`, `gradcam.png`, `vit.png`, `combined.png`, `composite5_pred.png` (classification stage)
- `{name}_ResUNet_PRED_CH1.png`, `{name}_ResUNet_PRED_CH2.png`, `{name}_ResUNet_OVERLAY.png`, `{name}_PANEL.png` (staining stage)

Global summaries: `results/results.xlsx` and `results/colorization_results.xlsx`.

---

## Architecture summary

| Component        | Architecture                                                                 |
|-------------------|--------------------------------------------------------------------------------|
| Classifier        | ResNet50 (2048-d) + DeiT3-Base/16 (768-d) → concat → Linear(2816→1) + Sigmoid  |
| Staining (GAN)    | U-Net with CBAM + residual bottleneck (generator) + PatchGAN (discriminator)   |

Detailed documentation for each component (hyperparameters, CLAHE/RemoveBrightBorderFlood preprocessing, image formats) is available in the `classification/` and `staining/` READMEs respectively.

