# GAN Organoid Fluorescence Generator

Image-to-image translation model that predicts a fluorescent chamber-formation
stain from label-free brightfield microscopy images, using a U-Net
generator with CBAM attention and a PatchGAN discriminator.

Two separate generators are trained (one per class, **Chamber Forming** and
**Chamber Nonforming**).

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

`BASE_DIR` defaults to `/Documents/Organoids`. Change it in the configuration
block at the bottom of `train.py` and the constants block at the top of
`evaluate.py`.

---

## Image specifications

| Property  | Requirement                                                                 |
|-----------|------------------------------------------------------------------------------|
| Format    | `.tif` (TIFF)                                                                |
| Color     | RGB or grayscale (grayscale is stacked to 3ch)                              |
| Size      | Any — resized to **256 × 256** before training/inference                    |
| Bit depth | 8-bit recommended; 16-bit will be read by PIL                               |
| Naming    | The image filenames are used to match the input (CH3) and target (CH1/CH2) images. The target file must have the same filename as its CH3 counterpart with `CH3` swapped for the target channel (e.g. `Image_XY35_CH3.tif` → `Image_XY35_CH1.tif`), so pairs can be matched. |

---

## Switching class / target channel

Training and evaluation each process **one class per run**, controlled by a
single variable. The input channel is always CH3, but the **target**
channel is configurable: CH1 for green fluorescence, CH2 for red.

### In `train.py`

Find the configuration block near the bottom of the file:

```python
CLASS_TO_TRAIN = "forming"  # "forming" or "nonforming"

CHANNEL_TAG = "CH3"       # grayscale / brightfield input channel subfolder
TARGET_TAG  = "CH1"       # fluorescent target channel subfolder — "CH1" (green) or "CH2" (red)
```

Set `CLASS_TO_TRAIN` to `"forming"` or `"nonforming"`, then run the script.
Run it twice (once per value) to train both models.

Set `TARGET_TAG` to `"CH1"` to train a generator that predicts green
fluorescence, or `"CH2"` to predict red fluorescence.

### In `evaluate.py`

Find the constants block near the top of the file:

```python
CLASS_TO_EVAL = "forming"  # "forming" or "nonforming"

CHANNEL_TAG = "CH3"
TARGET_TAG  = "CH1"        # must match the TARGET_TAG used to train the checkpoint being evaluated
```

Set `CLASS_TO_EVAL` to match the checkpoint you want to evaluate, and make
sure `TARGET_TAG` matches whichever channel that checkpoint was trained
against.

---

## Train / validation split

- Split: **80% train / 20% test**, per class.
- Fixed seed `42` everywhere — the split is always identical across runs.
- `evaluate.py` uses the same seed and split ratio as
  `train.py`, so it always evaluates on the exact 20% held-out set used
  during that class's training.

---

## Running training

```bash
python train.py
```

Outputs:
- `best_model_forming.keras` / `best_model_nonforming.keras` — best generator
  checkpoint for the trained class (saved whenever test SSIM improves)

---

## Running evaluation

```bash
python evaluate.py
```

Outputs (inside `val_staining_reports/`):
- `results_<class>.xlsx` — per-image PSNR/SSIM on the held-out test set
- `test_set_<class>/input/`, `target/`, `generated/` — every held-out test
  image (brightfield input, real stain target, generated stain), each
  resized back to that image's original dimensions and saved under its
  original filename

---

## Model architecture

**Generator** — U-Net with CBAM attention and a residual bottleneck:

```
Input brightfield (256 × 256 × 3)
        │
Encoder:  64 → 128 → 256 → 512 → 512
          (strided Conv2D + InstanceNorm + LeakyReLU)
        │
Bottleneck: Conv2D(512, dropout) → CBAM → Residual block(512)
        │
Decoder:  512 → 512 → 256 → 128 → 64
          (transposed Conv2D + InstanceNorm + ReLU + skip concat;
           CBAM applied on the top 3 decoder stages)
        │
Conv2DTranspose(3) + Sigmoid
        │
Generated stain (256 × 256 × 3)
```

**Discriminator** — PatchGAN:

```
[Input, Target or Generated] (256×256×3 each) → Concatenate
        │
Conv(64) → Conv(128) → Conv(256) → Conv(512)
(strided, LeakyReLU)
        │
Conv(1) → patch-wise real/fake logits
```

**Losses**:
- Generator: adversarial BCE (from logits) + `100.0 × L1` reconstruction loss
  against the real target
- Discriminator: average of real/fake BCE (from logits)

---

## Hyperparameters

| Parameter                | Value              | Location                                              |
|---------------------------|--------------------|--------------------------------------------------------|
| Epochs (forming)          | 300                | `train.py` — `FORMING_EPOCHS`                     |
| Epochs (nonforming)       | 300                | `train.py` — `NONFORMING_EPOCHS`                  |
| Batch size                | 8                  | `train.py` — `BATCH_SIZE`                         |
| Generator optimizer       | Adam (lr=2e-4, β1=0.5) | `train.py` — `gen_opt`                        |
| Discriminator optimizer   | Adam (lr=2e-4, β1=0.5) | `train.py` — `disc_opt`                       |
| L1 loss weight            | 100.0              | `train.py` — `generator_loss`                      |
| Image size                | 256 × 256          | `train.py` / `evaluate.py` — `IMG_SIZE`       |
| Contrast factor           | 0.5                | `train.py` / `evaluate.py` — `CONTRAST_FACTOR`|
| Test split                | 20%                | `train.py` / `evaluate.py` — `TEST_SIZE`      |
| Split/seed                | 42                 | `train.py` / `evaluate.py` — `SEED`           |

---

## Dependencies

```
tensorflow
numpy
pandas
Pillow
scikit-learn
matplotlib
openpyxl
```
