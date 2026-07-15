import os
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# ============================================================
# Config
# ============================================================
# Mirrors Staining.py's config. Only ONE class's model is evaluated per run
# (matching Staining.py's single-class-per-run training), controlled by
# CLASS_TO_EVAL below.

BASE_DIR = "/Documents/Organoids"

FORMING_LABEL    = "Chamber Forming"
NONFORMING_LABEL = "Chamber Nonforming"

CHANNEL_TAG = "CH3"   # grayscale / brightfield input channel subfolder
TARGET_TAG  = "CH1"   # fluorescent target ("stain") channel subfolder

IMG_SIZE = 256
CONTRAST_FACTOR = 0.5   # must match load_pair's contrast_factor in Staining.py

SEED = 42
TEST_SIZE = 0.2         # must match run_training_phase's test_size in Staining.py

# Which class's trained generator to evaluate: "forming" or "nonforming".
CLASS_TO_EVAL = "forming"

MODEL_PATHS = {
    "forming": "best_model_forming.keras",
    "nonforming": "best_model_nonforming.keras",
}

OUT_DIR = "val_staining_reports"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Custom layers
# ============================================================
# Duplicated from Staining.py so the saved .keras model (which references
# these custom layers) can be loaded here without importing Staining.py.

@register_keras_serializable()
class ReduceMeanLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=-1, keepdims=True)


@register_keras_serializable()
class ReduceMaxLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_max(inputs, axis=-1, keepdims=True)


@register_keras_serializable()
class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(shape=(input_shape[-1],), initializer='ones', trainable=True)
        self.offset = self.add_weight(shape=(input_shape[-1],), initializer='zeros', trainable=True)

    def call(self, x):
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        return self.scale * (x - mean) / tf.sqrt(var + self.epsilon) + self.offset

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


CUSTOM_OBJECTS = {
    "ReduceMeanLayer": ReduceMeanLayer,
    "ReduceMaxLayer": ReduceMaxLayer,
    "InstanceNormalization": InstanceNormalization,
}

# ============================================================
# Data gathering (identical to Staining.py's load_image_pairs)
# ============================================================
#
# Folder layout:
#   BASE_DIR/
#     Chamber Forming/
#       CH3/   *.tif   (grayscale / brightfield input, e.g. Image_XY35_CH3(4).tif)
#       CH1/   *.tif   (fluorescent target / "stain", e.g. Image_XY35_CH1(4).tif)
#     Chamber Nonforming/
#       CH3/   *.tif
#       CH1/   *.tif

def load_image_pairs(base_dir, class_label, channel_tag='CH3', target_tag='CH1'):
    """
    Gathers (input, target) image path pairs for one class folder, using
    the same base_dir/<class_label>/<channel>/*.tif structure and glob-based
    listing as Classification.py's load_image_paths(), and the same matching
    logic used by Staining.py's load_image_pairs():
      1. Try an exact basename match in target_dir first.
      2. Fall back to swapping channel_tag -> target_tag in the filename
         (e.g. Image_XY35_CH3(4).tif -> Image_XY35_CH1(4).tif).
    """
    input_dir = os.path.join(base_dir, class_label, channel_tag)
    target_dir = os.path.join(base_dir, class_label, target_tag)

    if not os.path.isdir(input_dir):
        print(f"[!] Input directory does not exist: {input_dir}")
    if not os.path.isdir(target_dir):
        print(f"[!] Target directory does not exist: {target_dir}")

    input_files = sorted(
        p for p in glob.glob(os.path.join(input_dir, "*"))
        if p.lower().endswith(('.tif', '.tiff'))
    )

    print(f"[{class_label}] Found {len(input_files)} file(s) in {input_dir}")

    matched_input_files = []
    matched_target_files = []

    for input_path in input_files:
        filename = os.path.basename(input_path)

        target_path = os.path.join(target_dir, filename)
        if not os.path.exists(target_path):
            swapped_filename = filename.replace(channel_tag, target_tag)
            target_path = os.path.join(target_dir, swapped_filename)

        if os.path.exists(target_path):
            matched_input_files.append(input_path)
            matched_target_files.append(target_path)
        else:
            print(f"[!] No match found for {input_path} -> expected {target_path}")

    print(f"[{class_label}] Matched {len(matched_input_files)} pair(s)")

    return matched_input_files, matched_target_files


def load_pair_arrays(gray_path, green_path, img_size=IMG_SIZE, contrast_factor=CONTRAST_FACTOR):
    """
    Loads and preprocesses one (input, target) pair, identical to
    Staining.py's load_pair, but takes plain Python str paths (no
    tf.py_function wrapping) since evaluation runs eagerly per-image.
    """
    gray_image = Image.open(gray_path).convert("RGB")
    green_image = Image.open(green_path).convert("RGB")

    gray_image = np.array(gray_image)
    green_image = np.array(green_image)

    gray_image = tf.image.resize(gray_image, [img_size, img_size])
    green_image = tf.image.resize(green_image, [img_size, img_size])

    gray_image = tf.cast(gray_image, tf.float32) / 255.0
    green_image = tf.cast(green_image, tf.float32) / 255.0

    gray_image = tf.image.adjust_contrast(gray_image, contrast_factor)
    green_image = tf.image.adjust_contrast(green_image, contrast_factor)

    return gray_image, green_image


# ============================================================
# Image export
# ============================================================

def save_image_array(img_array, save_path):
    """Saves a float [0,1] HxWx3 tensor/array as a standard image file."""
    arr = np.clip(img_array.numpy() if hasattr(img_array, "numpy") else img_array, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(save_path)


# ============================================================
# Evaluation
# ============================================================

def main():
    if CLASS_TO_EVAL not in ("forming", "nonforming"):
        raise ValueError(
            f"CLASS_TO_EVAL must be 'forming' or 'nonforming', got {CLASS_TO_EVAL!r}"
        )

    class_label = FORMING_LABEL if CLASS_TO_EVAL == "forming" else NONFORMING_LABEL
    model_path = MODEL_PATHS[CLASS_TO_EVAL]

    # 1) Gather pairs the same way Staining.py does, then re-create the
    #    same train/test split (same order + same random_state) so we
    #    evaluate on the exact held-out test set from training.
    gray_files, green_files = load_image_pairs(BASE_DIR, class_label, CHANNEL_TAG, TARGET_TAG)

    print(f"{CLASS_TO_EVAL.capitalize()} pairs: {len(gray_files)}")
    if len(gray_files) == 0:
        raise RuntimeError(
            "No matched image pairs were found. Check BASE_DIR, "
            "FORMING_LABEL/NONFORMING_LABEL, and CHANNEL_TAG/TARGET_TAG."
        )

    labels = [1 if CLASS_TO_EVAL == "forming" else 0] * len(gray_files)

    _, gray_test, _, green_test, _, _ = train_test_split(
        gray_files, green_files, labels, test_size=TEST_SIZE, random_state=SEED, shuffle=True
    )

    print(f"Evaluating on {len(gray_test)} held-out test pair(s)")

    # 2) Load the trained generator
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train it first with Staining.py.")

    generator = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)

    # 3) Run inference + compute per-image PSNR/SSIM, saving every test-set
    #    image (input / target / generated) to disk as we go.
    test_set_dir = os.path.join(OUT_DIR, f"test_set_{CLASS_TO_EVAL}")
    input_dir = os.path.join(test_set_dir, "input")
    target_dir = os.path.join(test_set_dir, "target")
    generated_dir = os.path.join(test_set_dir, "generated")
    for d in (input_dir, target_dir, generated_dir):
        os.makedirs(d, exist_ok=True)

    results = []
    for gray_path, green_path in zip(gray_test, green_test):
        gray_img, green_img = load_pair_arrays(gray_path, green_path)

        gray_batch = tf.expand_dims(gray_img, axis=0)
        generated = generator(gray_batch, training=False)[0]

        psnr = float(tf.image.psnr(green_img, generated, max_val=1.0).numpy())
        ssim = float(tf.image.ssim(green_img, generated, max_val=1.0).numpy())

        # Resize input, target, and generated images all back up to the
        # original input file's dimensions before saving (metrics above are
        # still computed at the model's native IMG_SIZE x IMG_SIZE resolution).
        orig_w, orig_h = Image.open(gray_path).size
        gray_resized = tf.image.resize(gray_img, [orig_h, orig_w], method='bicubic')
        green_resized = tf.image.resize(green_img, [orig_h, orig_w], method='bicubic')
        generated_resized = tf.image.resize(generated, [orig_h, orig_w], method='bicubic')

        basename = os.path.splitext(os.path.basename(gray_path))[0]
        save_image_array(gray_resized, os.path.join(input_dir, f"{basename}.png"))
        save_image_array(green_resized, os.path.join(target_dir, f"{basename}.png"))
        save_image_array(generated_resized, os.path.join(generated_dir, f"{basename}.png"))

        results.append({
            "input_path": gray_path,
            "target_path": green_path,
            "psnr": psnr,
            "ssim": ssim,
        })

    print(f"✅ Saved {len(results)} test-set image(s) to {test_set_dir}")

    df = pd.DataFrame(results)

    # 4) Aggregate metrics
    print(f"\nMean PSNR: {df['psnr'].mean():.4f}")
    print(f"Mean SSIM: {df['ssim'].mean():.4f}")

    xlsx_path = os.path.join(OUT_DIR, f"results_{CLASS_TO_EVAL}.xlsx")
    df.to_excel(xlsx_path, index=False)
    print(f"\n✅ Saved: {xlsx_path}")


if __name__ == "__main__":
    main()