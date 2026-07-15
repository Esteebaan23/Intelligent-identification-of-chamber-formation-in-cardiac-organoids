import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

CLASSIFICATION_RESULTS_XLSX = os.path.join("results", "results.xlsx")
OUT_ROOT = "results"

MODEL_PATHS = {
    "Chamber Forming": "best_model_forming.keras",
    "Chamber Nonforming": "best_model_nonforming.keras",
}

CHANNEL_TAG = "CH3"
TARGET_TAG = "CH1"

IMG_SIZE = 256
CONTRAST_FACTOR = 0.5
SEED = 42

GENERATED_DIR = os.path.join(OUT_ROOT, "generated")

np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)

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

def find_matching_target(input_path, channel_tag=CHANNEL_TAG, target_tag=TARGET_TAG):
    input_dir = os.path.dirname(input_path)
    parent_dir = os.path.dirname(input_dir)
    channel_dirname = os.path.basename(input_dir)

    if channel_dirname != channel_tag:
        return None

    target_dir = os.path.join(parent_dir, target_tag)
    if not os.path.isdir(target_dir):
        return None

    filename = os.path.basename(input_path)

    target_path = os.path.join(target_dir, filename)
    if os.path.exists(target_path):
        return target_path

    swapped_filename = filename.replace(channel_tag, target_tag)
    target_path = os.path.join(target_dir, swapped_filename)
    if os.path.exists(target_path):
        return target_path

    return None

def load_and_preprocess(path, img_size=IMG_SIZE, contrast_factor=CONTRAST_FACTOR):
    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size

    arr = np.array(img)
    arr = tf.image.resize(arr, [img_size, img_size])
    arr = tf.cast(arr, tf.float32) / 255.0
    arr = tf.image.adjust_contrast(arr, contrast_factor)

    return arr, (orig_w, orig_h)

def save_image_array(img_array, save_path):
    arr = np.clip(img_array.numpy() if hasattr(img_array, "numpy") else img_array, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(save_path)

def main():
    if not os.path.exists(CLASSIFICATION_RESULTS_XLSX):
        raise FileNotFoundError(
            f"Could not find {CLASSIFICATION_RESULTS_XLSX}. Run the classification "
            "pipeline first (it produces results.xlsx with each image's predicted "
            "class), then run this script."
        )

    df = pd.read_excel(CLASSIFICATION_RESULTS_XLSX)

    required_cols = {"file_name", "input_path", "pred_name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{CLASSIFICATION_RESULTS_XLSX} is missing expected column(s): {missing}")

    generators = {}
    for class_name, model_path in MODEL_PATHS.items():
        if os.path.exists(model_path):
            print(f"Loading generator for '{class_name}': {model_path}")
            generators[class_name] = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
        else:
            print(f"[!] Model not found for '{class_name}': {model_path} (images of this class will be skipped)")

    rows = []
    missing_paths = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Staining"):
        class_name = row.pred_name
        generator = generators.get(class_name)

        if generator is None:
            print(f"[!] Skipping {row.file_name}: no generator available for class '{class_name}'")
            continue

        if not os.path.exists(row.input_path):
            print(f"[!] Skipping {row.file_name}: input file not found at {row.input_path} "
                  f"(results.xlsx may be stale — try re-running the classification pipeline)")
            missing_paths.append(row.input_path)
            continue

        gray_img, (orig_w, orig_h) = load_and_preprocess(row.input_path)
        gray_batch = tf.expand_dims(gray_img, axis=0)
        generated = generator(gray_batch, training=False)[0]

        generated_resized = tf.image.resize(generated, [orig_h, orig_w], method='bicubic')

        basename = os.path.splitext(row.file_name)[0]
        collected_png = os.path.join(GENERATED_DIR, f"{basename}.png")
        save_image_array(generated_resized, collected_png)

        psnr_val, ssim_val = None, None
        target_path = find_matching_target(row.input_path)
        if target_path is not None:
            target_img, _ = load_and_preprocess(target_path)
            psnr_val = float(tf.image.psnr(target_img, generated, max_val=1.0).numpy())
            ssim_val = float(tf.image.ssim(target_img, generated, max_val=1.0).numpy())

        rows.append({
            "file_name": row.file_name,
            "input_path": row.input_path,
            "pred_name": class_name,
            "generated_collected_png": collected_png,
            "target_path": target_path,
            "psnr": psnr_val,
            "ssim": ssim_val,
        })

    out_df = pd.DataFrame(rows)
    xlsx_path = os.path.join(OUT_ROOT, "staining_results.xlsx")
    out_df.to_excel(xlsx_path, index=False)

    print(f"\nStaining done. Stained {len(out_df)} / {len(df)} image(s).")
    print(f"All generated images collected in: {GENERATED_DIR}")
    if len(out_df):
        with_metrics = out_df["psnr"].notna().sum()
        print(f"{with_metrics} / {len(out_df)} image(s) had a matching ground-truth target "
              f"(PSNR/SSIM reported); the rest are unlabeled production images.")
    if missing_paths:
        print(f"[!] {len(missing_paths)} image(s) were skipped because their input file was missing. "
              f"Re-run the classification pipeline to refresh results.xlsx if filenames changed.")
    print(f"Excel: {xlsx_path}")

if __name__ == "__main__":
    main()