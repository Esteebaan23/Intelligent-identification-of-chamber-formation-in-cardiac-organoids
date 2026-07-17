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
    "Chamber Forming": {
        "CH1": "best_model_forming_CH1.keras",
        "CH2": "best_model_forming_CH2.keras",
    },
    "Chamber Nonforming": {
        "CH1": "best_model_nonforming_CH1.keras",
        "CH2": "best_model_nonforming_CH2.keras",
    },
}

CHANNEL_TAG = "CH3"
TARGET_CHANNELS = ["CH1", "CH2"]

IMG_SIZE = 256
CONTRAST_FACTOR = 0.5
SEED = 42

GENERATED_DIR = os.path.join(OUT_ROOT, "generated")

np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs(OUT_ROOT, exist_ok=True)
for _channel in TARGET_CHANNELS:
    os.makedirs(os.path.join(GENERATED_DIR, _channel), exist_ok=True)

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
    for class_name, channel_paths in MODEL_PATHS.items():
        for channel, model_path in channel_paths.items():
            if os.path.exists(model_path):
                print(f"Loading generator for '{class_name}' / {channel}: {model_path}")
                generators[(class_name, channel)] = tf.keras.models.load_model(
                    model_path, custom_objects=CUSTOM_OBJECTS
                )
            else:
                print(f"[!] Model not found for '{class_name}' / {channel}: {model_path} "
                      f"(this channel will be skipped for images of this class)")

    rows = []
    missing_paths = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Staining"):
        class_name = row.pred_name

        if not os.path.exists(row.input_path):
            print(f"[!] Skipping {row.file_name}: input file not found at {row.input_path} "
                  f"(results.xlsx may be stale — try re-running the classification pipeline)")
            missing_paths.append(row.input_path)
            continue

        gray_img, (orig_w, orig_h) = load_and_preprocess(row.input_path)
        gray_batch = tf.expand_dims(gray_img, axis=0)
        basename = os.path.splitext(row.file_name)[0]

        result_row = {
            "file_name": row.file_name,
            "input_path": row.input_path,
            "pred_name": class_name,
        }

        for channel in TARGET_CHANNELS:
            generator = generators.get((class_name, channel))
            if generator is None:
                print(f"[!] Skipping {channel} for {row.file_name}: no generator available for class '{class_name}'")
                result_row[f"generated_{channel}_png"] = None
                continue

            generated = generator(gray_batch, training=False)[0]
            generated_resized = tf.image.resize(generated, [orig_h, orig_w], method='bicubic')

            output_basename = basename.replace(CHANNEL_TAG, channel)
            collected_png = os.path.join(GENERATED_DIR, channel, f"{output_basename}.png")
            save_image_array(generated_resized, collected_png)
            result_row[f"generated_{channel}_png"] = collected_png

        rows.append(result_row)


    out_df = pd.DataFrame(rows)
    xlsx_path = os.path.join(OUT_ROOT, "staining_results.xlsx")
    out_df.to_excel(xlsx_path, index=False)

    print(f"\nStaining done. Processed {len(out_df)} / {len(df)} image(s).")
    print(f"All generated images collected in: {GENERATED_DIR} (one subfolder per channel)")
    if missing_paths:
        print(f"[!] {len(missing_paths)} image(s) were skipped because their input file was missing. "
              f"Re-run the classification pipeline to refresh results.xlsx if filenames changed.")
    print(f"Excel: {xlsx_path}")

if __name__ == "__main__":
    main()