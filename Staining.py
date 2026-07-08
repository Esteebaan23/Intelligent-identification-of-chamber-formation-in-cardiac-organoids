import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable

# ==== Config ====
IMG_SIZE    = 256                           # model inference resolution
EXCEL_PATH  = "results/results.xlsx"        # CH3 classification output
OUTPUT_ROOT = "results"
EPS         = 1e-5

# ---------------------- Custom Keras layers ----------------------
@register_keras_serializable()
class ReduceMeanLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=-1, keepdims=True)

@register_keras_serializable()
class ReduceMaxLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_max(inputs, axis=-1, keepdims=True)

class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
    def build(self, input_shape):
        self.scale  = self.add_weight(shape=(input_shape[-1],), initializer='ones',  trainable=True)
        self.offset = self.add_weight(shape=(input_shape[-1],), initializer='zeros', trainable=True)
    def call(self, x):
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        return self.scale * (x - mean) / tf.sqrt(var + self.epsilon) + self.offset

def cbam_block(x, ratio=8):
    channel  = x.shape[-1]
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling2D()(x)
    shared   = tf.keras.layers.Dense(channel // ratio, activation='relu')
    avg_out  = tf.keras.layers.Dense(channel)(shared(avg_pool))
    max_out  = tf.keras.layers.Dense(channel)(shared(max_pool))
    ca = tf.keras.layers.Activation('sigmoid')(tf.keras.layers.Add()([avg_out, max_out]))
    ca = tf.keras.layers.Reshape((1, 1, channel))(ca)
    x  = tf.keras.layers.Multiply()([x, ca])
    avg_pool = ReduceMeanLayer()(x)
    max_pool = ReduceMaxLayer()(x)
    sa = tf.keras.layers.Conv2D(1, 7, padding='same', activation='sigmoid')(
        tf.keras.layers.Concatenate()([avg_pool, max_pool]))
    return tf.keras.layers.Multiply()([x, sa])

# ---------------------- Helpers ----------------------
def as_gray_uint8(arr):
    if arr.ndim == 2:
        g = arr
    else:
        g = 0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]
    return np.clip(g, 0, 255).astype(np.uint8)

def predict_from_ch3(gen, ch3_path, orig_size):
    """Run ResUNet inference on a CH3 image and return output at original size (lossless PNG quality)."""
    # Downscale to model input size
    img = Image.open(ch3_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    pred = gen(tf.expand_dims(arr, 0), training=False)[0].numpy()  # [H,W,C] in [0,1]

    # Upscale back to original size using LANCZOS (highest quality, lossless output)
    pred_u8  = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
    pred_pil = Image.fromarray(pred_u8).resize(orig_size, Image.LANCZOS)
    return np.array(pred_pil)

def make_overlay(base_rgb, ch1_gray=None, ch2_gray=None,
                 alpha=0.45,       # blend weight (0 = no overlay, 1 = full overlay)
                 green_gain=1.8,   # CH1 signal intensity (green channel)
                 red_gain=1.8,     # CH2 signal intensity (red channel)
                 gamma=0.9,        # gamma correction (lower = brighter)
                 contrast=1.05):   # global contrast multiplier
    base = base_rgb.astype(np.float32)
    mask = np.zeros_like(base, dtype=np.float32)

    if ch1_gray is not None:
        g = as_gray_uint8(ch1_gray).astype(np.float32)
        g = np.power(g / 255.0, gamma) * 255.0
        mask[..., 1] = np.clip(g * green_gain, 0, 255)

    if ch2_gray is not None:
        r = as_gray_uint8(ch2_gray).astype(np.float32)
        r = np.power(r / 255.0, gamma) * 255.0
        mask[..., 0] = np.clip(r * red_gain, 0, 255)

    out = base.copy()
    active = (mask.sum(axis=-1) > 5)  # blend only where signal exists
    out[active] = (1 - alpha) * base[active] + alpha * mask[active]
    return np.clip(out * contrast, 0, 255).astype(np.uint8)

def save_png(img_u8, path_no_ext):
    path = f"{path_no_ext}.png"
    Image.fromarray(img_u8).save(path)
    return path

# ---------------------- Model loading (ResUNet only) ----------------------
CLASS_KEYS = ["Chamber Forming", "Chamber Nonforming"]

def load_resunet_models():
    """
    Expected file layout:
      Files/CH1_ResUNet_forming_model.keras
      Files/CH1_ResUNet_nonforming_model.keras
      Files/CH2_ResUNet_forming_model.keras
      Files/CH2_ResUNet_nonforming_model.keras
    """
    base_dir = "Files"
    models   = {}
    for cls in CLASS_KEYS:
        cls_key       = "forming" if "Forming" in cls else "nonforming"
        models[cls]   = {}
        for ch in ["CH1", "CH2"]:
            model_path = os.path.join(base_dir, f"{ch}_ResUNet_{cls_key}_model.keras")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            models[cls][ch] = load_model(
                model_path,
                custom_objects={
                    "InstanceNormalization": InstanceNormalization,
                    "cbam_block": cbam_block,
                }
            )
            print(f"Loaded: {model_path}")
    return models

# ---------------------- Main ----------------------
def main():
    df     = pd.read_excel(EXCEL_PATH)
    models = load_resunet_models()
    print()
    metrics = []

    for _, row in df.iterrows():
        ch3_path  = row["input_path"]
        label     = row["pred_name"]
        out_dir   = row["output_folder"]

        if not os.path.exists(ch3_path):
            print(f"Skipping (not found): {ch3_path}")
            continue

        ch3_img   = np.array(Image.open(ch3_path).convert("RGB"))
        orig_size = (ch3_img.shape[1], ch3_img.shape[0])  # (width, height) for PIL
        base_name = os.path.splitext(os.path.basename(ch3_path))[0]

        print(f"Processing {base_name} -> {label}")

        gen_ch1  = models[label]["CH1"]
        gen_ch2  = models[label]["CH2"]

        pred_ch1 = predict_from_ch3(gen_ch1, ch3_path, orig_size)
        pred_ch2 = predict_from_ch3(gen_ch2, ch3_path, orig_size)
        overlay  = make_overlay(ch3_img, pred_ch1, pred_ch2)

        ch1_path     = save_png(pred_ch1, os.path.join(out_dir, f"{base_name}_ResUNet_PRED_CH1"))
        ch2_path     = save_png(pred_ch2, os.path.join(out_dir, f"{base_name}_ResUNet_PRED_CH2"))
        overlay_path = save_png(overlay,  os.path.join(out_dir, f"{base_name}_ResUNet_OVERLAY"))

        # Side-by-side panel: original | pred_CH1 | pred_CH2 | overlay
        H, W    = ch3_img.shape[:2]
        panel   = np.ones((H, 4*W, 3), dtype=np.uint8) * 255
        for j, im in enumerate([ch3_img, pred_ch1, pred_ch2, overlay]):
            panel[:, j*W:(j+1)*W] = im
        panel_path = save_png(panel, os.path.join(out_dir, f"{base_name}_PANEL"))

        metrics.append({
            "image":        base_name,
            "class":        label,
            "folder":       out_dir,
            "pred_ch1":     ch1_path,
            "pred_ch2":     ch2_path,
            "overlay":      overlay_path,
            "panel":        panel_path,
        })

    out_excel = os.path.join(OUTPUT_ROOT, "colorization_results.xlsx")
    pd.DataFrame(metrics).to_excel(out_excel, index=False)
    print(f"\nColorization done. Results in: {OUTPUT_ROOT}")
    print(f"Excel: {out_excel}")

if __name__ == "__main__":
    main()
