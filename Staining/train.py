import os
import glob
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import register_keras_serializable

device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

IMG_SIZE = 256

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

def cbam_block(x, ratio=8):
    channel = x.shape[-1]

    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)
    shared = layers.Dense(channel // ratio, activation='relu')
    avg_out = layers.Dense(channel)(shared(avg_pool))
    max_out = layers.Dense(channel)(shared(max_pool))
    ca = layers.Activation('sigmoid')(layers.Add()([avg_out, max_out]))
    ca = layers.Reshape((1, 1, channel))(ca)
    x = layers.Multiply()([x, ca])

    avg_pool = ReduceMeanLayer()(x)
    max_pool = ReduceMaxLayer()(x)
    sa = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(
        layers.Concatenate()([avg_pool, max_pool]))
    x = layers.Multiply()([x, sa])

    return x

def residual_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.LayerNormalization()(x)
    x = layers.Add()([shortcut, x])
    return layers.ReLU()(x)

def conv_block(x, filters, use_dropout=False):
    x = layers.Conv2D(filters, 4, strides=2, padding='same', use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    if use_dropout:
        x = layers.Dropout(0.5)(x)
    return x

def deconv_block(x, skip, filters, use_dropout=False, use_cbam=False):
    x = layers.Conv2DTranspose(filters, 4, strides=2, padding='same', use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)
    if use_dropout:
        x = layers.Dropout(0.5)(x)
    x = layers.Concatenate()([x, skip])
    if use_cbam:
        x = cbam_block(x)
    return x

def build_generator(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    inputs = layers.Input(shape=input_shape)

    e1 = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
    e1 = InstanceNormalization()(e1)
    e1 = layers.LeakyReLU(0.2)(e1)

    e2 = conv_block(e1, 128)
    e3 = conv_block(e2, 256)
    e4 = conv_block(e3, 512)
    e5 = conv_block(e4, 512)

    b = conv_block(e5, 512, use_dropout=True)
    b = cbam_block(b)
    b = residual_block(b, 512)

    d1 = deconv_block(b, e5, 512, use_dropout=True, use_cbam=True)
    d2 = deconv_block(d1, e4, 512, use_dropout=True, use_cbam=True)
    d3 = deconv_block(d2, e3, 256, use_cbam=True)
    d4 = deconv_block(d3, e2, 128)
    d5 = deconv_block(d4, e1, 64)

    x = layers.Conv2DTranspose(3, 4, strides=2, padding='same',
                                activation='sigmoid', dtype='float32')(d5)

    return models.Model(inputs, x)

def build_discriminator(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    inp = layers.Input(shape=input_shape)
    tar = layers.Input(shape=input_shape)
    x = layers.Concatenate()([inp, tar])

    x = layers.Conv2D(64, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(512, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)

    out = layers.Conv2D(1, 4, strides=1, padding='same', use_bias=True)(x)

    return models.Model([inp, tar], out)

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_out, gen_out, target):
    adv = bce(tf.ones_like(fake_out), fake_out)
    l1 = tf.reduce_mean(tf.abs(target - gen_out))
    return adv + 100.0 * l1

def discriminator_loss(real_out, fake_out):
    real_loss = bce(tf.ones_like(real_out), real_out)
    fake_loss = bce(tf.zeros_like(fake_out), fake_out)
    return (real_loss + fake_loss) * 0.5

def compute_psnr_ssim_on_test(generator, test_ds):
    psnr_metric = tf.metrics.Mean()
    ssim_metric = tf.metrics.Mean()

    for input_gray, target_green, _ in test_ds:
        generated = generator(input_gray, training=False)
        psnr = tf.image.psnr(target_green, generated, max_val=1.0)
        ssim = tf.image.ssim(target_green, generated, max_val=1.0)
        psnr_metric.update_state(psnr)
        ssim_metric.update_state(ssim)

    return psnr_metric.result().numpy(), ssim_metric.result().numpy()

@tf.function
def train_step(input_gray, target_green, gen, discriminator, gen_opt, disc_opt):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_green = gen(input_gray, training=True)

        real_out = discriminator([input_gray, target_green], training=True)
        fake_out = discriminator([input_gray, fake_green], training=True)

        if isinstance(real_out, list):
            real_out = real_out[0]
        if isinstance(fake_out, list):
            fake_out = fake_out[0]

        g_loss = generator_loss(fake_out, fake_green, target_green)
        d_loss = discriminator_loss(real_out, fake_out)

    gen_grad = gen_tape.gradient(g_loss, gen.trainable_variables)
    disc_grad = disc_tape.gradient(d_loss, discriminator.trainable_variables)

    gen_opt.apply_gradients(zip(gen_grad, gen.trainable_variables))
    disc_opt.apply_gradients(zip(disc_grad, discriminator.trainable_variables))

    return g_loss, d_loss

def load_image_pairs(base_dir, class_label, channel_tag='CH3', target_tag='CH1'):
    
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

def load_pair(gray_path, green_path, img_size=IMG_SIZE, contrast_factor=0.5):
    gray_path = gray_path.numpy().decode('utf-8')
    green_path = green_path.numpy().decode('utf-8')

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

def augment(gray, green):
    if tf.random.uniform(()) > 0.5:
        gray = tf.image.flip_left_right(gray)
        green = tf.image.flip_left_right(green)
    if tf.random.uniform(()) > 0.5:
        gray = tf.image.flip_up_down(gray)
        green = tf.image.flip_up_down(green)
    if tf.random.uniform(()) > 0.5:
        gray = tf.image.rot90(gray)
        green = tf.image.rot90(green)
    if tf.random.uniform(()) > 0.5:
        crop_scale = tf.random.uniform((), 0.8, 1.0)
        crop_size = tf.cast(crop_scale * IMG_SIZE, tf.int32)
        gray = tf.image.random_crop(gray, size=[crop_size, crop_size, 3])
        green = tf.image.random_crop(green, size=[crop_size, crop_size, 3])
        gray = tf.image.resize(gray, [IMG_SIZE, IMG_SIZE])
        green = tf.image.resize(green, [IMG_SIZE, IMG_SIZE])
    if tf.random.uniform(()) > 0.5:
        delta = tf.random.uniform((), -0.1, 0.1)
        gray = tf.clip_by_value(gray + delta, 0.0, 1.0)
        green = tf.clip_by_value(green + delta, 0.0, 1.0)
    if tf.random.uniform(()) > 0.5:
        factor = tf.random.uniform((), 0.9, 1.1)
        gray = tf.image.adjust_contrast(gray, factor)
        green = tf.image.adjust_contrast(green, factor)

    return gray, green

def create_dataset(gray_files, green_files, labels, batch_size=8, training=True):
    def wrapped_load_pair(g, gr, l):
        gray_img, green_img = tf.py_function(load_pair, [g, gr], [tf.float32, tf.float32])
        gray_img.set_shape([IMG_SIZE, IMG_SIZE, 3])
        green_img.set_shape([IMG_SIZE, IMG_SIZE, 3])
        if training:
            gray_img, green_img = augment(gray_img, green_img)
        return gray_img, green_img, l

    dataset = tf.data.Dataset.from_tensor_slices((gray_files, green_files, labels))
    dataset = dataset.map(wrapped_load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache().shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def run_training_phase(name, gray_files, green_files, labels, epochs,
                        model_out_path, batch_size=8):
    gray_train, gray_test, green_train, green_test, labels_train, labels_test = train_test_split(
        gray_files, green_files, labels, test_size=0.2, random_state=42, shuffle=True
    )

    train_ds = create_dataset(gray_train, green_train, labels_train, batch_size, training=True)
    test_ds = create_dataset(gray_test, green_test, labels_test, batch_size, training=False)

    gen = build_generator()
    disc = build_discriminator()
    gen_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    disc_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    print(f"=== Training phase: {name} ({len(gray_train)} train / {len(gray_test)} test) ===")
    best_psnr, best_ssim = 0.0, 0.0

    for epoch in range(epochs):
        print(f"[{name}] Epoch {epoch + 1}/{epochs}")
        for input_gray, target_green, _ in train_ds:
            g_loss, d_loss = train_step(input_gray, target_green, gen, disc, gen_opt, disc_opt)
        print(f"  Generator Loss: {g_loss:.4f} | Discriminator Loss: {d_loss:.4f}")

        if epoch % 10 == 0:
            test_psnr, test_ssim = compute_psnr_ssim_on_test(gen, test_ds)
            print(f"  Test PSNR: {test_psnr:.4f} | Test SSIM: {test_ssim:.4f}")
            if test_psnr > best_psnr:
                best_psnr = test_psnr
            if test_ssim > best_ssim:
                best_ssim = test_ssim
                gen.save(model_out_path)
                print(f"  New best model saved -> {model_out_path}")

    return gen

# Configuration (See README.md for details.)

BASE_DIR = "/Documents/Organoids"

FORMING_LABEL    = "Chamber Forming"
NONFORMING_LABEL = "Chamber Nonforming"

CHANNEL_TAG = "CH3"
TARGET_TAG  = "CH1"

NONFORMING_EPOCHS = 300
FORMING_EPOCHS = 300
BATCH_SIZE = 8

NONFORMING_MODEL_OUT = "best_model_nonforming.keras"
FORMING_MODEL_OUT = "best_model_forming.keras"

CLASS_TO_TRAIN = "forming"

def main():
    if CLASS_TO_TRAIN not in ("forming", "nonforming"):
        raise ValueError(
            f"CLASS_TO_TRAIN must be 'forming' or 'nonforming', got {CLASS_TO_TRAIN!r}"
        )

    if CLASS_TO_TRAIN == "forming":
        class_label = FORMING_LABEL
        class_value = 1
        epochs = FORMING_EPOCHS
        model_out_path = FORMING_MODEL_OUT
    else:
        class_label = NONFORMING_LABEL
        class_value = 0
        epochs = NONFORMING_EPOCHS
        model_out_path = NONFORMING_MODEL_OUT

    gray_files, green_files = load_image_pairs(
        BASE_DIR, class_label, CHANNEL_TAG, TARGET_TAG)

    print(f"{CLASS_TO_TRAIN.capitalize()} pairs: {len(gray_files)}")

    if len(gray_files) == 0:
        raise RuntimeError(
            "No matched image pairs were found for the selected class. "
            "Check that BASE_DIR, FORMING_LABEL/NONFORMING_LABEL, and "
            "CHANNEL_TAG/TARGET_TAG point to real subfolders "
            "containing .tif files, e.g. BASE_DIR/'Chamber Forming'/CH3/*.tif "
            "and BASE_DIR/'Chamber Forming'/CH1/*.tif. See the [!] messages "
            "printed above for the exact paths that were checked."
        )

    labels = [class_value] * len(gray_files)

    run_training_phase(
        CLASS_TO_TRAIN, gray_files, green_files, labels,
        epochs, model_out_path, BATCH_SIZE,
    )

if __name__ == "__main__":
    main()