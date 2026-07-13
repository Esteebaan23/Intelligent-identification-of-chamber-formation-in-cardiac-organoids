#THIS IS THE CODE!!!!

import os
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

class InstanceNormalization(layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(shape=(input_shape[-1],), initializer='ones', trainable=True)
        self.offset = self.add_weight(shape=(input_shape[-1],), initializer='zeros', trainable=True)

    def call(self, x):
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        return self.scale * (x - mean) / tf.sqrt(var + self.epsilon) + self.offset

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

def transformer_block(x, num_heads=4, ff_dim=256, dropout_rate=0.1):
    # Multi-head self-attention
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1]//num_heads)(x, x)
    attn_output = layers.Dropout(dropout_rate)(attn_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

    # Feed-forward network
    ff_output = layers.Dense(ff_dim, activation='relu')(out1)
    ff_output = layers.Dropout(dropout_rate)(ff_output)
    ff_output = layers.Dense(x.shape[-1])(ff_output)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ff_output)

    return out2

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
### ------------------------
### Discriminator
### ------------------------

def build_discriminator(input_shape=(256, 256, 3)):
    inp = layers.Input(shape=input_shape)
    tar = layers.Input(shape=input_shape)
    x = layers.Concatenate()([inp, tar])

    # 1st conv block
    x = layers.Conv2D(64, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)

    # 2nd conv block
    x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)

    # 3rd conv block
    x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)

    # 4th conv block
    x = layers.Conv2D(512, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)

    # Output layer
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
"""
def match_grayscale_with_green(all_file_paths):
    matched_gray_files = []
    matched_green_files = []

    # Remove hidden MacOS files
    all_file_paths = [f for f in all_file_paths if not os.path.basename(f).startswith("._")]

    for filename in all_file_paths:
        if "CH4" in os.path.basename(filename):
            green_filename = filename.replace("CH4", "CH2")
            if green_filename in all_file_paths:
                matched_gray_files.append(filename)
                matched_green_files.append(green_filename)
            else:
                print(f"[!] No match found for {filename} → expected {green_filename}")

    return matched_gray_files, matched_green_files

# --- Main ---
base_dir = "/home/STUDENTS/ijw0032/Downloads/data_with_fluorescence/CH1_CH2_angello_data"  # ← change this to your base directory path

forming_files = []
nonforming_files = []

# Walk through all subdirectories
for root, dirs, files in os.walk(base_dir):
    folder_name = os.path.basename(root)
    all_file_paths = [os.path.join(root, f) for f in files if f.lower().endswith(".tif")]

    if folder_name == "chamber":
        forming_files.extend(all_file_paths)
    elif folder_name in ["non-chamber", "nonchamber"]:
        nonforming_files.extend(all_file_paths)
    else:
        continue  # skip irrelevant folders

# Match grayscale with green separately
matched_gray_files, matched_green_files = match_grayscale_with_green(nonforming_files)

matched_gray_files1, matched_green_files1 = match_grayscale_with_green(forming_files)


labels = []
labels1 = []

for path in matched_gray_files:
    labels.append(0)


for path in matched_gray_files1:
    labels1.append(1)
    """


forming_dir = "/home/STUDENTS/ijw0032/Downloads/data_with_fluorescence/COLORIZED_chamber_forming"
nonforming_dir = "/home/STUDENTS/ijw0032/Downloads/data_with_fluorescence/COLORIZED_chamber_nonforming"

forming_files = [os.path.join(forming_dir, f) for f in os.listdir(forming_dir) if f.lower().endswith('.tif')]
nonforming_files = [os.path.join(nonforming_dir, f) for f in os.listdir(nonforming_dir) if f.lower().endswith('.tif')]

def match_grayscale_with_green(all_file_paths):
    matched_gray_files = []
    matched_green_files = []

    for filename in all_file_paths:
        if "CH4" in os.path.basename(filename):
            green_filename = filename.replace("CH4", "CH1")
            
            if green_filename in all_file_paths:
                matched_gray_files.append(filename)
                matched_green_files.append(green_filename)
            else:
                print(f"[!] No match found for {filename} → expected {green_filename}")
    
    return matched_gray_files, matched_green_files

matched_gray_files, matched_green_files = match_grayscale_with_green(nonforming_files)
matched_gray_files1, matched_green_files1 = match_grayscale_with_green(forming_files)


labels = []
labels1 = []

for path in matched_gray_files:
    labels.append(0)


for path in matched_gray_files1:
    labels1.append(1)

print(f"matched_gray_files: {len(matched_gray_files)}")
print(f"matched_gray_files1: {len(matched_gray_files1)}")


def load_pair(gray_path, green_path):
    gray_path = gray_path.numpy().decode('utf-8')
    green_path = green_path.numpy().decode('utf-8')

    gray_image = Image.open(gray_path).convert("RGB")
    green_image = Image.open(green_path).convert("RGB")
    
    gray_image = np.array(gray_image)
    green_image = np.array(green_image)

    if len(gray_image.shape) == 2:
        gray_image = np.expand_dims(gray_image, axis=-1)
    if len(green_image.shape) == 3 and green_image.shape[2] == 4:
        green_image = green_image[:, :, :3]

    gray_image = tf.image.resize(gray_image, [IMG_SIZE, IMG_SIZE])
    green_image = tf.image.resize(green_image, [IMG_SIZE, IMG_SIZE])

    gray_image = tf.cast(gray_image, tf.float32) / 255.0
    green_image = tf.cast(green_image, tf.float32) / 255.0

    contrast_factor = 0.5
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
        gray = tf.image.random_crop(gray, size=[crop_size, crop_size, 1])
        green = tf.image.random_crop(green, size=[crop_size, crop_size, 1])
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

def wrapped_load_pair(g, gr, l):
    gray_img, green_img = tf.py_function(
        load_pair, [g, gr, l], [tf.float32, tf.float32]
    )

    gray_img.set_shape([256, 256, 3])
    green_img.set_shape([256, 256, 3])

    gray_img, green_img = augment(gray_img, green_img)

    return gray_img, green_img, l

def create_dataset(gray_files, green_files, labels, batch_size=8):
    def wrapped_load_pair(g, gr, l):
        gray_img, green_img = tf.py_function(load_pair, [g, gr], [tf.float32, tf.float32])
        gray_img.set_shape([256, 256, 3])
        green_img.set_shape([256, 256, 3])
        return gray_img, green_img, l

    dataset = tf.data.Dataset.from_tensor_slices((gray_files, green_files, labels))
    dataset = dataset.map(wrapped_load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache().shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

gray_train, gray_test, green_train, green_test, labels_train, labels_test = train_test_split(
    matched_gray_files, matched_green_files, labels, test_size=0.2, random_state=42, shuffle=True
)

gray_train1, gray_test1, green_train1, green_test1, labels_train1, labels_test1 = train_test_split(
    matched_gray_files1, matched_green_files1, labels1, test_size=0.2, random_state=42, shuffle=True
)

nonforming_ds = create_dataset(gray_train, green_train, labels_train, batch_size=8)
forming_ds = create_dataset(gray_train1, green_train1, labels_train1, batch_size=8)
test_ds1 = create_dataset(gray_test, green_test, labels_test, batch_size=8)
test_ds2 = create_dataset(gray_test1, green_test1, labels_test1, batch_size=8)

gen1 = build_generator()
disc1 = build_discriminator()

gen2 = build_generator()
disc2 = build_discriminator()

gen_opt1 = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_opt1 = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

gen_opt2 = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_opt2 = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

def warm_up(model, optimizer, input_shapes):
    dummy_inputs = [tf.random.normal(shape) for shape in input_shapes]

    with tf.GradientTape() as tape:
        output = model(dummy_inputs, training=True)
        loss = tf.reduce_mean(output if isinstance(output, tf.Tensor) else output[0])
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


warm_up(gen1, gen_opt1, [(1, 256, 256, 3)])

warm_up(disc1, disc_opt1, [(1, 256, 256, 3), (1, 256, 256, 3)])

warm_up(gen2, gen_opt2, [(1, 256, 256, 3)])
warm_up(disc2, disc_opt2, [(1, 256, 256, 3), (1, 256, 256, 3)])


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


NONFORMING_EPOCHS = 300
FORMING_EPOCHS = 300

print("=== Phase 1: Training on non-forming samples ===")
best_psnr, best_ssim = 0.0, 0.0  # initialize best metrics

for epoch in range(NONFORMING_EPOCHS):
    print(f"Pretrain Epoch {epoch + 1}/{NONFORMING_EPOCHS}")
    
    for input_gray, target_green, labels in nonforming_ds:
        g_loss, d_loss = train_step(input_gray, target_green, gen1, disc1, gen_opt1, disc_opt1)

    print(f"Generator Loss: {g_loss:.4f} | Discriminator Loss: {d_loss:.4f}")
    
    if epoch % 10 == 0:
        test_psnr, test_ssim = compute_psnr_ssim_on_test(gen1, test_ds1)
        print(f"[....] Test PSNR: {test_psnr:.4f} | Test SSIM: {test_ssim:.4f}")
        if test_psnr > best_psnr:
            best_psnr = test_psnr
        if test_ssim > best_ssim:
            best_ssim = test_ssim
            gen1.save("CH1_ResUNet_nonforming_model.keras")

print("=== Best Evaluation on Nonforming Test Set ===")
print(f"Best Test PSNR: {best_psnr:.4f}")
print(f"Best Test SSIM: {best_ssim:.4f}")

print("\n=== Phase 2: Training on forming samples ===")
best_psnr, best_ssim = 0.0, 0.0  # reset for forming phase

for epoch in range(FORMING_EPOCHS):
    print(f"Finetune Epoch {epoch + 1}/{FORMING_EPOCHS}")
    
    for input_gray, target_green, labels in forming_ds:
        g_loss, d_loss = train_step(input_gray, target_green, gen2, disc2, gen_opt2, disc_opt2)

    print(f"Generator Loss: {g_loss:.4f} | Discriminator Loss: {d_loss:.4f}")
    
    if epoch % 10 == 0:
        test_psnr, test_ssim = compute_psnr_ssim_on_test(gen2, test_ds2)
        print(f"[....] Test PSNR: {test_psnr:.4f} | Test SSIM: {test_ssim:.4f}")
        if (test_psnr > best_psnr) and (test_ssim > best_ssim):
            best_psnr, best_ssim = test_psnr, test_ssim
            print(f"New best model at epoch {epoch + 1} "
                  f"(PSNR: {best_psnr:.4f}, SSIM: {best_ssim:.4f})")
            gen2.save("CH1_ResUNet_forming_model.keras")

print("=== Best Evaluation on Forming Test Set ===")
print(f"Best Test PSNR: {best_psnr:.4f}")
print(f"Best Test SSIM: {best_ssim:.4f}")