import os, glob, math
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
from collections import deque

import timm
import torch
import torch.nn as nn
from torchvision import transforms, models

import matplotlib.pyplot as plt

# ---------------------- Settings ----------------------
INPUT_DIR   = "input"                    # folder with input images
MODEL_PATH  = "Files/best_model_CH3.pth" # classification checkpoint
OUT_ROOT    = "results"                  # output root folder
CHANNEL_TAG = "CH3"                      # filter: only files whose name contains CH3
APPLY_CLAHE = True
THRESH      = 0.5                        # probability threshold for class 1 (Nonforming)
SEED        = 42
BATCH_SIZE  = 1                          # one image at a time
# ----------------------------------------------------

os.makedirs(OUT_ROOT, exist_ok=True)
np.random.seed(SEED); torch.manual_seed(SEED)

CLASS_NAMES = {0: "Chamber Forming", 1: "Chamber Nonforming"}


# ---------------------- Border cleaning ----------------------
class RemoveBrightBorderFlood:
    def __init__(self, band_frac=0.23, quantiles=(0.90,0.88,0.85,0.80),
                 min_remove_px=500, morph_dilate_iter=1,
                 fill_mode="mean", inpaint_radius=3):
        self.band_frac = float(band_frac)
        self.quantiles = tuple(quantiles)
        self.min_remove_px = int(min_remove_px)
        self.morph_dilate_iter = int(morph_dilate_iter)
        self.fill_mode = fill_mode
        self.inpaint_radius = int(inpaint_radius)

    def _flood_bright_band(self, gray, band_mask, thr):
        h, w = gray.shape
        allowed = ((band_mask > 0) & (gray >= thr)).astype(np.uint8)
        if not allowed.any():
            return np.zeros_like(allowed, dtype=np.uint8)
        visited = np.zeros_like(allowed, dtype=np.uint8)
        dq = deque()
        for x in range(w):
            if allowed[0, x]:   visited[0, x] = 1; dq.append((x, 0))
            if allowed[h-1, x]: visited[h-1, x] = 1; dq.append((x, h-1))
        for y in range(h):
            if allowed[y, 0]:   visited[y, 0] = 1; dq.append((0, y))
            if allowed[y, w-1]: visited[y, w-1] = 1; dq.append((w-1, y))
        while dq:
            x, y = dq.popleft()
            for nx, ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                if 0 <= nx < w and 0 <= ny < h:
                    if allowed[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = 1
                        dq.append((nx, ny))
        return visited

    def __call__(self, pil_img):
        img = np.array(pil_img)
        if img.ndim == 2: img = np.stack([img]*3, axis=-1)
        h, w, _ = img.shape
        mside = min(h, w)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)

        band = max(1, int(mside * self.band_frac))
        band_mask = np.zeros_like(gray, dtype=np.uint8)
        band_mask[:band,:]=1; band_mask[-band:,:]=1; band_mask[:,:band]=1; band_mask[:,-band:]=1

        remove_mask = np.zeros_like(gray, dtype=np.uint8)
        band_vals = gray[band_mask > 0]
        thr_seq = [np.quantile(band_vals, q) for q in self.quantiles] if band_vals.size else []
        for thr in thr_seq:
            visited = self._flood_bright_band(gray, band_mask, thr)
            if visited.sum() >= self.min_remove_px:
                remove_mask = visited
                break
        if remove_mask.any() and self.morph_dilate_iter > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            remove_mask = cv2.dilate(remove_mask, kernel, iterations=self.morph_dilate_iter)

        mask255 = (remove_mask * 255).astype(np.uint8)
        if self.fill_mode == "inpaint" and mask255.any():
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            out = cv2.inpaint(bgr, mask255, self.inpaint_radius, cv2.INPAINT_TELEA)
            out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        else:
            m = (remove_mask[...,None]).astype(np.float32)
            fill = img.mean(axis=(0,1), keepdims=True).astype(np.float32) if self.fill_mode=="mean" \
                   else np.zeros((1,1,3), np.float32)
            out = (img.astype(np.float32)*(1.0-m) + fill*m).astype(np.uint8)
        return Image.fromarray(out)

def get_cleaner():
    return RemoveBrightBorderFlood(
        band_frac=0.23,
        quantiles=(0.90,0.88,0.85,0.80),
        min_remove_px=500,
        morph_dilate_iter=1,
        fill_mode="mean",
        inpaint_radius=3
    )

# ---------------------- Model ----------------------
class HybridResNetViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet_out = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.vit = timm.create_model('deit3_base_patch16_224', pretrained=True)
        self.vit_out = self.vit.head.in_features
        self.vit.head = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(self.resnet_out + self.vit_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.resnet(x)  # [B, 2048]
        x2 = self.vit(x)     # [B, 768]
        x  = torch.cat((x1, x2), dim=1)
        return self.fc(x)    # probability of class 1 (Nonforming)

# ---------------------- Grad-CAM & Attention ----------------------
def gradcam_resnet(model, img_tensor, device):
    features, gradients = [], []
    def fw_hook(m, i, o): features.append(o)
    def bw_hook(m, gi, go): gradients.append(go[0])
    handle_fw = model.resnet.layer4.register_forward_hook(fw_hook)
    try:
        handle_bw = model.resnet.layer4.register_full_backward_hook(bw_hook)
    except:
        handle_bw = model.resnet.layer4.register_backward_hook(bw_hook)

    output = model(img_tensor.to(device))
    model.zero_grad()
    output.backward(torch.ones_like(output))

    grads = gradients[0].detach().cpu()[0]   # [C,H,W]
    feats = features[0].detach().cpu()[0]    # [C,H,W]
    weights = torch.mean(grads, dim=(1, 2))  # [C]

    cam = torch.zeros(feats.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * feats[i]
    cam = cam.clamp(min=0)
    cam = cam / (cam.max() + 1e-6)

    handle_fw.remove(); handle_bw.remove()
    return cam.numpy()  # (H, W)

def vit_attention_map(model, img_tensor, device):
    attns = []
    def hook_fn(m, i, o): attns.append(o)
    handle = model.vit.blocks[-1].attn.register_forward_hook(hook_fn)
    _ = model(img_tensor.to(device))
    handle.remove()

    attn = attns[0]
    if attn.dim() == 4 and attn.shape[-1] == attn.shape[-2]:
        attn_cls = attn[0].mean(0)[0, 1:]  # mean over heads, CLS to tokens
    elif attn.dim() == 3:
        attn_cls = attn.mean(0)[0, 1:]
    else:
        x = attn
        while x.dim() > 2: x = x.mean(0)
        attn_cls = x

    num_patches = attn_cls.shape[0]
    best_h, best_w, min_diff = None, None, float('inf')
    for h in range(1, int(math.sqrt(num_patches)) + 2):
        if num_patches % h == 0:
            w = num_patches // h
            if abs(h - w) < min_diff:
                best_h, best_w = h, w; min_diff = abs(h - w)
    if best_h is None:
        raise ValueError(f"No rectangular shape for {num_patches} tokens.")
    attn_map = attn_cls.reshape(best_h, best_w).detach().cpu().numpy()
    attn_map = attn_map - attn_map.min()
    attn_map = attn_map / (attn_map.max() + 1e-6)
    return attn_map

# ---------------------- Upscale & overlays ----------------------
def upscale_map_smooth(map01, out_w, out_h, sigma=1.5):
    m = map01.astype(np.float32)
    if m.max() > 1.0 or m.min() < 0.0:
        mn, mx = float(m.min()), float(m.max())
        m = (m - mn) / (mx - mn + 1e-6) if mx > mn else np.zeros_like(m)
    m = cv2.resize(m, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    m = cv2.GaussianBlur(m, (0,0), sigmaX=sigma, sigmaY=sigma)
    m = (m - m.min()) / (m.max() - m.min() + 1e-6)
    return m

def overlay_heatmap(base_rgb_uint8, map01, alpha=0.45):
    heat = cv2.applyColorMap((map01*255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base_rgb_uint8, 1.0, heat, alpha, 0)

def plot_and_save(original_pil, cleaned_pil, cam_small, attn_small,
                  prob, pred, save_path, dpi=220):
    original = np.array(original_pil.convert("RGB"))
    cleaned  = np.array(cleaned_pil.convert("RGB"))
    H, W = original.shape[:2]

    cam_hr  = upscale_map_smooth(cam_small,  W, H, sigma=1.8)
    attn_hr = upscale_map_smooth(attn_small, W, H, sigma=1.2)
    comb_hr = (cam_hr + attn_hr) * 0.5

    grad_on_clean = overlay_heatmap(cleaned, cam_hr,  alpha=0.45)
    attn_on_clean = overlay_heatmap(cleaned, attn_hr, alpha=0.45)
    comb_on_clean = overlay_heatmap(cleaned, comb_hr, alpha=0.45)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), dpi=dpi)
    axes[0].imshow(original);        axes[0].set_title("Original");      axes[0].axis('off')
    axes[1].imshow(cleaned);         axes[1].set_title("Cleaned");       axes[1].axis('off')
    axes[2].imshow(grad_on_clean);   axes[2].set_title(f"Grad-CAM\nPred:{pred} ({prob:.2f})"); axes[2].axis('off')
    axes[3].imshow(attn_on_clean);   axes[3].set_title("ViT Attention"); axes[3].axis('off')
    axes[4].imshow(comb_on_clean);   axes[4].set_title("Combined");      axes[4].axis('off')
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return grad_on_clean, attn_on_clean, comb_on_clean

# ---------------------- Utils ----------------------
def safe_stem(path):
    base = os.path.splitext(os.path.basename(path))[0]
    return base.replace(os.sep, "_")

def build_transformers():
    base_tf = transforms.Compose([transforms.Resize((224,224)),
                                   transforms.ToTensor()])
    norm_tf = transforms.Normalize([0.5]*3, [0.5]*3)
    return base_tf, norm_tf

def read_and_clean(path, apply_clahe=True, cleaner=None):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(path)
    if apply_clahe:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
        img_bgr = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    if cleaner is not None:
        pil = cleaner(pil)
    return pil

def collect_input_images(input_dir, channel_tag="CH3"):
    valid_exts = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")
    paths = set()

    # Search directly in input_dir
    for ext in valid_exts:
        for p in glob.glob(os.path.join(input_dir, ext)):
            paths.add(p)

    # Search in input_dir/CH3 subfolder if it exists
    ch3_dir = os.path.join(input_dir, "CH3")
    if os.path.exists(ch3_dir):
        for ext in valid_exts:
            for p in glob.glob(os.path.join(ch3_dir, ext)):
                paths.add(p)

    # Keep only files whose name contains the channel tag
    if channel_tag:
        paths = [p for p in paths if channel_tag.lower() in os.path.basename(p).lower()]
    else:
        paths = list(paths)

    return sorted(paths)

# ---------------------- Main ----------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = HybridResNetViT().to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    base_tf, norm_tf = build_transformers()
    cleaner = get_cleaner()

    out_forming    = os.path.join(OUT_ROOT, CLASS_NAMES[0])
    out_nonforming = os.path.join(OUT_ROOT, CLASS_NAMES[1])
    os.makedirs(out_forming, exist_ok=True)
    os.makedirs(out_nonforming, exist_ok=True)

    img_list = collect_input_images(INPUT_DIR, CHANNEL_TAG)
    if len(img_list) == 0:
        print(f"No images found in '{INPUT_DIR}' or '{INPUT_DIR}/CH3' matching tag '{CHANNEL_TAG}'.")
        return

    rows = []

    for path in tqdm(img_list, desc="Inference"):
        original_pil = Image.open(path).convert("RGB")
        cleaned_pil  = read_and_clean(path, apply_clahe=APPLY_CLAHE, cleaner=cleaner)

        with torch.no_grad():
            x = base_tf(cleaned_pil)
            x = norm_tf(x).unsqueeze(0).to(device)
            prob = float(model(x).squeeze(1).item())
        pred = int(prob > THRESH)
        pred_name = CLASS_NAMES[pred]

        stem    = safe_stem(path)
        out_dir = os.path.join(OUT_ROOT, pred_name, stem)
        os.makedirs(out_dir, exist_ok=True)

        orig_png = os.path.join(out_dir, "original.png")
        cln_png  = os.path.join(out_dir, "cleaned.png")
        original_pil.save(orig_png)
        cleaned_pil.save(cln_png)

        x_single   = norm_tf(base_tf(cleaned_pil)).unsqueeze(0).to(device)
        cam_small  = gradcam_resnet(model, x_single, device)
        attn_small = vit_attention_map(model, x_single, device)

        composite_png = os.path.join(out_dir, f"composite5_pred{pred}.png")
        grad_on_clean, attn_on_clean, comb_on_clean = plot_and_save(
            original_pil, cleaned_pil, cam_small, attn_small, prob, pred, composite_png
        )
        grad_png = os.path.join(out_dir, "gradcam.png")
        vit_png  = os.path.join(out_dir, "vit.png")
        comb_png = os.path.join(out_dir, "combined.png")
        Image.fromarray(grad_on_clean).save(grad_png)
        Image.fromarray(attn_on_clean).save(vit_png)
        Image.fromarray(comb_on_clean).save(comb_png)

        rows.append({
            "file_name":        os.path.basename(path),
            "input_path":       path,
            "pred_id":          pred,
            "pred_name":        pred_name,
            "prob_nonforming":  prob,
            "threshold":        THRESH,
            "output_folder":    out_dir,
            "original_png":     orig_png,
            "cleaned_png":      cln_png,
            "gradcam_png":      grad_png,
            "vit_png":          vit_png,
            "combined_png":     comb_png,
            "composite5_png":   composite_png,
        })

    df = pd.DataFrame(rows)
    xlsx_path = os.path.join(OUT_ROOT, "results.xlsx")
    df.to_excel(xlsx_path, index=False)
    print(f"\nClassification done. Results in: {OUT_ROOT}")
    print(f"Excel: {xlsx_path}")

if __name__ == "__main__":
    main()
