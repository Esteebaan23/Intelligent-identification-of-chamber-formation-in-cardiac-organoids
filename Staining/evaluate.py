import os, glob, math, random
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
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix)

import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR    = "/Documents/Organoids"
CHANNEL     = "CH3"
MODEL_PATH  = "best_model_CH3.pth"
OUT_DIR     = "val_hybrid_reports"
BATCH_SIZE  = 1
SEED        = 42
TEST_SIZE   = 0.2
APPLY_CLAHE = True  # always mirrors training (CLAHE applied for all channels)
THRESH      = 0.5


os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#  Utils dataset/split
def load_image_paths(base_dir, channel='CH3'):
    classes = ['Chamber Forming', 'Chamber Nonforming']
    data = []
    for label in classes:
        img_dir = os.path.join(base_dir, label, channel)
        for img_path in glob.glob(f"{img_dir}/*.tif"):
            data.append((img_path, 0 if label == 'Chamber Forming' else 1))
    return data

def split_data(image_list, test_size=0.2, seed=42):
    labels = [label for _, label in image_list]
    tr, va = train_test_split(image_list, test_size=test_size, stratify=labels, random_state=seed)
    pd.DataFrame(tr, columns=["path","label"]).to_csv(os.path.join(OUT_DIR, "split_train.csv"), index=False)
    pd.DataFrame(va, columns=["path","label"]).to_csv(os.path.join(OUT_DIR, "split_val.csv"), index=False)
    return tr, va

# ------------------ Corner cleaning ------------------
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
            fill = img.mean(axis=(0,1), keepdims=True).astype(np.float32) if self.fill_mode=="mean" else np.zeros((1,1,3), np.float32)
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

# ----------------------- Dataset -----------------------
class OrganoidDataset(Dataset):
    def __init__(self, data, transform=None, apply_clahe=True, cleaner=None):
        self.data = data
        self.transform = transform
        self.apply_clahe = apply_clahe
        self.cleaner = cleaner

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(img_path)
        if self.apply_clahe:
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
            img_bgr = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)

        if self.cleaner is not None:
            pil = self.cleaner(pil)

        tensor = transforms.ToTensor()(pil)
        return pil, tensor, label, img_path

def get_val_transform():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

#  Model
class HybridResNetViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet_out = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.vit = timm.create_model('deit3_base_patch16_224', pretrained=True)
        self.vit_out = self.vit.head.in_features
        self.vit.head = nn.Identity()

        self.fc = nn.Sequential(nn.Linear(self.resnet_out + self.vit_out, 1),
                                nn.Sigmoid())

    def forward(self, x):
        x1 = self.resnet(x)  # [B, 2048]
        x2 = self.vit(x)     # [B, 768]
        x  = torch.cat((x1, x2), dim=1)
        return self.fc(x)

# ----------------- Grad-CAM & Attention -----------------
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

    grads = gradients[0].detach().cpu()[0]          # [C,H,W]
    feats = features[0].detach().cpu()[0]           # [C,H,W]
    weights = torch.mean(grads, dim=(1, 2))         # [C]

    cam = torch.zeros(feats.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * feats[i]
    cam = cam.clamp(min=0)
    cam = cam / (cam.max() + 1e-6)

    handle_fw.remove(); handle_bw.remove()
    return cam.numpy()

def vit_attention_map(model, img_tensor, device):
    attns = []
    def hook_fn(m, i, o): attns.append(o)
    handle = model.vit.blocks[-1].attn.register_forward_hook(hook_fn)
    _ = model(img_tensor.to(device))
    handle.remove()

    attn = attns[0]
    if attn.dim() == 4 and attn.shape[-1] == attn.shape[-2]:
        attn_cls = attn[0].mean(0)[0, 1:]
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

# Upscale
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

def plot_and_save_with_clean(original_pil, cleaned_pil, cam_small, attn_small,
                             prob, pred, true_label, save_path, dpi=220):
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
    axes[0].imshow(original);        axes[0].set_title("Original");  axes[0].axis('off')
    axes[1].imshow(cleaned);         axes[1].set_title("Cleaned");   axes[1].axis('off')
    axes[2].imshow(grad_on_clean);   axes[2].set_title(f"Grad-CAM\nPred:{pred} ({prob:.2f}) | True:{true_label}"); axes[2].axis('off')
    axes[3].imshow(attn_on_clean);   axes[3].set_title("ViT Attention"); axes[3].axis('off')
    axes[4].imshow(comb_on_clean);   axes[4].set_title("Combined");  axes[4].axis('off')
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def collate_keep_pil(batch):
    cleaned_pils = [b[0] for b in batch]
    xs          = torch.stack([b[1] for b in batch], dim=0)
    labels      = torch.tensor([int(b[2]) for b in batch])
    paths       = [b[3] for b in batch]
    return cleaned_pils, xs, labels, paths


# Val
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Data + split
    image_paths = load_image_paths(BASE_DIR, CHANNEL)
    train_data, val_data = split_data(image_paths, TEST_SIZE, SEED)

    # 2) Datasets / loaders (val)
    cleaner = get_cleaner() if CHANNEL == 'CH3' else None
    base_tf  = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    norm_tf  = transforms.Normalize([0.5]*3, [0.5]*3)

    class ValWrapper(Dataset):
        def __init__(self, data, apply_clahe, cleaner, base_tf, norm_tf):
            self.inner = OrganoidDataset(data, transform=None, apply_clahe=apply_clahe, cleaner=cleaner)
            self.base_tf = base_tf; self.norm_tf = norm_tf
        def __len__(self): return len(self.inner)
        def __getitem__(self, idx):
            cleaned_pil, _, label, path = self.inner[idx]
            x = self.base_tf(cleaned_pil)
            x = self.norm_tf(x)
            return cleaned_pil, x, label, path

    val_set = ValWrapper(val_data, APPLY_CLAHE, cleaner, base_tf, norm_tf)
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_keep_pil
    )

    # 3) Model
    model = HybridResNetViT().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 4) Eval
    results = []
    y_true, y_prob = [], []

    for cleaned_pils, xs, labels, paths in tqdm(val_loader, desc="Validating"):
        with torch.no_grad():
            probs = model(xs.to(device)).squeeze(1).detach().cpu().numpy()
        preds = (probs > THRESH).astype(int)

        for i in range(len(paths)):
            prob = float(probs[i])
            pred = int(preds[i])
            true_label = int(labels[i].item())
            path = paths[i]
            cleaned_pil = cleaned_pils[i]

            y_true.append(true_label)
            y_prob.append(prob)
            results.append({
                "path": path, "pred": pred, "true": true_label,
                "prob": prob, "correct": int(pred == true_label)
            })

    y_pred = (np.array(y_prob) > THRESH).astype(int)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='binary', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='binary', zero_division=0)

    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    # ---- Classification report ----
    target_names = ["Chamber Forming (0)", "Chamber Nonforming (1)"]
    cls_rep = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print("\nClassification Report:\n", cls_rep)
    with open(os.path.join(OUT_DIR, "classification_report.txt"), "w") as f:
        f.write(cls_rep)

    # ---- Confusion matrix ----
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix (rows=true, cols=pred):\n", cm)
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(os.path.join(OUT_DIR, "confusion_matrix.csv"), index=True)

    plt.figure(figsize=(5,4), dpi=180)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"))
    plt.close()

    df = pd.DataFrame(results)
    xlsx_path = os.path.join(OUT_DIR, "results.xlsx")
    df.to_excel(xlsx_path, index=False)
    print(f"\n✅ Saved: {xlsx_path}")
    print(f"✅ Report:  {os.path.join(OUT_DIR, 'classification_report.txt')}")
    print(f"✅ Matrix:   {os.path.join(OUT_DIR, 'confusion_matrix.png')}")

    # 5) Misclassified
    errors = [r for r in results if r["correct"] == 0]
    if not errors:
        print("\n🎉 No misclassifications.")
        return

    os.makedirs(os.path.join(OUT_DIR, "misclassified"), exist_ok=True)
    for k, row in enumerate(tqdm(errors, desc="Exporting figures")):
        img_path   = row["path"]
        prob       = row["prob"]
        pred       = row["pred"]
        true_label = row["true"]

        original_pil = Image.open(img_path).convert("RGB")
        cleaned_pil  = cleaner(original_pil) if cleaner is not None else original_pil

        x = norm_tf(base_tf(cleaned_pil)).unsqueeze(0).to(device)

        cam_small  = gradcam_resnet(model, x, device)
        attn_small = vit_attention_map(model, x, device)

        out_path = os.path.join(OUT_DIR, "misclassified", f"misclf_{k:04d}_pred{pred}_true{true_label}.png")
        plot_and_save_with_clean(original_pil, cleaned_pil, cam_small, attn_small,
                                 prob, pred, true_label, out_path)

    print(f"\n Visualizations on: {os.path.join(OUT_DIR, 'misclassified')}")


if __name__ == "__main__":
    main()
