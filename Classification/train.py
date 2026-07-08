import os
import random
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
import timm
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from collections import deque
import glob


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_image_paths(base_dir, channel='CH3'):
    classes = ['Chamber Forming', 'Chamber Nonforming']
    data = []
    for label in classes:
        img_dir = os.path.join(base_dir, label, channel)
        for img_path in glob.glob(f"{img_dir}/*.tif"):
            data.append((img_path, 0 if label == 'Chamber Forming' else 1))
    return data

def split_data(image_list, test_size=0.2, seed=42):
    train_data, val_data = train_test_split(
        image_list,
        test_size=test_size,
        stratify=[label for _, label in image_list],
        random_state=seed,
    )
    return train_data, val_data

class OrganoidDataset(Dataset):
    def __init__(self, data, transform=None, apply_clahe=True):
        self.data = data
        self.transform = transform
        self.apply_clahe = apply_clahe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))

        if self.apply_clahe:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if self.transform:
            img = self.transform(img)

        return img, label



class RemoveBrightBorderFlood:

    def __init__(self,
                 band_frac=0.23,
                 quantiles=(0.90,0.88,0.85,0.80),
                 min_remove_px=500,
                 morph_dilate_iter=1,
                 fill_mode="mean",
                 inpaint_radius=3):
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
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        h, w, _ = img.shape
        mside = min(h, w)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)


        band = max(1, int(mside * self.band_frac))
        band_mask = np.zeros_like(gray, dtype=np.uint8)
        band_mask[:band,:]  = 1
        band_mask[-band:,:] = 1
        band_mask[:,:band]  = 1
        band_mask[:,-band:] = 1

        remove_mask = np.zeros_like(gray, dtype=np.uint8)


        band_vals = gray[band_mask > 0]
        if band_vals.size == 0:
            thr_seq = []
        else:
            thr_seq = [np.quantile(band_vals, q) for q in self.quantiles]

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
            if self.fill_mode == "mean":
                fill = img.mean(axis=(0,1), keepdims=True).astype(np.float32)
            else:
                fill = np.zeros((1,1,3), dtype=np.float32)
            out = (img.astype(np.float32)*(1.0-m) + fill*m).astype(np.uint8)

        return Image.fromarray(out)




def get_transforms(channel='CH3'):
    # CH3: RemoveBrightBorderFlood. CH1/CH2: CLAHE only (handled in OrganoidDataset).
    if channel == 'CH3':
        cleaner = RemoveBrightBorderFlood(
            band_frac=0.23,
            quantiles=(0.90,0.88,0.85,0.80),
            min_remove_px=500,
            morph_dilate_iter=1,
            fill_mode="mean"
        )
        train_transform = transforms.Compose([
            transforms.Lambda(lambda im: cleaner(im)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        val_transform = transforms.Compose([
            transforms.Lambda(lambda im: cleaner(im)),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
    else:  # CH1, CH2
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        val_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
    return train_transform, val_transform



class HybridResNetViT(nn.Module):
    def __init__(self):
        super().__init__()

        # ResNet50
        self.resnet = models.resnet50(pretrained=True)
        self.resnet_out = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        # DeiT3 ViT
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
        x = torch.cat((x1, x2), dim=1)  # [B, 2816]
        return self.fc(x)




def train_model(model, train_loader, val_loader, save_path, num_epochs=10):
    flag = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.BCELoss()

    best_acc = 0
    best_f1 = 0
    history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs, labels = imgs.to(device), labels.float().to(device)

            outputs = model(imgs).squeeze()
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

            preds = (outputs > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_acc, precision, recall, f1, _ = simple_val(model, val_loader, device)

        history.append({
            "epoch": epoch + 1,
            "train_loss": running_loss / total,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        })

        print(f"Epoch {epoch+1}: Train Acc = {train_acc:.4f} | Val Acc = {val_acc:.4f} | F1 = {f1:.4f}")

        if val_acc > best_acc and f1 > best_f1:
            best_acc = val_acc
            best_f1 = f1
            torch.save(model.state_dict(), save_path)
            if (best_acc > 0.85 and best_f1 > 0.85):
                flag = 1
                print("Flag = 1")
            print(f"✅ Saved best model with val acc: {val_acc:.4f}")

        torch.cuda.empty_cache()



    df = pd.DataFrame(history)
    df.to_excel(save_path.replace('.pth', '_history.xlsx'), index=False)
    if (flag == 1):
        os.system("python3 Analysis2.py")
    else:
        os.system("python3 Hybrid.py")


from sklearn.metrics import precision_score, recall_score, f1_score

def simple_val(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.cpu().numpy()

            outputs = model(imgs).squeeze().cpu().numpy()
            preds = (outputs > 0.5).astype(int)

            y_true.extend(labels)
            y_pred.extend(preds)

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

    return acc, precision, recall, f1, (y_true, y_pred)


# ── Configuration ─────────────────────────────────────────────────────────────
# To switch channels, change CHANNEL and BASE_DIR. See README.md for details.
CHANNEL  = 'CH3'                   # Options: 'CH1', 'CH2', 'CH3'
BASE_DIR = "/Documents/Organoids"

# ── Training ───────────────────────────────────────────────────────────────────
print(f"\n=== Training channel {CHANNEL} ===")
image_paths          = load_image_paths(BASE_DIR, CHANNEL)
train_data, val_data = split_data(image_paths, seed=SEED)
train_tf, val_tf     = get_transforms(CHANNEL)

train_set = OrganoidDataset(train_data, transform=train_tf)
val_set   = OrganoidDataset(val_data,   transform=val_tf)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_set,   batch_size=16, shuffle=False, num_workers=2)

model           = HybridResNetViT()
checkpoint_path = f"best_model_{CHANNEL}.pth"
train_model(model, train_loader, val_loader, checkpoint_path, num_epochs=50)



