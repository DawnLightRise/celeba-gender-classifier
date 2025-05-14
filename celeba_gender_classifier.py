"""
Chen Tang, 05/13/2025

celeba_gender_classifier.py

Train and evaluate a MobileNetV2-based sex classifier on CelebA.

Usage example:
python celeba_gender_classifier.py \
  --data-dir ./ \
  --output-dir ./checkpoints \
  --batch-size 64 \
  --img-size 128 \
  --warmup-epochs 3 \
  --finetune-epochs 7 \
  --lr-head 1e-3 \
  --lr-ft 1e-4 \
  --weight-decay 1e-5 \
  --seed 42
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ['PYTHONHASHSEED']       = str(seed)

class CelebADataset(Dataset):
    """Dataset for CelebA sex classification."""
    def __init__(self, csv_attr, img_dir, csv_split, split, transform):
        df = pd.read_csv(csv_attr)
        df['Male'] = (df['Male'] == 1).astype(int)
        parts = pd.read_csv(csv_split)
        df = df.merge(parts, on='image_id')
        mapping = {'train':0, 'valid':1, 'test':2}
        df = df[df['partition'] == mapping[split]].reset_index(drop=True)
        self.df       = df
        self.img_dir  = img_dir
        self.transform= transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.img_dir, row['image_id'])
        img  = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, int(row['Male'])

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Eval', leave=False):
            imgs = imgs.to(device)
            out  = model(imgs).argmax(dim=1).cpu().tolist()
            all_preds.extend(out)
            all_labels.extend(labels.tolist())
    return (
        accuracy_score(all_labels, all_preds),
        precision_score(all_labels, all_preds),
        recall_score(all_labels, all_preds),
        f1_score(all_labels, all_preds),
    )

def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, labels in tqdm(loader, desc='Train', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',      type=str, required=True,
                        help='Path to folder containing imgs and CSVs')
    parser.add_argument('--output-dir',    type=str, default='./checkpoints',
                        help='Directory to save best model')
    parser.add_argument('--batch-size',    type=int, default=64)
    parser.add_argument('--img-size',      type=int, default=128)
    parser.add_argument('--warmup-epochs', type=int, default=3,
                        help='Epochs training head only')
    parser.add_argument('--finetune-epochs', type=int, default=7,
                        help='Epochs fine-tuning full model')
    parser.add_argument('--lr-head',       type=float, default=1e-3)
    parser.add_argument('--lr-ft',         type=float, default=1e-4)
    parser.add_argument('--weight-decay',  type=float, default=1e-5)
    parser.add_argument('--seed',          type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    img_dir   = os.path.join(args.data_dir, 'img_align_celeba')
    attr_csv  = os.path.join(args.data_dir, 'list_attr_celeba.csv')
    split_csv = os.path.join(args.data_dir, 'list_eval_partition.csv')
    os.makedirs(args.output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_ds = CelebADataset(attr_csv, img_dir, split_csv, 'train', transform)
    valid_ds = CelebADataset(attr_csv, img_dir, split_csv, 'valid', transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.hub.load('pytorch/vision:v0.13.1', 'mobilenet_v2', pretrained=True)
    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, 2)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    # Phase 1: head warm-up
    for p in model.features.parameters():
        p.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr_head)
    print("=== Warm-up Phase ===")
    for e in range(1, args.warmup_epochs+1):
        loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        acc, prec, rec, f1 = evaluate(model, valid_loader, device)
        print(f"Epoch {e}/{args.warmup_epochs}  Loss {loss:.4f}  "
              f"Acc {acc:.4f}  F1 {f1:.4f}")

    # Phase 2: fine-tune full network
    for p in model.features.parameters():
        p.requires_grad = True
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr_ft,
                           weight_decay=args.weight_decay)
                           
    print("\n=== Fine-tune Phase ===")
    best_acc, best_path = 0.0, None
    for e in range(1, args.finetune_epochs+1):
        loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        acc, prec, rec, f1 = evaluate(model, valid_loader, device)
        print(f"Epoch {e}/{args.finetune_epochs}  Loss {loss:.4f}  "
              f"Acc {acc:.4f}  F1 {f1:.4f}")
        if acc > best_acc:
            best_acc, best_path = acc, os.path.join(args.output_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_path)

    print(f"\nBest valid acc: {best_acc:.4f}  (saved to {best_path})")
    
    test_ds    = CelebADataset(attr_csv, img_dir, split_csv, 'test', transform)
    test_loader= DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.to(device).eval()

    test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, device)
    print("\n=== Test Set Metrics ===")
    print(f"Accuracy : {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall   : {test_rec:.4f}")
    print(f"F1-score : {test_f1:.4f}")

if __name__ == '__main__':
    main()
