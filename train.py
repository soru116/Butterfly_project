import os
import csv

import torch
from tqdm import tqdm

from dataset import (
    build_train_val_test_loaders,
    build_external_test_loader,
)
from model import build_model, DEVICE

# ============================
# 設定
# ============================

TRAIN_DIR = "butterfly_dataset/train"
TEST_DIR = "butterfly_dataset/test"

MODEL_SAVE_PATH = "vit_cls_best.pth"

BATCH_SIZE = 16
NUM_WORKERS = 0          # Windows/AMD 建議 0
NUM_EPOCHS = 10
IMAGE_SIZE = 224


# ============================
# 基本 train / val loop
# ============================
def train_one_epoch(epoch, vit, classifier, train_loader, optimizer, criterion):
    vit.train()
    classifier.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        out = vit(pixel_values=imgs)
        cls = out.last_hidden_state[:, 0, :]
        logits = classifier(cls)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(1)
        total_loss += loss.item() * imgs.size(0)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

        pbar.set_postfix(loss=total_loss / total,
                         acc=correct / total)

    return total_loss / total, correct / total


def validate_one_epoch(epoch, vit, classifier, val_loader, criterion):
    vit.eval()
    classifier.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Val   Epoch {epoch}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            out = vit(pixel_values=imgs)
            cls = out.last_hidden_state[:, 0, :]
            logits = classifier(cls)

            loss = criterion(logits, labels)

            preds = logits.argmax(1)
            total_loss += loss.item() * imgs.size(0)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

            pbar.set_postfix(val_loss=total_loss / total,
                             val_acc=correct / total)

    return total_loss / total, correct / total


# ============================
# internal test 評估（70/20/10 的 10%）
# ============================
def evaluate_test_split(vit, classifier, test_loader_raw):
    vit.eval()
    classifier.eval()

    TP = TN = FP = FN = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader_raw, desc="Internal Test Evaluating"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            out = vit(pixel_values=imgs)
            cls = out.last_hidden_state[:, 0, :]
            logits = classifier(cls)

            preds = logits.argmax(1)

            # 假設 1 = Positive, 0 = Negative
            TP += ((labels == 1) & (preds == 1)).sum().item()
            TN += ((labels == 0) & (preds == 0)).sum().item()
            FP += ((labels == 0) & (preds == 1)).sum().item()
            FN += ((labels == 1) & (preds == 0)).sum().item()
            total += labels.size(0)

    accuracy = (TP + TN) / total if total > 0 else 0.0
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    print("\n===== Internal Test Split Evaluation (10%) =====")
    print(f"TP = {TP}")
    print(f"TN = {TN}")
    print(f"FP = {FP}")
    print(f"FN = {FN}")
    print("----------------------------------")
    print(f"Accuracy = {accuracy:.4f}")
    print(f"TPR (Sensitivity) = {TPR:.4f}")
    print(f"TNR (Specificity) = {TNR:.4f}")
    print("==================================\n")


# ============================
# 對 data/test 做預測 → CSV
# ============================
def predict_external_test(vit, classifier, test_loader, out_csv="testresult.csv"):
    vit.eval()
    classifier.eval()

    predictions = []

    with torch.no_grad():
        for imgs, paths in tqdm(test_loader, desc="Predicting external test"):
            imgs = imgs.to(DEVICE)

            out = vit(pixel_values=imgs)
            cls = out.last_hidden_state[:, 0, :]
            logits = classifier(cls)
            preds = logits.argmax(1).cpu().tolist()

            for p, path in zip(preds, paths):
                fname = os.path.basename(path)
                predictions.append([fname, p])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label"])
        w.writerows(predictions)

    print(f"推論完成，{out_csv} 已建立於：{os.getcwd()}")


def main():
    # Dataloaders：70/20/10 split + external test loader
    train_loader, val_loader, test_loader_raw = build_train_val_test_loaders(
        TRAIN_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE,
    )
    external_test_loader = build_external_test_loader(
        TEST_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE,
    )

    # Model
    vit, classifier, optimizer, criterion = build_model()

    # Train loop
    best_val_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            epoch, vit, classifier, train_loader, optimizer, criterion
        )
        val_loss, val_acc = validate_one_epoch(
            epoch, vit, classifier, val_loader, criterion
        )

        print(f"Epoch {epoch}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "vit": vit.state_dict(),
                    "classifier": classifier.state_dict(),
                },
                MODEL_SAVE_PATH,
            )
            print(f"Saved best model! val_acc={val_acc:.4f}")

    print("最佳驗證準確度:", best_val_acc)

    # Internal test split 評估
    evaluate_test_split(vit, classifier, test_loader_raw)

    # 對 data/test 做預測 → testresult.csv
    predict_external_test(vit, classifier, external_test_loader, out_csv="Test_Dataset.csv")


if __name__ == "__main__":
    main()
