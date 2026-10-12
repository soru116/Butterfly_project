import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 2
LR = 2e-5
WEIGHT_DECAY = 1e-2


def build_model():
    """
    建立 ViT backbone + MLP classifier + optimizer + loss
    回傳：vit, classifier, optimizer, criterion
    """
    vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    vit.to(DEVICE)

    hidden = vit.config.hidden_size
    classifier = nn.Sequential(
        nn.Linear(hidden, hidden // 2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden // 2, NUM_CLASSES),
    ).to(DEVICE)

    params = list(vit.parameters()) + list(classifier.parameters())
    optimizer = optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    return vit, classifier, optimizer, criterion
