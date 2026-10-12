import os
import glob
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image


def get_transforms(image_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


class SubsetWithTransform(Dataset):
    """讓每個 subset 使用自己的 transform"""
    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        path, label = self.base_dataset.samples[real_idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


class ExternalTestDataset(Dataset):
    """data/test 用的 dataset（沒有 label）"""
    def __init__(self, folder, transform=None):
        all_paths = glob.glob(os.path.join(folder, "*"))
        self.paths = [
            p for p in all_paths
            if p.lower().endswith(".jpg") and os.access(p, os.R_OK)
        ]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, path


def build_train_val_test_loaders(
    train_dir: str,
    batch_size: int,
    num_workers: int,
    image_size: int
):
    """
    使用 random_split → 再重新建立 transform 版本的 dataset
    """
    train_tf, val_tf = get_transforms(image_size)

    base = datasets.ImageFolder(train_dir)

    total = len(base)
    n_train = int(0.70 * total)
    n_val = int(0.15 * total)
    n_test = total - n_train - n_val

    train_subset, val_subset, test_subset = random_split(
        base, [n_train, n_val, n_test]
    )

    train_indices = train_subset.indices
    val_indices = val_subset.indices
    test_indices = test_subset.indices

    train_ds = SubsetWithTransform(base, train_indices, transform=train_tf)
    val_ds = SubsetWithTransform(base, val_indices, transform=val_tf)
    test_ds = SubsetWithTransform(base, test_indices, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Total images: {total}")
    print(f"Train = {n_train}, Val = {n_val}, Test = {n_test}")
    print("Class mapping:", base.class_to_idx)

    return train_loader, val_loader, test_loader


def build_external_test_loader(test_dir, batch_size, num_workers, image_size):
    _, val_tf = get_transforms(image_size)
    ds = ExternalTestDataset(test_dir, transform=val_tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
