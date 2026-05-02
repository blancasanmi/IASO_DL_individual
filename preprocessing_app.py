import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import app_
app = app_.app
image = app_.image
volume = app_.volume
VOLUME_PATH = app_.VOLUME_PATH
TENSORS_PATH = app_.TENSORS_PATH

@app.function(
    image=image,
    cpu=8,
    timeout=3600,
    volumes={VOLUME_PATH: volume},
)
def prepare_data():
    import os, pickle
    import pandas as pd
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    from sklearn.model_selection import train_test_split
    from PIL import Image

    path = os.path.join(VOLUME_PATH, "dataset") # where I will mount the dataset with preprocessing for later use
    os.makedirs(TENSORS_PATH, exist_ok=True)

    IMAGENET_MEAN = [0.485, 0.456, 0.406] # the normalization values based on ImageNet stats
    IMAGENET_STD  = [0.229, 0.224, 0.225]
    IMG_SIZE = 224

    transform_train = transforms.Compose([ # transformations on train set 
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    transform_val = transforms.Compose([ 
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # ── Dataset ──
    class PlantDataset(Dataset):
        def __init__(self, df, img_dir, transform=None):
            self.df = df.reset_index(drop=True)
            self.img_dir = img_dir
            self.transform = transform
            unique_labels = sorted(self.df["labels"].unique())
            self.label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            image = Image.open(os.path.join(self.img_dir, row["image"])).convert("RGB")
            label = torch.tensor(self.label_to_idx[row["labels"]], dtype=torch.long)
            if self.transform:
                image = self.transform(image)
            return image, label

    train_df = pd.read_csv(os.path.join(path, 'train.csv'))
    train_split, temp = train_test_split(train_df, test_size=0.4, random_state=42, stratify=train_df["labels"]) # to ensure class frequencies stay the same across splits 
    val_split, test_split = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["labels"])

    # ── Save label mapping (needed at training time) ──
    unique_labels = sorted(train_df["labels"].unique())
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    with open(os.path.join(TENSORS_PATH, "label_to_idx.pkl"), "wb") as f:
        pickle.dump(label_to_idx, f)

    # ── Process and save each split ──
    for split_name, split_df, transform in [
        ("train", train_split, transform_train),
        ("val",   val_split,   transform_val),
        ("test",  test_split,  transform_val),
    ]:
        print(f"Processing {split_name} ({len(split_df)} images)...")
        dataset = PlantDataset(split_df, os.path.join(path, "train_images/"), transform)
        loader  = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8)

        all_images, all_labels = [], []
        for images, labels in loader:
            all_images.append(images)
            all_labels.append(labels)

        # Save as a single tensor file per split
        torch.save({
            "images": torch.cat(all_images),  # [N, 3, 224, 224]
            "labels": torch.cat(all_labels),  # [N]
        }, os.path.join(TENSORS_PATH, f"{split_name}.pt"))
        print(f"  ✅ saved {split_name}.pt")

    volume.commit()
    print("All splits saved to volume.")
