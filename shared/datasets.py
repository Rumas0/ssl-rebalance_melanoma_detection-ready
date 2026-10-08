import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ----- transforms (identical to v1) -----
def make_transforms(image_size=224):
    normal_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    mel_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(0.4, 0.4, 0.3, 0.1),
        transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return normal_transform, mel_transform, test_transform


class AugmentedDataset(Dataset):
    """v1 dataset: MEL oversampling + MEL-specific augmentation. Unchanged."""
    def __init__(self, df, image_dir, mel_multiplier=5,
                 transform_normal=None, transform_mel=None):
        self.image_dir = image_dir
        self.classes = sorted(df['label'].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform_normal = transform_normal
        self.transform_mel = transform_mel

        mel_rows = df[df['label'] == 'MEL']
        other_rows = df[df['label'] != 'MEL']
        mel_expanded = pd.concat([mel_rows] * mel_multiplier, ignore_index=True)
        self.expanded_df = pd.concat([mel_expanded, other_rows], ignore_index=True)
        self.expanded_df = self.expanded_df.sample(frac=1, random_state=42).reset_index(drop=True)

    def __len__(self):
        return len(self.expanded_df)

    def __getitem__(self, idx):
        row = self.expanded_df.iloc[idx]
        img = Image.open(f"{self.image_dir}/{row['image']}.jpg").convert('RGB')
        label = self.class_to_idx[row['label']]
        if row['label'] == 'MEL' and self.transform_mel is not None:
            img = self.transform_mel(img)
        elif self.transform_normal is not None:
            img = self.transform_normal(img)
        return img, label


class StandardDataset(Dataset):
    """v1 ablation dataset: no oversampling, no MEL-specific augmentation."""
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.classes = sorted(df['label'].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(f"{self.image_dir}/{row['image']}.jpg").convert('RGB')
        label = self.class_to_idx[row['label']]
        if self.transform:
            img = self.transform(img)
        return img, label


class SyntheticAugmentedDataset(Dataset):
    """
    v2 dataset: real training data + synthetic MEL images.

    Combines:
      - The real MEL rows, oversampled `mel_multiplier` times (as in v1)
      - The real BKL + NV rows, once each
      - The synthetic MEL images, once each (or oversampled `synth_multiplier` times)

    Synthetic images are read from a flat directory where filenames are
    arbitrary but every file is a melanoma. This is the only safe way to use
    them — do NOT put them in the same folder as real images and infer labels
    from filenames.
    """
    def __init__(self, real_df, synthetic_dir, image_dir, mel_multiplier=5,
                 synth_multiplier=1, transform_normal=None, transform_mel=None,
                 synth_transform=None):
        self.image_dir = image_dir
        self.synthetic_dir = synthetic_dir
        self.classes = sorted(real_df['label'].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform_normal = transform_normal
        self.transform_mel = transform_mel
        self.synth_transform = synth_transform if synth_transform else transform_mel

        # real rows, MEL oversampled
        mel_rows = real_df[real_df['label'] == 'MEL']
        other_rows = real_df[real_df['label'] != 'MEL']
        mel_expanded = pd.concat([mel_rows] * mel_multiplier, ignore_index=True)
        self.real_df = pd.concat([mel_expanded, other_rows], ignore_index=True)

        # synthetic MEL image paths
        synth_files = sorted([
            f for f in os.listdir(synthetic_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.synthetic_files = synth_files * synth_multiplier
        self.n_real = len(self.real_df)
        self.n_synth = len(self.synthetic_files)

    def __len__(self):
        return self.n_real + self.n_synth

    def __getitem__(self, idx):
        if idx < self.n_real:
            row = self.real_df.iloc[idx]
            img = Image.open(f"{self.image_dir}/{row['image']}.jpg").convert('RGB')
            label = self.class_to_idx[row['label']]
            if row['label'] == 'MEL' and self.transform_mel is not None:
                img = self.transform_mel(img)
            elif self.transform_normal is not None:
                img = self.transform_normal(img)
            return img, label
        else:
            fname = self.synthetic_files[idx - self.n_real]
            img = Image.open(os.path.join(self.synthetic_dir, fname)).convert('RGB')
            if self.synth_transform is not None:
                img = self.synth_transform(img)
            return img, self.class_to_idx['MEL']