import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class ImglistDataset(Dataset):
    """
    Dataset backed by a plain-text image list file. Each line has the format:
        <relative_image_path>  <integer_class_label>

    Image paths are resolved in priority order:
      1. As an absolute path (used directly when it exists on disk).
      2. Joined with data_root.
      3. Joined with data_root/images/ (legacy fallback layout).

    If the imglist file itself is missing, the dataset enters MOCK mode and
    returns random noise images. This allows a quick pipeline smoke-test
    without the full dataset on disk.
    """

    def __init__(self, imglist_path: str, data_root: str, transform=None, allow_mock: bool = False):
        self.transform  = transform
        self.data_root  = data_root
        self.samples    = []
        self.mock       = False

        if not os.path.exists(imglist_path):
            if not allow_mock:
                raise FileNotFoundError(f"Imglist not found: {imglist_path}")
            print(f"[Warning] Imglist not found: {imglist_path}. Running in MOCK mode.")
            self.mock    = True
            self.samples = [('mock', 0)] * 100
            return

        with open(imglist_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.samples.append((parts[0], int(parts[1])))

        print(f"Loaded {len(self.samples):,} samples from {imglist_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        if self.mock:
            dummy = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
            return (self.transform(dummy) if self.transform else dummy), 0

        img_path, label = self.samples[idx]
        try:
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
            elif os.path.exists(os.path.join(self.data_root, img_path)):
                image = Image.open(os.path.join(self.data_root, img_path)).convert('RGB')
            else:
                image = Image.open(
                    os.path.join(self.data_root, 'images', img_path)
                ).convert('RGB')
        except Exception:
            # Return a black placeholder rather than crashing the batch.
            image = Image.new('RGB', (224, 224), color='black')

        return (self.transform(image) if self.transform else image), label


class OODDatasetWrapper(Dataset):
    """
    Wraps any Dataset and overrides every label:
      - is_ood=True  → label becomes -1 (the sentinel value for OOD samples).
      - is_ood=False → label is kept as the original class index (ID samples).

    Downstream code can therefore distinguish ID from OOD simply by checking
    whether label == -1, without needing a separate boolean mask array.
    """

    def __init__(self, dataset: Dataset, is_ood: bool = False):
        self.dataset = dataset
        self.is_ood  = is_ood

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img, (-1 if self.is_ood else label)

    def __len__(self) -> int:
        return len(self.dataset)


def build_combined_loader(cfg, id_imglist: str, ood_imglist: str, preprocess) -> DataLoader:
    """
    Build a shuffled DataLoader over the concatenation of one ID dataset and
    one OOD dataset, with OOD labels replaced by -1.

    Shuffling ensures that the combined stream interleaves ID and OOD samples
    throughout, which is important for streaming adaptation dynamics (a purely
    ID-then-OOD ordering would cause all beta progress to accumulate at the
    wrong phase).

    Parameters
    ----------
    cfg         : Config object with data_root, batch_size, num_workers.
    id_imglist  : Absolute path to the ID imglist file.
    ood_imglist : Absolute path to the OOD imglist file.
    preprocess  : CLIP preprocessing transform.

    Returns
    -------
    DataLoader over the combined ConcatDataset.
    """
    allow_mock = getattr(cfg, 'allow_mock_data', False)
    id_ds  = ImglistDataset(id_imglist,  cfg.data_root, transform=preprocess, allow_mock=allow_mock)
    ood_ds = ImglistDataset(ood_imglist, cfg.data_root, transform=preprocess, allow_mock=allow_mock)

    combined = ConcatDataset([
        OODDatasetWrapper(id_ds,  is_ood=False),
        OODDatasetWrapper(ood_ds, is_ood=True),
    ])

    return DataLoader(
        combined,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
