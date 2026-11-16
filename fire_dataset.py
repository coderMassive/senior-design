
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset

NODATA = -9999.0
LAND_COVER = "land_cover"
NUM_LC_CLASSES = 17  # IGBP classes 1..17


class FirePatchDataset(Dataset):
    """Reads the folders written by fire_patch.py (row_<N>/).

    Each sample is a dict:
      x          float32 (C, H, W)  continuous bands, normalized, 0 where missing
      valid      bool    (C, H, W)  True where the original pixel had data
      land_cover int64   (H, W)     IGBP class 1..17, 0 where missing
      row        int64              row index of the fire in the FPA file
      y          float32            only if labels are given
    If one_hot=True, land cover is also appended to x as 17 channels.
    """

    def __init__(
        self,
        root,
        labels=None,
        stats=None,
        one_hot=False,
        augment=False,
        cache=True,
    ):
        self.root = Path(root)
        self.labels = labels  # dict: row index -> target value
        self.one_hot = one_hot
        self.augment = augment
        self.cache = cache
        self._cache = {}

        self.dirs = sorted(
            d for d in self.root.glob("row_*")
            if (d / "environment_stack.tif").exists()
            and (d / "metadata.json").exists()
        )
        if labels is not None:
            self.dirs = [
                d for d in self.dirs if self._row(d) in self.labels
            ]
        if not self.dirs:
            raise FileNotFoundError(f"No usable row_* folders in {root}")

        first = json.loads((self.dirs[0] / "metadata.json").read_text())
        self.bands = first["bands"]
        self.lc_index = self.bands.index(LAND_COVER)
        self.cont_index = [
            i for i, b in enumerate(self.bands) if b != LAND_COVER
        ]
        self.cont_names = [self.bands[i] for i in self.cont_index]

        # stats = (mean, std) arrays for continuous bands.
        # Compute on the training set and pass the same stats to val/test.
        self.stats = stats if stats is not None else self.compute_stats()

    @staticmethod
    def _row(folder):
        return int(folder.name.split("_")[1])

    def _read(self, i):
        if i in self._cache:
            return self._cache[i]
        folder = self.dirs[i]
        with rasterio.open(folder / "environment_stack.tif") as src:
            data = src.read().astype("float32")
        if self.cache:
            self._cache[i] = data
        return data

    def compute_stats(self):
        """Mean and std per continuous band, ignoring missing pixels."""
        total = np.zeros(len(self.cont_index))
        total_sq = np.zeros(len(self.cont_index))
        count = np.zeros(len(self.cont_index))
        for i in range(len(self.dirs)):
            data = self._read(i)
            for k, band in enumerate(self.cont_index):
                values = data[band]
                values = values[values != NODATA]
                total[k] += values.sum()
                total_sq[k] += (values.astype("float64") ** 2).sum()
                count[k] += values.size
        count = np.maximum(count, 1)
        mean = total / count
        std = np.sqrt(np.maximum(total_sq / count - mean**2, 0)) + 1e-6
        return mean.astype("float32"), std.astype("float32")

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, i):
        data = self._read(i)

        valid_all = data != NODATA
        cont = data[self.cont_index].copy()
        valid = valid_all[self.cont_index]

        mean, std = self.stats
        cont = (cont - mean[:, None, None]) / std[:, None, None]
        cont[~valid] = 0.0

        lc = data[self.lc_index].copy()
        lc_valid = valid_all[self.lc_index]
        lc = np.where(lc_valid, lc, 0).astype("int64")

        if self.augment:
            # Same random flip and 90 degree turn for every array.
            k = np.random.randint(4)
            flip = np.random.rand() < 0.5
            arrays = [cont, valid, lc]
            out = []
            for a in arrays:
                a = np.rot90(a, k, axes=(-2, -1))
                if flip:
                    a = np.flip(a, axis=-1)
                out.append(np.ascontiguousarray(a))
            cont, valid, lc = out

        x = torch.from_numpy(cont)
        valid = torch.from_numpy(valid)
        lc = torch.from_numpy(lc)

        if self.one_hot:
            onehot = torch.nn.functional.one_hot(
                lc, NUM_LC_CLASSES + 1
            )[..., 1:]  # drop the "missing" class 0
            x = torch.cat([x, onehot.permute(2, 0, 1).float()], dim=0)

        row = self._row(self.dirs[i])
        sample = {
            "x": x,
            "valid": valid,
            "land_cover": lc,
            "row": torch.tensor(row),
        }
        if self.labels is not None:
            sample["y"] = torch.tensor(self.labels[row], dtype=torch.float32)
        return sample


def make_loaders(root, labels, batch_size=16, val_frac=0.2, seed=0, **kw):
    """Split by fire, compute stats on train only, return both loaders."""
    probe = FirePatchDataset(root, labels=labels, **kw)
    n = len(probe)
    order = np.random.default_rng(seed).permutation(n)
    n_val = int(n * val_frac)
    val_idx, train_idx = order[:n_val], order[n_val:]

    train_ds = FirePatchDataset(root, labels=labels, augment=True, **kw)
    train_ds.dirs = [probe.dirs[i] for i in train_idx]
    train_ds.stats = train_ds.compute_stats()

    val_ds = FirePatchDataset(
        root, labels=labels, stats=train_ds.stats, **kw
    )
    val_ds.dirs = [probe.dirs[i] for i in val_idx]

    train = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val = DataLoader(val_ds, batch_size=batch_size)
    return train, val


if __name__ == "__main__":
    # Example: labels maps FPA row index -> target (for example log fire size).
    labels = {0: 1.2, 1: 3.4}
    train, val = make_loaders("fire_patches", labels, batch_size=2, val_frac=0.5)
    batch = next(iter(train))
    print({k: tuple(v.shape) for k, v in batch.items()})
