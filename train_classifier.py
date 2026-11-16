"""Train a fire vs no-fire patch classifier.

Positives: row_N folders (made by fire_patch.py).
Negatives: neg_N folders (made by make_negatives.py).
Each fire and its matching negative always land on the same side of the split.
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fire_dataset import FirePatchDataset
from train_fire import auc, block


class PatchDataset(Dataset):
    SIZE = 80  # 20 km at 250 m per pixel

    def __init__(self, base, train):
        self.base, self.train = base, train

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        s = self.base[i]
        x = torch.cat([s["x"], s["valid"].float()], dim=0)
        # Earth Engine sometimes returns 81 pixels instead of 80. Crop or
        # pad to a fixed size so every patch can be batched together.
        x = x[:, :self.SIZE, :self.SIZE]
        pad_h, pad_w = self.SIZE - x.shape[1], self.SIZE - x.shape[2]
        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, (0, max(pad_w, 0), 0, max(pad_h, 0)))
        label = 1.0 if self.base.dirs[i].name.startswith("row_") else 0.0
        if self.train:
            k = int(torch.randint(4, (1,)))
            x = torch.rot90(x, k, (1, 2))
            if torch.rand(1) < 0.5:
                x = x.flip(-1)
        return x.contiguous(), torch.tensor(label)


class PatchNet(nn.Module):
    """Small CNN: patch in, one fire logit out."""

    def __init__(self, in_ch, width=32):
        super().__init__()
        self.net = nn.Sequential(
            block(in_ch, width),
            nn.MaxPool2d(2),
            block(width, width * 2),
            nn.MaxPool2d(2),
            block(width * 2, width * 4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(width * 4, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    scores, labels = [], []
    for x, y in loader:
        scores += torch.sigmoid(model(x.to(device))).cpu().tolist()
        labels += y.tolist()
    scores, labels = np.array(scores), np.array(labels)
    acc = ((scores > 0.5) == (labels > 0.5)).mean()
    return auc(scores, labels), acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="fire_patches")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    torch.manual_seed(args.seed)

    dummy = (np.zeros(1, "float32"), np.ones(1, "float32"))
    probe = FirePatchDataset(args.root, one_hot=True, stats=dummy)
    root = probe.root

    # Only keep rows that have both a fire patch and a matching negative.
    rows = sorted(
        FirePatchDataset._row(d) for d in probe.dirs
        if (root / f"neg_{FirePatchDataset._row(d)}" / "metadata.json").exists()
    )
    if len(rows) < 10:
        raise SystemExit("Need more fire and neg_ pairs. Run make_negatives.py.")
    rng = np.random.default_rng(args.seed)
    rows = list(rng.permutation(rows))
    n_val = max(1, int(len(rows) * args.val_frac))
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    def dirs_for(rs):
        out = []
        for r in rs:
            out += [root / f"row_{r}", root / f"neg_{r}"]
        return out

    train_base = FirePatchDataset(args.root, one_hot=True, stats=dummy)
    train_base.dirs = dirs_for(train_rows)
    train_base.stats = train_base.compute_stats()  # train only stats

    val_base = FirePatchDataset(args.root, one_hot=True, stats=train_base.stats)
    val_base.dirs = dirs_for(val_rows)

    train_ds = PatchDataset(train_base, train=True)
    val_ds = PatchDataset(val_base, train=False)
    train_dl = DataLoader(
        train_ds, args.batch_size, shuffle=True,
        drop_last=len(train_ds) > args.batch_size,
    )
    val_dl = DataLoader(val_ds, args.batch_size)

    in_ch = train_ds[0][0].shape[0]
    print(
        f"input channels: {in_ch}, train patches: {len(train_ds)}, "
        f"val patches: {len(val_ds)} (half fire, half no fire)"
    )

    model = PatchNet(in_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        score, acc = evaluate(model, val_dl, device)
        history.append(score)
        print(
            f"epoch {epoch:3d}  loss {total / len(train_ds):.4f}  "
            f"val AUC {score:.3f}  val acc {acc:.3f}"
        )

    torch.save(
        {"model": model.state_dict(), "stats": train_base.stats, "in_ch": in_ch},
        "fire_classifier.pt",
    )
    print(f"mean val AUC over last 5 epochs: {np.mean(history[-5:]):.3f}")


if __name__ == "__main__":
    main()
