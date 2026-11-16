"""Plot validation patches with the true label and the model's predicted label.

Each row is one patch. Each column is one band. The row label shows
true label, predicted label, and the model's fire probability.
Green text means the model was right, red means it was wrong.

Examples:
  python show_predictions.py
  python show_predictions.py --n 16 --show wrong
  python show_predictions.py --bands ndvi elevation temperature_max land_cover
"""
import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from fire_dataset import NODATA, FirePatchDataset
from train_classifier import PatchDataset, PatchNet

CMAPS = {
    "ndvi": "YlGn", "evi": "YlGn", "land_cover": "tab20",
    "dead_fuel_moisture_100h": "Blues", "dead_fuel_moisture_1000h": "Blues",
    "temperature_max": "inferno", "precipitation": "Blues",
    "wind_speed": "viridis", "relative_humidity_min": "Blues",
    "elevation": "terrain", "slope": "magma",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="fire_patches")
    p.add_argument("--model", default="fire_classifier.pt")
    p.add_argument("--n", type=int, default=12, help="number of patches")
    p.add_argument("--show", choices=["random", "wrong", "right"], default="random")
    p.add_argument(
        "--bands", nargs="+",
        default=["ndvi", "land_cover", "dead_fuel_moisture_100h", "elevation"],
    )
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="predictions.png")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device, weights_only=False)

    dummy = (np.zeros(1, "float32"), np.ones(1, "float32"))
    probe = FirePatchDataset(args.root, one_hot=True, stats=dummy)
    root = probe.root

    # Use the same validation fires as training.
    val_rows = ckpt.get("val_rows")
    if val_rows is None:
        # Older checkpoint: rebuild the split. Only matches if the data
        # folders have not changed since training.
        rows = sorted(
            FirePatchDataset._row(d) for d in probe.dirs
            if (root / f"neg_{FirePatchDataset._row(d)}" / "metadata.json").exists()
        )
        rows = list(np.random.default_rng(args.seed).permutation(rows))
        val_rows = rows[: max(1, int(len(rows) * args.val_frac))]

    base = FirePatchDataset(args.root, one_hot=True, stats=ckpt["stats"])
    base.dirs = [d for r in val_rows for d in (root / f"row_{r}", root / f"neg_{r}")]
    ds = PatchDataset(base, train=False)

    model = PatchNet(ckpt["in_ch"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        probs = [
            torch.sigmoid(model(ds[i][0][None].to(device))).item()
            for i in range(len(ds))
        ]
    probs = np.array(probs)
    truth = np.array([int(ds[i][1].item()) for i in range(len(ds))])
    pred = (probs > 0.5).astype(int)
    correct = pred == truth
    print(f"validation accuracy: {correct.mean():.3f} on {len(ds)} patches")

    pool = np.arange(len(ds))
    if args.show == "wrong":
        pool = pool[~correct]
    elif args.show == "right":
        pool = pool[correct]
    if len(pool) == 0:
        raise SystemExit("No patches match that filter.")
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(pool, size=min(args.n, len(pool)), replace=False)

    names = {0: "no fire", 1: "fire"}
    cols = len(args.bands)
    fig, axes = plt.subplots(
        len(chosen), cols, figsize=(2.6 * cols + 1.5, 2.6 * len(chosen)),
        squeeze=False,
    )
    for r, i in enumerate(chosen):
        data = base._read(i)[:, :80, :80]
        for c, band in enumerate(args.bands):
            ax = axes[r][c]
            raw = data[base.bands.index(band)]
            masked = np.ma.masked_where(raw == NODATA, raw)
            kw = {"cmap": CMAPS.get(band, "viridis"), "interpolation": "nearest"}
            if band == "land_cover":
                kw.update(vmin=1, vmax=17)
            ax.imshow(masked, **kw)
            ax.plot(40, 40, "r+", markersize=10, markeredgewidth=1.5)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(band, fontsize=8)
        label = (
            f"true: {names[truth[i]]}\npred: {names[pred[i]]}\n"
            f"p(fire) = {probs[i]:.2f}\n{base.dirs[i].name}"
        )
        axes[r][0].set_ylabel(
            label, rotation=0, ha="right", va="center", fontsize=8,
            labelpad=8, color="green" if correct[i] else "red",
        )

    fig.suptitle("Validation patches (red + marks the patch center)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print("saved", args.out)


if __name__ == "__main__":
    main()
