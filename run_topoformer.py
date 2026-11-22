import argparse
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from materials import asset, canonical, map_paths, read_config
from topoformer.data import datasets
from topoformer.data.datasets import AlfredoFDDataset, make_loaders
from topoformer.data.split_builders import (
    build_briancancer_train_test, build_nctcrc100k_trainval_crcval7k_test,
)
from topoformer.data.topo_maps import TopoMapStore
from topoformer.engine.preflight import preflight
from topoformer.engine.runtime import build_train_args, save_npz, setup_logger
from topoformer.model.checkpoints import dino_sha256, load_selected_weights
from topoformer.model.disentanglement import set_seed, train
from topoformer.model.model_builder import build_topoformer_model

ROOT = Path(__file__).resolve().parent
SEEDS = {"briancancer": (42, 123, 666), "crc": (12, 42, 666)}


def load_images(config, dataset):
    paths = config["datasets"][dataset]
    if dataset == "crc":
        split = build_nctcrc100k_trainval_crcval7k_test(paths["train"], paths["test"])
    else:
        split = build_briancancer_train_test(paths["train"], paths["test"])
    images = np.asarray(split.paths, dtype=str)
    labels = np.asarray(split.label_names)[split.y_all]
    train_idx = split.train_idx
    test_idx = split.test_idx

    classes = list(split.label_names)
    reference = paths.get("split_manifest")
    if reference:
        all_paths = map_paths(images, config)
        lookup = {name: i for i, name in enumerate(all_paths)}
        with np.load(reference, allow_pickle=False) as manifest:
            train_idx, val_idx, test_idx = (
                np.asarray([lookup[name] for name in map_paths(manifest[key], config)])
                for key in ("train", "val", "test")
            )
        np.testing.assert_array_equal(np.sort(test_idx), np.sort(split.test_idx))
        covered = np.concatenate((train_idx, val_idx, test_idx))
        if len(covered) != len(images) or len(set(covered)) != len(images):
            raise ValueError("Split manifest does not cover the dataset exactly once")
    elif split.val_idx is not None:
        train_idx, val_idx = split.train_idx, split.val_idx
    else:
        raise ValueError("CRC requires its original split manifest to preserve order across servers")

    return map_paths(images, config), np.searchsorted(classes, labels), classes, {
        "train": train_idx, "val": val_idx, "test": test_idx,
    }


def build_model(config, store, classes, device):
    sys.path.insert(0, str(ROOT / "Backbone" / "dinov3" / "dinov3"))
    return build_topoformer_model(
        device=device, fd_label_names=classes, dino_local_ckpt=config["dinov3"],
        topo_c=store.topo_c, convnext_ckpt=config["convnext"], cti_grid=store.topo_hw[0],
    )


def load_checkpoint(model, run, config=None):
    load_selected_weights(model, run / "best_val.pt", config)
    model.eval()


def check_run(run, dataset, threshold, seed):
    metadata = json.loads((run / "run.json").read_text(encoding="utf-8"))
    expected = {"dataset": dataset, "threshold": threshold, "seed": seed, "variant": "full",
                "cti": "gated-local-cls",
                "checkpoint": "raw-at-best-ema-val-epoch"}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Run metadata does not match the requested TopoFormer run: {run}")


def check_manifest(run, images, indices, config):
    with np.load(run / "splits.npz", allow_pickle=False) as manifest:
        for split in ("train", "val", "test"):
            np.testing.assert_array_equal(map_paths(manifest[split], config), images[indices[split]])


def run_model(args):
    config = read_config(args.config)
    run = args.run
    set_seed(args.seed)
    # A process gets one physical GPU through CUDA_VISIBLE_DEVICES.
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    datasets.NUM_WORKERS = args.workers
    base = asset(config["datasets"][args.dataset]["maps"], args.threshold)
    path_key = partial(canonical, config=config)
    is_export = args.action == "export"
    splits = args.split
    test_only = is_export and splits == ["test"]
    store = TopoMapStore(base, path_key=path_key, test_only=test_only)
    images, y, classes, indices = load_images(config, args.dataset)
    subjects = indices["test"] if test_only else np.concatenate(list(indices.values()))
    store.check_alignment(images[subjects])
    if is_export:
        check_run(run, args.dataset, args.threshold, args.seed)
        check_manifest(run, images, indices, config)
    model = build_model(config, store, classes, device)
    batch = 8
    train_args = build_train_args(device, False, args.epochs, 5e-5, "lora", 0, 16, 16.0, 0.05)
    if args.action in ("preflight", "train"):
        train_loader, val_loader, _ = make_loaders(
            images, y, indices["train"], indices["test"], store, device, batch,
            val_idx=indices["val"], balance_sampler=True,
        )
        hook = getattr(model.vit.encoder, "_topo_hook", None)
        preflight(model, train_loader, device, amp=train_args.amp)
        if args.action == "preflight":
            return
        if config.get("compact_checkpoints"):
            train_args.cti_dino_sha256 = dino_sha256(config)
        run.mkdir(parents=True, exist_ok=False)
        (run / "run.json").write_text(json.dumps({
            "dataset": args.dataset, "threshold": args.threshold, "seed": args.seed, "variant": "full",
            "cti": "gated-local-cls",
            "checkpoint": "raw-at-best-ema-val-epoch", "classes": classes, "epochs": args.epochs,
        }, indent=2) + "\n", encoding="utf-8")
        np.savez_compressed(run / "splits.npz", **{key: images[idx] for key, idx in indices.items()})
        train_args.cti_audit = hook
        train_args.cti_output = run
        history, best = train(model, train_loader, val_loader, device, train_args, setup_logger("topoformer"))
        if len(history) != args.epochs or best is None:
            raise RuntimeError(f"Expected {args.epochs} complete epochs and a validation checkpoint")
        best_epoch = max(history, key=lambda row: row["bal_acc_A"])["epoch"]
        (run / "best_epoch.txt").write_text(str(best_epoch), encoding="utf-8")
        print(f"TRAIN_COMPLETE {run} epoch={best_epoch}")
        return

    load_checkpoint(model, run, config)
    best_epoch = int((run / "best_epoch.txt").read_text())
    output = run
    for split in splits:
        file = output / f"dinov3FD_{args.dataset}_{split}.npz"
        if file.exists():
            raise FileExistsError(f"Export already exists: {file}")
    for split in splits:
        loader = DataLoader(
            AlfredoFDDataset(indices[split], images, y, store, device, return_path=True),
            batch_size=batch, shuffle=False, num_workers=args.workers, pin_memory=True,
        )
        save_npz(split, loader, model, device, str(output), args.dataset, dict(enumerate(classes)), classes,
                 "dinov3_vit7b16", config["dinov3"], batch, best_epoch, base,
                 store.topo_c, store.topo_hw)


def main():
    parser = argparse.ArgumentParser(description="Train TopoFormer or export its features")
    parser.add_argument("action", choices=("preflight", "train", "export"))
    parser.add_argument("dataset", choices=SEEDS)
    parser.add_argument("threshold", type=int, choices=(160, 180, 200))
    parser.add_argument("seed", type=int)
    parser.add_argument("run", type=Path)
    parser.add_argument("--config", default="config.local.json")
    parser.add_argument("--split", choices=("train", "val", "test"), nargs="+", default=["test"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be an unsigned 32-bit integer")
    if len(set(args.split)) != len(args.split):
        parser.error("Specify each export split once")
    if args.action == "train" and args.run.exists():
        parser.error("Training needs a new directory. Use export to reuse a checkpoint.")
    run_model(args)


if __name__ == "__main__":
    main()
