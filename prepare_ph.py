import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from run_topoformer import load_images
from materials import asset, read_config
from ph import GV_DIM, WINDOWS
from ph.features import extract


def main():
    parser = argparse.ArgumentParser(description="Prepare PH maps and pooled features at the chosen threshold")
    parser.add_argument("dataset", choices=("briancancer", "crc"))
    parser.add_argument("threshold", type=int, choices=(160, 180, 200))
    parser.add_argument("--config", default="config.local.json")
    args = parser.parse_args()
    config = read_config(args.config)
    data = config["datasets"][args.dataset]
    base = asset(data["maps"], args.threshold)
    pool = asset(data["pooled"], args.threshold)
    images, y, classes, indices = load_images(config, args.dataset)
    splits = {"train": np.concatenate((indices["train"], indices["val"])), "test": indices["test"]}
    dirs = [Path(f"{base}_L{i}.mmap") for i in range(4)]
    if Path(pool).exists() or any(path.exists() or path.with_suffix(".npz").exists() for path in dirs):
        raise FileExistsError("PH target already exists; choose a new destination or reuse the existing cache")
    work = [path.with_suffix(".partial") for path in dirs]
    for path in work:
        path.mkdir(parents=True)
    metadata = {
        "phc_img_hw": 224 if args.dataset == "crc" else 128, "phc_threshold": args.threshold,
        "phc_dilation_kernel": 2, "phc_dilation_iterations": 1, "vec_res": 20,
        "scales": [(width, width) for width in WINDOWS], "persist_type": "alpha", "vec_type": "PI",
        "max_hw": None, "max_windows_per_scale": 256, "full_windows_only": True, "seed": 42,
        "use_phc_fpn": True, "fpn_upsample_mode": "nearest", "fpn_smooth": False,
        "map_align_to_l0": True,
    }
    grid = 14 if args.dataset == "crc" else 8
    for level, path in enumerate(work):
        np.save(path / "meta.npy", [json.dumps(dict(metadata, level_id=level))])
        (path / "full_grid.json").write_text(
            json.dumps({name: [grid, grid] for name in splits}), encoding="utf-8",
        )
    vectors = {}
    for name, index in splits.items():
        paths = images[index]
        arrays = [
            np.lib.format.open_memmap(
                directory / f"{name}_phc_map_level.npy", mode="w+", dtype="float32",
                shape=(len(index), 400, grid // 2 ** level, grid // 2 ** level),
            )
            for level, directory in enumerate(work)
        ]
        for item in work:
            np.save(item / f"{name}_files.npy", paths)
        pooled = np.empty((len(index), GV_DIM), dtype=np.float32)
        for i, path in enumerate(tqdm(paths, desc=name)):
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise OSError(path)
            maps, pooled[i] = extract(gray, args.dataset, args.threshold)
            for array, level in zip(arrays, maps):
                array[i] = level
        for array in arrays:
            array.flush()
        del array, arrays
        vectors.update({f"{name}_files": paths, f"{name}_labels": np.asarray(classes)[y[index]], f"{name}_phc": pooled})
    output = Path(pool)
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix(".partial.npz")
    np.savez_compressed(pending, **vectors, phc_threshold=args.threshold)
    for source, target in zip(work, dirs):
        source.rename(target)
    pending.rename(output)


if __name__ == "__main__":
    main()
