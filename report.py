import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from materials import asset, map_paths, read_config
from ph import GV_DIM

METRICS = ("accuracy", "balanced_accuracy", "macro_precision", "macro_specificity_ovr", "macro_f1", "macro_auroc_ovr")
ALIASES = {"astrocytoma": "astro", "glioblastoma": "gbm", "oligodendroglioma": "oligo"}


def measures(y, probability):
    nclass = probability.shape[1]
    pred = probability.argmax(axis=1)
    matrix = np.bincount(y * nclass + pred, minlength=nclass * nclass).reshape(nclass, nclass)
    positives = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    tp = matrix.diagonal()
    if (positives == 0).any() or (positives == len(y)).any():
        raise ValueError("Metrics require complete test label coverage")
    return np.asarray([
        tp.sum() / len(y), np.mean(tp / positives),
        np.mean(tp / np.maximum(predicted, 1)),
        np.mean(1 - (predicted - tp) / (len(y) - positives)),
        np.mean(2 * tp / (predicted + positives)),
        roc_auc_score(y, probability, multi_class="ovr", average="macro"),
    ])


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_feature(path, config, feature):
    with np.load(path, allow_pickle=True) as cache:
        ids = map_paths(cache["paths"].astype(str), config)
        x = cache[feature].astype(np.float32)
        y = cache["y"].astype(np.int64)
        classes = cache["classes"].astype(str)
    if len(set(ids)) != len(ids) or len(x) != len(y) or len(x) != len(ids) or not np.isfinite(x).all():
        raise ValueError(f"Invalid feature cache: {path}")
    return x, y, ids, classes


def ph_rows(path, split, ids, classes, y, config):
    with np.load(path, allow_pickle=True) as cache:
        labels = cache[f"{split}_labels"].astype(str)
        keys = map_paths(cache[f"{split}_files"].astype(str), config)
        lookup = {key: idx for idx, key in enumerate(keys)}
        if len(lookup) != len(keys):
            raise ValueError(f"Duplicate PH IDs in {path}")
        take = [lookup[name] for name in ids]
        actual = [ALIASES.get(name.lower(), name) for name in labels[take]]
        np.testing.assert_array_equal(actual, classes[y])
        values = cache[f"{split}_phc"][take].astype(np.float32)
    if values.ndim != 2 or values.shape[1] != GV_DIM:
        raise ValueError(f"Expected {GV_DIM} pooled features from four fused TopoFPN scales in {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite pooled PH in {path}")
    return values


def unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def save_predictions(output, prob, y, ids, classes):
    output.mkdir(parents=True, exist_ok=True)
    name = output / "predictions.npz"
    tmp = output / "predictions.tmp.npz"
    np.savez_compressed(tmp, paths=ids, y_true=y, y_pred=prob.argmax(axis=1), probabilities=prob, classes=classes)
    tmp.replace(name)
    result = dict(zip(METRICS, measures(y, prob)))
    result["weighted_f1"] = f1_score(y, prob.argmax(axis=1), average="weighted")
    return result


def score(args):
    config = read_config(args.config)
    data = config["datasets"][args.dataset]
    run = args.run
    metadata = json.loads((run / "run.json").read_text(encoding="utf-8"))
    if (any(metadata.get(key) != getattr(args, key) for key in ("dataset", "threshold", "seed"))
            or metadata.get("variant") != "full" or metadata.get("cti") != "gated-local-cls"):
        raise ValueError(f"Wrong TopoFormer run metadata: {run}")

    prefix = f"dinov3FD_{args.dataset}"
    with np.load(run / "splits.npz", allow_pickle=False) as z:
        expected = {name: map_paths(z[name], config) for name in ("train", "val", "test")}
    if len(set(np.concatenate(tuple(expected.values())))) != sum(map(len, expected.values())):
        raise ValueError("The training/validation/test split overlaps")

    x_train, y_train, train_ids, train_classes = load_feature(run / f"{prefix}_train.npz", config, "feat")
    np.testing.assert_array_equal(np.sort(train_ids), np.sort(expected["train"]))
    pool = asset(data["pooled"], args.threshold)
    h_train = ph_rows(pool, "train", train_ids, train_classes, y_train, config)

    mu, sd = x_train.mean(axis=0), x_train.std(axis=0) + 1e-6
    lower = h_train.min(axis=0)
    span = h_train.max(axis=0) - lower
    span[span == 0] = 1e-6

    def fuse(x, h):
        return np.concatenate((unit((x - mu) / sd), 0.3 * unit((h - lower) / span)), axis=1)

    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight="balanced", solver="lbfgs")
    clf.fit(fuse(x_train, h_train), y_train)
    np.testing.assert_array_equal(clf.classes_, np.arange(len(train_classes)))
    del x_train, h_train

    for split in args.split:
        output = run / "scores" / split
        x, y, ids, classes = load_feature(run / f"{prefix}_{split}.npz", config, "feat")
        np.testing.assert_array_equal(np.sort(ids), np.sort(expected[split]))
        np.testing.assert_array_equal(train_classes, classes)
        h = ph_rows(pool, "train" if split == "val" else "test", ids, classes, y, config)
        prob = clf.predict_proba(fuse(x, h))
        result = {
            "dataset": args.dataset, "seed": args.seed, "threshold": args.threshold, "split": split,
            "n": len(y), "fusion_feature": "feat", "topology_dim": GV_DIM,
            "primary": save_predictions(output, prob, y, ids, classes),
        }
        write_json(output / "metrics.json", result)
        print(output / "metrics.json")


def main():
    parser = argparse.ArgumentParser(description="Fit and evaluate the four-scale TopoFormer late-fusion classifier")
    parser.add_argument("dataset", choices=("briancancer", "crc"))
    parser.add_argument("threshold", type=int, choices=(160, 180, 200))
    parser.add_argument("seed", type=int)
    parser.add_argument("run", type=Path)
    parser.add_argument("--config", default="config.local.json")
    parser.add_argument("--split", choices=("val", "test"), nargs="+", default=["test"])
    args = parser.parse_args()
    if len(set(args.split)) != len(args.split):
        parser.error("Specify each score split once")
    score(args)


if __name__ == "__main__":
    main()
