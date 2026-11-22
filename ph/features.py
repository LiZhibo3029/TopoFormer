import cv2
import numpy as np
from gudhi.representations import PersistenceImage

from ph import WINDOWS
from ph.alpha_pi import alphacomplex, preprocess


def upsample_nearest(map_chw: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Matches torch.nn.functional.interpolate(mode='nearest') index mapping."""
    h, w = map_chw.shape[-2:]
    th, tw = hw
    if (h, w) == (th, tw):
        return map_chw
    iy = np.minimum((np.arange(th, dtype=np.int64) * h) // th, h - 1)
    ix = np.minimum((np.arange(tw, dtype=np.int64) * w) // tw, w - 1)
    return map_chw[:, iy[:, None], ix[None, :]]


def extract(gray, dataset, threshold):
    """Apply the same alpha/PI and nearest-neighbor FPN recipe as the saved caches."""
    size = 224 if dataset == "crc" else 128
    gray = cv2.resize(gray.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    prep = preprocess(thresh=threshold, kernel_size=2, iterate=1)
    image = prep.dilate(prep.threshold(gray))
    raw = []
    for width in WINDOWS:
        h = size // width
        maps = np.empty((400, h, h), dtype=np.float32)
        for i in range(h):
            for j in range(h):
                diagram = alphacomplex(image[i * width:(i + 1) * width, j * width:(j + 1) * width])
                pi = PersistenceImage(resolution=(20, 20), bandwidth=1.0)
                maps[:, i, j] = pi.fit_transform(diagram)[0]
        raw.append(maps)

    fused = list(raw)
    for level in range(len(fused) - 2, -1, -1):
        fused[level] = fused[level] + upsample_nearest(fused[level + 1], fused[level].shape[-2:])

    # Pool the native grids before expanding their cells for the map cache.
    # In particular, the 3x3 CRC grid does not expand uniformly to 14x14.
    pooled = np.concatenate([level.mean(axis=(1, 2)) for level in fused])
    return fused, pooled
