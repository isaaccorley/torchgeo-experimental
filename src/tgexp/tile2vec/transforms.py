import numpy as np


def clip_and_scale_image(
    img: np.ndarray,
    img_type: str = "naip",
    clip_min: float = 0.0,
    clip_max: float = 10000.0,
) -> np.ndarray:
    """Clips and scales bands to between [0, 1] for NAIP, RGB, and Landsat
    satellite images. Clipping applies for Landsat only.
    """
    if img_type in ("naip", "rgb"):
        return img / 255
    elif img_type == "landsat":
        return np.clip(img, clip_min, clip_max) / (clip_max - clip_min)


class GetBands:
    """Gets the first X bands of the tile triplet."""

    def __init__(self, bands: int) -> None:
        assert bands >= 0, "Must get at least 1 band"
        self.bands = bands

    def __call__(self, sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        a, n, d = (sample["anchor"], sample["neighbor"], sample["distant"])
        # Tiles are already in [c, w, h] order
        a, n, d = (a[: self.bands, :, :], n[: self.bands, :, :], d[: self.bands, :, :])
        sample = {"anchor": a, "neighbor": n, "distant": d}
        return sample


class RandomFlipAndRotate:
    """Does data augmentation during training by randomly flipping (horizontal
    and vertical) and randomly rotating (0, 90, 180, 270 degrees). Keep in mind
    that pytorch samples are CxWxH.
    """

    def __call__(self, sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        a, n, d = (sample["anchor"], sample["neighbor"], sample["distant"])
        # Randomly horizontal flip
        if np.random.rand() < 0.5:
            a = np.flip(a, axis=2).copy()
        if np.random.rand() < 0.5:
            n = np.flip(n, axis=2).copy()
        if np.random.rand() < 0.5:
            d = np.flip(d, axis=2).copy()
        # Randomly vertical flip
        if np.random.rand() < 0.5:
            a = np.flip(a, axis=1).copy()
        if np.random.rand() < 0.5:
            n = np.flip(n, axis=1).copy()
        if np.random.rand() < 0.5:
            d = np.flip(d, axis=1).copy()
        # Randomly rotate
        rotations = np.random.choice([0, 1, 2, 3])
        if rotations > 0:
            a = np.rot90(a, k=rotations, axes=(1, 2)).copy()
        rotations = np.random.choice([0, 1, 2, 3])
        if rotations > 0:
            n = np.rot90(n, k=rotations, axes=(1, 2)).copy()
        rotations = np.random.choice([0, 1, 2, 3])
        if rotations > 0:
            d = np.rot90(d, k=rotations, axes=(1, 2)).copy()
        sample = {"anchor": a, "neighbor": n, "distant": d}
        return sample


class ClipAndScale:
    """Clips and scales bands to between [0, 1] for NAIP, RGB, and Landsat
    satellite images. Clipping applies for Landsat only.
    """

    def __init__(self, img_type: str) -> None:
        assert img_type in ["naip", "rgb", "landsat"]
        self.img_type = img_type

    def __call__(self, sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        a, n, d = (
            clip_and_scale_image(sample["anchor"], self.img_type),
            clip_and_scale_image(sample["neighbor"], self.img_type),
            clip_and_scale_image(sample["distant"], self.img_type),
        )
        sample = {"anchor": a, "neighbor": n, "distant": d}
        return sample
