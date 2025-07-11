import os
import random

import numpy as np
import rasterio


def load_img(
    img_file: str, val_type: str = "uint8", bands_only: bool = False, num_bands: int = 4
) -> np.ndarray:
    """Loads an image using gdal, returns it as an array."""
    assert val_type in ["uint8", "float32"]
    with rasterio.open(img_file) as src:
        if val_type == "uint8":
            img = src.read().astype(np.uint8)
        elif val_type == "float32":
            img = src.read().astype(np.float32)
    img = np.moveaxis(img, 0, -1)
    if bands_only:
        img = img[:, :, :num_bands]
    return img


def get_triplet_imgs(
    img_dir: str, img_ext: str = ".tif", n_triplets: int = 1000
) -> np.ndarray:
    """Returns a numpy array of dimension (n_triplets, 2). First column is
    the img name of anchor/neighbor tiles and second column is img name
    of distant tiles.
    """
    img_names = []
    for filename in os.listdir(img_dir):
        if filename.endswith(img_ext):
            img_names.append(filename)
    img_triplets = list(map(lambda _: random.choice(img_names), range(2 * n_triplets)))
    img_triplets = np.array(img_triplets)
    return img_triplets.reshape((-1, 2))


def get_triplet_tiles(
    tile_dir: str,
    img_dir: str,
    img_triplets: np.ndarray,
    tile_size: int = 50,
    neighborhood: int = 100,
    val_type: str = "uint8",
    bands_only: bool = False,
    save: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    if not os.path.exists(tile_dir):
        os.makedirs(tile_dir)
    size_even = tile_size % 2 == 0
    tile_radius = tile_size // 2

    n_triplets = img_triplets.shape[0]
    unique_imgs = np.unique(img_triplets)
    tiles = np.zeros((n_triplets, 3, 2), dtype=np.int16)

    for img_name in unique_imgs:
        print(f"Sampling image {img_name}")
        if img_name[-3:] == "npy":
            img = np.load(img_name)
        else:
            img = load_img(
                os.path.join(img_dir, img_name),
                val_type=val_type,
                bands_only=bands_only,
            )
        img_padded = np.pad(
            img,
            pad_width=[(tile_radius, tile_radius), (tile_radius, tile_radius), (0, 0)],
            mode="reflect",
        )
        img_shape = img_padded.shape

        for idx, row in enumerate(img_triplets):
            if row[0] == img_name:
                xa, ya = sample_anchor(img_shape, tile_radius)
                xn, yn = sample_neighbor(img_shape, xa, ya, neighborhood, tile_radius)

                if verbose:
                    print(f"    Saving anchor and neighbor tile #{idx}")
                    print(f"    Anchor tile center:{(xa, ya)}")
                    print(f"    Neighbor tile center:{(xn, yn)}")
                if save:
                    tile_anchor = extract_tile(img_padded, xa, ya, tile_radius)
                    tile_neighbor = extract_tile(img_padded, xn, yn, tile_radius)
                    if size_even:
                        tile_anchor = tile_anchor[:-1, :-1]
                        tile_neighbor = tile_neighbor[:-1, :-1]
                    np.save(os.path.join(tile_dir, f"{idx}anchor.npy"), tile_anchor)
                    np.save(os.path.join(tile_dir, f"{idx}neighbor.npy"), tile_neighbor)

                tiles[idx, 0, :] = xa - tile_radius, ya - tile_radius
                tiles[idx, 1, :] = xn - tile_radius, yn - tile_radius

                if row[1] == img_name:
                    # distant image is same as anchor/neighbor image
                    xd, yd = sample_distant_same(
                        img_shape, xa, ya, neighborhood, tile_radius
                    )
                    if verbose:
                        print(f"    Saving distant tile #{idx}")
                        print(f"    Distant tile center:{(xd, yd)}")
                    if save:
                        tile_distant = extract_tile(img_padded, xd, yd, tile_radius)
                        if size_even:
                            tile_distant = tile_distant[:-1, :-1]
                        np.save(
                            os.path.join(tile_dir, f"{idx}distant.npy"), tile_distant
                        )
                    tiles[idx, 2, :] = xd - tile_radius, yd - tile_radius

            elif row[1] == img_name:
                # distant image is different from anchor/neighbor image
                xd, yd = sample_distant_diff(img_shape, tile_radius)
                if verbose:
                    print(f"    Saving distant tile #{idx}")
                    print(f"    Distant tile center:{(xd, yd)}")
                if save:
                    tile_distant = extract_tile(img_padded, xd, yd, tile_radius)
                    if size_even:
                        tile_distant = tile_distant[:-1, :-1]
                    np.save(os.path.join(tile_dir, f"{idx}distant.npy"), tile_distant)
                tiles[idx, 2, :] = xd - tile_radius, yd - tile_radius

    return tiles


def sample_anchor(img_shape: tuple[int, int, int], tile_radius: int) -> tuple[int, int]:
    w_padded, h_padded, c = img_shape
    w = w_padded - 2 * tile_radius
    h = h_padded - 2 * tile_radius

    xa = np.random.randint(0, w) + tile_radius
    ya = np.random.randint(0, h) + tile_radius
    return xa, ya


def sample_neighbor(
    img_shape: tuple[int, int, int],
    xa: int,
    ya: int,
    neighborhood: int,
    tile_radius: int,
) -> tuple[int, int]:
    w_padded, h_padded, c = img_shape
    w = w_padded - 2 * tile_radius
    h = h_padded - 2 * tile_radius

    xn = np.random.randint(
        max(xa - neighborhood, tile_radius), min(xa + neighborhood, w + tile_radius)
    )
    yn = np.random.randint(
        max(ya - neighborhood, tile_radius), min(ya + neighborhood, h + tile_radius)
    )
    return xn, yn


def sample_distant_same(
    img_shape: tuple[int, int, int],
    xa: int,
    ya: int,
    neighborhood: int,
    tile_radius: int,
) -> tuple[int, int]:
    w_padded, h_padded, c = img_shape
    w = w_padded - 2 * tile_radius
    h = h_padded - 2 * tile_radius

    xd, yd = xa, ya
    while (xd >= xa - neighborhood) and (xd <= xa + neighborhood):
        xd = np.random.randint(0, w) + tile_radius
    while (yd >= ya - neighborhood) and (yd <= ya + neighborhood):
        yd = np.random.randint(0, h) + tile_radius
    return xd, yd


def sample_distant_diff(
    img_shape: tuple[int, int, int], tile_radius: int
) -> tuple[int, int]:
    return sample_anchor(img_shape, tile_radius)


def extract_tile(
    img_padded: np.ndarray, x0: int, y0: int, tile_radius: int
) -> np.ndarray:
    """Extracts a tile from a (padded) image given the row and column of
    the center pixel and the tile size. E.g., if the tile
    size is 15 pixels per side, then the tile radius should be 7.
    """
    w_padded, h_padded, c = img_padded.shape
    row_min = x0 - tile_radius
    row_max = x0 + tile_radius
    col_min = y0 - tile_radius
    col_max = y0 + tile_radius
    assert row_min >= 0, f"Row min: {row_min}"
    assert row_max <= w_padded, f"Row max: {row_max}"
    assert col_min >= 0, f"Col min: {col_min}"
    assert col_max <= h_padded, f"Col max: {col_max}"
    tile = img_padded[row_min : row_max + 1, col_min : col_max + 1, :]
    return tile
