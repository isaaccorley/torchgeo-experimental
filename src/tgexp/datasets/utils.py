from collections.abc import Sequence

from torch.utils.data import Dataset, random_split


def dataset_split(
    dataset: Dataset, val_pct: float, test_pct: float | None = None
) -> list[Sequence[int]]:
    """Split a dataset into training, validation, and optionally test sets and returns the indices.

    Parameters
    ----------
    dataset : Dataset
        The dataset to be split.
    val_pct : float
        The proportion of the dataset to include in the validation split (between 0 and 1).
    test_pct : float, optional
        The proportion of the dataset to include in the test split (between 0 and 1). If None, only training and validation splits are returned.

    Returns
    -------
    list of Sequence[int]
        A list containing the indices for each split. If `test_pct` is None, returns [train_indices, val_indices]. Otherwise, returns [train_indices, val_indices, test_indices].

    """
    if test_pct is None:
        val_length = int(len(dataset) * val_pct)
        train_length = len(dataset) - val_length
        train_dataset, val_dataset = random_split(dataset, [train_length, val_length])
        return [train_dataset.indices, val_dataset.indices]
    else:
        val_length = int(len(dataset) * val_pct)
        test_length = int(len(dataset) * test_pct)
        train_length = len(dataset) - (val_length + test_length)
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_length, val_length, test_length]
        )
        return [train_dataset.indices, val_dataset.indices, test_dataset.indices]
