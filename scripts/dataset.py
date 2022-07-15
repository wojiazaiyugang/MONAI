import numpy as np
from torch.utils.data import Dataset


class RandomSubItemListDataset(Dataset):
    """
    Assume the original item is list/tuple type, and when the item list length is too large,
    the monai collate_fn will make the final batch too large. This Dataset will randomly
    return a subset within each item.

    Note,
        The next __getitem__ may return subset with common elements. The class can be used for
        training even if each epoch may have already seen samples.
    """

    def __init__(self, dataset, max_len=1) -> None:
        super().__init__()
        self.dataset = dataset
        self.max_len = max_len
        assert self.max_len >= 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item_list = self.dataset[index]
        isinstance(item_list, (list, tuple))
        assert len(item_list) > 0
        num = len(item_list)
        perm = np.random.permutation(num)
        indices = perm[:min(self.max_len, num)]
        sub_items = [item_list[i] for i in indices]
        return sub_items