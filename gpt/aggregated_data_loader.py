from loguru import logger
import torch
from torch.utils.data import Dataset

from gpt.fast_data_loader import MapfArrowDataset


class AggregatedMapfArrowDataset(Dataset):
    def __init__(self, folder_paths, device, batch_sizes):
        """
        Aggregates datasets from multiple folders into a single dataset.

        Args:
            folder_paths (list): List of folder paths containing datasets.
            device (str): Device to load the data onto (e.g., 'cuda:0').
            batch_sizes (list): List of batch sizes for each dataset.
        """
        assert len(folder_paths) == len(batch_sizes), "Each dataset must have a corresponding batch size."

        self.datasets = []
        self.batch_sizes = batch_sizes

        for folder_path, batch_size in zip(folder_paths, batch_sizes):
            dataset = MapfArrowDataset(folder_path, device, batch_size)
            self.datasets.append(dataset)

        self.device = device

    def __iter__(self):
        """
        Creates an iterator that yields data batches from the aggregated datasets with specified batch sizes.
        """
        dataset_iters = [iter(dataset) for dataset in self.datasets]

        while True:
            batch_inputs, batch_targets = zip(*[next(dataset_iter) for dataset_iter in dataset_iters])
            yield torch.cat(batch_inputs, dim=0), torch.cat(batch_targets, dim=0)

    def get_full_dataset_size(self):
        return sum(dataset.get_full_dataset_size() for dataset in self.datasets)

    def get_shard_size(self):
        return sum(dataset.get_shard_size() for dataset in self.datasets)


def main():
    folder_paths = ["../dataset/train", "../dataset/validation"]
    # folder_paths = ["../dataset/train", "../dagger"]
    batch_sizes = [24, 8]  # Exact batch sizes for train and validation datasets
    aggregated_dataset = AggregatedMapfArrowDataset(folder_paths, device='cuda:0', batch_sizes=batch_sizes)
    data = iter(aggregated_dataset)

    logger.info(aggregated_dataset.get_full_dataset_size())
    logger.info(aggregated_dataset.get_shard_size())

    x = 0
    while True:
        x += 1
        qx, qy = next(data)
        # logger.info(str(qx.shape) + ' ' + str(qy.shape))


if __name__ == "__main__":
    main()
