import glob
import os
import time

import numpy as np
import pyarrow as pa
import torch
import random
from loguru import logger
from torch.utils.data import Dataset


ACTION_HORIZON = 3


class MapfArrowDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, device, batch_size):
        self.all_data_files = self.file_paths = sorted(glob.glob(os.path.join(folder_path, "*.arrow")))
        self.device = device
        self.batch_size = batch_size
        self.dtype = torch.int8

        ddp_local_rank = int(device.split(':')[-1])
        ddp_world_size = os.environ.get("WORLD_SIZE")
        random.shuffle(self.file_paths)
        if "dagger" in folder_path or "ddg" in folder_path:
            self.file_paths = sorted(self.file_paths,
                                     key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]),
                                     reverse=True)
        if "validation" not in folder_path and ddp_local_rank is not None and ddp_world_size is not None:
            self.file_paths = self.file_paths[int(ddp_local_rank)::int(ddp_world_size)]

        # pre-allocate memory for the input and target tensors (same file size)
        sample_input_tensors, sample_gt_actions = self._get_data_from_file(self.file_paths[0])

        self.input_tensors = torch.empty(sample_input_tensors.shape, dtype=self.dtype, device=self.device)
        self.target_tensors = torch.full(
            sample_input_tensors.shape + (ACTION_HORIZON,),
            -1,
            dtype=self.dtype,
            device=self.device,
        )

        logger.info(
            f"Single file tensor size: {self.input_tensors.numel() * self.input_tensors.element_size() / 1e9:.4f} GB")

    @staticmethod
    def _get_data_from_file(file_path):
        with pa.memory_map(file_path) as source:
            table = pa.ipc.open_file(source).read_all()
            input_tensors = table["input_tensors"].to_numpy()
            gt_actions = table["gt_actions"].to_numpy()

        # shuffle data within the current file
        indices = np.random.permutation(len(input_tensors))
        input_tensors = np.stack(input_tensors[indices])
        # `gt_actions` comes as an object array of Python lists; stack into a dense int8 array
        gt_actions = np.stack(gt_actions[indices]).astype(np.int8, copy=False)

        return input_tensors, gt_actions

    def load_and_transfer_data_file(self, filename):
        start_time = time.monotonic()

        input_tensors, gt_actions = self._get_data_from_file(filename)

        self.input_tensors.copy_(torch.tensor(input_tensors, dtype=self.dtype), non_blocking=True)
        gt_actions_tensor = torch.tensor(gt_actions, dtype=self.dtype)
        if gt_actions_tensor.ndim == 1:
            gt_actions_tensor = gt_actions_tensor.unsqueeze(-1)
        if gt_actions_tensor.size(-1) != ACTION_HORIZON:
            raise ValueError(
                f"Expected gt_actions last dimension to be {ACTION_HORIZON}, got {gt_actions_tensor.size(-1)}"
            )
        self.target_tensors[:, -1, :].copy_(gt_actions_tensor, non_blocking=True)
        finish_time = time.monotonic() - start_time
        logger.debug(f'Data from {filename} for {self.device} device prepared in ~{round(finish_time, 5)}s')

    def __iter__(self):
        while True:
            for file_path in self.file_paths:
                self.load_and_transfer_data_file(file_path)
                num_samples = len(self.input_tensors)
                if num_samples < self.batch_size:
                    raise KeyError('The dataset is too small to sample a single batch.')
                for i in range(0, num_samples - num_samples % self.batch_size, self.batch_size):
                    yield self.input_tensors[i:i + self.batch_size], self.target_tensors[i:i + self.batch_size]

    def get_shard_size(self):
        return len(self.input_tensors) * len(self.file_paths)

    def get_full_dataset_size(self):
        return len(self.input_tensors) * len(self.all_data_files)


def main():
    # folder_path = "../dataset/validation"
    folder_path = "../dataset/train"
    dataset = MapfArrowDataset(folder_path, device='cuda:0', batch_size=3 * 256)
    data = iter(dataset)
    x = 0
    logger.info(dataset.get_full_dataset_size())
    logger.info(dataset.get_shard_size())

    while True:
        x += 1
        qx, qy = next(data)
        logger.info(str(qx.shape) + ' ' + str(qy.shape))


if __name__ == "__main__":
    main()
