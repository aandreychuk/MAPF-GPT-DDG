import glob
import os
import time

import numpy as np
import pyarrow as pa
import torch
import random
from loguru import logger
from torch.utils.data import Dataset


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
        
        # Determine action horizon dynamically from the loaded data
        self.action_horizon = sample_gt_actions.shape[-1] if sample_gt_actions.ndim > 1 else 1

        self.input_tensors = torch.empty(sample_input_tensors.shape, dtype=self.dtype, device=self.device)
        self.target_tensors = torch.full(
            sample_input_tensors.shape + (self.action_horizon,),
            -1,
            dtype=self.dtype,
            device=self.device,
        )

        logger.info(
            f"Single file tensor size: {self.input_tensors.numel() * self.input_tensors.element_size() / 1e9:.4f} GB")
        logger.info(f"Detected action horizon: {self.action_horizon}")

    @staticmethod
    def _get_data_from_file(file_path):
        with pa.memory_map(file_path) as source:
            table = pa.ipc.open_file(source).read_all()
            input_tensors = table["input_tensors"].to_numpy()
            gt_actions = table["gt_actions"].to_numpy()

        # Flatten the data: convert from List[timestep][agent][...] to List[agent_obs][...]
        flattened_inputs = []
        flattened_actions = []
        
        for t in range(len(input_tensors)):
            for agent_idx in range(len(input_tensors[t])):
                flattened_inputs.append(input_tensors[t][agent_idx])
                flattened_actions.append(gt_actions[t][agent_idx])

        # Convert to numpy arrays
        flattened_inputs = np.array(flattened_inputs)
        flattened_actions = np.array(flattened_actions)

        # shuffle data within the current file
        indices = np.random.permutation(len(flattened_inputs))
        flattened_inputs = flattened_inputs[indices]
        flattened_actions = flattened_actions[indices]

        return flattened_inputs, flattened_actions

    def load_and_transfer_data_file(self, filename):
        start_time = time.monotonic()

        input_tensors, gt_actions = self._get_data_from_file(filename)

        self.input_tensors.copy_(torch.tensor(input_tensors, dtype=self.dtype), non_blocking=True)
        gt_actions_tensor = torch.tensor(gt_actions, dtype=self.dtype)
        
        # Ensure gt_actions has the correct shape for action_horizon
        if gt_actions_tensor.ndim == 1:
            gt_actions_tensor = gt_actions_tensor.unsqueeze(-1)
        if gt_actions_tensor.size(-1) != self.action_horizon:
            raise ValueError(
                f"Expected gt_actions last dimension to be {self.action_horizon}, got {gt_actions_tensor.size(-1)}"
            )
        
        # Copy the action tensor to the target tensor
        self.target_tensors.copy_(gt_actions_tensor, non_blocking=True)
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
