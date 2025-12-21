import glob
import os
import time
import threading
from queue import Queue

import numpy as np
import pyarrow as pa
import torch

from loguru import logger
from torch.utils.data import Dataset


class MapfArrowDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, device, batch_size):
        self.all_data_files = self.file_paths = sorted(glob.glob(os.path.join(folder_path, "*.arrow")))
        self.device = device
        self.batch_size = batch_size
        self.dtype = torch.int8

        ddp_local_rank = os.environ.get("LOCAL_RANK")
        ddp_world_size = os.environ.get("WORLD_SIZE")
        # Divide files among DDP workers for training

        if not self.all_data_files:
            raise FileNotFoundError(f"No '.arrow' files found in the provided path: '{folder_path}'")

        if "train" in self.file_paths[0] and ddp_local_rank is not None and ddp_world_size is not None:
            ddp_local_rank, ddp_world_size = int(ddp_local_rank), int(ddp_world_size)
            files_per_worker = len(self.file_paths) // ddp_world_size
            start_index = ddp_local_rank * files_per_worker
            end_index = start_index + files_per_worker
            self.file_paths = self.file_paths[start_index:end_index]

        # pre-allocate memory for the input and target tensors (same file size)
        sample_input_tensors, sample_target_tensors, sample_agents_in_obs, sample_rel_coords = self._get_data_from_file(self.file_paths[0])
        self.input_tensors = torch.empty(sample_input_tensors.shape, dtype=self.dtype, device=self.device)
        self.target_tensors = torch.full(sample_target_tensors.shape, -1, dtype=self.dtype, device=self.device)
        self.agents_in_obs = torch.empty(sample_agents_in_obs.shape, dtype=self.dtype, device=self.device)
        self.agents_rel_coords = torch.empty(sample_rel_coords.shape, dtype=self.dtype, device=self.device)

        logger.info(
            f"Single file tensor size: {self.input_tensors.numel() * self.input_tensors.element_size() / 1e9:.4f} GB")
        
        # Pre-allocated buffers for async loading of next file
        self.next_input_tensors = torch.empty(sample_input_tensors.shape, dtype=self.dtype, device=self.device)
        self.next_target_tensors = torch.full(sample_target_tensors.shape, -1, dtype=self.dtype, device=self.device)
        self.next_agents_in_obs = torch.empty(sample_agents_in_obs.shape, dtype=self.dtype, device=self.device)
        self.next_agents_rel_coords = torch.empty(sample_rel_coords.shape, dtype=self.dtype, device=self.device)
        
        # Threading support for async loading
        self._preload_thread = None
        self._preload_queue = Queue(maxsize=1)  # Queue to signal when preload is complete
        self._preload_lock = threading.Lock()
        self._next_file_ready = False

    @staticmethod
    def _get_data_from_file(file_path, shuffle_data=True, max_num_neighbors=13):
        with pa.memory_map(file_path) as source:
            table = pa.ipc.open_file(source).read_all()
            input_tensors_raw = table["input_tensors"].to_numpy(zero_copy_only=False)
            gt_actions_raw = table["gt_actions"].to_numpy(zero_copy_only=False)
            agents_in_obs_raw = table["agents_in_obs"].to_numpy(zero_copy_only=False)
            agents_rel_coords_raw = table["agents_rel_coords"].to_numpy(zero_copy_only=False)

        # input_tensors: List[t][agent][256] -> np.array[t, agent, 256]
        input_tensors = np.array(
            [[np.array(inner, dtype=np.int8) for inner in batch] for batch in input_tensors_raw],
            dtype=np.int8,
        )

        # gt_actions: List[timestep][agent][10] -> full action horizon per agent
        # Result shape: [num_timesteps, num_agents, horizon]
        gt_actions = np.array(
            [
                [np.array(agent_actions, dtype=np.int8) for agent_actions in actions_at_t]
                for actions_at_t in gt_actions_raw
            ],
            dtype=np.int8,
        )

        # agents_in_obs_raw: List[t][agent][int8] (variable length neighbor lists)
        # We need a dense tensor: [t, agent, max_num_neighbors], padded with -1.
        num_timesteps = len(agents_in_obs_raw)
        num_agents = len(agents_in_obs_raw[0]) if num_timesteps > 0 else 0

        agents_in_obs = np.full(
            (num_timesteps, num_agents, max_num_neighbors),
            fill_value=-1,
            dtype=np.int8,
        )

        for t, obs in enumerate(agents_in_obs_raw):
            for a, o in enumerate(obs):
                length = min(len(o), max_num_neighbors)
                if length > 0:
                    agents_in_obs[t, a, :length] = np.array(o[:length], dtype=np.int8)


        agents_rel_coords = np.array(
            [[np.array(coords, dtype=np.int8) for coords in coords_at_t] for coords_at_t in agents_rel_coords_raw],
            dtype=np.int8,
        )

        # shuffle data within the current file
        if shuffle_data:
            indices = np.random.permutation(len(input_tensors))
            input_tensors = input_tensors[indices]
            gt_actions = gt_actions[indices]
            agents_in_obs = agents_in_obs[indices]
            agents_rel_coords = agents_rel_coords[indices]

        return input_tensors, gt_actions, agents_in_obs, agents_rel_coords

    def load_and_transfer_data_file(self, filename, use_next_buffers=False):
        start_time = time.monotonic()

        input_tensors, gt_actions, agents_in_obs, agents_rel_coords = self._get_data_from_file(filename)

        if use_next_buffers:
            target_input = self.next_input_tensors
            target_target = self.next_target_tensors
            target_agents = self.next_agents_in_obs
            target_coords = self.next_agents_rel_coords
        else:
            target_input = self.input_tensors
            target_target = self.target_tensors
            target_agents = self.agents_in_obs
            target_coords = self.agents_rel_coords

        target_input.copy_(torch.tensor(input_tensors, dtype=self.dtype), non_blocking=True)
        target_target.copy_(torch.tensor(gt_actions, dtype=self.dtype), non_blocking=True)
        target_agents.copy_(torch.tensor(agents_in_obs, dtype=self.dtype), non_blocking=True)
        target_coords.copy_(torch.tensor(agents_rel_coords, dtype=self.dtype), non_blocking=True)

        finish_time = time.monotonic() - start_time
        logger.debug(f'Data from {filename} for {self.device} device prepared in ~{round(finish_time, 5)}s')
        
        if use_next_buffers:
            with self._preload_lock:
                self._next_file_ready = True
            self._preload_queue.put(True)  # Signal that preload is complete

    def _preload_next_file(self, filename):
        """Load next file asynchronously into next_* buffers"""
        try:
            self.load_and_transfer_data_file(filename, use_next_buffers=True)
        except Exception as e:
            logger.error(f"Error preloading file {filename}: {e}")
            with self._preload_lock:
                self._next_file_ready = False
            self._preload_queue.put(False)  # Signal error

    def _start_preload(self, filename):
        """Start async preloading of next file"""
        with self._preload_lock:
            self._next_file_ready = False
        # Clear queue in case there's a stale signal
        while not self._preload_queue.empty():
            try:
                self._preload_queue.get_nowait()
            except:
                break
        
        if self._preload_thread is not None and self._preload_thread.is_alive():
            self._preload_thread.join(timeout=0.1)  # Wait briefly for previous preload to finish
        
        self._preload_thread = threading.Thread(target=self._preload_next_file, args=(filename,), daemon=True)
        self._preload_thread.start()

    def _swap_buffers(self):
        """Swap current and next buffers after next file is loaded"""
        with self._preload_lock:
            if not self._next_file_ready:
                return False
            
            # Swap buffers
            self.input_tensors, self.next_input_tensors = self.next_input_tensors, self.input_tensors
            self.target_tensors, self.next_target_tensors = self.next_target_tensors, self.target_tensors
            self.agents_in_obs, self.next_agents_in_obs = self.next_agents_in_obs, self.agents_in_obs
            self.agents_rel_coords, self.next_agents_rel_coords = self.next_agents_rel_coords, self.agents_rel_coords
            
            self._next_file_ready = False
            return True

    def __iter__(self):
        while True:
            for file_idx, file_path in enumerate(self.file_paths):
                # Check if we have a preloaded file ready (works for wrap-around too)
                if self._swap_buffers():
                    # Successfully swapped, next file was already loaded
                    pass
                else:
                    # Load current file synchronously (first iteration or preload didn't complete)
                    self.load_and_transfer_data_file(file_path)
                
                # Start preloading next file asynchronously (if not the last file)
                next_file_idx = (file_idx + 1) % len(self.file_paths)
                if len(self.file_paths) > 1:  # Only preload if there are multiple files
                    next_file_path = self.file_paths[next_file_idx]
                    self._start_preload(next_file_path)
                
                # Yield batches from current file
                num_batches = (len(self.input_tensors) + self.batch_size - 1) // self.batch_size
                for batch_idx in range(num_batches):
                    i = batch_idx * self.batch_size
                    yield (self.input_tensors[i:i + self.batch_size], 
                           self.target_tensors[i:i + self.batch_size], 
                           self.agents_in_obs[i:i + self.batch_size],
                           self.agents_rel_coords[i:i + self.batch_size])
                    
                    # On the last batch of current file, check if preload is ready
                    # If not ready yet, wait for it (but this is non-blocking for training
                    # since we're done with current file anyway)
                    if batch_idx == num_batches - 1 and len(self.file_paths) > 1:
                        # Check if preload is already complete
                        with self._preload_lock:
                            is_ready = self._next_file_ready
                        
                        if not is_ready:
                            # Wait for preload to finish (with timeout to avoid infinite wait)
                            try:
                                result = self._preload_queue.get(timeout=30.0)  # 30 second timeout
                                if not result:
                                    logger.warning(f"Preload failed for {next_file_path}, will load synchronously")
                            except:
                                logger.warning(f"Preload timeout for {next_file_path}, will load synchronously")
                        
                        # Ensure thread has finished
                        if self._preload_thread is not None:
                            self._preload_thread.join(timeout=1.0)

    def get_shard_size(self):
        return len(self.input_tensors) * len(self.file_paths)

    def get_full_dataset_size(self):
        return len(self.input_tensors) * len(self.all_data_files)


def main():
    folder_path = "../dataset/train/mazes"
    dataset = MapfArrowDataset(folder_path, device='cuda:0', batch_size=32)
    data = iter(dataset)
    x = 0
    logger.info(dataset.get_full_dataset_size())
    logger.info(dataset.get_shard_size())

    while True:
        x += 1
        observations, actions, agent_chat_ids, agents_rel_coords = next(data)
        logger.info(str(observations.shape) + ' ' + str(actions.shape) + ' ' + str(agent_chat_ids.shape) + ' ' + str(agents_rel_coords.shape))
        logger.info('Tokenized observation example:' + str(observations[0][0]))
        logger.info('Action:' + str(actions[0][0]))
        logger.info('Chat ids:' + str(agent_chat_ids[0][0]))
        logger.info('Rel coords:' + str(agents_rel_coords[0][0]))
        exit(0)


if __name__ == "__main__":
    main()
