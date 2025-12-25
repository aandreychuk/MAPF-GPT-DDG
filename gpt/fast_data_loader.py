import glob
import os
import time
import threading
from queue import Queue as ThreadQueue
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
import pyarrow as pa
import torch

from loguru import logger
from torch.utils.data import Dataset

# Shared thread pool for I/O operations
# This prevents I/O saturation when multiple loaders run simultaneously
_shared_io_executor = None
_shared_io_executor_lock = threading.Lock()
_MAX_CONCURRENT_IO = 2  # Limit concurrent file operations to prevent I/O saturation

def _get_shared_io_executor():
    """Get or create shared thread pool executor for I/O operations"""
    global _shared_io_executor
    with _shared_io_executor_lock:
        if _shared_io_executor is None:
            _shared_io_executor = ThreadPoolExecutor(
                max_workers=_MAX_CONCURRENT_IO,
                thread_name_prefix="DataLoaderIO"
            )
    return _shared_io_executor


class MapfArrowDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, device, batch_size, field_of_view_size=None, 
                 preload_coordinator=None, loader_id=None):
        self.all_data_files = self.file_paths = sorted(glob.glob(os.path.join(folder_path, "*.arrow")))
        self.device = device
        self.batch_size = batch_size
        self.dtype = torch.int8
        self.field_of_view_size = field_of_view_size  # Store FOV size for slicing redundant tokens
        self.preload_coordinator = preload_coordinator
        self.loader_id = loader_id if loader_id is not None else id(self)  # Unique ID for this loader

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
        sample_input_tensors, sample_target_tensors, sample_agents_in_obs, sample_rel_coords = self._get_data_from_file(
            self.file_paths[0], field_of_view_size=self.field_of_view_size
        )
        self.input_tensors = torch.empty(sample_input_tensors.shape, dtype=self.dtype, device=self.device)
        self.target_tensors = torch.full(sample_target_tensors.shape, -1, dtype=self.dtype, device=self.device)
        self.agents_in_obs = torch.empty(sample_agents_in_obs.shape, dtype=self.dtype, device=self.device)
        self.agents_rel_coords = torch.empty(sample_rel_coords.shape, dtype=self.dtype, device=self.device)

        logger.info(
            f"Single file tensor size: {self.input_tensors.numel() * self.input_tensors.element_size() / 1e9:.4f} GB")
        
        # Pre-allocated buffers for double buffering (current and next only)
        # Next buffer: preloaded file ready to swap when current finishes
        self.next_input_tensors = torch.empty(sample_input_tensors.shape, dtype=self.dtype, device=self.device)
        self.next_target_tensors = torch.full(sample_target_tensors.shape, -1, dtype=self.dtype, device=self.device)
        self.next_agents_in_obs = torch.empty(sample_agents_in_obs.shape, dtype=self.dtype, device=self.device)
        self.next_agents_rel_coords = torch.empty(sample_rel_coords.shape, dtype=self.dtype, device=self.device)
        
        # Preload state for next file (managed by coordinator if available)
        self._preload_future = None  # Future from coordinator for next file
        self._preload_result_queue = None
        self._preload_error_queue = None
        self._preload_filename = None
        
        # Track batch progress for priority calculation
        self._current_batch_idx = 0
        self._total_batches_in_file = 0
        
        # Track if next buffer is ready
        self._next_ready = False  # True when next file is loaded and ready

    @staticmethod
    def _get_data_from_file(file_path, shuffle_data=True, max_num_neighbors=13, field_of_view_size=None):
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
        
        # Discard redundant tokens from FOV: keep only first field_of_view_size tokens
        # If dataset has 128 FOV tokens and we want 121, slice to [:121]
        if field_of_view_size is not None and input_tensors.shape[2] > field_of_view_size:
            original_size = input_tensors.shape[2]
            input_tensors = input_tensors[:, :, :field_of_view_size]
            logger.debug(f"Sliced input_tensors from {original_size} to {field_of_view_size} tokens (removed {original_size - field_of_view_size} redundant tokens)")

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

    def load_and_transfer_data_file(self, filename):
        """Load data from file and transfer to GPU (reuses existing GPU buffers)"""
        start_time = time.monotonic()

        input_tensors, gt_actions, agents_in_obs, agents_rel_coords = self._get_data_from_file(
            filename, field_of_view_size=self.field_of_view_size
        )

        # Copy directly to existing GPU buffers (no intermediate tensor allocation)
        # torch.from_numpy() shares memory with numpy array (zero-copy on CPU if contiguous)
        # Since numpy arrays are already int8, torch.from_numpy() creates int8 tensor directly
        self.input_tensors.copy_(torch.from_numpy(input_tensors), non_blocking=True)
        self.target_tensors.copy_(torch.from_numpy(gt_actions), non_blocking=True)
        self.agents_in_obs.copy_(torch.from_numpy(agents_in_obs), non_blocking=True)
        self.agents_rel_coords.copy_(torch.from_numpy(agents_rel_coords), non_blocking=True)

        finish_time = time.monotonic() - start_time
        logger.debug(f'Data from {filename} for {self.device} device prepared in ~{round(finish_time, 5)}s')
    
    def _start_preload(self, filename):
        """
        Start async preloading of next file using centralized coordinator with priority.
        
        Args:
            filename: Path to file to preload into next buffer
        """
        # Prevent duplicate preload requests for the same file
        if self._preload_filename == filename and self._preload_future is not None:
            # Already preloading this file, just update priority if needed
            if self.preload_coordinator is not None:
                batches_remaining = max(0, self._total_batches_in_file - self._current_batch_idx)
                self.preload_coordinator.update_priority(self.loader_id, batches_remaining)
            return
        
        # Wait for previous preload to finish (if different file)
        if self._preload_future is not None:
            try:
                self._preload_future.result(timeout=0.1)
            except:
                pass  # Timeout or error, will be handled in _check_and_load_next_buffer
        
        # Create queues for communication with coordinator worker
        self._preload_result_queue = ThreadQueue()
        self._preload_error_queue = ThreadQueue()
        
        # Calculate priority based on batches remaining
        # Lower priority value = higher priority (will be preloaded first)
        # Priority = batches remaining in current file
        batches_remaining = max(0, self._total_batches_in_file - self._current_batch_idx)
        priority = batches_remaining
        
        if self.preload_coordinator is not None:
            # Use centralized coordinator with priority (preferred - better I/O management)
            self._preload_future = self.preload_coordinator.request_preload(
                self.loader_id,
                filename,
                self.field_of_view_size,
                self._preload_result_queue,
                self._preload_error_queue,
                priority=priority
            )
        else:
            # Fallback to per-loader executor (for standalone usage)
            self._preload_future = _get_shared_io_executor().submit(
                self._preload_worker_thread,
                filename,
                self._preload_result_queue,
                self._preload_error_queue
            )
        
        self._preload_filename = filename
    
    def _preload_worker_thread(self, file_path, result_queue, error_queue):
        """Fallback worker function for standalone usage (without coordinator)"""
        try:
            start_time = time.monotonic()
            
            # Do CPU-bound work in worker thread (file I/O and numpy operations)
            input_tensors, gt_actions, agents_in_obs, agents_rel_coords = self._get_data_from_file(
                file_path, field_of_view_size=self.field_of_view_size
            )
            
            finish_time = time.monotonic() - start_time
            logger.debug(f'Worker thread: Data from {file_path} processed in ~{round(finish_time, 5)}s')
            
            # Put numpy arrays directly in queue
            result_queue.put({
                'input_tensors': input_tensors,
                'gt_actions': gt_actions,
                'agents_in_obs': agents_in_obs,
                'agents_rel_coords': agents_rel_coords,
                'success': True
            })
            
        except Exception as e:
            logger.error(f"Error in preload worker thread for file {file_path}: {e}")
            error_queue.put({'success': False, 'error': str(e)})

    def _check_and_load_next_buffer(self):
        """
        Check if next file preload is complete and load it into next buffer.
        Returns True if buffer was loaded, False otherwise.
        """
        if self._next_ready:
            return True  # Already loaded
        
        # Check if worker has completed
        if self._preload_future is not None and self._preload_future.done():
            # Check for errors first
            if self._preload_error_queue is not None and not self._preload_error_queue.empty():
                try:
                    error_info = self._preload_error_queue.get_nowait()
                    logger.warning(f"Preload failed for {self._preload_filename}: {error_info.get('error', 'Unknown error')}")
                except:
                    pass
                # Clean up
                self._cleanup_preload_worker()
                return False
            
            # Get results from worker
            if self._preload_result_queue is not None and not self._preload_result_queue.empty():
                try:
                    result = self._preload_result_queue.get_nowait()
                    if result.get('success', False):
                        # Transfer numpy arrays directly to existing GPU buffers
                        self.next_input_tensors.copy_(
                            torch.from_numpy(result['input_tensors']),
                            non_blocking=True
                        )
                        self.next_target_tensors.copy_(
                            torch.from_numpy(result['gt_actions']),
                            non_blocking=True
                        )
                        self.next_agents_in_obs.copy_(
                            torch.from_numpy(result['agents_in_obs']),
                            non_blocking=True
                        )
                        self.next_agents_rel_coords.copy_(
                            torch.from_numpy(result['agents_rel_coords']),
                            non_blocking=True
                        )
                        
                        self._next_ready = True
                        self._cleanup_preload_worker()
                        return True
                except Exception as e:
                    logger.warning(f"Error retrieving preload results: {e}")
                    self._cleanup_preload_worker()
                    return False
        
        return False
    
    def _swap_buffers(self):
        """
        Swap current and next buffers (double buffering).
        After swap, next buffer is empty and ready for new preload.
        Returns True if swap was successful, False otherwise.
        """
        # Check if next buffer is ready
        if self._check_and_load_next_buffer():
            # Swap current and next buffers
            self.input_tensors, self.next_input_tensors = self.next_input_tensors, self.input_tensors
            self.target_tensors, self.next_target_tensors = self.next_target_tensors, self.target_tensors
            self.agents_in_obs, self.next_agents_in_obs = self.next_agents_in_obs, self.agents_in_obs
            self.agents_rel_coords, self.next_agents_rel_coords = self.next_agents_rel_coords, self.agents_rel_coords
            
            # Mark next as not ready (it's now current, and the old current is now next but empty)
            self._next_ready = False
            return True
        
        return False
    
    def _cleanup_preload_worker(self):
        """Clean up preload worker and queues for next file"""
        if self._preload_future is not None:
            # Future cleanup - check for exceptions but don't wait
            try:
                self._preload_future.result(timeout=0.0)
            except:
                pass  # Ignore exceptions/timeouts
            # Cancel in coordinator if using one
            if self.preload_coordinator is not None:
                self.preload_coordinator.cancel_preload(self.loader_id)
            self._preload_future = None
        self._preload_result_queue = None
        self._preload_error_queue = None
        self._preload_filename = None

    def __iter__(self):
        while True:
            for file_idx, file_path in enumerate(self.file_paths):
                # Check if we have a preloaded file ready (works for wrap-around too)
                if self._swap_buffers():
                    # Successfully swapped, next file was already loaded and is now current
                    # After swap, we're effectively on file_idx+1, so next file is file_idx+2
                    next_file_idx = (file_idx + 2) % len(self.file_paths)
                    if len(self.file_paths) > 1:
                        next_file_path = self.file_paths[next_file_idx]
                        self._start_preload(next_file_path)
                else:
                    # Load current file synchronously (first iteration or preload didn't complete)
                    self.load_and_transfer_data_file(file_path)
                    
                    # Start preloading next file immediately after loading current
                    next_file_idx = (file_idx + 1) % len(self.file_paths)
                    if len(self.file_paths) > 1:
                        next_file_path = self.file_paths[next_file_idx]
                        self._start_preload(next_file_path)
                
                # Calculate total batches for this file
                self._total_batches_in_file = (len(self.input_tensors) + self.batch_size - 1) // self.batch_size
                self._current_batch_idx = 0
                
                # Yield batches from current file
                for batch_idx in range(self._total_batches_in_file):
                    self._current_batch_idx = batch_idx
                    i = batch_idx * self.batch_size
                    yield (self.input_tensors[i:i + self.batch_size], 
                           self.target_tensors[i:i + self.batch_size], 
                           self.agents_in_obs[i:i + self.batch_size],
                           self.agents_rel_coords[i:i + self.batch_size])
                    
                    # Update priority periodically (every 10 batches) to avoid overhead
                    # Loaders closer to finishing get higher priority
                    if self.preload_coordinator is not None:
                        # Only update priority periodically to avoid lock contention
                        if batch_idx % 10 == 0 or batch_idx == self._total_batches_in_file - 1:
                            batches_remaining = self._total_batches_in_file - batch_idx - 1
                            if batches_remaining >= 0:
                                self.preload_coordinator.update_priority(self.loader_id, batches_remaining)
                    
                    # On the last batch of current file, ensure next buffer is ready
                    if batch_idx == self._total_batches_in_file - 1 and len(self.file_paths) > 1:
                        # Check if next buffer is ready (with timeout)
                        if not self._next_ready:
                            if self._preload_future is not None:
                                try:
                                    self._preload_future.result(timeout=30.0)  # 30 second timeout
                                    self._check_and_load_next_buffer()
                                except Exception as e:
                                    logger.warning(f"Preload timeout/error for {next_file_path}: {e}, will load synchronously")
                                    self._cleanup_preload_worker()

    def get_shard_size(self):
        return len(self.input_tensors) * len(self.file_paths)

    def get_full_dataset_size(self):
        return len(self.input_tensors) * len(self.all_data_files)
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self._cleanup_preload_worker()
        except:
            pass  # Ignore errors during cleanup


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
