import glob
import os
import time
import multiprocessing
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pyarrow as pa
import torch

from loguru import logger
from torch.utils.data import Dataset

# Set multiprocessing start method for better compatibility with CUDA
# Only set if not already set and not running under torchrun/DDP
# torchrun/DDP manages the start method, so we shouldn't override it
if hasattr(multiprocessing, 'get_start_method'):
    try:
        # Check if we're in a DDP context (torchrun sets these env vars)
        is_ddp = os.environ.get('WORLD_SIZE') is not None or os.environ.get('RANK') is not None
        if not is_ddp:
            current_method = multiprocessing.get_start_method(allow_none=True)
            if current_method is None:
                # Only set if not already set and not in DDP context
                multiprocessing.set_start_method('spawn', force=False)
    except (RuntimeError, ValueError):
        # Already set or not available, ignore
        pass


class MapfArrowDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, device, batch_size, field_of_view_size=None):
        self.all_data_files = self.file_paths = sorted(glob.glob(os.path.join(folder_path, "*.arrow")))
        self.device = device
        self.batch_size = batch_size
        self.dtype = torch.int8
        self.field_of_view_size = field_of_view_size  # Store FOV size for slicing redundant tokens

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
        
        # Pre-allocated buffers for async loading of next file
        self.next_input_tensors = torch.empty(sample_input_tensors.shape, dtype=self.dtype, device=self.device)
        self.next_target_tensors = torch.full(sample_target_tensors.shape, -1, dtype=self.dtype, device=self.device)
        self.next_agents_in_obs = torch.empty(sample_agents_in_obs.shape, dtype=self.dtype, device=self.device)
        self.next_agents_rel_coords = torch.empty(sample_rel_coords.shape, dtype=self.dtype, device=self.device)
        
        # Multiprocessing support for async loading
        # Note: We don't use Manager() to avoid conflicts with torchrun/DDP
        # Worker processes communicate via Queue and SharedMemory only
        self._preload_process = None
        self._preload_queue = Queue(maxsize=1)  # Queue to signal when preload is complete (legacy, kept for compatibility)
        self._field_of_view_size_shared = field_of_view_size  # Store for worker process
        # Queues for worker process communication (initialized when needed)
        self._preload_result_queue = None
        self._preload_error_queue = None
        self._preload_filename = None
        # Store shapes and dtypes for shared memory allocation
        self._sample_shapes_and_dtypes = [
            (sample_input_tensors.shape, sample_input_tensors.dtype),
            (sample_target_tensors.shape, sample_target_tensors.dtype),
            (sample_agents_in_obs.shape, sample_agents_in_obs.dtype),
            (sample_rel_coords.shape, sample_rel_coords.dtype),
        ]

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

    def load_and_transfer_data_file(self, filename, use_next_buffers=False):
        """Load data from file and transfer to GPU. Can be called from main process or worker."""
        start_time = time.monotonic()

        input_tensors, gt_actions, agents_in_obs, agents_rel_coords = self._get_data_from_file(
            filename, field_of_view_size=self.field_of_view_size
        )

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
        
        # Note: use_next_buffers=True path is legacy and not used with multiprocessing implementation
        if use_next_buffers:
            self._preload_queue.put(True)  # Signal that preload is complete (legacy)

    @staticmethod
    def _preload_worker(file_path, field_of_view_size, result_queue, error_queue, shapes_and_dtypes):
        """Worker process function: loads file and processes numpy arrays, uses shared memory to avoid pickling overhead"""
        shared_memories = []
        try:
            start_time = time.monotonic()
            
            # Do CPU-bound work in worker process (file I/O and numpy operations)
            input_tensors, gt_actions, agents_in_obs, agents_rel_coords = MapfArrowDataset._get_data_from_file(
                file_path, field_of_view_size=field_of_view_size
            )
            
            finish_time = time.monotonic() - start_time
            logger.debug(f'Worker: Data from {file_path} processed in ~{round(finish_time, 5)}s')
            
            # Use shared memory to avoid pickling overhead
            # Create shared memory blocks for each array
            arrays_data = []
            for arr, (shape, dtype) in zip(
                [input_tensors, gt_actions, agents_in_obs, agents_rel_coords],
                shapes_and_dtypes
            ):
                arr_bytes = arr.nbytes
                shm = SharedMemory(create=True, size=arr_bytes)
                shared_memories.append(shm)
                
                # Copy array data to shared memory
                shared_arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                shared_arr[:] = arr[:]
                
                arrays_data.append({
                    'name': shm.name,
                    'shape': shape,
                    'dtype': dtype
                })
            
            # Send shared memory info (lightweight, no data copying)
            result_queue.put({
                'arrays': arrays_data,
                'success': True
            })
            
            # Note: shared memory will be cleaned up by main process after reading
            # Worker process keeps references to prevent premature cleanup
            
        except Exception as e:
            logger.error(f"Error in preload worker for file {file_path}: {e}")
            # Clean up shared memory on error
            for shm in shared_memories:
                try:
                    shm.close()
                    shm.unlink()
                except:
                    pass
            error_queue.put({'success': False, 'error': str(e)})

    def _start_preload(self, filename):
        """Start async preloading of next file using multiprocessing"""
        
        # Clear queue in case there's a stale signal
        while not self._preload_queue.empty():
            try:
                self._preload_queue.get_nowait()
            except:
                break
        
        # Wait for previous process to finish
        if self._preload_process is not None and self._preload_process.is_alive():
            self._preload_process.join(timeout=0.1)
            if self._preload_process.is_alive():
                # Force terminate if still alive after timeout
                self._preload_process.terminate()
                self._preload_process.join(timeout=1.0)
        
        # Create queues for communication with worker process
        result_queue = Queue()
        error_queue = Queue()
        
        # Start worker process to do CPU-bound work
        self._preload_process = Process(
            target=self._preload_worker,
            args=(filename, self._field_of_view_size_shared, result_queue, error_queue, self._sample_shapes_and_dtypes),
            daemon=True
        )
        self._preload_process.start()
        
        # Store queues for later retrieval
        self._preload_result_queue = result_queue
        self._preload_error_queue = error_queue
        self._preload_filename = filename

    def _swap_buffers(self):
        """Swap current and next buffers after next file is loaded"""
        # Check if worker process has completed and get results
        if self._preload_process is not None and not self._preload_process.is_alive():
            # Check for errors first
            if self._preload_error_queue is not None and not self._preload_error_queue.empty():
                try:
                    error_info = self._preload_error_queue.get_nowait()
                    logger.warning(f"Preload failed for {self._preload_filename}: {error_info.get('error', 'Unknown error')}")
                except:
                    pass
                # Clean up
                self._cleanup_preload_process()
                return False
            
            # Get results from worker process
            if self._preload_result_queue is not None and not self._preload_result_queue.empty():
                try:
                    result = self._preload_result_queue.get_nowait()
                    if result.get('success', False):
                        # Read from shared memory (no pickling/unpickling overhead)
                        shared_memories = []
                        try:
                            # Read arrays directly from shared memory into torch tensors
                            # This avoids creating intermediate numpy copies
                            arr_infos = result['arrays']
                            
                            # Create numpy views of shared memory and convert directly to torch
                            shm0 = SharedMemory(name=arr_infos[0]['name'])
                            shared_memories.append(shm0)
                            arr0 = np.ndarray(arr_infos[0]['shape'], dtype=arr_infos[0]['dtype'], buffer=shm0.buf)
                            self.next_input_tensors.copy_(
                                torch.from_numpy(arr0).to(dtype=self.dtype), 
                                non_blocking=True
                            )
                            
                            shm1 = SharedMemory(name=arr_infos[1]['name'])
                            shared_memories.append(shm1)
                            arr1 = np.ndarray(arr_infos[1]['shape'], dtype=arr_infos[1]['dtype'], buffer=shm1.buf)
                            self.next_target_tensors.copy_(
                                torch.from_numpy(arr1).to(dtype=self.dtype), 
                                non_blocking=True
                            )
                            
                            shm2 = SharedMemory(name=arr_infos[2]['name'])
                            shared_memories.append(shm2)
                            arr2 = np.ndarray(arr_infos[2]['shape'], dtype=arr_infos[2]['dtype'], buffer=shm2.buf)
                            self.next_agents_in_obs.copy_(
                                torch.from_numpy(arr2).to(dtype=self.dtype), 
                                non_blocking=True
                            )
                            
                            shm3 = SharedMemory(name=arr_infos[3]['name'])
                            shared_memories.append(shm3)
                            arr3 = np.ndarray(arr_infos[3]['shape'], dtype=arr_infos[3]['dtype'], buffer=shm3.buf)
                            self.next_agents_rel_coords.copy_(
                                torch.from_numpy(arr3).to(dtype=self.dtype), 
                                non_blocking=True
                            )
                        finally:
                            # Clean up shared memory immediately after reading
                            for shm in shared_memories:
                                try:
                                    shm.close()
                                    shm.unlink()
                                except:
                                    pass
                        
                        # Swap buffers
                        self.input_tensors, self.next_input_tensors = self.next_input_tensors, self.input_tensors
                        self.target_tensors, self.next_target_tensors = self.next_target_tensors, self.target_tensors
                        self.agents_in_obs, self.next_agents_in_obs = self.next_agents_in_obs, self.agents_in_obs
                        self.agents_rel_coords, self.next_agents_rel_coords = self.next_agents_rel_coords, self.agents_rel_coords
                        
                        # Clean up process
                        self._cleanup_preload_process()
                        
                        return True
                except Exception as e:
                    logger.warning(f"Error retrieving preload results: {e}")
                    self._cleanup_preload_process()
                    return False
        
        return False
    
    def _cleanup_preload_process(self):
        """Clean up preload process and queues"""
        if self._preload_process is not None:
            if self._preload_process.is_alive():
                self._preload_process.terminate()
            self._preload_process.join(timeout=1.0)
            self._preload_process = None
        self._preload_result_queue = None
        self._preload_error_queue = None
        self._preload_filename = None

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
                        # Wait for preload process to finish (with timeout)
                        if self._preload_process is not None:
                            self._preload_process.join(timeout=30.0)  # 30 second timeout
                            
                            if self._preload_process.is_alive():
                                logger.warning(f"Preload timeout for {next_file_path}, will load synchronously")
                                self._preload_process.terminate()
                                self._preload_process.join(timeout=1.0)
                                self._preload_process = None
                            else:
                                # Process finished, try to swap buffers
                                self._swap_buffers()

    def get_shard_size(self):
        return len(self.input_tensors) * len(self.file_paths)

    def get_full_dataset_size(self):
        return len(self.input_tensors) * len(self.all_data_files)
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self._cleanup_preload_process()
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
