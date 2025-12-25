from loguru import logger
import torch
import threading
from concurrent.futures import ThreadPoolExecutor

from torch.utils.data import Dataset
from gpt.fast_data_loader import MapfArrowDataset


class PreloadCoordinator:
    """
    Centralized coordinator for managing preload requests across multiple loaders.
    Prevents I/O saturation by limiting concurrent file operations and prioritizing
    loaders that are closer to needing their next file (based on batch progress).
    """
    def __init__(self, max_concurrent_io=2):
        self.executor = ThreadPoolExecutor(
            max_workers=max_concurrent_io,
            thread_name_prefix="DataLoaderIO"
        )
        self.pending_preloads = {}  # loader_id -> Future
        self.pending_requests = {}  # loader_id -> (priority, request_info)
        self.active_preloads = set()  # loader_ids currently being preloaded
        self.lock = threading.Lock()
        self._load_file_func = None  # Will be set by the dataset
        self.max_concurrent_io = max_concurrent_io
    
    def register_load_function(self, load_func):
        """Register the function to use for loading files"""
        self._load_file_func = load_func
    
    def request_preload(self, loader_id, file_path, field_of_view_size, result_queue, error_queue, priority=0):
        """
        Request a preload for a specific loader with a priority.
        Lower priority value = higher priority (will be preloaded first).
        Priority should be based on batches remaining in current file.
        
        Args:
            loader_id: Unique identifier for the loader
            file_path: Path to file to preload
            field_of_view_size: FOV size parameter
            result_queue: Queue to put results in
            error_queue: Queue to put errors in
            priority: Priority value (lower = higher priority, based on batches remaining)
        
        Returns the Future object for tracking completion.
        """
        with self.lock:
            # Cancel existing preload for this loader if any
            if loader_id in self.pending_preloads:
                old_future = self.pending_preloads[loader_id]
                if not old_future.done():
                    old_future.cancel()
                del self.pending_preloads[loader_id]
                self.active_preloads.discard(loader_id)
            
            # Store request with priority
            self.pending_requests[loader_id] = (
                priority,
                (loader_id, file_path, field_of_view_size, result_queue, error_queue)
            )
            
            # Try to start preloads (will prioritize based on priority)
            self._process_pending_requests()
            
            # Return future if it was started, otherwise None (will be started later)
            return self.pending_preloads.get(loader_id)
    
    def _process_pending_requests(self):
        """Process pending requests, starting highest priority ones up to max_concurrent_io limit"""
        # Clean up completed preloads first
        self._cleanup_completed()
        
        # Sort pending requests by priority (lower priority value = higher priority)
        sorted_requests = sorted(
            self.pending_requests.items(),
            key=lambda x: x[1][0]  # Sort by priority
        )
        
        # Start preloads up to the limit
        for loader_id, (priority, request_info) in sorted_requests:
            if len(self.active_preloads) >= self.max_concurrent_io:
                break  # Reached limit
            
            if loader_id not in self.active_preloads:
                # Start this preload
                loader_id, file_path, field_of_view_size, result_queue, error_queue = request_info
                
                future = self.executor.submit(
                    self._preload_worker,
                    file_path,
                    field_of_view_size,
                    result_queue,
                    error_queue
                )
                self.pending_preloads[loader_id] = future
                self.active_preloads.add(loader_id)
                del self.pending_requests[loader_id]
    
    def update_priority(self, loader_id, new_priority):
        """
        Update the priority of a pending preload request.
        If the preload is already active, this has no effect.
        Only updates if the request is still pending (not yet started).
        """
        with self.lock:
            # Only update if request is still pending (not yet active)
            if loader_id in self.pending_requests and loader_id not in self.active_preloads:
                # Update priority and re-sort
                old_priority, request_info = self.pending_requests[loader_id]
                self.pending_requests[loader_id] = (new_priority, request_info)
                # Re-process to potentially start higher priority requests
                self._process_pending_requests()
    
    def _preload_worker(self, file_path, field_of_view_size, result_queue, error_queue):
        """Worker function that loads file and processes numpy arrays"""
        try:
            import time
            import os
            start_time = time.monotonic()
            
            # Use the registered load function
            input_tensors, gt_actions, agents_in_obs, agents_rel_coords = self._load_file_func(
                file_path, field_of_view_size=field_of_view_size
            )
            
            finish_time = time.monotonic() - start_time
            thread_id = threading.current_thread().ident
            logger.debug(f'Coordinator worker (thread {thread_id}): Data from {os.path.basename(file_path)} processed in ~{round(finish_time, 5)}s')
            
            # Put numpy arrays directly in queue
            result_queue.put({
                'input_tensors': input_tensors,
                'gt_actions': gt_actions,
                'agents_in_obs': agents_in_obs,
                'agents_rel_coords': agents_rel_coords,
                'success': True
            })
            
        except Exception as e:
            logger.error(f"Error in coordinator preload worker for file {file_path}: {e}")
            error_queue.put({'success': False, 'error': str(e)})
    
    def _cleanup_completed(self):
        """Remove completed futures from tracking and start next pending requests"""
        completed = []
        for loader_id, future in self.pending_preloads.items():
            if future.done():
                completed.append(loader_id)
        
        for loader_id in completed:
            del self.pending_preloads[loader_id]
            self.active_preloads.discard(loader_id)
        
        # After cleanup, try to start more pending requests
        if completed:
            self._process_pending_requests()
    
    def cancel_preload(self, loader_id):
        """Cancel a pending preload for a specific loader"""
        with self.lock:
            # Cancel active preload if any
            if loader_id in self.pending_preloads:
                future = self.pending_preloads[loader_id]
                if not future.done():
                    future.cancel()
                del self.pending_preloads[loader_id]
                self.active_preloads.discard(loader_id)
            
            # Remove pending request if any
            if loader_id in self.pending_requests:
                del self.pending_requests[loader_id]
    
    def shutdown(self):
        """Shutdown the executor and cancel all pending preloads"""
        with self.lock:
            for future in self.pending_preloads.values():
                if not future.done():
                    future.cancel()
            self.pending_preloads.clear()
            self.pending_requests.clear()
            self.active_preloads.clear()
        self.executor.shutdown(wait=False)


class AggregatedMapfArrowDataset(Dataset):
    def __init__(self, folder_paths, device, batch_sizes, field_of_view_size=None, max_concurrent_io=2):
        """
        Aggregates datasets from multiple folders into a single dataset.

        Args:
            folder_paths (list): List of folder paths containing datasets.
            device (str): Device to load the data onto (e.g., 'cuda:0').
            batch_sizes (list): List of batch sizes for each dataset.
            field_of_view_size (int, optional): Number of FOV tokens to keep. If None, keeps all tokens.
            max_concurrent_io (int): Maximum concurrent I/O operations across all loaders (default: 2).
        """
        assert len(folder_paths) == len(batch_sizes), "Each dataset must have a corresponding batch size."

        # Create centralized preload coordinator
        self.preload_coordinator = PreloadCoordinator(max_concurrent_io=max_concurrent_io)
        
        # Register the load function from MapfArrowDataset
        self.preload_coordinator.register_load_function(MapfArrowDataset._get_data_from_file)

        self.datasets = []
        self.batch_sizes = batch_sizes

        # Create datasets and pass the coordinator
        # Apply batch start offsets to desynchronize loaders and prevent simultaneous file loading
        # Each loader skips a fraction of batches in the first file, spreading out file transitions
        num_loaders = len(folder_paths)
        for idx, (folder_path, batch_size) in enumerate(zip(folder_paths, batch_sizes)):
            # Calculate offset: loader 0 skips 0, loader 1 skips 1/N, loader 2 skips 2/N, etc.
            # This spreads out file transitions across time
            batch_start_offset = idx / num_loaders if num_loaders > 1 else 0.0
            
            dataset = MapfArrowDataset(
                folder_path, 
                device, 
                batch_size, 
                field_of_view_size=field_of_view_size,
                preload_coordinator=self.preload_coordinator,
                loader_id=idx,
                batch_start_offset=batch_start_offset
            )
            self.datasets.append(dataset)

        self.device = device
    
    def __del__(self):
        """Cleanup coordinator on deletion"""
        try:
            if hasattr(self, 'preload_coordinator'):
                self.preload_coordinator.shutdown()
        except:
            pass

    def __iter__(self):
        """
        Creates an iterator that yields data batches from the aggregated datasets with specified batch sizes.
        
        Note: Each underlying dataset automatically preloads its next file asynchronously
        while processing the current file, preventing blocking during file transitions.
        This is especially important when datasets have different batch sizes and thus
        process files at different rates.
        
        Handles datasets with different numbers of agents by padding smaller ones to match the maximum.
        """
        dataset_iters = [iter(dataset) for dataset in self.datasets]

        while True:
            batches = [next(dataset_iter) for dataset_iter in dataset_iters]
            batch_inputs, batch_targets, batch_agents, batch_rel_coords = zip(*batches)
            
            # Find maximum number of agents across all batches
            max_num_agents = max(batch.shape[1] for batch in batch_inputs)
            
            # Pad batches that have fewer agents
            padded_inputs = []
            padded_targets = []
            padded_agents = []
            padded_rel_coords = []
            
            for obs, actions, agents, rel_coords in zip(batch_inputs, batch_targets, batch_agents, batch_rel_coords):
                B, C, *rest_dims = obs.shape
                
                if C < max_num_agents:
                    # Pad observations: pad with zeros (or empty_token_code if you have one)
                    pad_size = max_num_agents - C
                    obs_pad = torch.zeros(B, pad_size, *rest_dims, dtype=obs.dtype, device=obs.device)
                    obs = torch.cat([obs, obs_pad], dim=1)
                    
                    # Pad actions: pad with -1 (ignore_index)
                    actions_pad = torch.full((B, pad_size, *actions.shape[2:]), -1, dtype=actions.dtype, device=actions.device)
                    actions = torch.cat([actions, actions_pad], dim=1)
                    
                    # Pad agent_chat_ids: pad with -1 (empty_connection_code)
                    # agents shape: [B, C, L] where L is number of neighbors
                    agents_pad = torch.full((B, pad_size, agents.shape[2]), -1, dtype=agents.dtype, device=agents.device)
                    agents = torch.cat([agents, agents_pad], dim=1)
                    
                    # Pad agents_rel_coords: pad with appropriate value
                    # rel_coords shape: [B, C, L*2]
                    rel_coords_pad = torch.full((B, pad_size, rel_coords.shape[2]), -1, dtype=rel_coords.dtype, device=rel_coords.device)
                    rel_coords = torch.cat([rel_coords, rel_coords_pad], dim=1)
                
                padded_inputs.append(obs)
                padded_targets.append(actions)
                padded_agents.append(agents)
                padded_rel_coords.append(rel_coords)
            
            yield (torch.cat(padded_inputs, dim=0), 
                   torch.cat(padded_targets, dim=0), 
                   torch.cat(padded_agents, dim=0),
                   torch.cat(padded_rel_coords, dim=0))

    def get_full_dataset_size(self):
        return sum(dataset.get_full_dataset_size() for dataset in self.datasets)

    def get_shard_size(self):
        return sum(dataset.get_shard_size() for dataset in self.datasets)


def main():
    folder_paths = ["../dataset/train/mazes", "../dataset/train/random", "../dataset/train/house"]
    batch_sizes = [16, 8, 8]  # Exact batch sizes for train and validation datasets
    aggregated_dataset = AggregatedMapfArrowDataset(folder_paths, device='cuda:0', batch_sizes=batch_sizes)
    data = iter(aggregated_dataset)

    logger.info(aggregated_dataset.get_full_dataset_size())
    logger.info(aggregated_dataset.get_shard_size())

    x = 0
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