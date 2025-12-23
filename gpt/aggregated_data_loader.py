from loguru import logger
import torch

from torch.utils.data import Dataset
from gpt.fast_data_loader import MapfArrowDataset


class AggregatedMapfArrowDataset(Dataset):
    def __init__(self, folder_paths, device, batch_sizes, field_of_view_size=None):
        """
        Aggregates datasets from multiple folders into a single dataset.

        Args:
            folder_paths (list): List of folder paths containing datasets.
            device (str): Device to load the data onto (e.g., 'cuda:0').
            batch_sizes (list): List of batch sizes for each dataset.
            field_of_view_size (int, optional): Number of FOV tokens to keep. If None, keeps all tokens.
        """
        assert len(folder_paths) == len(batch_sizes), "Each dataset must have a corresponding batch size."

        self.datasets = []
        self.batch_sizes = batch_sizes

        for folder_path, batch_size in zip(folder_paths, batch_sizes):
            dataset = MapfArrowDataset(folder_path, device, batch_size, field_of_view_size=field_of_view_size)
            self.datasets.append(dataset)

        self.device = device

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