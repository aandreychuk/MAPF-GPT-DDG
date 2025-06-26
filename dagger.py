import argparse
from pogema_toolbox.registry import ToolboxRegistry
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
import glob

def run_dagger(dagger_type, num_workers, device_id, seed, file_size, run_infinite=False):
    def run_worker(worker_id):
        if 'warehouse' in dagger_type:
            subprocess.run(f"python worker.py --worker_id {worker_id} --device_id {device_id} --map_seed {0} --scenario_seed {seed + worker_id * 1000} --dataset_path {path_to_dataset} --dagger_type {dagger_type} --path_to_weights {path_to_weights} --num_agents {','.join(map(str, [32, 64, 96, 128, 160, 192]))} --file_size {file_size}", shell=True)
        else:
            subprocess.run(f"python worker.py --worker_id {worker_id} --device_id {device_id} --map_seed {seed + worker_id * 1000} --scenario_seed {0} --dataset_path {path_to_dataset} --dagger_type {dagger_type} --path_to_weights {path_to_weights} --file_size {file_size}", shell=True)
    
    path_to_weights = f"out/ckpt_{dagger_type}.pt"
    path_to_dataset = f"dataset/{dagger_type}"
    ToolboxRegistry.setup_logger('INFO')
    while True:
        checkpoint_files = glob.glob(f"out/ckpt_{dagger_type}_*0.pt")
        if dagger_type == 'dagger' or dagger_type == 'ddg':
            checkpoint_files = [f for f in checkpoint_files if 'warehouse' not in f]
        if checkpoint_files:
            path_to_weights = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            executor.map(run_worker, range(num_workers))
        seed += 1000*num_workers
        if not run_infinite:
            break

def main():
    parser = argparse.ArgumentParser(description='Dagger')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers (default: %(default)d)')
    parser.add_argument('--dagger_type', type=str, choices=['ddg', 'ddg_warehouse', 'dagger', 'dagger_warehouse'], default='ddg', help='Dagger type (default: %(default)s)')
    parser.add_argument('--device_id', type=int, default=0, help='Device ID (default: %(default)d)')
    parser.add_argument('--seed', type=int, default=0, help='Seed (default: %(default)d)')
    parser.add_argument('--file_size', type=int, default=50 * 2 ** 11, help='File size (default: %(default)d)')
    parser.add_argument('--run_infinite', action='store_true', help='Run infinite (default: %(default)d)')
    args = parser.parse_args()
    run_dagger(args.dagger_type, args.num_workers, args.device_id, args.seed, args.file_size, run_infinite=args.run_infinite)

if __name__ == '__main__':
    main()
