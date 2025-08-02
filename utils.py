# utils.py

import io
import os
import time
from collections import defaultdict, deque
import datetime
import sys # Import sys

import torch
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

# --- Function to suppress printing ---
# (You might need this elsewhere if using distributed training)
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
# ---------------------------------

def init_distributed_mode(args):
    # Default values for non-distributed case
    args.distributed = False
    args.rank = 0
    args.local_rank = 0
    args.world_size = 1

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Launched by torchrun or similar
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.distributed = True # Set distributed flag
        print(f"Distributed: RANK={args.rank}, WORLD_SIZE={args.world_size}, LOCAL_RANK={args.local_rank}")

    elif 'SLURM_PROCID' in os.environ:
        # Launched by SLURM
        try:
            args.rank = int(os.environ['SLURM_PROCID'])
            args.local_rank = args.rank % torch.cuda.device_count()
            # SLURM typically sets these, but double-check your environment
            args.world_size = int(os.environ['SLURM_NTASKS'])
            # Make sure dist_url is set correctly for SLURM in args or env
            # Often requires MASTER_ADDR, MASTER_PORT to be set based on the first node
            if not hasattr(args, 'dist_url') or args.dist_url == 'env://':
                 # Attempt basic env setup if not provided externally
                 if 'MASTER_ADDR' not in os.environ:
                     print("Warning: MASTER_ADDR not set for SLURM, distributed init might fail.", file=sys.stderr)
                     # Set a default or raise error depending on your SLURM setup needs
                 if 'MASTER_PORT' not in os.environ:
                     # Set a default port if needed
                     os.environ['MASTER_PORT'] = '29500' # Example port
                     print(f"Warning: MASTER_PORT not set for SLURM, using default {os.environ['MASTER_PORT']}.", file=sys.stderr)
                 args.dist_url = 'env://' # Proceed with env://

            args.distributed = True
            print(f"Distributed (SLURM): RANK={args.rank}, WORLD_SIZE={args.world_size}, LOCAL_RANK={args.local_rank}")
        except KeyError as e:
            print(f"Error initializing SLURM distributed mode: Missing environment variable {e}", file=sys.stderr)
            args.distributed = False # Fallback to non-distributed
        except Exception as e:
            print(f"Error initializing SLURM distributed mode: {e}", file=sys.stderr)
            args.distributed = False # Fallback to non-distributed

    else:
        # Not in a distributed setting detected by environment variables
        print('Not using (explicitly detected) distributed mode.')
        args.distributed = False
        # Keep default rank/local_rank/world_size

    # --- Initialize process group ONLY if distributed ---
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.dist_backend = 'nccl' # Or 'gloo' if needed
        print(f'| distributed init (rank {args.rank}/{args.world_size}) - backend: {args.dist_backend}, init_method: {args.dist_url}', flush=True)
        try:
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                     world_size=args.world_size, rank=args.rank)
            dist.barrier() # Sync processes after init
            # Setup printing suppression on non-master ranks
            setup_for_distributed(args.rank == 0)
        except Exception as e:
             print(f"!!!!! ERROR initializing process group !!!!!", file=sys.stderr)
             print(f"Args: rank={args.rank}, world_size={args.world_size}, local_rank={args.local_rank}, dist_url={args.dist_url}", file=sys.stderr)
             print(f"Error: {e}", file=sys.stderr)
             # Decide how to handle failure: exit, or try to continue non-distributed?
             # For simplicity, let's try to revert to non-distributed
             print("!!!!! Attempting to fallback to non-distributed mode !!!!!", file=sys.stderr)
             args.distributed = False
             args.rank = 0
             args.local_rank = 0
             args.world_size = 1
             # Reset device maybe? Might not be necessary if cuda calls handle index 0 fine.
             # torch.cuda.set_device(args.local_rank)

    # --- ALWAYS return args ---
    return args