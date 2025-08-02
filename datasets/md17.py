import torch
import numpy as np
import os
import pickle as pkl
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm


class MD17(InMemoryDataset):
    """
    MD17 Dataset, adapted for PyTorch Geometric Data objects.
    The task is to predict the final positions of atoms given their initial positions and velocities.
    This class handles loading, splitting, and processing for a single molecule type.

    Args:
        root (string): Root directory where the dataset should be saved.
        molecule_type (string): The name of the molecule to load (e.g., 'aspirin', 'benzene').
        split (string): The split to load ('train', 'valid', 'test').
        max_samples (int): Maximum number of samples to use from the split.
        delta_frame (int): The time step difference for dynamics prediction.
    """

    def __init__(self, root, molecule_type, split, max_samples=1000, delta_frame=1, transform=None, pre_transform=None,# ★ 新增一个参数，可以是一个字典，来为不同split指定大小
                 split_sizes={'train': 10000, 'valid': 2000, 'test': 2000},
                 pre_filter=None):
        self.molecule_type = molecule_type
        self.split = split
        self.max_samples = max_samples
        self.delta_frame = delta_frame
        self.split_sizes = split_sizes # ★ 保存新的参数

        # 将 root 路径转换为绝对路径，避免相对路径问题
        super(MD17, self).__init__(os.path.abspath(root), transform, pre_transform, pre_filter)

        # ★★★ 关键修改：检查文件是否存在，如果不存在就强制处理 ★★★
        if not os.path.exists(self.processed_paths[0]):
            print(f"Processed file not found for {self.molecule_type} - {self.split}. Forcing reprocessing...")
            self.process()
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        return [f'{self.molecule_type}_dft.npz']

    @property
    def processed_file_names(self):
        train_size_str = f"{self.split_sizes.get('train', 0) // 1000}k" # e.g., '10k' or '100k'
        return [f'md17_{self.molecule_type}_{self.split}_train{train_size_str}.pt']

    def download(self):
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        if not os.path.exists(raw_path):
            raise FileNotFoundError(
                f"Raw data file not found: {raw_path}. "
                f"Please download the MD17 datasets (e.g., from http://www.quantum-machine.org/gdml/data/) "
                f"and place the .npz files in {self.raw_dir}"
            )

    def process(self):
        print(f"Running process() for {self.molecule_type} with custom split sizes: {self.split_sizes}...")
        data_npz = np.load(os.path.join(self.raw_dir, self.raw_file_names[0]))
        positions = torch.from_numpy(data_npz['R']).float()
        atomic_numbers = torch.from_numpy(data_npz['z']).long()

        # ================== ★★★ 核心修改 2: 修改 process() 的划分逻辑 ★★★ ==================
        train_size = self.split_sizes.get('train', 20000)
        val_size = self.split_sizes.get('valid', 2000)
        test_size = self.split_sizes.get('test', 2000)
        
        split_id = f"{train_size//1000}k-{val_size//1000}k-{test_size//1000}k" # e.g., 100k-2k-2k
        split_path = os.path.join(self.raw_dir, f'{self.molecule_type}_split_{split_id}.pkl')
        
        try:
            with open(split_path, 'rb') as f:
                split_indices = pkl.load(f)
                print(f"Loaded existing {split_id} split for {self.molecule_type}")
        except FileNotFoundError:
            print(f"Creating new {split_id} split for {self.molecule_type}")
            np.random.seed(42)
            num_total = len(positions)
            all_indices = np.arange(num_total)
            np.random.shuffle(all_indices)
            
            required_samples = train_size + val_size + test_size
            if num_total < required_samples:
                 raise ValueError(f"Molecule {self.molecule_type} has only {num_total} samples, not enough for the specified split.")

            train_idx = all_indices[:train_size]
            val_idx = all_indices[train_size : train_size + val_size]
            test_idx = all_indices[train_size + val_size : train_size + val_size + test_size]
            split_indices = (train_idx, val_idx, test_idx)
            with open(split_path, 'wb') as f:
                pkl.dump(split_indices, f)

        split_map = {'train': 0, 'valid': 1, 'test': 2}
        indices = split_indices[split_map[self.split]]

        # The max_samples argument in __init__ still acts as a final limiter
        # But for fair comparison, we should set it high enough in the run script
        # so it doesn't interfere with this 500/2k/2k split.
        indices = indices[:self.max_samples]

        # Calculate velocities via finite difference: v(t) = x(t) - x(t-1)
        # Ensure we don't index before 0
        min_required_index = 1 + self.delta_frame  # We need t, t-1, and t+delta_frame
        valid_indices_mask = indices >= min_required_index
        valid_indices = indices[valid_indices_mask]

        pos_initial = positions[valid_indices - self.delta_frame]
        pos_final = positions[valid_indices]
        # velocity at (t - delta_frame)
        vel_initial = pos_initial - positions[valid_indices - self.delta_frame - 1]

        print(
            f"Processing {self.molecule_type} - {self.split} split. Using {len(pos_initial)} valid samples after filtering for dynamics.")

        data_list = []
        for i in tqdm(range(len(pos_initial)), desc=f"Creating {self.split} data for {self.molecule_type}"):
            data = Data(
                pos=pos_initial[i],
                z=atomic_numbers,
                vel=vel_initial[i],
                y=pos_final[i],
            )
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])