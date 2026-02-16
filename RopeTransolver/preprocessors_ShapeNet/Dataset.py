import os
import torch
import numpy as np

from torch.utils.data import Dataset

def sato_collate_fn(batch):
    lengths = [item['x'].shape[0] for item in batch]
    min_len = min(lengths)

    batch_x = []
    batch_y = []

    for item in batch:
        x = item['x']
        y = item['y']

        if x.shape[0] > min_len:
            # Randomly sample min_len indices
            idx = torch.randperm(x.shape[0])[:min_len]
            x = x[idx]
            y = y[idx]

        batch_x.append(x)
        batch_y.append(y)

    return {
        'x': torch.stack(batch_x),
        'y': torch.stack(batch_y),
    }


class SATO_Dataset(Dataset):
    def __init__(self, data_list, config=None, is_train=True):
        self.data_list = data_list
        self.config = config
        self.is_train = is_train

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        x = data['Surface_data']['Surface_points']
        y = data['Surface_data']['Surface_feature']  # Assuming the first feature is the target

        # Move downsample logic here to allow parallel processing by DataLoader workers
        if self.config is not None and hasattr(self.config.model, 'down_sample'):
            # Use numpy random generation which is process-safe in workers
            num_points = x.shape[0]
            sample_size = int(num_points * self.config.model.down_sample)

            # Generate indices (on CPU)
            sampled_indices = np.random.choice(num_points, sample_size, replace=False)

            x = x[sampled_indices]
            y = y[sampled_indices]

        return {'x': x, 'y': y}

class VTKDataset():
    def __init__(self):
        pass

    def get_all_file_paths(self, directory):
        file_paths = []
        points_dir = os.path.join(directory, "feature")
        for file in os.listdir(points_dir):
            full_path = os.path.join(points_dir, file)
            if os.path.isfile(full_path):  # 只保留文件
                file_paths.append(full_path)
        return file_paths

    # generate data dictionary
    def get_data_dict(self, directory):
        # read all SurfacePressure file names
        Surface_file_paths = self.get_all_file_paths(directory)

        # load train/test/val index
        with open(os.path.join(directory, 'train_val_test_splits/train_design_ids.txt'), 'r') as file:
            train_index = [line.strip().split("_")[-1] for line in file]
        with open(os.path.join(directory, 'train_val_test_splits/test_design_ids.txt'), 'r') as file:
            test_index = [line.strip().split("_")[-1] for line in file]
        with open(os.path.join(directory, 'train_val_test_splits/val_design_ids.txt'), 'r') as file:
            val_index = [line.strip().split("_")[-1] for line in file]

        train_data_lst, test_data_lst, val_data_lst = [], [], []
        for file_path in Surface_file_paths:
            index = file_path.split("_")[-1].split(".")[0]
            Surface_points = np.load(os.path.join(directory,  'points', f'nodes_{index}.npy'))
            Surface_feature = np.load(os.path.join(directory,  'feature', f'features_{index}.npy'))

            Surface_points = torch.Tensor(Surface_points).float()
            Surface_feature = torch.Tensor(Surface_feature).float()
            Surface_data = {
                'Surface_points': Surface_points,
                'Surface_feature': Surface_feature,
            }

            data = {'index': index, 'Surface_data': Surface_data}

            if index in train_index:
                train_data_lst.append(data)
            elif index in test_index:
                test_data_lst.append(data)
            elif index in val_index:
                val_data_lst.append(data)
        return train_data_lst, test_data_lst, val_data_lst

