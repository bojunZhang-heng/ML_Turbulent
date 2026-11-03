import os
import torch
import numpy as np

from torch.utils.data import Dataset

def sato_collate_fn(batch):
    lengths = [item['x_pres'].shape[0] for item in batch]
    min_len = min(lengths)

    batch_x_pres = []
    batch_x_wss = []
    batch_y_pres = []
    batch_y_wss = []

    for item in batch:
        x_pres = item['x_pres']
        x_wss = item['x_wss']
        y_pres = item['y_pres']
        y_wss = item['y_wss']

        if x_pres.shape[0] > min_len:
            # Randomly sample min_len indices
            idx = torch.randperm(x_pres.shape[0])[:min_len]
            x_pres = x_pres[idx]
            y_pres = y_pres[idx]
            y_wss = y_wss[idx]

        batch_x_pres.append(x_pres)
        batch_x_wss.append(x_wss)
        batch_y_pres.append(y_pres)
        batch_y_wss.append(y_wss)

    return {
        'x_pres': torch.stack(batch_x_pres),
        'x_wss': torch.stack(batch_x_wss),
        'y_pres': torch.stack(batch_y_pres),
        'y_wss': torch.stack(batch_y_wss),
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
        x_pres = data['Surface_data']['Surface_pres_points']
        x_wss = data['Surface_data']['Surface_wss_points']
        y_pres = data['Surface_data']['Surface_pressure']
        y_wss = data['Surface_data']['Surface_wss']

        # Move downsample logic here to allow parallel processing by DataLoader workers
        if self.config is not None and hasattr(self.config.model, 'down_sample'):
            # Use numpy random generation which is process-safe in workers
            num_points = x_pres.shape[0]
            sample_size = int(num_points * self.config.model.down_sample)

            # Generate indices (on CPU)
            sampled_indices = np.random.choice(num_points, sample_size, replace=False)

            x_pres = x_pres[sampled_indices]
            x_wss = x_wss[sampled_indices]
            y_pres = y_pres[sampled_indices]
            y_wss = y_pres[sampled_indices]

        return {'x_pres': x_pres, 'x_wss': x_wss, 'y_pres': y_pres, 'y_wss': y_wss}


class VTKDataset():
    def __init__(self):
        pass

    def get_all_file_paths(self, directory):
        file_paths = []
        points_dir = os.path.join(directory, "SurfaceWSS", "points_v2")
        for file in os.listdir(points_dir):
            full_path = os.path.join(points_dir, file)
            if os.path.isfile(full_path):  # 只保留文件
                file_paths.append(full_path)

        return file_paths

    # generate data dictionary
    def get_data_dict(self, directory):
        # read all SurfacePressure file names
        SurfacePressure_file_paths = self.get_all_file_paths(directory)

        # load train/test/val index
        with open(os.path.join(directory, 'train_val_test_splits/train_design_ids.txt'), 'r') as file:
            train_index = [line.strip().split("_")[-1] for line in file]
        with open(os.path.join(directory, 'train_val_test_splits/test_design_ids.txt'), 'r') as file:
            test_index = [line.strip().split("_")[-1] for line in file]
        with open(os.path.join(directory, 'train_val_test_splits/val_design_ids.txt'), 'r') as file:
            val_index = [line.strip().split("_")[-1] for line in file]

        train_data_lst, test_data_lst, val_data_lst = [], [], []
        for file_path in SurfacePressure_file_paths:
            index = file_path.split("_")[-1].split(".")[0]
            Surface_pres_points = np.load(os.path.join(directory, 'SurfacePressure', 'points_v2', f'points_{index}.npy'))
            Surface_pressure = np.load(os.path.join(directory, 'SurfacePressure', 'pressure_v2', f'pressure_{index}.npy'))
            Surface_wss_points = np.load(os.path.join(directory, 'SurfaceWSS', 'points_v2', f'points_{index}.npy'))
            Surface_wss = np.load(os.path.join(directory, 'SurfaceWSS', 'wss_v2', f'wss_{index}.npy'))

            Surface_pres_points = torch.Tensor(Surface_pres_points).float()
            Surface_pressure = torch.Tensor(Surface_pressure).float()
            Surface_wss_points = torch.Tensor(Surface_wss_points).float()
            Surface_wss = torch.Tensor(Surface_wss).float()

            Surface_data = {
                'Surface_pres_points': Surface_pres_points,
                'Surface_pressure': Surface_pressure,
                'Surface_wss_points': Surface_wss_points,
                'Surface_wss': Surface_wss,
            }

            data = {'index': index, 'Surface_data': Surface_data}

            if index in train_index:
                train_data_lst.append(data)
            elif index in test_index:
                test_data_lst.append(data)
            elif index in val_index:
                val_data_lst.append(data)

        return train_data_lst, test_data_lst, val_data_lst

