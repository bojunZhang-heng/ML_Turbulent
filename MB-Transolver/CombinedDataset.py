import torch
from torch.utils.data import Dataset

class CombinedDataset(Dataset):
    def __init__(self, pressure_dataset, wss_dataset, cad_dataset, volume_dataset):
        self.pressure_dataset = pressure_dataset
        self.wss_dataset = wss_dataset
        self.cad_dataset = cad_dataset
        self.volume_dataset = volume_dataset

    def __len__(self):
        return len(self.pressure_dataset)  # Assuming all datasets have the same length
    # The length about different dataset is not the same

    def __getitem__(self, idx):
        pressure_data = self.pressure_dataset[idx]
        wss_data = self.wss_dataset[idx]
        cad_data = self.cad_dataset[idx]
        volume_data = self.volume_dataset[idx]

        # Combine the data into a single dictionary
        combined_data = {
            'surf_position': pressure_data[0],
            'surf_pressure': pressure_data[1],
            'surf_wss': wss_data[1],
            'geometry_position': cad_data,
            'volume_position': volume_data[0],
            'volume_pressure': volume_data[1],
            'volume_wss': volume_data[2],
            'volume_vel': volume_data[3],
        }
        return combined_data

