import torch
from torch.utils.data import Dataset

class CombinedDataset(Dataset):
    def __init__(self, pressure_dataset, wss_dataset, cad_dataset):
        self.pressure_dataset = pressure_dataset
        self.wss_dataset = wss_dataset
        self.cad_dataset = cad_dataset

    def __len__(self):
        return len(self.pressure_dataset)  # Assuming all datasets have the same length
    # The length about different dataset is not the same

    def __getitem__(self, idx):
        pressure_data = self.pressure_dataset[idx]
        wss_data = self.wss_dataset[idx]
        cad_data = self.cad_dataset[idx]

        # Combine the data into a single dictionary
        combined_data = {
            'pressure': pressure_data,
            'WallShearStress': wss_data,
            'cad': cad_data
        }
        return combined_data

