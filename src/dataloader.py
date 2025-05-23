"""Module for managing data loading"""

import torch
import os
from torch.utils.data import DataLoader, random_split
from utils.data_io import TensorDataset
from config import DataLoaderConfig
import argparse


class MyDataLoader:    
    def __init__(self, config=None, **kwargs):
        self.config = config if config else DataLoaderConfig()
        
        # Override configuration parameters with specified arguments
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self.train_dataset = None
        self.valid_dataset = None
        self.train_dataloader = None
        self.valid_dataloader = None
        
        self._prepare_datasets()
        self._prepare_dataloaders()
    
    def _prepare_datasets(self):
        if self.config.RESTART_MODE:
            self.train_dataset = TensorDataset(filenames=[
                os.path.join(self.config.TEMP_DIR, self.config.TRAIN_IMAGES_FILE)
            ])
            self.valid_dataset = TensorDataset(filenames=[
                os.path.join(self.config.TEMP_DIR, self.config.VALID_IMAGES_FILE)
            ])
        
        else:
            dataset = TensorDataset(filenames=[
                os.path.join(self.config.DATA_DIR, self.config.TRAIN_IMAGES_FILE)
            ])
            dataset_size = len(dataset)
            valid_size = int(self.config.VALID_SPLIT_RATIO * dataset_size)
            train_size = dataset_size - valid_size
            self.train_dataset, self.valid_dataset = random_split(dataset, [train_size, valid_size])

            # Save dataset information to file (create temp directory if it doesn't exist)
            os.makedirs(self.config.TEMP_DIR, exist_ok=True)
            torch.save(
                torch.cat([torch.unsqueeze(self.train_dataset[i], dim=0) for i in range(len(self.train_dataset))], dim=0), 
                os.path.join(self.config.TEMP_DIR, self.config.TRAIN_IMAGES_FILE)
            )
            torch.save(
                torch.cat([torch.unsqueeze(self.valid_dataset[i], dim=0) for i in range(len(self.valid_dataset))], dim=0), 
                os.path.join(self.config.TEMP_DIR, self.config.VALID_IMAGES_FILE)
            )
    
    def _prepare_dataloaders(self):
        self.train_dataloader = DataLoader(
            self.train_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True, 
            pin_memory=True,
            num_workers=self.config.NUM_WORKERS
        )
        self.valid_dataloader = DataLoader(
            self.valid_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False, 
            pin_memory=True,
            num_workers=self.config.NUM_WORKERS
        )
    
    def get_train_dataloader(self):
        return self.train_dataloader
    
    def get_valid_dataloader(self):
        return self.valid_dataloader


def get_dataloaders_from_args():
    parser = argparse.ArgumentParser(description='DataLoader Configuration')
    parser.add_argument('--batch_size', type=int, default=DataLoaderConfig.BATCH_SIZE)
    parser.add_argument('--restart_mode', action='store_true')
    parser.add_argument('--data_dir', type=str, default=DataLoaderConfig.DATA_DIR)
    parser.add_argument('--temp_dir', type=str, default=DataLoaderConfig.TEMP_DIR)
    parser.add_argument('--valid_split_ratio', type=float, default=DataLoaderConfig.VALID_SPLIT_RATIO)
    parser.add_argument('--NUM_WORKERS', type=int, default=DataLoaderConfig.NUM_WORKERS)
    
    args = parser.parse_args()
    
    # Convert arguments to dictionary
    config_kwargs = {
        'BATCH_SIZE': args.batch_size,
        'RESTART_MODE': args.restart_mode,
        'DATA_DIR': args.data_dir,
        'TEMP_DIR': args.temp_dir,
        'VALID_SPLIT_RATIO': args.valid_split_ratio,
        'NUM_WORKERS': args.NUM_WORKERS
    }
    
    # Create and return data loaders
    data_loader = MyDataLoader(**config_kwargs)
    return data_loader.get_train_dataloader(), data_loader.get_valid_dataloader()


# Sample execution when run as a module directly
if __name__ == "__main__":
    # Create data loader with default settings
    data_loader = DataLoader()
    train_dataloader = data_loader.get_train_dataloader()
    valid_dataloader = data_loader.get_valid_dataloader()
    
    print(f"Training dataloader: batch size {train_dataloader.batch_size}, "
          f"number of batches {len(train_dataloader)}")
    print(f"Validation dataloader: batch size {valid_dataloader.batch_size}, "
          f"number of batches {len(valid_dataloader)}")