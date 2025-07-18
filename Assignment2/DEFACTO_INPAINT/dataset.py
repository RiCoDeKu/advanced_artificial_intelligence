import torch
import os
from torch.utils.data import DataLoader, random_split
from mylib.data_io import TensorDataset


def prepare_datasets(restart_mode, data_dir, train_input_file, train_target_file, 
                    valid_input_file, valid_target_file, validation_ratio=0.01):    
    if restart_mode:
        train_dataset = TensorDataset(filenames=[
            os.path.join('./temp/', train_input_file),
            os.path.join('./temp/', train_target_file)
        ])
        valid_dataset = TensorDataset(filenames=[
            os.path.join('./temp/', valid_input_file),
            os.path.join('./temp/', valid_target_file)
        ])
        train_size = len(train_dataset)
        valid_size = len(valid_dataset)
    
    else:
        dataset = TensorDataset(filenames=[
            os.path.join(data_dir, train_input_file),
            os.path.join(data_dir, train_target_file)
        ])

        dataset_size = len(dataset)
        valid_size = int(validation_ratio * dataset_size)  # 全体の validation_ratio % を検証用に
        train_size = dataset_size - valid_size  # 残りを学習用に
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

        # データセット情報をファイルに保存
        os.makedirs('./temp/', exist_ok=True)  # tempディレクトリがない場合は作成
        torch.save(torch.cat([torch.unsqueeze(train_dataset[i][0], dim=0) for i in range(len(train_dataset))], dim=0), os.path.join('./temp/', train_input_file))
        torch.save(torch.cat([torch.unsqueeze(train_dataset[i][1], dim=0) for i in range(len(train_dataset))], dim=0), os.path.join('./temp/', train_target_file))
        torch.save(torch.cat([torch.unsqueeze(valid_dataset[i][0], dim=0) for i in range(len(valid_dataset))], dim=0), os.path.join('./temp/', valid_input_file))
        torch.save(torch.cat([torch.unsqueeze(valid_dataset[i][1], dim=0) for i in range(len(valid_dataset))], dim=0), os.path.join('./temp/', valid_target_file))

    return train_dataset, valid_dataset, train_size, valid_size

def create_dataloaders(train_dataset, valid_dataset, batch_size, num_workers=8, pin_memory=True):
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    return train_dataloader, valid_dataloader


