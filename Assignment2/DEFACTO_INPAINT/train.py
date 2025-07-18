import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import torch
import torch.nn as nn
import torch.optim as optim
from model_base import ForgeryDetector
from trainer import trainer
from dataset import prepare_datasets, create_dataloaders

# ------------------- Configuration ------------------- #

DEVICE = 'cuda:0'
RESTART_MODE = False
USE_AMP = True
FLOAT_DTYPE = torch.float16 
LOSS_SCALER = torch.amp.grad_scaler.GradScaler(enabled=USE_AMP, device='cuda', init_scale=2**16)
ADAM_EPS = 1e-4 if USE_AMP and (FLOAT_DTYPE == torch.float16) else 1e-8
N_EPOCHS = 20
BATCH_SIZE = 100

# データセット
DATA_DIR = './Datasets/'
TRAIN_INPUT_IMAGES_FILE  = 'defacto_train_input_images.pt'
TRAIN_TARGET_IMAGES_FILE = 'defacto_train_target_images.pt'
VALID_INPUT_IMAGES_FILE  = 'defacto_valid_input_images.pt'
VALID_TARGET_IMAGES_FILE = 'defacto_valid_target_images.pt'


# 画像サイズ
H = 128 
W = 128 
C = 3 

# 学習結果の保存先
MODEL_DIR = './defacto_models/'
MODEL_FILE = os.path.join(MODEL_DIR, 'forgery_detector_model.pth')

# 中断／再開の際に用いる一時ファイル
CHECKPOINT_EPOCH = os.path.join('./temp/', 'checkpoint_epoch.pkl')
CHECKPOINT_MODEL = os.path.join('./temp/', 'checkpoint_model.pth')
CHECKPOINT_OPT = os.path.join('./temp/', 'checkpoint_opt.pth')

# ------------------- Prepare Dataset ------------------- #

print('Preparing datasets...')
train_dataset, valid_dataset, train_size, valid_size = prepare_datasets(
    restart_mode=RESTART_MODE,
    data_dir=DATA_DIR,
    train_input_file=TRAIN_INPUT_IMAGES_FILE,
    train_target_file=TRAIN_TARGET_IMAGES_FILE,
    valid_input_file=VALID_INPUT_IMAGES_FILE,
    valid_target_file=VALID_TARGET_IMAGES_FILE
)
print(f'Train dataset size: {train_size}, Valid dataset size: {valid_size}')
print('Creating dataloaders...')
train_dataloader, valid_dataloader = create_dataloaders(
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    batch_size=BATCH_SIZE
)
print('Dataloaders prepared successfully.')

# ------------------- Train ------------------- #
print('Starting training...')
model = ForgeryDetector(C=C, H=H, W=W).to(DEVICE)


optimizer = optim.Adam(model.parameters(), eps=ADAM_EPS)
loss_func = nn.BCELoss()

# トレーニングの実行
model, loss_viz = trainer(
    model=model,
    train_dataloader=train_dataloader,
    valid_dataloader=valid_dataloader,
    optimizer=optimizer,
    loss_func=loss_func,
    device=DEVICE,
    n_epochs=N_EPOCHS,
    restart_mode=RESTART_MODE,
    checkpoint_epoch=CHECKPOINT_EPOCH,
    checkpoint_model=CHECKPOINT_MODEL,
    checkpoint_opt=CHECKPOINT_OPT,
    use_amp=USE_AMP,
    float_dtype=FLOAT_DTYPE,
    loss_scaler=LOSS_SCALER,
    model_dir=MODEL_DIR,
    model_file=MODEL_FILE
)