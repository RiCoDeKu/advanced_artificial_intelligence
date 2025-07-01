import os
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
from dataset import prepare_datasets, create_dataloaders
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

# ---------------- カスタム損失関数の定義 ----------------
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.dice_loss = utils.losses.DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.__name__ = 'dice_bce_loss'
        
    def forward(self, inputs, targets):
        dice_loss = self.dice_loss(inputs, targets)
        bce_loss = self.bce_loss(inputs, targets)
        # Dice LossとBCE Lossを同じ重みで組み合わせる
        return dice_loss + bce_loss

# ---------------- Configuration ----------------

BATCH_SIZE = 10  # バッチサイズの定義
DEVICE = "cuda"

DATA_DIR = './Datasets/'
TRAIN_INPUT_IMAGES_FILE  = 'defacto_train_input_images.pt'
TRAIN_TARGET_IMAGES_FILE = 'defacto_train_target_images.pt'
VALID_INPUT_IMAGES_FILE  = 'defacto_valid_input_images.pt'
VALID_TARGET_IMAGES_FILE = 'defacto_valid_target_images.pt'

MODEL_DIR = './defacto_models/'
MODEL_FILE = os.path.join(MODEL_DIR, 'UnetPlusPlus_DiceBCE_model.pth')
METRICS_DIR = './metrics/'
os.makedirs(METRICS_DIR, exist_ok=True)

# ---------------- データセットとデータローダーの準備 ----------------
train_dataset, valid_dataset, train_size, valid_size = prepare_datasets(
    restart_mode=False,
    data_dir=DATA_DIR,
    train_input_file=TRAIN_INPUT_IMAGES_FILE,
    train_target_file=TRAIN_TARGET_IMAGES_FILE,
    valid_input_file=VALID_INPUT_IMAGES_FILE,
    valid_target_file=VALID_TARGET_IMAGES_FILE
)
train_dataloader, valid_dataloader = create_dataloaders(
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    batch_size=BATCH_SIZE
)

# ---------------- モデル設定 ----------------
# UnetPlusPlus from scratch - EfficientNet-B4 without pretrained weights (encoder_weights=None)
ENCODER = "efficientnet-b4"
ENCODER_WEIGHTS = None  # スクラッチから学習するため、事前学習済みの重みを使用しない
ACTIVATION = "sigmoid"
CLASS_NUM = 1  # segmentationの正解ラベル数

model = smp.UnetPlusPlus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS,  # 事前学習なし（スクラッチから学習）
    classes=CLASS_NUM, 
    activation=ACTIVATION,
)

# ---------------- 学習設定 ----------------
# カスタム損失関数 (Dice Loss + BCE Loss) を使用
loss = DiceBCELoss()
metrics = [
    utils.metrics.IoU(threshold=0.5),
    utils.metrics.Precision(threshold=0.5),
    utils.metrics.Recall(threshold=0.5)
]
optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])

train_epoch = utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

max_score = 0

# 学習の進捗を記録するリスト
train_losses = []
valid_losses = []
train_iou_scores = []
valid_iou_scores = []
train_precision_scores = []
valid_precision_scores = []
train_recall_scores = []
valid_recall_scores = []
epochs = []

for i in range(0, 40):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_dataloader)
    valid_logs = valid_epoch.run(valid_dataloader)
    
    # 学習の進捗を記録
    epochs.append(i)
    train_losses.append(train_logs['dice_bce_loss'])  # カスタム損失関数名を反映
    valid_losses.append(valid_logs['dice_bce_loss'])  # カスタム損失関数名を反映
    train_iou_scores.append(train_logs['iou_score'])
    valid_iou_scores.append(valid_logs['iou_score'])
    train_precision_scores.append(train_logs['precision'])
    valid_precision_scores.append(valid_logs['precision'])
    train_recall_scores.append(train_logs['recall'])
    valid_recall_scores.append(valid_logs['recall'])

    # モデルの保存、学習率の変更など
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, MODEL_FILE)
        print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')

# トレーニング終了後にグラフを生成
plt.figure(figsize=(12, 10))

# 損失関数のグラフ
plt.subplot(2, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, valid_losses, label='Valid Loss')
plt.title('Dice + BCE Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# IoUスコアのグラフ
plt.subplot(2, 2, 2)
plt.plot(epochs, train_iou_scores, label='Train IoU')
plt.plot(epochs, valid_iou_scores, label='Valid IoU')
plt.title('IoU Score vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('IoU Score')
plt.legend()
plt.grid(True)

# Precisionスコアのグラフ
plt.subplot(2, 2, 3)
plt.plot(epochs, train_precision_scores, label='Train Precision')
plt.plot(epochs, valid_precision_scores, label='Valid Precision')
plt.title('Precision vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)

# Recallスコアのグラフ
plt.subplot(2, 2, 4)
plt.plot(epochs, train_recall_scores, label='Train Recall')
plt.plot(epochs, valid_recall_scores, label='Valid Recall')
plt.title('Recall vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(METRICS_DIR, 'training_DiceBCE_metrics.png'))
plt.close()

print(f"メトリクスのグラフを {os.path.join(METRICS_DIR, 'training_DiceBCE_metrics.png')} に保存しました。")
