import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import torch


# 前回の試行の続きを行いたい場合は True にする -> 再開モードになる．
# なお，Colab環境で再開モードを利用する場合は，前回終了時に temp ディレクトリの中身を自分の Google Drive に退避しておき，
# それを改めて /content/AI_advanced/temp 以下にあらかじめ移しておく必要がある．
RESTART_MODE = False

DEVICE = 'cuda:0'

# 高速化・省メモリ化のために半精度小数を用いた混合精度学習を行うか否か（Trueの場合は行う）
USE_AMP = True
FLOAT_DTYPE = torch.float16 # 混合精度学習を行う場合の半精度小数の型．環境によっては torch.bfloat16 にした方が良好な性能になる（ただしColabのT4 GPU環境ではムリ）．

# 混合精度学習の設定
if DEVICE == 'cpu':
    USE_AMP = False # CPU使用時は強制的に混合精度学習をOFFにする
LOSS_SCALER = torch.amp.grad_scaler.GradScaler(enabled=USE_AMP, device='cuda', init_scale=2**16)
ADAM_EPS = 1e-4 if USE_AMP and (FLOAT_DTYPE == torch.float16) else 1e-8

# 全ての訓練データを一回ずつ使用することを「1エポック」として，何エポック分学習するか
# 再開モードの場合も, このエポック数の分だけ追加学習される（N_EPOCHSは最終エポック番号ではない）
N_EPOCHS = 20

# 学習時のバッチサイズ
BATCH_SIZE = 100

# データセットの存在するフォルダ・ファイル名
DATA_DIR = './data/CelebA'
TRAIN_IMAGES_FILE = 'tinyCelebA_train_images.pt'
VALID_IMAGES_FILE = 'tinyCelebA_valid_images.pt'

# 画像サイズ

H = 128 # 縦幅
W = 128 # 横幅
C = 3 # チャンネル数（カラー画像なら3，グレースケール画像なら1）

# 特徴ベクトルの次元数
N = 128

# 学習結果の保存先フォルダ
MODEL_DIR = './models/Celeb/AE'

# 学習結果のニューラルネットワークの保存先
MODEL_FILE_ENC = os.path.join(MODEL_DIR, f'{N}_encoder.pth') # エンコーダ
MODEL_FILE_DEC = os.path.join(MODEL_DIR, f'{N}_decoder.pth') # デコーダ

# 中断／再開の際に用いる一時ファイルの保存先
CHECKPOINT_EPOCH = os.path.join('./temp/', 'checkpoint_epoch.pkl')
CHECKPOINT_ENC_MODEL = os.path.join('./temp/', 'checkpoint_enc_model.pth')
CHECKPOINT_DEC_MODEL = os.path.join('./temp/', 'checkpoint_dec_model.pth')
CHECKPOINT_ENC_OPT = os.path.join('./temp/', 'checkpoint_enc_opt.pth')
CHECKPOINT_DEC_OPT = os.path.join('./temp/', 'checkpoint_dec_opt.pth')

import torch
import torch.nn as nn
import torch.nn.functional as F


# Residual Block
# 入力特徴マップと出力特徴マップのチャンネル数は同一であることを前提とする
class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding, activation=F.relu):
        super(ResBlock, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(num_features=channels)
        self.bn2 = nn.BatchNorm2d(num_features=channels)
    def forward(self, x):
        h = self.activation(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.activation(h + x)


# 顔画像を N 次元の特徴ベクトルへと圧縮するニューラルネットワーク
# AutoEncoderのエンコーダ部分のサンプル
class FaceEncoder(nn.Module):

    # C: 入力顔画像のチャンネル数（1または3と仮定）
    # H: 入力顔画像の縦幅（8の倍数と仮定）
    # W: 入力顔画像の横幅（8の倍数と仮定）
    # N: 出力の特徴ベクトルの次元数
    def __init__(self, C, H, W, N):
        super(FaceEncoder, self).__init__()

        # 畳込み層1～3
        # カーネルサイズ4，ストライド幅2，パディング1の設定なので，これらを通すことにより特徴マップの縦幅・横幅がそれぞれ 1/2 になる
        # 3つ適用することになるので，最終的には都合 1/8 になる -> ゆえに，入力顔画像の縦幅と横幅を各々8の倍数と仮定している
        self.conv1 = nn.Conv2d(in_channels=C, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)

        # 畳込み層4
        # カーネルサイズ3，ストライド幅1，パディング1の設定なので，これを通しても特徴マップの縦幅・横幅は変化しない
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        #self.conv4 = ResBlock(channels=64, kernel_size=3, stride=1, padding=1) # 例: Residual Block を使用する場合はこのように記載

        # バッチ正規化層
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.bn4 = nn.BatchNorm2d(num_features=64)

        # 平坦化
        self.flat = nn.Flatten()

        # 全結合層1
        # 畳込み層1～3を通すことにより特徴マップの縦幅・横幅は都合 1/8 になっている．
        # その後，さらに self.conv4 を通してから全結合層を適用する予定なので，入力側のユニット数は 64*(H/8)*(W/8) = H*W
        self.fc1 = nn.Linear(in_features=H*W, out_features=2048)

        # 全結合層2
        self.fc2 = nn.Linear(in_features=2048, out_features=N)

    def forward(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(x)))
        h = F.leaky_relu(self.bn2(self.conv2(h)))
        h = F.leaky_relu(self.bn3(self.conv3(h)))
        h = F.leaky_relu(self.bn4(self.conv4(h)))
        #h = self.conv4(h) # 例: Residual Block を使用する場合はこのように記載（Residual Blockの内部でバッチ正規化と活性化関数を適用しているので，外側では適用しない）
        h = self.flat(h)
        h = F.leaky_relu(self.fc1(h))
        z = self.fc2(h)
        return z


# N 次元の特徴ベクトルから顔画像を生成するニューラルネットワーク
# AutoEncoderのデコーダ部分のサンプル
class FaceDecoder(nn.Module):

    # C: 出力顔画像のチャンネル数（1または3と仮定）
    # H: 出力顔画像の縦幅（8の倍数と仮定）
    # W: 出力顔画像の横幅（8の倍数と仮定）
    # N: 入力の特徴ベクトルの次元数
    def __init__(self, C, H, W, N):
        super(FaceDecoder, self).__init__()
        self.W = W
        self.H = H

        # 全結合層1,2
        # パーセプトロン数は FaceEncoder の全結合層と真逆に設定
        self.fc2 = nn.Linear(in_features=N, out_features=2048)
        self.fc1 = nn.Linear(in_features=2048, out_features=H*W)

        # 転置畳込み層1～4
        # カーネルサイズ，ストライド幅，パディングは FaceEncoder の畳込み層1～4と真逆に設定
        self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1)

        # バッチ正規化層
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.bn1 = nn.BatchNorm2d(num_features=8)

        # 畳込み層
        # 転置畳込み層の出力には checker board artifact というノイズが乗りやすいので，最後に畳込み層を通しておく
        self.conv = nn.Conv2d(in_channels=8, out_channels=C, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        h = F.leaky_relu(self.fc2(z))
        h = F.leaky_relu(self.fc1(h))
        h = torch.reshape(h, (len(h), 64, self.H//8, self.W//8)) # 一列に並んだユニットを 64*(H/8)*(W/8) の特徴マップに並べ直す
        h = F.leaky_relu(self.bn4(self.deconv4(h)))
        h = F.leaky_relu(self.bn3(self.deconv3(h)))
        h = F.leaky_relu(self.bn2(self.deconv2(h)))
        h = F.leaky_relu(self.bn1(self.deconv1(h)))
        y = torch.sigmoid(self.conv(h))
        return y
    
    import torch
from torch.utils.data import DataLoader, random_split
from src.utils.data_io import TensorDataset


# 再開モードの場合は，前回使用したデータセットをロードして使用する
if RESTART_MODE:

    # テンソルファイルを読み込み，前回使用したデータセットを用意
    train_dataset = TensorDataset(filenames=[
        os.path.join('./temp/', TRAIN_IMAGES_FILE)
    ])
    valid_dataset = TensorDataset(filenames=[
        os.path.join('./temp/', VALID_IMAGES_FILE)
    ])
    train_size = len(train_dataset)
    valid_size = len(valid_dataset)

# そうでない場合は，新たにデータセットを読み込む
else:

    # テンソルファイルを読み込み, 訓練データセットを用意
    dataset = TensorDataset(filenames=[
        os.path.join(DATA_DIR, TRAIN_IMAGES_FILE)
    ])

    # 訓練データセットを分割し，一方を検証用に回す
    dataset_size = len(dataset)
    valid_size = int(0.002 * dataset_size) # 全体の 0.2% を検証用に -> tinyCelebA の画像は全部で 16000 枚なので，検証用画像は 16000*0.002=32 枚
    train_size = dataset_size - valid_size # 残りの 99.8% を学習用に
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    # データセット情報をファイルに保存
    torch.save(torch.cat([torch.unsqueeze(train_dataset[i], dim=0) for i in range(len(train_dataset))], dim=0), os.path.join('./temp/', TRAIN_IMAGES_FILE))
    torch.save(torch.cat([torch.unsqueeze(valid_dataset[i], dim=0) for i in range(len(valid_dataset))], dim=0), os.path.join('./temp/', VALID_IMAGES_FILE))


# 訓練データおよび検証用データをミニバッチに分けて使用するための「データローダ」を用意
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8)

