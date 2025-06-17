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


# 顔画像を N 次元の特徴ベクトルへと圧縮するニューラルネットワーク（VAE版）
# Variational AutoEncoder のエンコーダ部分のサンプル
class FaceEncoderV(nn.Module):

    # N: 出力の特徴ベクトルの次元数
    def __init__(self, C, H, W, N, use_BatchNorm=False):
        super(FaceEncoderV, self).__init__()

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
        self.use_BatchNorm = use_BatchNorm
        if use_BatchNorm:
            self.bn1 = nn.BatchNorm2d(num_features=16)
            self.bn2 = nn.BatchNorm2d(num_features=32)
            self.bn3 = nn.BatchNorm2d(num_features=64)
            self.bn4 = nn.BatchNorm2d(num_features=64)

        # 平坦化
        self.flat = nn.Flatten()

        # 全結合層
        # 畳込み層1～3を通すことにより特徴マップの縦幅・横幅は都合 1/8 になっている．
        # その後，さらに self.conv4 を通してから全結合層を適用する予定なので，入力側のユニット数は 64*(H/8)*(W/8) = H*W
        self.fc1 = nn.Linear(in_features=H*W, out_features=2048)
        self.fc_mu = nn.Linear(in_features=2048, out_features=N)
        self.fc_lnvar = nn.Linear(in_features=2048, out_features=N)

    def forward(self, x):
        if self.use_BatchNorm:
            h = F.leaky_relu(self.bn1(self.conv1(x)))
            h = F.leaky_relu(self.bn2(self.conv2(h)))
            h = F.leaky_relu(self.bn3(self.conv3(h)))
            h = F.leaky_relu(self.bn4(self.conv4(h)))
        else:
            h = F.leaky_relu(self.conv1(x))
            h = F.leaky_relu(self.conv2(h))
            h = F.leaky_relu(self.conv3(h))
            h = F.leaky_relu(self.conv4(h))
        h = self.flat(h)
        h = F.leaky_relu(self.fc1(h))
        mu = self.fc_mu(h)
        lnvar = self.fc_lnvar(h)
        eps = torch.randn_like(mu) # mu と同じサイズの標準正規乱数を生成
        z = mu + eps * torch.exp(0.5 * lnvar)
        return z, mu, lnvar


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
    