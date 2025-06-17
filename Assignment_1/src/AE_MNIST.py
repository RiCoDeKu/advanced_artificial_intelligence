import torch
import torch.nn as nn
import torch.nn.functional as F

# MNIST画像を N 次元の特徴ベクトルへと圧縮するニューラルネットワーク
# AutoEncoder のエンコーダ部分のサンプル
class MNISTEncoder(nn.Module):

    # N: 出力の特徴ベクトルの次元数
    def __init__(self, N, use_BatchNorm=False):
        super(MNISTEncoder, self).__init__()

        # 畳込み層1,2
        # カーネルサイズ4，ストライド幅2，パディング1の設定なので，これらを通すことにより特徴マップの縦幅・横幅がそれぞれ 1/2 になる
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=4, stride=2, padding=1)

        # 畳込み層3
        # カーネルサイズ3，ストライド幅1，パディング1の設定なので，これを通しても特徴マップの縦幅・横幅は変化しない
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)

        # バッチ正規化層
        self.use_BatchNorm = use_BatchNorm
        if use_BatchNorm:
            self.bn1 = nn.BatchNorm2d(num_features=4) # num_features は conv1 の out_channels と同じ値に
            self.bn2 = nn.BatchNorm2d(num_features=8) # num_features は conv2 の out_channels と同じ値に
            self.bn3 = nn.BatchNorm2d(num_features=8) # num_features は conv3 の out_channels と同じ値に

        # 平坦化
        self.flat = nn.Flatten()

        ## Live Share!!!!!!!!!!!!!!!!!!
        # 全結合層
        # 畳込み層1, 2を通すことにより特徴マップの縦幅・横幅は 28/4 = 7 になっている．
        # その後，さらに self.conv3 を通してから全結合層を適用する予定なので，入力側のパーセプトロン数は 8*7*7
        self.fc = nn.Linear(in_features=8*7*7, out_features=N)

    def forward(self, x):
        if self.use_BatchNorm:
            h = F.leaky_relu(self.bn1(self.conv1(x)))
            h = F.leaky_relu(self.bn2(self.conv2(h)))
            h = F.leaky_relu(self.bn3(self.conv3(h)))
        else:
            h = F.leaky_relu(self.conv1(x))
            h = F.leaky_relu(self.conv2(h))
            h = F.leaky_relu(self.conv3(h))
        h = self.flat(h)
        z = self.fc(h)
        return z

# N 次元の特徴ベクトルからMNIST風画像を生成するニューラルネットワーク
# AutoEncoder および Variational AutoEncoder のデコーダ部分のサンプル（デコーダ部分は通常の AE と VAE で全く同じ）
class MNISTDecoder(nn.Module):

    # N: 入力の特徴ベクトルの次元数
    def __init__(self, N, use_BatchNorm=False):
        super(MNISTDecoder, self).__init__()

        # 全結合層
        # パーセプトロン数は MNISTEncoder の全結合層と真逆に設定
        self.fc = nn.Linear(in_features=N, out_features=8*7*7)

        # 転置畳込み層1～3
        # カーネルサイズ，ストライド幅，パディングは MNISTEncoder の畳込み層1～3と真逆に設定
        self.deconv3 = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=4, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=4, stride=2, padding=1)

        # 畳込み層
        # 転置畳込み層の出力には checker board artifact というノイズが乗りやすいので，最後に畳込み層を通しておく
        self.conv = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1)

        # バッチ正規化層
        self.use_BatchNorm = use_BatchNorm
        if use_BatchNorm:
            self.bn3 = nn.BatchNorm2d(num_features=8)
            self.bn2 = nn.BatchNorm2d(num_features=4)
            self.bn1 = nn.BatchNorm2d(num_features=4)

    def forward(self, z):
        h = F.leaky_relu(self.fc(z))
        h = torch.reshape(h, (len(h), 8, 7, 7)) # 一列に並んだユニットを 8*7*7 の特徴マップに並べ直す
        if self.use_BatchNorm:
            h = F.leaky_relu(self.bn3(self.deconv3(h)))
            h = F.leaky_relu(self.bn2(self.deconv2(h)))
            h = F.leaky_relu(self.bn1(self.deconv1(h)))
        else:
            h = F.leaky_relu(self.deconv3(h))
            h = F.leaky_relu(self.deconv2(h))
            h = F.leaky_relu(self.deconv1(h))
        y = torch.sigmoid(self.conv(h))
        return y
    