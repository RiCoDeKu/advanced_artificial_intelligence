import torch
import torch.nn as nn
import torch.nn.functional as F


# 畳込み，バッチ正規化，ReLUをセットで行うクラス
class myConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(myConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


# 逆畳込み，バッチ正規化，ReLUをセットで行うクラス
class myConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(myConvTranspose2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


# defacto の画像中から改ざん領域を推定するニューラルネットワーク
class ForgeryDetector(nn.Module):

    # C: 入力画像のチャンネル数（1または3と仮定）
    # H: 入力画像の縦幅（8の倍数と仮定）
    # W: 入力画像の横幅（8の倍数と仮定）
    def __init__(self, C, H, W):
        super(ForgeryDetector, self).__init__()

        # 畳込み層1
        # カーネルサイズ3，ストライド幅1，パディング1の設定なので，これを通しても特徴マップの縦幅・横幅は変化しない
        self.conv1 = myConv2d(in_channels=C, out_channels=16, kernel_size=3, stride=1, padding=1)

        # 畳込み層2～4
        # カーネルサイズ4，ストライド幅2，パディング1の設定なので，これらを通すことにより特徴マップの縦幅・横幅がそれぞれ 1/2 になる
        # 3つ適用することになるので，最終的には都合 1/8 になる -> ゆえに，入力画像の縦幅と横幅を各々8の倍数と仮定している
        self.conv2 = myConv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv3 = myConv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv4 = myConv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)

        # 逆畳込み層5～7
        # カーネルサイズ4，ストライド幅2，パディング1の設定なので，これらを通すことにより特徴マップの縦幅・横幅がそれぞれ 2 倍になる
        # 3つ適用することになるので，最終的には元の大きさに戻る
        self.deconv5 = myConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.deconv6 = myConvTranspose2d(in_channels=128, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.deconv7 = myConvTranspose2d(in_channels=64, out_channels=16, kernel_size=4, stride=2, padding=1)

        # 畳込み層8
        # カーネルサイズ3，ストライド幅1，パディング1の設定なので，これを通しても特徴マップの縦幅・横幅は変化しない
        self.conv8 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.conv4(h3)
        h = self.deconv5(h4)
        h = torch.cat([h, h3], dim=1) # U-net 型の skip connection
        h = self.deconv6(h)
        h = torch.cat([h, h2], dim=1) # U-net 型の skip connection
        h = self.deconv7(h)
        h = torch.cat([h, h1], dim=1) # U-net 型の skip connection
        y = torch.sigmoid(self.conv8(h))
        return y