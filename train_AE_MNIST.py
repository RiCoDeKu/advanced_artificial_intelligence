import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from src.utils.data_io import TensorDataset
from src.utils.visualizers import LossVisualizer
from src.utils.data_io import show_images
from src.utils.utility import save_checkpoint, load_checkpoint
from src.AE_MNIST import MNISTEncoder, MNISTDecoder

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

def main():
	# 特徴ベクトルの次元数
	N = 32
	print(f"Using feature vector dimension N={N}")
	train(N)

def train(N = 32):
	# 特徴ベクトルの次元数
	print(f"Using feature vector dimension N={N}")
	N_EPOCHS = 20
	USE_BATCH_NORM = True

	DEVICE = 'cuda:0'
	RESTART_MODE = False
	BATCH_SIZE = 100

	# データセットの存在するフォルダ・ファイル名
	DATA_DIR = './data/MNIST'
	TRAIN_IMAGES_FILE = 'MNIST_train_images.pt'
	VALID_IMAGES_FILE = 'MNIST_valid_images.pt'

	# 学習結果の保存先フォルダ 変更すること
	MODEL_DIR = './models/MNIST/AE/'

	# 学習結果のニューラルネットワークの保存先
	MODEL_FILE_ENC = os.path.join(MODEL_DIR, f'{N}_encoder.pth') # エンコーダ
	MODEL_FILE_DEC = os.path.join(MODEL_DIR, f'{N}_decoder.pth') # デコーダ

	# 中断／再開の際に用いる一時ファイルの保存先
	#####       一旦置いとく・・・・        #####
	CHECKPOINT_EPOCH = os.path.join('./temp/', 'checkpoint_epoch.pkl')
	CHECKPOINT_ENC_MODEL = os.path.join('./temp/', 'checkpoint_enc_model.pth')
	CHECKPOINT_DEC_MODEL = os.path.join('./temp/', 'checkpoint_dec_model.pth')
	CHECKPOINT_ENC_OPT = os.path.join('./temp/', 'checkpoint_enc_opt.pth')
	CHECKPOINT_DEC_OPT = os.path.join('./temp/', 'checkpoint_dec_opt.pth')


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
		valid_size = int(0.01 * dataset_size) # 全体の 1% を検証用に
		train_size = dataset_size - valid_size # 残りの 99% を学習用に
		train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

		# データセット情報をファイルに保存
		torch.save(torch.cat([torch.unsqueeze(train_dataset[i], dim=0) for i in range(len(train_dataset))], dim=0), os.path.join('./temp/', TRAIN_IMAGES_FILE))
		torch.save(torch.cat([torch.unsqueeze(valid_dataset[i], dim=0) for i in range(len(valid_dataset))], dim=0), os.path.join('./temp/', VALID_IMAGES_FILE))


	# 訓練データおよび検証用データをミニバッチに分けて使用するための「データローダ」を用意
	train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8)
	valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8)

	# ./temp ディレクトリの中身を全て削除
	if os.path.exists('./temp'):
		shutil.rmtree('./temp')
		os.makedirs('./temp')
		
	# エポック番号
	INIT_EPOCH = 0 # 初期値
	LAST_EPOCH = INIT_EPOCH + N_EPOCHS # 最終値

	# ニューラルネットワークの作成
	enc_model = MNISTEncoder(N=N, use_BatchNorm=USE_BATCH_NORM).to(DEVICE)
	dec_model = MNISTDecoder(N=N, use_BatchNorm=USE_BATCH_NORM).to(DEVICE)

	# 最適化アルゴリズムの指定（ここでは SGD でなく Adam を使用）
	enc_optimizer = optim.Adam(enc_model.parameters())
	dec_optimizer = optim.Adam(dec_model.parameters())

	# 再開モードの場合は，前回チェックポイントから情報をロードして学習再開
	if RESTART_MODE:
		print('Loading checkpoint ...')
		INIT_EPOCH, LAST_EPOCH, enc_model, enc_optimizer = load_checkpoint(CHECKPOINT_EPOCH, CHECKPOINT_ENC_MODEL, CHECKPOINT_ENC_OPT, N_EPOCHS, enc_model, enc_optimizer)
		_, _, dec_model, dec_optimizer = load_checkpoint(CHECKPOINT_EPOCH, CHECKPOINT_DEC_MODEL, CHECKPOINT_DEC_OPT, N_EPOCHS, dec_model, dec_optimizer)
		print('')

	# 損失関数： 平均二乗誤差損失を使用（これが最適とは限らない．平均絶対誤差損失 nn.L1Loss() なども考えられる）
	loss_func = nn.MSELoss()

	# 損失関数値を記録する準備
	loss_viz = LossVisualizer(['train loss', 'valid loss'], init_epoch=INIT_EPOCH)

	# 勾配降下法による繰り返し学習
	for epoch in range(INIT_EPOCH, LAST_EPOCH):

		print('Epoch {0}:'.format(epoch + 1))

		# 学習
		enc_model.train()
		dec_model.train()
		sum_loss = 0
		for X in tqdm(train_dataloader):
			for param in enc_model.parameters():
				param.grad = None
			for param in dec_model.parameters():
				param.grad = None
			X = X.to(DEVICE)
			Z = enc_model(X) # 入力画像 X を現在のエンコーダに入力し，特徴ベクトル Z を得る
			Y = dec_model(Z) # 特徴ベクトル Z を現在のデコーダに入力し，復元画像 Y を得る
			loss = loss_func(X, Y) # 損失関数の現在値を計算
			loss.backward() # 誤差逆伝播法により，個々のパラメータに関する損失関数の勾配（偏微分）を計算
			enc_optimizer.step() # 勾配に沿ってパラメータの値を更新
			dec_optimizer.step() # 同上
			sum_loss += float(loss) * len(X)
		avg_loss = sum_loss / train_size
		loss_viz.add_value('train loss', avg_loss) # 訓練データに対する損失関数の値を記録
		print('train loss = {0:.6f}'.format(avg_loss))

		# 検証
		enc_model.eval()
		dec_model.eval()
		sum_loss = 0
		with torch.inference_mode():
			for X in tqdm(valid_dataloader):
				X = X.to(DEVICE)
				Z = enc_model(X)
				Y = dec_model(Z)
				loss = loss_func(X, Y)
				sum_loss += float(loss) * len(X)
		avg_loss = sum_loss / valid_size
		loss_viz.add_value('valid loss', avg_loss) # 検証用データに対する損失関数の値を記録
		print('valid loss = {0:.6f}'.format(avg_loss))
		print('')

		# 学習経過の表示
		if epoch == 0:
			show_images(X.to('cpu').detach(), num=BATCH_SIZE, title='original', save_fig=False, save_dir=MODEL_DIR)
		show_images(Y.to('cpu').detach(), num=BATCH_SIZE, title='epoch {0}'.format(epoch + 1), save_fig=False, save_dir=MODEL_DIR)

		# 現在の学習状態を一時ファイル（チェックポイント）に保存
		save_checkpoint(CHECKPOINT_EPOCH, CHECKPOINT_ENC_MODEL, CHECKPOINT_ENC_OPT, epoch+1, enc_model, enc_optimizer)
		save_checkpoint(CHECKPOINT_EPOCH, CHECKPOINT_DEC_MODEL, CHECKPOINT_DEC_OPT, epoch+1, dec_model, dec_optimizer)

	# 学習結果のニューラルネットワークモデルをファイルに保存
	enc_model = enc_model.to('cpu')
	dec_model = dec_model.to('cpu')
	torch.save(enc_model.state_dict(), MODEL_FILE_ENC)
	torch.save(dec_model.state_dict(), MODEL_FILE_DEC)

	# 損失関数の記録をファイルに保存
	loss_viz.save(v_file=os.path.join(MODEL_DIR, f'{N}_loss_graph.png'), h_file=os.path.join(MODEL_DIR, f'{N}_loss_history.csv'))


if __name__ == '__main__':
	main()