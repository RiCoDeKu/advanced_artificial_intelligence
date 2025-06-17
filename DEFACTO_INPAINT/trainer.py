import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from mylib.visualizers import LossVisualizer
from mylib.data_io import show_images
from mylib.utility import save_checkpoint, load_checkpoint
from metrics import region_consistency_metric

def trainer(model,
            train_dataloader,
            valid_dataloader,
            optimizer,
            loss_func,
            device, 
            n_epochs,
            restart_mode=False,
            checkpoint_epoch=None,
            checkpoint_model=None, 
            checkpoint_opt=None,
            use_amp=False,
            float_dtype=torch.float32,
            loss_scaler=None,
            model_dir=None,
            model_file=None):

    # エポック番号
    init_epoch = 0  # 初期値
    last_epoch = init_epoch + n_epochs  # 最終値
    
    # 再開モードの場合は，前回チェックポイントから情報をロードして学習再開
    if restart_mode:
        init_epoch, last_epoch, model, optimizer = load_checkpoint(
            checkpoint_epoch, checkpoint_model, checkpoint_opt, n_epochs, model, optimizer)
        print('')
    
    # 訓練データと検証データのサイズを取得
    train_size = len(train_dataloader.dataset)
    valid_size = len(valid_dataloader.dataset)
    
    # 損失関数値を記録する準備
    loss_viz = LossVisualizer(['train loss', 'valid loss', 'recall', 'precision', 'IoU'], init_epoch=init_epoch)
    
    # 勾配降下法による繰り返し学習
    for epoch in range(init_epoch, last_epoch):
        print('Epoch {0}:'.format(epoch + 1))
        
        # 学習
        model.train()
        sum_loss = 0
        for X, Y in tqdm(train_dataloader):
            for param in model.parameters():
                param.grad = None
            X = X.to(device)  # 入力画像
            Y = Y.to(device)  # 正解のマスク画像
            with torch.amp.autocast_mode.autocast(enabled=use_amp, device_type='cuda', dtype=float_dtype):
                Y_pred = model(X)  # 入力画像 X をニューラルネットワークに入力し，改ざん領域の推測値 Y_pred を得る
            loss = loss_func(Y_pred.to(torch.float32), Y)  # 損失関数の現在値を計算
            with torch.amp.autocast_mode.autocast(enabled=use_amp, device_type='cuda', dtype=float_dtype):
                if use_amp and loss_scaler is not None:
                    loss_scaler.scale(loss).backward()  # 誤差逆伝播法により，個々のパラメータに関する損失関数の勾配（偏微分）を計算
                    loss_scaler.step(optimizer)
                    loss_scaler.update()  # 勾配に沿ってパラメータの値を更新
                else:
                    loss.backward()
                    optimizer.step()
                sum_loss += float(loss) * len(X)
        avg_loss = sum_loss / train_size
        loss_viz.add_value('train loss', avg_loss)  # 訓練データに対する損失関数の値を記録
        print('train loss = {0:.6f}'.format(avg_loss))
        
        # 検証
        model.eval()
        sum_loss = 0
        sum_recall = 0
        sum_precision = 0
        sum_IoU = 0
        with torch.inference_mode():
            for X, Y in tqdm(valid_dataloader):
                X = X.to(device)  # 入力画像
                Y = Y.to(device)  # 正解のマスク画像
                Y_pred = model(X)
                loss = loss_func(Y_pred, Y)
                recall, precision, IoU = region_consistency_metric(Y_pred, Y)  # 評価指標の値を計算
                sum_recall += recall * len(X)
                sum_precision += precision * len(X)
                sum_IoU += IoU * len(X)
                sum_loss += float(loss) * len(X)
        avg_recall = sum_recall / valid_size
        avg_precision = sum_precision / valid_size
        avg_IoU = sum_IoU / valid_size
        avg_loss = sum_loss / valid_size
        loss_viz.add_value('valid loss', avg_loss)  # 検証用データに対する損失関数の値を記録
        loss_viz.add_value('recall', avg_recall)  # 検証用データに対する評価指標の値を記録
        loss_viz.add_value('precision', avg_precision)  # 同上
        loss_viz.add_value('IoU', avg_IoU)  # 同上
        print('valid loss = {0:.6f}'.format(avg_loss))
        print('recall = {0:.6f}'.format(avg_recall))
        print('precision = {0:.6f}'.format(avg_precision))
        print('IoU = {0:.6f}'.format(avg_IoU))
        print('')
        
        # 学習経過の表示
        if epoch == 0:
            show_images(Y.to('cpu').detach(), num=8, num_per_row=8, title='ground truth', save_fig=True, save_dir=model_dir)
        show_images(Y_pred.to('cpu').detach(), num=8, num_per_row=8, title='epoch {0}'.format(epoch + 1), save_fig=True, save_dir=model_dir)

        # 現在の学習状態を一時ファイルに保存
        save_checkpoint(checkpoint_epoch, checkpoint_model, checkpoint_opt, epoch+1, model, optimizer)

    model = model.to('cpu')  # モデルを CPU に移動
    torch.save(model.state_dict(), model_file)  # モデルのパラメータを保存

    loss_viz.save(v_file=os.path.join(model_dir, 'loss_graph.png'), h_file=os.path.join(model_dir, 'loss_history.csv'))

    return model, loss_viz