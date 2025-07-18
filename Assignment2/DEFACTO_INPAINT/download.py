import os
import subprocess
import torch
from torchvision import transforms
from mylib.data_io import CSVBasedDataset


def make_defacto_tensors(filename, dirname='./defacto'):
    transform = transforms.Grayscale()
    dataset = CSVBasedDataset(dirname=dirname, filename=os.path.join(dirname, filename), items=['Input Image', 'Ground Truth'], dtypes=['image', 'image'])
    data = [dataset[i] for i in range(len(dataset))]
    x = torch.cat([torch.unsqueeze(u, dim=0) for u, v in data], dim=0)
    y = torch.cat([torch.unsqueeze(transform(v), dim=0) for u, v in data], dim=0)
    del dataset, data
    return x, y


def main():
    if not os.path.isfile('defacto_train_input_images.pt'):
        # データセットをダウンロードして解凍
        subprocess.run(["wget", "https://tus.box.com/shared/static/v4giouhhlttk29vgoaxuf56rsuzcjkc2.gz", "-O", "defacto.tar.gz"], check=True)
        subprocess.run(["tar", "-zxf", "defacto.tar.gz"], check=True)
        subprocess.run(["rm", "-f", "defacto.tar.gz"], check=True)

        # 必要なディレクトリが存在することを確認
        os.makedirs('./Datasets', exist_ok=True)
        os.makedirs('./temp', exist_ok=True)

        # トレーニングデータの処理
        image_tensor, label_tensor = make_defacto_tensors(filename='train_list.csv')
        torch.save(image_tensor, './Datasets/defacto_train_input_images.pt')
        torch.save(label_tensor, './Datasets/defacto_train_target_images.pt')
        del image_tensor, label_tensor

        # テストデータの処理
        image_tensor, label_tensor = make_defacto_tensors(filename='test_list.csv')
        torch.save(image_tensor, './Datasets/defacto_test_input_images.pt')
        torch.save(label_tensor, './Datasets/defacto_test_target_images.pt')
        torch.save(image_tensor, './temp/defacto_test_input_images.pt')
        torch.save(label_tensor, './temp/defacto_test_target_images.pt')
        del image_tensor, label_tensor

        # 一時ファイルの削除
        subprocess.run(["rm", "-fr", "defacto"], check=True)


if __name__ == "__main__":
    main()