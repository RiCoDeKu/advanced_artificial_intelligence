# 物体改竄領域の推定


## モジュールインストール
```bash
uv pip install segmentation_models_pytorch
```

### 1. ディレクトリ構成
```
.
├── Datasets 
├── defacto_models # 学習済みモデル
├── temp 
├── mylib 
├── scratch_model/ # 一から学習する場合
│   ├── metrics.py　
│   ├── model_1.py
│   ├── model_base.py
│   ├── train.py
│   └── trainer.py
├── dataset.py # データローダー作成
├── download.py # データセットのダウンロード
├── evaluation.ipynb
├── hoge.ipynb
└── util.ipynb
```


## 2. 学習
```bash
python scratch_model/train.py 
```