# -*- coding: utf-8 -*-
"""
学習済みDQNモデルによる推論プログラム (inference.py)

このスクリプトは、学習済みのモデルファイル (.pth) を読み込み、
そのモデルを使ってCarRacing環境をプレイします。

実行方法:
1. このファイルを `inference.py` として保存します。
2. MODEL_PATH に、使用したい学習済みモデルのパスを指定します。
3. `python inference.py` を実行します。
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import cv2
import time

# --- 注意 ---
# このスクリプトを単体で実行するために、学習スクリプトと同じ
# クラス・関数定義をここに含める必要があります。

# ==================================
# モデル・環境の定義（学習スクリプトからコピー）
# ==================================
def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized_frame = cv2.resize(gray_frame, (84, 84), interpolation=cv2.INTER_AREA)
    return np.expand_dims(resized_frame, axis=0)

class FrameStack:
    def __init__(self, env, k):
        self.env = env
        self.k = k
        self.frames = deque([], maxlen=k)
        obs_shape = (k, 84, 84)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        self.action_space = self.env.action_space

    def reset(self):
        obs, info = self.env.reset()
        processed_obs = preprocess_frame(obs)
        for _ in range(self.k):
            self.frames.append(processed_obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed_obs = preprocess_frame(obs)
        self.frames.append(processed_obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=0)
    
    def close(self):
        self.env.close()

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x / 255.0)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

# ==================================
# 推論の実行
# ==================================
if __name__ == "__main__":
    # --- 設定 ---
    # 使用する学習済みモデルのパスを指定してください
    MODEL_PATH = "/home/yamaguchi/AAI/advanced_artificial_intelligence/Assignment3/car_racing_results/model_DQN_with_Huber_Loss.pth"
    NUM_EPISODES_TO_RUN = 5 # テスト実行するエピソード数

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 環境の初期化 (描画モードを "human" にする)
    base_env = gym.make("CarRacing-v3", continuous=False, render_mode="human")
    env = FrameStack(base_env, k=4)

    num_actions = env.action_space.n
    state_shape = env.observation_space.shape

    # モデルの構造を定義し、デバイスに送る
    model = DQN(state_shape, num_actions).to(device)

    # 学習済みの重みをロードする
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please make sure the path is correct and the model has been trained.")
        exit()

    # モデルを評価モードに設定 (重要: DropoutやBatchNormなどを無効化)
    model.eval()

    for episode in range(NUM_EPISODES_TO_RUN):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        print(f"\n--- Starting Episode {episode + 1} ---")

        while not done:
            # 状態をTensorに変換
            state_tensor = torch.tensor(np.array(state), device=device, dtype=torch.float32).unsqueeze(0)

            # モデルを使って最適な行動を推論 (探索はしない)
            with torch.no_grad(): # 勾配計算を無効化して計算を高速化
                q_values = model(state_tensor)
                action = q_values.max(1)[1].item() # 最もQ値が高い行動を選択

            # 選択した行動を実行
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            state = next_state
            
            # 描画が早すぎないように少し待機
            time.sleep(0.01)

        print(f"Episode {episode + 1} finished. Total Reward: {total_reward:.2f}")

    env.close()
    print("\nInference finished.")

