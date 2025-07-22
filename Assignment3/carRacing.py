# -*- coding: utf-8 -*-
"""
DQNによる自律走行AIの構築 (CarRacing) - Pythonスクリプト版

このスクリプトは、Gymnasiumの`CarRacing-v2`環境を対象に、
Deep Q-Network (DQN) を用いて自律走行AIを学習させます。

以下のステップで段階的にDQNモデルを改良し、その効果を検証します。
1. 基本的なDQNの実装 (損失関数: MSE)
2. 損失関数をHuber損失に変更
3. バッチサイズを64に変更
4. Double DQNを導入

実行前に、必要なライブラリをインストールしてください:
pip install gymnasium[box2d] pygame torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm numpy matplotlib opencv-python
"""

# ==================================
# ステップ0：ライブラリのインポート
# ==================================
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import os
import csv

# 不要な警告を非表示に
warnings.filterwarnings("ignore", category=UserWarning)

# ==================================
# ステップ1：環境の準備と状態の前処理
# ==================================

def preprocess_frame(frame):
    """フレームをグレースケール化し、リサイズする関数"""
    # (96, 96, 3) -> (96, 96)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # (96, 96) -> (84, 84)
    resized_frame = cv2.resize(gray_frame, (84, 84), interpolation=cv2.INTER_AREA)
    # (84, 84) -> (1, 84, 84) PyTorchの入力形式に合わせる
    return np.expand_dims(resized_frame, axis=0)

class FrameStack:
    """連続するフレームをスタックするためのラッパークラス"""
    def __init__(self, env, k):
        self.env = env
        self.k = k  # スタックするフレーム数
        self.frames = deque([], maxlen=k)
        
        # (k, 84, 84) の形状を持つ観測空間
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
        # deque内のフレームを結合して (k, 84, 84) のnumpy配列にする
        return np.concatenate(list(self.frames), axis=0)
    
    def close(self):
        self.env.close()

# ==================================
# ステップ2：DQNコンポーネントの実装
# ==================================

class ReplayBuffer:
    """経験を保存し、サンプリングするためのリプレイバッファ"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # 状態は (4, 84, 84) のnumpy配列
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    """状態を入力とし、Q値を出力するCNNモデル"""
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        # Nature-DQNで使われたCNNアーキテクチャを参考
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
        # 入力xは (バッチサイズ, 4, 84, 84)
        x = self.features(x / 255.0) # 値を正規化
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x
    
    def feature_size(self):
        # CNN部分の出力サイズを計算
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

# ==================================
# ステップ3：学習ロジックの実装
# ==================================

def compute_loss(batch_vars, policy_net, target_net, gamma, device, loss_fn, use_double_dqn=False):
    """損失を計算する関数"""
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch_vars

    # データをTensorに変換し、GPUへ送る
    state_batch = torch.tensor(state_batch, device=device, dtype=torch.float32)
    next_state_batch = torch.tensor(next_state_batch, device=device, dtype=torch.float32)
    action_batch = torch.tensor(action_batch, device=device, dtype=torch.long).unsqueeze(1)
    reward_batch = torch.tensor(reward_batch, device=device, dtype=torch.float32)
    done_batch = torch.tensor(done_batch, device=device, dtype=torch.float32)

    # 現在のQ値 (Q(s, a)) を計算
    q_values = policy_net(state_batch).gather(1, action_batch)

    # 目標Q値の計算
    with torch.no_grad():
        if use_double_dqn:
            # Double DQN: 行動選択はpolicy_net, 価値評価はtarget_net
            best_actions = policy_net(next_state_batch).argmax(1).unsqueeze(1)
            next_q_values = target_net(next_state_batch).gather(1, best_actions).squeeze(1)
        else:
            # Standard DQN: 行動選択も価値評価もtarget_net
            next_q_values = target_net(next_state_batch).max(1)[0]
        
        # 完了した状態のQ値は0とする
        expected_q_values = reward_batch + (gamma * next_q_values * (1 - done_batch))

    # 損失を計算
    loss = loss_fn(q_values, expected_q_values.unsqueeze(1))
    return loss

def select_action(state, policy_net, epsilon, num_actions, device):
    """ε-greedy法に基づいて行動を選択"""
    if random.random() > epsilon:
        with torch.no_grad():
            # (4, 84, 84) -> (1, 4, 84, 84)
            state_tensor = torch.tensor(np.array(state), device=device, dtype=torch.float32).unsqueeze(0)
            q_values = policy_net(state_tensor)
            return q_values.max(1)[1].item()
    else:
        return random.randrange(num_actions)

def plot_rewards(rewards, title, filename):
    """報酬をプロットしてファイルに保存する関数"""
    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.xlabel('Episode')
    plt.xlim(0, len(rewards) - 1)
    plt.ylabel('Total Reward')
    plt.plot(rewards, label='Total Reward per Episode')
    # 移動平均もプロットして傾向を見やすくする
    if len(rewards) >= 10:
        moving_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
        plt.plot(np.arange(9, len(rewards)), moving_avg, label='Moving Average (10 episodes)')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close() # メモリを解放するためにプロットを閉じる

def plot_all_rewards(results_dict, filename):
    """全ての実験結果を一つのグラフにプロットする関数"""
    plt.figure(figsize=(15, 8))
    plt.title('Comparison of All Training Configurations')
    
    for name, stats in results_dict.items():
        rewards = [s['TotalReward'] for s in stats]
        # 移動平均をプロットして比較しやすくする
        if len(rewards) >= 10:
            moving_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
            plt.plot(np.arange(9, len(rewards)), moving_avg, label=name)

    plt.xlabel('Episode')
    plt.ylabel('Total Reward (10-episode moving average)')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"\nComparison plot saved to {filename}")


# ==================================
# ステップ4：学習の実行
# ==================================

def train(config):
    """DQNモデルの学習を実行するメイン関数"""
    print(f"\n--- Training with config: {config['name']} ---")
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 環境の初期化
    base_env = gym.make("CarRacing-v3", render_mode=None, lap_complete_percent=0.95, domain_randomize=False, continuous=False)
    env = FrameStack(base_env, k=4)

    num_actions = env.action_space.n
    state_shape = env.observation_space.shape

    # ネットワークの初期化
    policy_net = DQN(state_shape, num_actions).to(device)
    target_net = DQN(state_shape, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval() # ターゲットネットワークは評価モード

    # オプティマイザとリプレイバッファ
    optimizer = optim.Adam(policy_net.parameters(), lr=config['learning_rate'])
    memory = ReplayBuffer(config['replay_buffer_size'])

    # ε（イプシロン）の設定
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 30000
    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * frame_idx / epsilon_decay)

    all_episode_stats = []
    frame_idx = 0

    for episode in range(config['num_episodes']):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        # tqdmのプログレスバーをエピソードごとに設定 (最大ステップ数は1000)
        with tqdm(total=1000, desc=f"Episode {episode+1}/{config['num_episodes']}", unit=" step") as pbar:
            while True:
                frame_idx += 1
                episode_steps += 1
                epsilon = epsilon_by_frame(frame_idx)
                action = select_action(state, policy_net, epsilon, num_actions, device)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                memory.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                # プログレスバーの更新と報酬の表示
                pbar.set_postfix(reward=f"{episode_reward:.2f}")
                pbar.update(1)

                if len(memory) > config['batch_size']:
                    batch_vars = memory.sample(config['batch_size'])
                    use_ddqn = config.get('use_double_dqn', False)
                    loss = compute_loss(batch_vars, policy_net, target_net, config['gamma'], device, config['loss_fn'], use_ddqn)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    # 勾配クリッピング
                    for param in policy_net.parameters():
                        if param.grad is not None:
                            param.grad.data.clamp_(-1, 1)
                    optimizer.step()

                if frame_idx % config['target_update_freq'] == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                if done:
                    # エピソードが早く終了した場合、バーの残りを埋める
                    pbar.update(1000 - pbar.n)
                    break
        
        all_episode_stats.append({
            "Episode": episode + 1,
            "TotalReward": episode_reward,
            "FinalEpsilon": epsilon,
            "TotalFrames": frame_idx,
            "EpisodeSteps": episode_steps
        })
        # tqdmの表示と重複するため、エピソードごとのprintは詳細情報のみに
        # print(f"Episode {episode+1}/{config['num_episodes']} finished. Total Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}, Steps: {episode_steps}")
        
    env.close()
    
    # --- 結果の保存 ---
    output_dir = "car_racing_results"
    os.makedirs(output_dir, exist_ok=True)
    config_name_safe = config['name'].replace(' ', '_').replace('&', '_and_')

    # CSVファイルに結果を保存
    csv_filename = os.path.join(output_dir, f"results_{config_name_safe}.csv")
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_episode_stats[0].keys())
        writer.writeheader()
        writer.writerows(all_episode_stats)
    print(f"Results CSV saved to {csv_filename}")

    # 報酬グラフの保存
    all_rewards = [s['TotalReward'] for s in all_episode_stats]
    plot_filename = os.path.join(output_dir, f"rewards_{config_name_safe}.png")
    plot_rewards(all_rewards, f"Final Rewards: {config['name']}", plot_filename)
    print(f"Reward plot saved to {plot_filename}")

    # モデルの保存
    model_filename = os.path.join(output_dir, f"model_{config_name_safe}.pth")
    torch.save(policy_net.state_dict(), model_filename)
    print(f"Model saved to {model_filename}")

    print(f"Training finished for {config['name']}.")
    return all_episode_stats

# ==================================
# メイン実行ブロック
# ==================================
if __name__ == "__main__":
    # --- 実験設定 ---
    # デモ用にエピソード数を少なく設定。実際は1000以上を推奨
    NUM_EPISODES = 1000 
    
    all_training_results = {}

    # 工夫1：基本的なDQNモデルの実装
    config_mse = {
        'name': 'DQN with MSE Loss',
        'num_episodes': NUM_EPISODES,
        'replay_buffer_size': 10000,
        'batch_size': 32,
        'gamma': 0.99,
        'learning_rate': 1e-4,
        'target_update_freq': 1000,
        'loss_fn': nn.MSELoss(),
        'use_double_dqn': False
    }
    all_training_results[config_mse['name']] = train(config_mse)

    # 工夫2：損失関数をHuber損失に変更
    config_huber = {
        'name': 'DQN with Huber Loss',
        'num_episodes': NUM_EPISODES,
        'replay_buffer_size': 10000,
        'batch_size': 32,
        'gamma': 0.99,
        'learning_rate': 1e-4,
        'target_update_freq': 1000,
        'loss_fn': nn.SmoothL1Loss(),
        'use_double_dqn': False
    }
    all_training_results[config_huber['name']] = train(config_huber)

    # 工夫3：バッチサイズを変更
    config_batch64 = {
        'name': 'DQN with Huber Loss and Batch Size 64',
        'num_episodes': NUM_EPISODES,
        'replay_buffer_size': 10000,
        'batch_size': 64,
        'gamma': 0.99,
        'learning_rate': 1e-4,
        'target_update_freq': 1000,
        'loss_fn': nn.SmoothL1Loss(),
        'use_double_dqn': False
    }
    all_training_results[config_batch64['name']] = train(config_batch64)

    # 工夫4：Double DQNを導入
    config_double_dqn = {
        'name': 'Double DQN with Huber Loss',
        'num_episodes': NUM_EPISODES,
        'replay_buffer_size': 10000,
        'batch_size': 32,
        'gamma': 0.99,
        'learning_rate': 1e-4,
        'target_update_freq': 1000,
        'loss_fn': nn.SmoothL1Loss(),
        'use_double_dqn': True # Double DQNを有効化
    }
    all_training_results[config_double_dqn['name']] = train(config_double_dqn)

    print("\nAll training sessions are complete.")
    
    # --- 全結果の比較グラフを生成 ---
    output_dir = "car_racing_results"
    comparison_filename = os.path.join(output_dir, "all_configs_comparison.png")
    plot_all_rewards(all_training_results, comparison_filename)
    
    print("Check the 'car_racing_results' directory for saved models, reward plots, CSV results, and the final comparison plot.")
