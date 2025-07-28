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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from pathlib import Path

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
# 評価用関数
# ==================================
def evaluate_model(model, env, device, num_episodes=10, save_data=False):
    """
    モデルを評価し、統計情報を返す
    """
    model.eval()
    episode_rewards = []
    episode_lengths = []
    action_counts = {i: 0 for i in range(env.action_space.n)}
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            state_tensor = torch.tensor(np.array(state), device=device, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                q_values = model(state_tensor)
                action = q_values.max(1)[1].item()
            
            action_counts[action] += 1
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            episode_length += 1
            state = next_state
        
        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Length = {episode_length}")
    
    stats = {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'actions': action_counts,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'max_reward': np.max(episode_rewards),
        'min_reward': np.min(episode_rewards)
    }
    
    return stats

def visualize_results(all_results, output_dir="evaluation_results"):
    """
    評価結果を可視化する
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # フォントの設定
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # データの準備
    model_names = [name.replace('model_', '').replace('.pth', '') for name in all_results.keys()]
    
    # 1. 報酬分布のボックスプロット
    plt.figure(figsize=(10, 6))
    reward_data = []
    for model_name, stats in all_results.items():
        reward_data.append(stats['rewards'])
    
    plt.boxplot(reward_data, labels=model_names)
    plt.title('Reward Distribution Comparison')
    plt.ylabel('Total Reward')
    plt.xlabel('Models')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/reward_boxplot.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 平均報酬の棒グラフ
    plt.figure(figsize=(10, 6))
    mean_rewards = [stats['mean_reward'] for stats in all_results.values()]
    std_rewards = [stats['std_reward'] for stats in all_results.values()]
    
    bars = plt.bar(model_names, mean_rewards, yerr=std_rewards, capsize=5, alpha=0.7)
    plt.title('Mean Reward Comparison')
    plt.ylabel('Mean Total Reward')
    plt.xlabel('Models')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 値をバーの上に表示
    for i, (bar, mean_val) in enumerate(zip(bars, mean_rewards)):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std_rewards[i] + 10,
                f'{mean_val:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mean_reward_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. エピソード長の比較
    plt.figure(figsize=(10, 6))
    length_data = []
    for model_name, stats in all_results.items():
        length_data.append(stats['lengths'])
    
    plt.boxplot(length_data, labels=model_names)
    plt.title('Episode Length Comparison')
    plt.ylabel('Episode Length')
    plt.xlabel('Models')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/episode_length_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. 行動分布の比較（3モデル比較のため1フィギュア）
    plt.figure(figsize=(12, 6))
    action_names = ['Do Nothing', 'Steer Left', 'Steer Right', 'Gas', 'Brake']
    
    x = np.arange(len(action_names))
    width = 0.25
    
    colors = ['blue', 'orange', 'green']
    for i, (model_name, stats) in enumerate(all_results.items()):
        total_actions = sum(stats['actions'].values())
        action_probs = [stats['actions'][j] / total_actions * 100 for j in range(len(action_names))]
        plt.bar(x + i * width, action_probs, width, 
               label=model_name.replace('model_', '').replace('.pth', ''), 
               alpha=0.7, color=colors[i])
    
    plt.title('Action Distribution Comparison (%)')
    plt.ylabel('Action Frequency (%)')
    plt.xlabel('Actions')
    plt.xticks(x + width, action_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/action_distribution_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. 報酬の時系列プロット（3モデル比較のため1フィギュア）
    plt.figure(figsize=(12, 6))
    for model_name, stats in all_results.items():
        plt.plot(stats['rewards'], 'o-', 
                label=model_name.replace('model_', '').replace('.pth', ''), alpha=0.7)
    
    plt.title('Reward per Episode')
    plt.xlabel('Episode Number')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/reward_per_episode.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. 統計情報のテーブル
    plt.figure(figsize=(10, 6))
    plt.axis('off')
    
    table_data = []
    for model_name, stats in all_results.items():
        table_data.append([
            model_name.replace('model_', '').replace('.pth', ''),
            f"{stats['mean_reward']:.1f}",
            f"{stats['std_reward']:.1f}",
            f"{stats['max_reward']:.1f}",
            f"{stats['min_reward']:.1f}",
            f"{stats['mean_length']:.0f}"
        ])
    
    table = plt.table(cellText=table_data,
                     colLabels=['Model', 'Mean', 'Std', 'Max', 'Min', 'Avg Length'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.5)
    plt.title('Performance Statistics', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_statistics.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. 個別の報酬分布ヒストグラム（各モデル個別）
    for model_name, stats in all_results.items():
        plt.figure(figsize=(8, 6))
        plt.hist(stats['rewards'], bins=10, alpha=0.7, edgecolor='black')
        model_display_name = model_name.replace('model_', '').replace('.pth', '')
        plt.title(f"Reward Distribution - {model_display_name}\nMean: {stats['mean_reward']:.1f} ± {stats['std_reward']:.1f}")
        plt.xlabel('Total Reward')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        safe_filename = model_display_name.replace(' ', '_').replace('(', '').replace(')', '')
        plt.savefig(f"{output_dir}/reward_dist_{safe_filename}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    # 8. 平均vs中央値の比較
    plt.figure(figsize=(10, 6))
    means = [stats['mean_reward'] for stats in all_results.values()]
    medians = [np.median(stats['rewards']) for stats in all_results.values()]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, means, width, label='Mean', alpha=0.7)
    plt.bar(x + width/2, medians, width, label='Median', alpha=0.7)
    
    plt.title('Mean vs Median Reward Comparison')
    plt.ylabel('Reward')
    plt.xlabel('Models')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mean_vs_median.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 9. 学習安定性の比較（変動係数）
    plt.figure(figsize=(10, 6))
    cv_values = [stats['std_reward'] / abs(stats['mean_reward']) if stats['mean_reward'] != 0 else 0 
                 for stats in all_results.values()]
    
    bars = plt.bar(model_names, cv_values, alpha=0.7, color='orange')
    plt.title('Coefficient of Variation (Lower = More Stable)')
    plt.ylabel('CV (Std/Mean)')
    plt.xlabel('Models')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 値をバーの上に表示
    for bar, cv in zip(bars, cv_values):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{cv:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/coefficient_of_variation.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 10. 成功率（正の報酬を得たエピソードの割合）
    plt.figure(figsize=(10, 6))
    success_rates = [(np.array(stats['rewards']) > 0).mean() * 100 
                     for stats in all_results.values()]
    
    bars = plt.bar(model_names, success_rates, alpha=0.7, color='green')
    plt.title('Success Rate (Episodes with Positive Reward)')
    plt.ylabel('Success Rate (%)')
    plt.xlabel('Models')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    # 値をバーの上に表示
    for bar, rate in zip(bars, success_rates):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/success_rate.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # CSVファイルに結果を保存
    results_df = pd.DataFrame([
        {
            'Model': model_name.replace('model_', '').replace('.pth', ''),
            'Mean_Reward': stats['mean_reward'],
            'Std_Reward': stats['std_reward'],
            'Max_Reward': stats['max_reward'],
            'Min_Reward': stats['min_reward'],
            'Median_Reward': np.median(stats['rewards']),
            'Q25_Reward': np.percentile(stats['rewards'], 25),
            'Q75_Reward': np.percentile(stats['rewards'], 75),
            'Mean_Length': stats['mean_length'],
            'Std_Length': stats['std_length'],
            'CV': stats['std_reward'] / abs(stats['mean_reward']) if stats['mean_reward'] != 0 else 0,
            'Success_Rate_%': (np.array(stats['rewards']) > 0).mean() * 100
        }
        for model_name, stats in all_results.items()
    ])
    
    results_df.to_csv(f"{output_dir}/evaluation_summary.csv", index=False)
    print(f"\n結果が {output_dir}/ ディレクトリに保存されました。")
    print("保存されたファイル:")
    print(f"  - {output_dir}/reward_boxplot.png")
    print(f"  - {output_dir}/mean_reward_comparison.png")
    print(f"  - {output_dir}/episode_length_comparison.png")
    print(f"  - {output_dir}/action_distribution_comparison.png")
    print(f"  - {output_dir}/reward_per_episode.png")
    print(f"  - {output_dir}/performance_statistics.png")
    print(f"  - {output_dir}/reward_dist_[model_name].png (各モデル)")
    print(f"  - {output_dir}/mean_vs_median.png")
    print(f"  - {output_dir}/coefficient_of_variation.png")
    print(f"  - {output_dir}/success_rate.png")
    print(f"  - {output_dir}/evaluation_summary.csv")
    
    return results_df
    
    # 2. 詳細な報酬分布のヒストグラム
    plt.figure(figsize=(12, 8))
    for i, (model_name, stats) in enumerate(all_results.items()):
        if i >= 4:  # 最大4つのサブプロットまで
            break
        plt.subplot(2, 2, i+1)
        plt.hist(stats['rewards'], bins=10, alpha=0.7, edgecolor='black')
        model_display_name = model_name.replace('model_', '').replace('.pth', '')
        plt.title(f"{model_display_name}\nMean: {stats['mean_reward']:.1f} ± {stats['std_reward']:.1f}")
        plt.xlabel('Total Reward')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/reward_distributions.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. モデル間の詳細比較
    plt.figure(figsize=(16, 6))
    
    # 3-1. 報酬の統計比較（平均、中央値、四分位範囲）
    plt.subplot(1, 3, 1)
    model_names = [name.replace('model_', '').replace('.pth', '') for name in all_results.keys()]
    means = [stats['mean_reward'] for stats in all_results.values()]
    medians = [np.median(stats['rewards']) for stats in all_results.values()]
    q25 = [np.percentile(stats['rewards'], 25) for stats in all_results.values()]
    q75 = [np.percentile(stats['rewards'], 75) for stats in all_results.values()]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, means, width, label='Mean', alpha=0.7)
    plt.bar(x + width/2, medians, width, label='Median', alpha=0.7)
    
    plt.title('Mean vs Median Reward Comparison')
    plt.ylabel('Reward')
    plt.xlabel('Models')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3-2. 学習安定性の比較（変動係数）
    plt.subplot(1, 3, 2)
    cv_values = [stats['std_reward'] / abs(stats['mean_reward']) if stats['mean_reward'] != 0 else 0 
                 for stats in all_results.values()]
    
    bars = plt.bar(model_names, cv_values, alpha=0.7, color='orange')
    plt.title('Coefficient of Variation\n(Lower = More Stable)')
    plt.ylabel('CV (Std/Mean)')
    plt.xlabel('Models')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 値をバーの上に表示
    for bar, cv in zip(bars, cv_values):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{cv:.3f}', ha='center', va='bottom')
    
    # 3-3. 成功率（正の報酬を得たエピソードの割合）
    plt.subplot(1, 3, 3)
    success_rates = [(np.array(stats['rewards']) > 0).mean() * 100 
                     for stats in all_results.values()]
    
    bars = plt.bar(model_names, success_rates, alpha=0.7, color='green')
    plt.title('Success Rate\n(Episodes with Positive Reward)')
    plt.ylabel('Success Rate (%)')
    plt.xlabel('Models')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    # 値をバーの上に表示
    for bar, rate in zip(bars, success_rates):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/detailed_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/detailed_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. CSVファイルに結果を保存
    results_df = pd.DataFrame([
        {
            'Model': model_name.replace('model_', '').replace('.pth', ''),
            'Mean_Reward': stats['mean_reward'],
            'Std_Reward': stats['std_reward'],
            'Max_Reward': stats['max_reward'],
            'Min_Reward': stats['min_reward'],
            'Median_Reward': np.median(stats['rewards']),
            'Q25_Reward': np.percentile(stats['rewards'], 25),
            'Q75_Reward': np.percentile(stats['rewards'], 75),
            'Mean_Length': stats['mean_length'],
            'Std_Length': stats['std_length'],
            'CV': stats['std_reward'] / abs(stats['mean_reward']) if stats['mean_reward'] != 0 else 0,
            'Success_Rate_%': (np.array(stats['rewards']) > 0).mean() * 100
        }
        for model_name, stats in all_results.items()
    ])
    
    results_df.to_csv(f"{output_dir}/evaluation_summary.csv", index=False)
    print(f"\n結果が {output_dir}/ ディレクトリに保存されました。")
    print("保存されたファイル:")
    print(f"  - {output_dir}/comprehensive_evaluation.png")
    print(f"  - {output_dir}/reward_distributions.png")
    print(f"  - {output_dir}/detailed_comparison.png")
    print(f"  - {output_dir}/evaluation_summary.csv")
    
    return results_df

# ==================================
# 推論の実行
# ==================================
if __name__ == "__main__":
    # --- 設定 ---
    # 評価する学習済みモデルのパス（batch64は除外）
    MODEL_PATHS = {
        "model_Double_DQN_with_Huber_Loss.pth": "/home/kosuke/Nakamura/advanced_artificial_intelligence/Assignment3/car_racing_results/model_Double_DQN_with_Huber_Loss.pth",
        "model_DQN_with_Huber_Loss.pth": "/home/kosuke/Nakamura/advanced_artificial_intelligence/Assignment3/car_racing_results/model_DQN_with_Huber_Loss.pth",
        "model_DQN_with_MSE_Loss.pth": "/home/kosuke/Nakamura/advanced_artificial_intelligence/Assignment3/car_racing_results/model_DQN_with_MSE_Loss.pth"
    }
    
    NUM_EPISODES_TO_EVALUATE = 20  # 各モデルの評価エピソード数（統計的信頼性のため増加）
    VISUALIZE_GAMEPLAY = False     # ゲームプレイを表示するかどうか
    SAVE_RESULTS = True           # 結果を保存するかどうか

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 結果を格納する辞書
    all_results = {}

    # 各モデルを評価
    for model_name, model_path in MODEL_PATHS.items():
        print(f"\n{'='*50}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*50}")
        
        # モデルファイルが存在するかチェック
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}")
            continue
        
        # 環境の初期化
        render_mode = "human" if VISUALIZE_GAMEPLAY else None
        base_env = gym.make("CarRacing-v3", continuous=False, render_mode=render_mode)
        env = FrameStack(base_env, k=4)

        num_actions = env.action_space.n
        state_shape = env.observation_space.shape

        # モデルの構造を定義し、デバイスに送る
        model = DQN(state_shape, num_actions).to(device)

        # 学習済みの重みをロードする
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Successfully loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            env.close()
            continue

        # モデルを評価
        stats = evaluate_model(model, env, device, 
                             num_episodes=NUM_EPISODES_TO_EVALUATE, 
                             save_data=SAVE_RESULTS)
        
        all_results[model_name] = stats
        
        # 環境をクローズ
        env.close()
        
        print(f"\n{model_name} 評価結果:")
        print(f"  平均報酬: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"  最大報酬: {stats['max_reward']:.2f}")
        print(f"  最小報酬: {stats['min_reward']:.2f}")
        print(f"  平均エピソード長: {stats['mean_length']:.1f}")

    # 結果の可視化
    if all_results and SAVE_RESULTS:
        print(f"\n{'='*50}")
        print("結果の可視化と保存を開始...")
        print(f"{'='*50}")
        
        # 結果ディレクトリを作成
        output_dir = "evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # 可視化を実行
        results_df = visualize_results(all_results, output_dir)
        
        # 詳細結果をCSVに保存
        detailed_results = []
        for model_name, stats in all_results.items():
            for i, (reward, length) in enumerate(zip(stats['rewards'], stats['lengths'])):
                detailed_results.append({
                    'Model': model_name.replace('model_', '').replace('.pth', ''),
                    'Episode': i + 1,
                    'Total_Reward': reward,
                    'Episode_Length': length
                })
        
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv(f"{output_dir}/detailed_evaluation_results.csv", index=False)
        
        print(f"\n詳細な評価結果:")
        print(results_df.to_string(index=False))
    
    elif not all_results:
        print("\nエラー: 評価できるモデルがありませんでした。")
    
    print(f"\n{'='*50}")
    print("評価完了!")
    print(f"{'='*50}")

