import warnings; warnings.filterwarnings("ignore", message=".*pkg_resources.*")
from tqdm import tqdm
import gymnasium as gym
import csv

env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)

N_STEPS = 1000
N_EPISODES = 10  # 保存したいエピソード数

results = []

for episode in range(N_EPISODES):
	state, _ = env.reset()
	total_reward = 0
	steps = 0
	for t in tqdm(range(N_STEPS), desc=f"Episode {episode+1}"):
		action = env.action_space.sample()  # Random action
		state, reward, terminated, truncated, info = env.step(action)
		#print(f"Step {t+1}: Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
		total_reward += reward
		steps += 1
		if terminated or truncated:
			break
	results.append([episode+1, steps, total_reward])

# CSVに保存
with open("car_racing_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Episode", "Steps", "TotalReward"])
    writer.writerows(results)
