from joblib import load
from matplotlib import pyplot as plt

def plot_rewards(rew, averaging_window=50, title="Total reward per episode (online)", version=1):
        """
        Visually represent the learning history to standard output.
        """
        averages = []
        for i in range(1, len(rew) + 1):
            lower = max(0, i - averaging_window)
            averages.append(sum(rew[lower:i]) / (i - lower))
        plt.xlabel("Episode")
        plt.ylabel("Episode length with " + str(averaging_window) + "-running average")
        plt.title(title)
        plt.plot(averages, color="black")
        plt.scatter(range(len(rew)), rew, s=2)
        plt.savefig(f"v{version}_reward_hex.png")
        plt.show()

for i in range(1, 18):
    reward = load(f'v{i}_hex_episode_rewards.a2c')
    plot_rewards(reward, version=i)

    print(len(reward))
