import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym


def main():
    env = gym.make("FrozenLake-v1", is_slippery=False)
    win_pct = []
    scores = []
    episode_count = 1000
    for episode in range(episode_count):
        state = env.reset()
        done = False
        score = 0
        while not done:
            action = env.action_space.sample()
            n_state, reward, done, info, _ = env.step(action)
            score += reward
        scores.append(score)
        if episode % 10 == 0:
            win_pct.append(np.mean(scores[-10:]))
        print(f"Episode {episode} score: {score}")
    plt.plot(win_pct)
    plt.show()


if __name__ == "__main__":
    main()
