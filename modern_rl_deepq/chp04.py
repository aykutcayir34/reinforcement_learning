import matplotlib.pyplot as plt  # For plotting results
import numpy as np              # For numerical operations
import gymnasium as gym         # OpenAI Gym for reinforcement learning environments


def main():
    # Environment configuration
    env = gym.make("FrozenLake-v1", is_slippery=False)
    
    # Performance tracking variables
    win_pct = []    # Tracks win percentage over time
    scores = []     # Stores individual episode scores
    episode_count = 1000
    
    # Training loop
    for episode in range(episode_count):
        state = env.reset()
        done = False
        score = 0
        
        # Episode loop
        while not done:
            # TODO: Replace with actual policy/Q-learning
            action = env.action_space.sample()  # Currently using random actions
            
            # Take action and observe result
            n_state, reward, done, info, _ = env.step(action)
            score += reward
            
            # TODO: Add learning update step here
            # Example: Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[n_state]) - Q[state, action])
        
        # Track performance
        scores.append(score)
        if episode % 10 == 0:  # Calculate moving average every 10 episodes
            win_pct.append(np.mean(scores[-10:]))
        print(f"Episode {episode} score: {score}")
    
    # Visualize results
    plt.plot(win_pct)
    plt.title("Training Performance")
    plt.xlabel("Episode (x10)")
    plt.ylabel("Win Rate")
    plt.show()


if __name__ == "__main__":
    main()
