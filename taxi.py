#laura nyrhilä taxi.py code

import gymnasium as gym
import numpy as np
import time

# solve the problem manually
def manual_exploration():
    env = gym.make("Taxi-v3", render_mode='ansi')
    state, _ = env.reset()

    done = False
    print(env.render())  # map rendering

    while not done:
        try:
            #manual input from user for each action
            action = int(input("Enter action (0: south, 1: north, 2: east, 3: west, 4: pickup, 5: dropoff): "))
            if action not in range(6):
                print("Invalid action. Please enter a number between 0 and 5.")
                continue

            # Take action and observe the results
            state, reward, done, _, _ = env.step(action)
            print(env.render())  # Render the map environment after each step
            print(f"Reward: {reward}, Done: {done}")

        except ValueError:
            print("Please enter a valid integer action.")
    
    print("Manual exploration complete.")

# Q-learning algorithm to find the optimal policy
def q_learning_taxi():
    env = gym.make("Taxi-v3", render_mode='ansi')
    state_dim = env.observation_space.n  
    action_dim = env.action_space.n      
    
    # Hyperparameters
    alpha = 0.9      # Learning rate
    gamma = 0.9      # Discount factor
    epsilon = 1.0    # Initial exploration probability
    epsilon_min = 0.1  # Minimum epsilon
    epsilon_decay = 0.995  # Decay rate for epsilon
    num_episodes = 1000  # Number of episodes
    max_steps = 100  # Max steps per episode

    # Q-table initialization
    Q = np.zeros((state_dim, action_dim))

    # Training fro-loop
    for episode in range(num_episodes):
        state, _ = env.reset()  # reset starts with fresh environment
        done = False

        for step in range(max_steps):
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(Q[state, :])  # Exploit

            # Perform action and observe the new state and reward
            new_state, reward, done, _, _ = env.step(action)

            # Q-learning update rule
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state, :]))

            state = new_state

            if done:
                break

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Test the trained policy
    test_episodes = 10
    test_rewards = []
    test_steps = []
    
    for _ in range(test_episodes):
        state, _ = env.reset()  # resetting for fresh env
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done and episode_steps < max_steps:
            # Exploit the learned Q-table policy
            action = np.argmax(Q[state, :])
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            episode_steps += 1
            print(env.render())
            time.sleep(0.5)  # Slow down rendering for visibility

        test_rewards.append(episode_reward)
        test_steps.append(episode_steps)

    # Output average test performance
    print("Average Reward:", np.mean(test_rewards))
    print("Average number of actions:", np.mean(test_steps))

# Run the manual exploration first
manual_exploration()

# After manual exploration, run the Q-learning algorithm
q_learning_taxi()

