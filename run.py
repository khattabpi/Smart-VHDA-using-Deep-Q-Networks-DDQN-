import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import VHDAEnv
from agent import DDQNAgent
import matplotlib.pyplot as plt
import numpy as np

def main(episodes=500):

    env = VHDAEnv(max_steps=500)
    agent = DDQNAgent(state_size=8, action_size=2)

    episode_rewards = []
    avg_qos = []
    unho_list = []

    first_episode_stats = None

    print("Training started...")

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward

        stats = env.get_qos_stats()
        episode_rewards.append(total_reward)
        avg_qos.append(stats["Average QoS"])
        unho_list.append(stats["Unnecessary Handovers"])

        if ep == 0:
            first_episode_stats = (total_reward, stats["Average QoS"], stats["Unnecessary Handovers"])

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{episodes} | "
                  f"Reward: {total_reward:.3f} | "
                  f"Avg QoS: {stats['Average QoS']:.3f} | "
                  f"Unnecessary HO: {stats['Unnecessary Handovers']} | "
                  f"Epsilon: {agent.epsilon:.3f}")

        agent.decay_epsilon()

    last_stats = (episode_rewards[-1], avg_qos[-1], unho_list[-1])

    print("\n====== FIRST vs LAST EPISODE ======")
    print(f"Reward:         {first_episode_stats[0]:.3f}  ->  {last_stats[0]:.3f}")
    print(f"Avg QoS:        {first_episode_stats[1]:.3f}  ->  {last_stats[1]:.3f}")
    print(f"Unnecessary HO: {first_episode_stats[2]}  ->  {last_stats[2]}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    axes[0].plot(episode_rewards)
    axes[0].set_title("Total Reward per Episode (DDQN)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].grid(True)

    axes[1].plot(avg_qos, color='green')
    axes[1].set_title("Average QoS per Episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Avg QoS")
    axes[1].grid(True)

    axes[2].plot(unho_list, color='red')
    axes[2].set_title("Unnecessary Handovers per Episode")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Count")
    axes[2].grid(True)

    plt.tight_layout()
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Results')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "ddqn_results.png"))
    print("\nResults saved in Results/ddqn_results.png")
    plt.show()

main()