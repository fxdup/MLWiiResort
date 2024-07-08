from WiiSportsGym.HomeRunEnv.HomeRunEnv import HomeRunEnv
from RandomAgent import RandomAgent
num_episodes = 10


def main():
    env = HomeRunEnv()
    agent = RandomAgent(env.action_space)
    rewards = []

    for _ in range(num_episodes):
        env.reset()
        print('Episode start')
        episode_reward = 0
        while True:
            action = agent.act(env.get_observation())
            reward, done = env.step(action)  # random action
            episode_reward += reward
            if done:
                print('Reward: %d' % episode_reward)
                rewards.append(episode_reward)
                break
    print('Average reward: %.2f' % (sum(rewards) / len(rewards)))


if __name__ == '__main__':
    main()