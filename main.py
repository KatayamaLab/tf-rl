import sys
import argparse

import numpy as np
import tensorflow as tf
import gym

from q_network import QNetwork
from dqnagent import DQNAgent

FLAGS = None

def main(_):
    # set up an environment
    env = gym.make('CartPole-v1')
    # env = gym.make('Acrobot-v1')
    # env = gym.make('MountainCar-v0')
    obs = env.reset()
    print('Observation space: ', env.observation_space,
            'Action space: ', env.action_space,
            'Initial observation: ', obs)

    # set up an agent
    agent = DQNAgent(env,
                QNetwork,
                minibatch_size_limit=32,
                discount_factor=0.99,
                history_size=4,
                learning_rate=0.0005,
                target_update_step=100,
                initial_exploration=1.0,
                final_exploration=0.1,
                final_exploration_frame=10000,
                replay_start_size=500,
                replay_memory_size=10000,
                log_dir=FLAGS.log_dir)

    if FLAGS.restore_model_path:
        agent.restore_variables(FLAGS.restore_model_path)

    # training
    for episode in range(1, FLAGS.max_episodes+1):
        terminal = False
        agent.new_episode()
        total_reward = 0
        time = 0

        if FLAGS.no_train:
            while not terminal:
                a, s, r_t, terminal = agent.act()
                env.render()
                total_reward += r_t
                time += 1
        else:
            while not terminal:
                a, s, r_t, terminal = agent.act_and_train()
                env.render()
                total_reward += r_t
                time += 1

            if episode % 100 == 0:
                agent.save_variables(episode, FLAGS.save_model_dir)

        print('#', episode, 'R: ', total_reward)
        agent.write_summary(episode, total_reward)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_episodes', type=int, default=10000,
                      help='Number of steps to run trainer.')
    parser.add_argument('--log_dir', type=str, default='/tmp/tf-rl/log',
                      help='Summaries log directory')
    parser.add_argument('--save_model_dir', type=str, default='/tmp/tf-rl/model/',
                      help='Model path for save')
    parser.add_argument('--restore_model_path', type=str, default='',
                      help='Model path for restore')
    parser.add_argument('--no_train', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, no training mode.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
