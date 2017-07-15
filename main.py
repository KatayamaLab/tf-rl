import sys
import argparse

import numpy as np
import tensorflow as tf
import gym

from q_network import QNetwork
from dqnagent import DQNAgent

tf.app.flags.DEFINE_integer('max_episodes', 10000, 'Number of episodes (10000).')
tf.app.flags.DEFINE_string('log_dir', '/tmp/tf-rl/log', 'Summaries log directory (/tmp/tf-rl/log)')
tf.app.flags.DEFINE_string('save_model_dir', '/tmp/tf-rl/model/', 'Model path for save (/tmp/tf-rl/model/)')
tf.app.flags.DEFINE_integer('interval_to_save_model', 500, 'Interval to save model (500).')
tf.app.flags.DEFINE_string('restore_model_path', '', 'Model path for restore')
tf.app.flags.DEFINE_boolean('train', True, 'Training mode (default: true).')

def main(_):
    flags = tf.app.flags.FLAGS

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
                history_length=4,
                learning_rate=0.00025,
                target_update_step=100,
                initial_exploration=1.0,
                final_exploration=0.1,
                final_exploration_frame=100000,
                replay_start_size=5000,
                replay_memory_size=100000,
                log_dir=flags.log_dir)

    if flags.restore_model_path:
        agent.restore_variables(flags.restore_model_path)

    total_frames = 0

    # training
    for episode in range(1, flags.max_episodes+1):
        terminal = False
        agent.new_episode()
        total_reward = 0
        frames = 0

        while not terminal:
            if flags.train:
                a, s, r_t, terminal = agent.act_and_train()
            else:
                a, s, r_t, terminal = agent.act()
            env.render()
            total_reward += r_t
            frames += 1
            total_frames += 1

        if episode % flags.interval_to_save_model == 0:
            agent.save_variables(episode, flags.save_model_dir)

        print('Episode: ', episode, ' Frames: ', total_frames, ' R: ', total_reward)
        agent.write_summary(episode, total_reward)

if __name__ == '__main__':
    tf.app.run(main=main)
