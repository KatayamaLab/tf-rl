import sys
import argparse

import numpy as np
import tensorflow as tf
import gym

from q_network import QNetwork
from dqnagent import DQNAgent
import parameters as P


def main(_):
    flags = tf.app.flags.FLAGS

    # set up an environment
    env = gym.make('LunarLander-v2')
    #env = gym.make('CartPole-v1')
    # env = gym.make('Acrobot-v1')
    # env = gym.make('MountainCar-v0')
    if P.record:
        env = gym.wrappers.Monitor(env, P.record_path, force=True)

    # set up an agent
    agent = DQNAgent(env,
                QNetwork,
                minibatch_size_limit=32,
                replay_memory_size=1000000,
                history_length=1,
                target_update_step=200,
                discount_factor=0.99,
                learning_rate=0.0025,
                initial_exploration=1.0,
                final_exploration=0.01,
                final_exploration_frame=10000,
                replay_start_size=100,
                log_dir=P.log_dir)

    print('Observation space: ', env.observation_space,
            'Action space: ', env.action_space)

    if P.restore_model_path:
        agent.restore_variables(P.restore_model_path)

    total_frames = 0

    # training
    for episode in range(1, P.max_episodes+1):
        terminal = False
        total_reward = 0
        frames = 0

        while not terminal:
            if P.train:
                a, s, r_t, terminal, info = agent.act_and_train()
            else:
                a, s, r_t, terminal, info = agent.act()
            if P.render:
                env.render()
            total_reward += r_t
            frames += 1
            total_frames += 1

        if episode % P.interval_to_save_model == 0:
            agent.save_variables(episode, P.save_model_dir)

        print('Episode: ', episode, ' Frames: ', total_frames, ' R: ', total_reward, ' Epsilon: ', info['epsilon'])
        agent.write_summary(episode, total_reward)

        agent.new_episode()

if __name__ == '__main__':
    tf.app.run(main=main)
