import sys
import argparse

import numpy as np
import tensorflow as tf
import gym

from q_network import QNetwork
from dqnagent import DQNAgent

from pi_v_network import PiVNetwork
from a3cagent import A3CAgent

import config

# my original environment
import swingupcartpole

def main(_):
    flags = tf.app.flags.FLAGS

    # set up an environment
    env = gym.make(config.environment)

    if config.record:
        env = gym.wrappers.Monitor(env, config.record_path, force=True)

    # set up an agent
    if config.continuous_action:
        agent = A3CAgent(env,
                PiVNetwork,
                history_length=1,
                discount_factor=0.99,
                log_dir=config.log_dir)
    else:
        agent = DQNAgent(env,
                QNetwork,
                minibatch_size_limit=32,
                replay_memory_size=3000000,
                history_length=4,
                # replay_memory_size=1000000,
                # history_length=1,
                target_update_step=200,
                discount_factor=0.99,
                learning_rate=0.0025,
                initial_exploration=1.0,
                final_exploration=0.01,
                final_exploration_frame=100000,
                replay_start_size=1000,
                #final_exploration_frame=10000,
                #replay_start_size=100,
                log_dir=config.log_dir)

    print('Observation space: ', env.observation_space,
            'Action space: ', env.action_space)

    if config.restore_model_path:
        agent.restore_variables(config.restore_model_path)

    total_frames = 0

    # training
    for episode in range(1, config.max_episodes+1):
        terminal = False
        total_reward = 0
        frames = 0

        while not terminal:
            if config.train:
                a, s, r_t, terminal, info = agent.act_and_train()
            else:
                a, s, r_t, terminal, info = agent.act()
            if config.render:
                env.render()
            #print(s,r_t)
            total_reward += r_t
            frames += 1
            total_frames += 1

        if episode % config.interval_to_save_model == 0:
            agent.save_variables(episode, config.save_model_dir)

        #print('Episode: ', episode, ' Frames: ', total_frames, ' R: ', total_reward, ' Epsilon: ', info['epsilon'])
        print('Episode: ', episode, ' Frames: ', total_frames, ' R: ', total_reward)
        agent.write_summary(episode, total_reward)

if __name__ == '__main__':
    tf.app.run(main=main)
