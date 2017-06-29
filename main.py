import sys
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Open AI gym
import gym

from q_function import QFunction
from q_function import MySession
from dqnagent import DQNAgent

FLAGS = None

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    # set up an environment
    env = gym.make('CartPole-v1')
    # env = gym.make('Acrobot-v1')
    #env = gym.make('MountainCar-v0')

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    obs = env.reset()
    print('Observation space: ', env.observation_space,
            'Action space: ', env.action_space,
            'Initial observation: ', obs)

    history_size = 4

    sess = MySession(FLAGS.log_dir + '/log')
    q_function = QFunction("q_orig", sess(),
                obs_size * history_size, n_actions,
                learning_rate=0.0005)
    q_hat_function = QFunction("q_hat", sess(),
                obs_size * history_size, n_actions)
    sess.initialize_variables()

    if FLAGS.restore_model_path:
        sess.restore_variables(FLAGS.restore_model_path)

    # set up an agent
    agent = DQNAgent(env, q_function, q_hat_function,
                minibatch_size_limit=32,
                gamma=0.99,
                initial_exploration=0.3 ,
                final_exploration=0.01,
                final_exploration_update_step=50000,
                target_update_step=200,
                history_size=history_size,
                replay_buffuer_size=1000000)

    # training
    for episode in range(1,FLAGS.max_steps+1):
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

        if episode % 500 == 0:
            sess.save_variables(episode, FLAGS.save_model_path)

        print('#', episode, 'R: ', total_reward)
        sess.write_summary(episode, total_reward)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
    parser.add_argument('--max_steps', type=int, default=5000,
                      help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
    parser.add_argument('--log_dir', type=str, default='/tmp/tf-rl',
                      help='Summaries log directory')
    parser.add_argument('--restore_model_path', type=str, default='',
                      help='Model path for restore')
    parser.add_argument('--save_model_path', type=str, default='./model/model.ckpt',
                      help='Model path for save')
    parser.add_argument('--no_train', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, no training mode.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
