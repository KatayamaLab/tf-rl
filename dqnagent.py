import random
import os.path
import numpy as np
import tensorflow as tf

from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(
            self, env,
            QNetworkClass,
            minibatch_size_limit=32,
            replay_memory_size=1000000,
            history_length=4,
            target_update_step=10000,
            discount_factor=0.99,
            learning_rate=0.00025,
            initial_exploration=1.0,
            final_exploration=0.1,
            final_exploration_frame=1000000,
            replay_start_size=50000,
            log_dir=None):
        self.env = env
        self.new_episode()

        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.shape[0]
        self.n_input = self.n_states * history_length

        #setup tensorflow
        self.sess = tf.Session()

        self.q = QNetworkClass("q_orig", self.n_input, self.n_actions, learning_rate)
        self.q_hat = QNetworkClass("q_hat", self.n_input, self.n_actions)

        self.total_reward = tf.placeholder(tf.float32)
        tf.summary.scalar("TotalReward", self.total_reward)

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=0)
        self.summary = tf.summary.merge_all()

        if tf.gfile.Exists(log_dir):
            tf.gfile.DeleteRecursively(log_dir)
        tf.gfile.MakeDirs(log_dir)

        if log_dir:
            self.log_writer = tf.summary.FileWriter(log_dir, self.sess.graph, flush_secs=20)
        else:
            self.log_writer = None
        #store parameter
        self.minibatch_size_limit = minibatch_size_limit
        self.gamma = discount_factor

        self.replay_buffer = ReplayBuffer(replay_memory_size)

        self.target_update_step = target_update_step
        self.step = 0

        self.phi_t = np.zeros((1, self.n_input)). \
                                astype(np.float32)

        self.epsilon = initial_exploration
        self.replay_start_size = replay_start_size
        self.final_exploration = final_exploration
        self.epsilon_step = (initial_exploration - final_exploration) \
                            / final_exploration_frame

    def act(self):
        a_t = np.argmax(self._perform_q(self.phi_t))

        s_t_1, r_t, terminal, _ = self.env.step(a_t)
        phi_t_1 = np.hstack((
            self.phi_t[:, self.n_states:],
            s_t_1.astype(np.float32).reshape((1,-1))
        ))
        self.phi_t = phi_t_1

        return a_t, s_t_1, r_t, terminal, {'epsilon':self.epsilon}

    def act_and_train(self):
        # With probability epsilon select a random action
        # Otherwise select acionn from Q network
        if random.random() <= self.epsilon:
            a_t = random.randint(0, self.n_actions-1)
        else:
            a_t = np.argmax(self._perform_q(self.phi_t))

        # Execute action in emulator and observe reward and state
        s_t_1, r_t, terminal, _ = self.env.step(a_t)
        phi_t_1 = np.hstack((
            self.phi_t[:, self.n_states:],
            s_t_1.astype(np.float32).reshape((1,-1))
        ))

        # Store transition
        self.replay_buffer.append([self.phi_t, a_t, r_t, phi_t_1, terminal])
        self.phi_t = phi_t_1

        # After specified steps start experienced replay to update Q network
        if self.step >= self.replay_start_size:
            # sample minibatch
            y = np.zeros((0, self.n_actions))
            phi = np.zeros((0, self.n_input))
            minibatch = self.replay_buffer.sample(self.minibatch_size_limit)

            for phi_j, a_j, r_j, phi_j_1, terminal_j in minibatch:
                y_j = self._perform_q(phi_j)[0]
                if terminal_j:
                    y_j[a_j] = r_j
                else:
                    # DDQN
                    a = np.argmax(self._perform_q(phi_j_1))
                    y_j[a_j] = r_j + self.gamma * self._perform_q_hat(phi_j_1)[0,a]
                    # DQN
                    #y_j[a_j] = r_j + self.gamma * np.max(self._perform_q_hat(phi_j_1))
                y = np.vstack((y, y_j))
                phi = np.vstack((phi, phi_j))

            # Update Q network #TODO comversion to numpy array should be done in q network class
            self._train_q(np.array(phi, dtype=np.float32), np.array(y, dtype=np.float32))

            # Update target Q network every specific steps
            if self.step % self.target_update_step == 0:
                 self._update_q_hat()

            # Update Exploration ratio
            if self.epsilon > self.final_exploration:
                self.epsilon -= self.epsilon_step

        self.step += 1

        return a_t, s_t_1, r_t, terminal, {'epsilon':self.epsilon}

    def new_episode(self):
        self.env.reset()

    def write_summary(self, episode, total_reward):
        summary = self.sess.run(self.summary, feed_dict={self.total_reward: np.array(total_reward)})
        self.log_writer.add_summary(summary, episode)

    def save_variables(self, step, model_dir=None):
        if model_dir:
            if not tf.gfile.Exists(model_dir):
                tf.gfile.MakeDirs(model_dir)
            full_path = os.path.join(model_dir, 'model')
            self.saver.save(self.sess, full_path, global_step=step)
            print('save model to ' + full_path)

    def restore_variables(self, model_path=None):
        if model_path:
            self.saver.restore(self.sess, model_path)
            print('Restore model from ' + model_path)

    def _perform_q(self, x):
        return self.q(self.sess, x)

    def _perform_q_hat(self, x):
        return self.q_hat(self.sess, x)

    def _train_q(self, x, t):
        self.q.train(self.sess, x, t)

    def _update_q_hat(self):
        self.q_hat.set_variables(self.sess, self.q.read_variables(self.sess))

    def __del__(self):
        if self.log_writer:
            self.log_writer.close()
