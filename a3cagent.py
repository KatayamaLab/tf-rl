import random
import os.path
import numpy as np
import tensorflow as tf

class A3CAgent:
    def __init__(
            self, env,
            PiVNetworkClass,
            history_length=4,
            discount_factor=0.99,
            learning_rate=0.00025,
            log_dir=None):
        self.env = env
        self.new_episode()

        self.n_actions = env.action_space.shape[0] # TODO assume 1D acction now
        self.n_states = env.observation_space.shape[0] # TODO assume 1D state now
        self.n_input = self.n_states * history_length

        self.a_min, self.a_max = env.action_space.low, env.action_space.high

        #setup tensorflow
        self.sess = tf.Session()

        self.pi_v_network = PiVNetworkClass("q_orig", self.n_input, self.n_actions, learning_rate)

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

        #  parameters
        self.gamma = discount_factor

        self.step = 0

        self.phi = []
        self.a = []
        self.r = []

        self.phi_t = np.zeros((1, self.n_input), dtype=np.float32)

        #TODO inpremented as parameter
        self.t_max = 5

        self.t = 0
        self.T = 0
        self.t_start=0

    def act(self):
        a_t = np.argmax(self._perform_q(self.phi_t))

        s_t_1, r_t, terminal, _ = self.env.step(a_t)
        phi_t_1 = np.hstack((
            self.phi_t[:, self.n_states:],
            s_t_1.astype(np.float32).reshape((1,-1))
        ))
        self.phi_t = phi_t_1

        return a_t, s_t_1, r_t, terminal, {}

    def act_and_train(self):
        # Perform action according to policy
        mu, sigma = self.pi_v_network.predict_pi(self.sess, self.phi_t)
        #print(mu,sigma)

        self.phi.append(self.phi_t)

        a_t=np.random.normal(mu, np.sqrt(sigma))

        self.a.append(a_t)

        # Execute action in emulator and observe reward and state
        s_t_1, r_t, terminal, _ = self.env.step(np.clip(a_t, self.a_min, self.a_max))

        phi_t_1 = np.hstack((
            self.phi_t[:, self.n_states:],
            s_t_1.astype(np.float32).reshape((1,-1))
        ))
        self.phi_t = phi_t_1

        self.r.append(r_t)

        self.t += 1
        self.T += 1

        if terminal or self.t-self.t_start >= self.t_max:
            # print(mu, sigma,phi_t_1)
            if terminal: # for terminal
                R_t = 0.0
            else: # for non-terminal s_t// Bootstrap from last state
                R_t = self.pi_v_network.predict_V(self.sess, self.phi_t)

            R = np.zeros(self.t-self.t_start);
            for i in reversed(range(0, self.t-self.t_start)): # i: t-1...t_start
                R_t = self.r[i] + self.gamma * R_t
                R[i] = R_t
            self.pi_v_network.update(self.sess, self.phi, self.a, R)

                #Accumulate gradient wrt theta'
                #Accumulate gradient wrt theta

            # TODO Perform asynchronous updates

            # TODO reset
            self.phi = []
            self.a = []
            self.r = []
            self.t_start = self.t
            # reset gradient

        return a_t, s_t_1, r_t, terminal, {}

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
