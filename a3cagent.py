import random
import os.path
import numpy as np
import tensorflow as tf

from copy import deepcopy

class A3CAgent:
    def __init__(
            self, env,
            PiVNetworkClass,
            history_length=1,
            discount_factor=0.99,
            learning_rate=0.00025,
            log_dir=None):
        #self.env = env
        self.envs = []
        self.envs.append(env)
        for i in range(29):
            self.envs.append(deepcopy(self.envs[0]))
            self.envs[i].seed(i)

        self.new_episode()
        self.n_actions = env.action_space.shape[0] # TODO assume 1D acction now
        self.n_states = env.observation_space.shape[0] # TODO assume 1D state now
        self.n_input = self.n_states * history_length

        self.a_min, self.a_max = env.action_space.low, env.action_space.high

        self.log_writer = None

        #setup tensorflow
        self.sess = tf.Session()

        self.pi_v_network = PiVNetworkClass(self.n_input, self.n_actions, learning_rate)

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

        #  parameters
        self.gamma = discount_factor

        self.step = 0

        self.phi_t = [np.zeros((self.n_input), dtype=np.float32) for i in range(len(self.envs))]
        self.phi = [[] for i in range(len(self.envs))]
        self.a = [[] for i in range(len(self.envs))]
        self.r = [[] for i in range(len(self.envs))]
        self.R = [[] for i in range(len(self.envs))]

        #TODO inpremented as parameter
        self.t_max = 5
        self.t = [0 for i in range(len(self.envs))]
        self.t_start=[0 for i in range(len(self.envs))]
        self.T = 0

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
        for (j,env) in enumerate(self.envs):
            # Perform action according to policy
            mu, sigma = self.pi_v_network.predict_pi(self.sess, self.phi_t[j])

            a_t=np.clip(np.random.normal(mu, np.sqrt(sigma)), self.a_min, self.a_max)

            # Execute action in emulator and observe reward and state
            s_t_1, r_t, terminal, _ = self.envs[j].step(a_t)
            #self.envs[j].render()
            self.phi[j].append(self.phi_t[j])
            self.a[j].append(a_t)
            self.r[j].append(r_t)

            self.phi_t[j] = np.hstack((
                self.phi_t[j][self.n_states:],
                s_t_1.astype(np.float32)
            )).reshape((-1))

            self.t[j] += 1
            self.T += 1

            if terminal or self.t[j]-self.t_start[j] >= self.t_max:
                if terminal: # for terminal
                    R_t = 0.0
                else: # for non-terminal s_t// Bootstrap from last state
                    R_t = self.pi_v_network.predict_V(self.sess, self.phi_t[j])

                for i in reversed(range(0, self.t[j]-self.t_start[j])): # i: t-1...t_start
                    R_t = self.r[j][i] + self.gamma * R_t
                    self.R[j].append([R_t])
                #print(self.phi[i], mu, sigma)

                if j==len(self.envs)-1:

                    self.pi_v_network.update(self.sess,
                        np.array(self.phi).reshape(-1,len(self.phi[0][0])),
                        np.array(self.a).reshape(-1,len(self.a[0][0])),
                        np.array(self.R).reshape(-1,1)
                    )
                    # TODO Perform asynchronous updates
                    self.phi = [[] for i in range(len(self.envs))]
                    self.a = [[] for i in range(len(self.envs))]
                    self.r = [[] for i in range(len(self.envs))]
                    self.R = [[] for i in range(len(self.envs))]
                self.t_start[j] = self.t[j]

        return a_t, s_t_1, r_t, terminal, {}

    def new_episode(self):
        for env in self.envs:
            env.reset()

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
