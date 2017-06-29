import numpy as np
import random

from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, env, q_networks,
            minibatch_size_limit=32,
            gamma=0.99,
            initial_exploration=1.0,
            final_exploration=0.1,
            final_exploration_update_step=1000000,
            target_update_step=10000,
            history_size=1,
            replay_buffuer_size=1000000):
        self.env = env
        self.new_episode()

        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.shape[0]

        self.q_networks = q_networks

        self.minibatch_size_limit = minibatch_size_limit
        self.gamma = gamma

        self.replay_buffer = ReplayBuffer(replay_buffuer_size)

        self.target_update_step = target_update_step
        self.step = 0

        self.history = history_size

        self.phi_t = np.zeros((1, self.n_states * self.history)). \
                                astype(np.float32)

        self.epsilon = initial_exploration
        self.final_exploration = final_exploration
        self.epsilon_step = (initial_exploration - final_exploration) \
                            / final_exploration_update_step

    def act(self):
        a_t = np.argmax(self.q_networks.perform_q(self.phi_t))

        s_t_1, r_t, terminal, _ = self.env.step(a_t)
        phi_t_1 = np.hstack((
            self.phi_t[:, self.n_states:],
            s_t_1.astype(np.float32).reshape((1,-1))
        ))
        self.phi_t = phi_t_1

        return a_t, s_t_1, r_t, terminal

    def act_and_train(self):
        if random.random() <= self.epsilon:
            a_t = random.randint(0, self.n_actions-1)
        else:
            a_t = np.argmax(self.q_networks.perform_q(self.phi_t))
        if self.epsilon > self.final_exploration:
            self.epsilon -= self.epsilon_step

        s_t_1, r_t, terminal, _ = self.env.step(a_t)
        phi_t_1 = np.hstack((
            self.phi_t[:, self.n_states:],
            s_t_1.astype(np.float32).reshape((1,-1))
        ))

        self.replay_buffer.append([self.phi_t, a_t, r_t, phi_t_1, terminal])
        self.phi_t = phi_t_1

        y = np.zeros((0, self.n_actions))
        phi = np.zeros((0, self.n_states * self.history))

        minibatch = self.replay_buffer.sample(self.minibatch_size_limit)
        for phi_j, a_j, r_j, phi_j_1, terminal_j in minibatch:
            y_j = self.q_networks.perform_q(phi_j)[0]
            if terminal_j:
                y_j[a_j] = r_j
            else:
                # DDQN
                a = np.argmax(self.q_networks.perform_q(phi_j_1))
                y_j[a_j] = r_j + self.gamma * self.q_networks.perform_q_hat(phi_j_1)[0,a]
                # DQN
                #y_j[a_j] = r_j + self.gamma * np.max(self.q_networks.perform_q_hat(phi_j_1))
            y = np.vstack((y, y_j))
            phi = np.vstack((phi, phi_j))

        self.q_networks.train_q(np.array(phi, dtype=np.float32),
                                        np.array(y, dtype=np.float32))

        if self.step % self.target_update_step == 0:
             self.q_networks.update_q_hat()
        self.step += 1

        return a_t, s_t_1, r_t, terminal

    def new_episode(self):
        self.env.reset()
        #castomized ---env
        self.highest=-999
