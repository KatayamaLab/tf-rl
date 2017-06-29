import numpy as np
import tensorflow as tf

class QNetworks:
    def __init__(self, logdir, NNClass, n_obs, n_action, learning_rate=0.0005):
        self.sess = tf.Session()

        self.q = NNClass("q_orig", n_obs, n_action, learning_rate)
        self.q_hat = NNClass("q_hat", n_obs, n_action)

        self.total_reward = tf.placeholder(tf.float32)
        tf.summary.scalar("TotalReward", self.total_reward)

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=0)
        self.summary = tf.summary.merge_all()

        if tf.gfile.Exists(logdir):
            tf.gfile.DeleteRecursively(logdir)
        tf.gfile.MakeDirs(logdir)

        self.log_writer = tf.summary.FileWriter(logdir, self.sess.graph, flush_secs=20)

    def write_summary(self, episode, total_reward):
        summary = self.sess.run(self.summary, feed_dict={self.total_reward: np.array(total_reward)})
        self.log_writer.add_summary(summary, episode)

    def save_variables(self, step, model_path=None):
        if model_path:
            self.saver.save(self.sess, model_path, global_step=step)
            print('save model to '+model_path)

    def restore_variables(self, model_path=None):
        if model_path:
            self.saver.restore(self.sess, model_path)
            print('Restore model from '+model_path)

    def perform_q(self, x):
        return self.q(self.sess, x)

    def perform_q_hat(self, x):
        return self.q_hat(self.sess, x)

    def train_q(self, x, t):
        self.q.train(self.sess, x, t)

    def update_q_hat(self):
        self.q_hat.set_variables(self.sess, self.q.read_variables(self.sess))

    def __del__(self):
        self.log_writer.close()
