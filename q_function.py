import numpy as np
import tensorflow as tf

class MySession:
    def __init__(self,logdir):
        self.sess = tf.Session()
        self.total_reward = tf.placeholder(tf.float32)
        tf.summary.scalar("TotalReward", self.total_reward)
        self.logdir= logdir

    def initialize_variables(self):
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=0)
        self.log_writer = tf.summary.FileWriter(self.logdir, self.sess.graph, flush_secs=20)
        self.summary = tf.summary.merge_all()

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

    def __call__(self):
        return self.sess

    def __del__(self):
        self.log_writer.close()


class QFunction:
    def __init__(self, qname, sess, phi_dim, a_dim, learning_rate=0.00025):
        def layer(input_tensor, input_dim, output_dim, name='layer', act=tf.nn.relu):
            with tf.name_scope(qname+name):
                w = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=1))
                self.variable_list.append(w)
                b = tf.Variable(tf.truncated_normal([output_dim], stddev=1))
                self.variable_list.append(b)
                h = act(tf.nn.xw_plus_b(input_tensor, w, b))
                tf.summary.histogram('weight', w)
                tf.summary.histogram('bias', b)
            return h

        self.variable_list = []

        self.x = tf.placeholder(tf.float32, [None, phi_dim])
        h1 = layer(self.x, phi_dim, 32, 'Hidden1')
        h2 = layer(h1, 32, 32, 'Hidden2')
        self.y = layer(h2, 32, a_dim, 'Output', tf.identity)
        self.t = tf.placeholder(tf.float32, [None, a_dim])
        loss = tf.reduce_mean(tf.square(self.t - self.y))
        self.train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
        self.sess = sess

    def __call__(self, x):
        return self.sess.run(self.y, feed_dict={self.x: x})

    def train(self, x, t):
        self.sess.run(self.train_step,
            feed_dict={self.x: x, self.t: t})

    def read_variables(self):
        with self.sess.as_default():
            values = [valiable.eval() for valiable in self.variable_list]
        return values

    def set_variables(self, values):
        with self.sess.as_default():
            for valiable, value in zip(self.variable_list, values):
                valiable.load(value)
