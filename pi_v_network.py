import tensorflow as tf
import numpy as np

class PiVNetwork:
    def __init__(self, qname, phi_dim, a_dim, learning_rate=1.0e-5):
        def layer(input_tensor, input_dim, output_dim, name='layer', act=tf.identity):
            with tf.name_scope(qname+name):
                w = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.01))
                self.variable_list.append(w)
                b = tf.Variable(tf.truncated_normal([output_dim]))
                self.variable_list.append(b)
                h = act(tf.nn.xw_plus_b(input_tensor, w, b))
                tf.summary.histogram('weight', w)
                tf.summary.histogram('bias', b)
            return h

        self.variable_list = []

        self.x = tf.placeholder(tf.float32, [None,phi_dim])
        self.R = tf.placeholder(tf.float32, [None,1])
        self.a = tf.placeholder(tf.float32, [None,a_dim])

        self.phi_dim = phi_dim
        self.a_dim = a_dim

        h1 = layer(self.x, phi_dim, 200, 'Hidden1', act=tf.nn.relu)
        h2 = layer(h1, 200, 200, 'Hidden2', act=tf.nn.relu)

        self.mu = layer(h1, 200, a_dim, 'mu', act=tf.identity)
        self.sigma = layer(h1, 200, a_dim, 'sigma', act=tf.nn.softplus)
        self.V = layer(h1, 200, 1, 'v', act=tf.identity)

        #log and exp is canceled
        self.loss_pi = (-tf.square(self.a - self.mu) / (self.sigma) - tf.sqrt( 2.0 * 3.14159265 * self.sigma)) * (self.R - self.V)
        self.loss_V= tf.square(self.R - self.V)

        # TODO output gradients to copy to local gradients
        #self.global_pi_grad = tf.placeholder(tf.float32, [None, phi_dim])
        #self.global_v_grad = tf.placeholder(tf.float32, [None, phi_dim])

        # TODO need to change to RMSProp
        self.optimizer = tf.train.AdamOptimizer(0.0001)
        self.train_pi = self.optimizer.minimize(self.loss_pi)
        self.train_V = self.optimizer.minimize(self.loss_V)

        # TODO Make it ASyn
        #self.local_pi_grad = self.optimizer.compute_gradients(self.pi_loss)
        #self.local_v_grad = self.optimizer.compute_gradients(self.v_loss)

        #self.train_step = self.optimizer.apply_gradients(???e). \
        #    apply_gradients(grads_and_vars=(,))

        #self.apply_gradients = [self.optimizer.apply_gradients(pi_gradient),
        #                        self.op`
        #self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # def get_pi_gradient(self, s, a, r):
    #     return self.optimizer.compute_gradients(self.pi_loss)
    #     #return sess.run(self.local_pi_grad, feed_dict={}) #TODO
    #
    # def get_v_gradient(self, s, a, r):
    #     return sess.run(self.local_v_grad, feed_dict={}) #TODO
    #
    # def apply_gradient(self, pi_gradient, v_gradient):

    def update(self, sess, phi, a, R):
        # sess.run(self.train,
        sess.run([self.train_pi, self.train_V],
            feed_dict={
                self.x: np.array(phi, dtype=np.float32).reshape(-1, self.phi_dim),
                self.a: np.array(a, dtype=np.float32).reshape(-1, self.a_dim),
                self.R: np.array(R, dtype=np.float32).reshape(-1,1)
            })

    def predict_pi_and_V(self, sess, x):
        mu, sigma, V = sess.run([self.mu, self.sigma, self.V], feed_dict={self.x: x})
        return mu[0], sigma[0], V[0,0]

    def predict_pi(self, sess, x):
        mu, sigma = sess.run([self.mu, self.sigma], feed_dict={self.x: x})
        return mu[0], sigma[0]

    def predict_V(self, sess, x):
        V = sess.run(self.V, feed_dict={self.x: x})
        return V[0,0]

    def train(self, sess, pi_grad, v_grad):
        sess.run(self.train_step,
            feed_dict={self.pi_grad: pi_grad, self.v_grad: v_grad})

    def read_variables(self, sess):
        with sess.as_default():
            values = [valiable.eval() for valiable in self.variable_list]
        return values

    def set_variables(self, sess, values):
        with sess.as_default():
            for valiable, value in zip(self.variable_list, values):
                valiable.load(value)
