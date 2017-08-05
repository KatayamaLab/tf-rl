import tensorflow as tf
import numpy as np

class PiVNetwork:
    def __init__(self, phi_dim, a_dim, learning_rate=1.0e-5):
        def layer(input_tensor, input_dim, output_dim, name='layer', act=tf.identity):
            with tf.name_scope(name):
                d = 1.0 / np.sqrt(input_dim)
                #w = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
                w = tf.Variable(tf.random_uniform([input_dim, output_dim], -d, d))
                self.variable_list.append(w)
                # b = tf.Variable(tf.truncated_normal([output_dim], stddev=0.1))
                b = tf.Variable(tf.random_uniform([output_dim], -d, d))
                self.variable_list.append(b)
                h = act(tf.nn.xw_plus_b(input_tensor, w, b))
                tf.summary.histogram('weight', w)
                tf.summary.histogram('bias', b)
            return h

        self.variable_list = []

        self.x = tf.placeholder(tf.float32, [None, phi_dim])
        self.a = tf.placeholder(tf.float32, [None, a_dim])
        self.R = tf.placeholder(tf.float32, [None, 1])

        self.phi_dim = phi_dim
        self.a_dim = a_dim

        h1 = layer(self.x, phi_dim, 100, 'Hidden1', act=tf.nn.relu)
        h2 = layer(self.x, phi_dim, 100, 'Hidden1', act=tf.nn.relu)
        #h2 = layer(h1, 200, 200, 'Hidden2', act=tf.nn.relu)

        self.mu = layer(h1, 100, a_dim, 'mu', act=tf.identity)
        self.sigma = layer(h2, 100, a_dim, 'sigma', act=tf.nn.softplus)
        self.V = layer(h2, 100, 1, 'v', act=tf.identity)


        self.loss_pi = tf.log(
                1/ tf.sqrt( 2.0 * 3.1415926535 * self.sigma ) *
                tf.exp(-tf.square(self.a - self.mu) / (2 * self.sigma) )
            ) * tf.stop_gradient(self.R - self.V)
        self.entropy = - (tf.log(2.0 * 3.1415926535 * self.sigma) + 1.0)

        self.loss_V= tf.square(self.R - self.V)


        # TODO need to change to RMSProp
        # self.optimizer = tf.train.AdamOptimizer(0.000001)
        self.optimizer = tf.train.RMSPropOptimizer(
                        learning_rate=0.00001,
                        decay=0.99,
                        momentum=0.0,
                        epsilon=0.1)

        #self.network_params = tf.trainable_variables()

        self.grad_vals_pi = self.optimizer.compute_gradients(
            tf.reduce_sum( -self.loss_pi + 0.001 * self.entropy)
        )
        self.grads_pi=[]
        self.vals_pi=[]
        for grad,val in self.grad_vals_pi:
            if grad is not None:
                self.grads_pi.append(grad)
                self.vals_pi.append(val)

        self.grad_vals_V = self.optimizer.compute_gradients(
            tf.reduce_sum(0.5 * self.loss_V)
        )
        self.grads_V=[]
        self.vals_V=[]
        for grad,val in self.grad_vals_V:
            if grad is not None:
                self.grads_V.append(grad)
                self.vals_V.append(val)

        self.grad_vals_pi_ = [(vals, grads) for vals, grads in zip(self.grads_pi, self.vals_pi)]
        self.grad_vals_V_ = [(vals, grads) for vals, grads in zip(self.grads_V, self.vals_V)]

        self.train_pi = self.optimizer.apply_gradients(self.grad_vals_pi_)
        self.train_V = self.optimizer.apply_gradients(self.grad_vals_V_)

        # TODO Make it ASyn


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
        sess.run([self.train_pi,self.train_V],
            feed_dict={
                self.x: phi,
                self.a: a,
                self.R: R
            })

    def gradients(self, sess, phi, a, R):
        return sess.run([self.grads_pi, self.grads_V],
            feed_dict={
                self.x: phi,
                self.a: a,
                self.R: R
            })

    def predict_pi_and_V(self, sess, x):
        mu, sigma, V = sess.run([self.mu, self.sigma, self.V], feed_dict={self.x: x})
        return mu[0], sigma[0], V

    def predict_pi(self, sess, phi):
        mu, sigma = sess.run([self.mu, self.sigma], feed_dict={self.x: [phi]})
        return mu[0], sigma[0]

    def predict_V(self, sess, phi):
        V = sess.run(self.V, feed_dict={self.x: [phi]})
        return V

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
