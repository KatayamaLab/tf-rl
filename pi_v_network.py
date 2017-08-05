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

        h1 = layer(self.x, phi_dim, 50, 'Hidden1', act=tf.nn.relu)
        h2 = layer(self.x, phi_dim, 50, 'Hidden1', act=tf.nn.relu)
        #h2 = layer(h1, 200, 200, 'Hidden2', act=tf.nn.relu)

        self.mu = layer(h1, 50, a_dim, 'mu', act=tf.identity)
        self.sigma = layer(h1, 50, a_dim, 'sigma', act=tf.nn.softplus)
        self.V = layer(h2, 50, 1, 'v', act=tf.identity)


        self.loss_pi = tf.reduce_sum(tf.log(
                1/ tf.sqrt( 2.0 * 3.1415926535 * self.sigma ) *
                tf.exp(-tf.square(self.a - self.mu) / (2 * self.sigma) )
            ) * (self.R - tf.stop_gradient(self.V)))
        self.entropy = - tf.reduce_sum(tf.log(2.0 * 3.1415926535 * self.sigma) + 1.0)

        self.loss_V= tf.reduce_sum(tf.square(self.R - self.V))


        # TODO need to change to RMSProp
        self.optimizer = tf.train.AdamOptimizer(0.00001)
        # self.optimizer = tf.train.RMSPropOptimizer(
        #                 learning_rate=0.00001,
        #                 decay=0.99,
        #                 momentum=0.0,
        #                 epsilon=0.1)

        #self.network_params = tf.trainable_variables()

        self.grad_vals = self.optimizer.compute_gradients(
            -self.loss_pi - 0.001 * self.entropy + 0.5 * self.loss_V
        )
        self.grads=[]
        self.vals=[]
        for grad,val in self.grad_vals:
            if grad is not None:
                self.grads.append(grad)
                self.vals.append(val)
        self.grad_vals_ = [(vals, grads) for vals, grads in zip(self.grads, self.vals)]

        #self.train = self.optimizer.apply_gradients(self.grad_vals_)
        self.train = self.optimizer.minimize(-self.loss_pi  - 0.001 * self.entropy + 0.5 * self.loss_V)
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
        sess.run(self.train,
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
