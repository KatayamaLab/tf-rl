import tensorflow as tf
import numpy as np

class PiVNetwork:
    def __init__(self, sess, phi_dim, a_dim, learning_rate=1.0e-5):
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
        self.sess = sess
        self.variable_list = []

        self.x = tf.placeholder(tf.float32, [None, phi_dim])
        self.a = tf.placeholder(tf.float32, [None, a_dim])
        self.R = tf.placeholder(tf.float32, [None, 1])

        self.phi_dim = phi_dim
        self.a_dim = a_dim

        h1 = layer(self.x, phi_dim, 200, 'Hidden1', act=tf.nn.relu6)
        #lstm1 = tf.contrib.rnn.BasicLSTMCell(128, forget_bias=1.0)
        #h12 = lstm1(self.x, )

        h2 = layer(self.x, phi_dim, 200, 'Hidden2', act=tf.nn.relu6)

        self.mu = layer(h1, 200, a_dim, 'mu', act=tf.tanh)
        self.sigma = tf.clip_by_value(layer(h1, 200, a_dim, 'sigma', act=tf.nn.softplus), 1e-6,1e+6)
        self.V = layer(h2, 200, 1, 'v', act=tf.identity)

        # self.loss_pi = tf.reduce_mean(tf.log(
        #         1/ tf.sqrt( 2.0 * 3.1415926535 * self.sigma ) *
        #         tf.exp(-tf.square(self.a - self.mu) / (2 * self.sigma) )
        #     ) * tf.stop_gradient(self.R - self.V))
        self.pi = tf.contrib.distributions.Normal(self.mu, self.sigma)
        self.loss_pi = tf.reduce_mean(
                self.pi.log_prob(self.a)
             * (self.R - tf.stop_gradient(self.V)))

        self.entropy = tf.reduce_mean(self.pi.entropy())

        self.loss_V= tf.reduce_mean(tf.square(self.R - self.V) )


        # TODO need to change to RMSProp
        # self.optimizer = tf.train.AdamOptimizer(0.001)
        self.optimizer_pi = tf.train.RMSPropOptimizer(
                        learning_rate=0.0001)
        self.optimizer_V = tf.train.RMSPropOptimizer(
                        learning_rate=0.001)
        #                decay=0.99,  momentum=0.0,    epsilon=0.1)

        self.train_pi = self.optimizer_pi.minimize(-self.loss_pi - 0.001 * self.entropy)
        self.train_V = self.optimizer_V.minimize(self.loss_V)

        # self.total_loss = - self.loss_pi - 0.0001 * self.entropy + 0.5 * self.loss_V
        # self.grad_vals = self.optimizer.compute_gradients(self.total_loss)
        # self.grads=[]
        # self.vals=[]
        # for grad,val in self.grad_vals:
        #     if grad is not None:
        #         self.grads.append(grad)
        #         self.vals.append(val)
        # self.grad_vals_ = [(vals, grads) for vals, grads in zip(self.grads, self.vals)]
        #self.train = self.optimizer.apply_gradients(self.grad_vals_)

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

    def update(self, phi, a, R):
        self.sess.run([self.train_pi, self.train_V],
            feed_dict={self.x: phi, self.a: a, self.R: R})

    def gradients(self, phi, a, R):
        return self.sess.run([self.grads_pi, self.grads_V],
            feed_dict={
                self.x: phi,
                self.a: a,
                self.R: R
            })

    def predict_pi_and_V(self, x):
        mu, sigma, V = self.sess.run([self.mu, self.sigma, self.V], feed_dict={self.x: x})
        sigma[0] += 1.0e-4
        return mu[0], sigma[0], V[0][0]

    def predict_pi(self, phi):
        mu, sigma = self.sess.run([self.mu, self.sigma], feed_dict={self.x: [phi]})
        return mu[0], sigma[0]

    def predict_V(self, phi):
        return self.sess.run(self.V, feed_dict={self.x: [phi]})[0][0]

    def choose_action(self, phi):
        return self.sess.run(self.pi.sample(), feed_dict={self.x: [phi]})[0]

    def train(self, pi_grad, v_grad):
        self.sess.run(self.train_step,
            feed_dict={self.pi_grad: pi_grad, self.v_grad: v_grad})

    def read_variables(self):
        with self.sess.as_default():
            values = [valiable.eval() for valiable in self.variable_list]
        return values

    def set_variables(self, values):
        with self.sess.as_default():
            for valiable, value in zip(self.variable_list, values):
                valiable.load(value)
