import tensorflow as tf

class NeuralNetwork:
    def __init__(self, qname, phi_dim, a_dim, learning_rate=0.00025):
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
        h1 = layer(self.x, phi_dim, 3200, 'Hidden1')
        h2 = layer(h1, 3200, 3200, 'Hidden2')
        h3 = layer(h2, 3200, 3200, 'Hidden2')
        h4 = layer(h3, 3200, 3200, 'Hidden2')
        h5 = layer(h4, 3200, 3200, 'Hidden2')
        h6 = layer(h5, 3200, 3200, 'Hidden2')
        self.y = layer(h6, 3200, a_dim, 'Output', tf.identity)
        self.t = tf.placeholder(tf.float32, [None, a_dim])
        loss = tf.reduce_mean(tf.square(self.t - self.y))
        self.train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

    def __call__(self, sess, x):
        return sess.run(self.y, feed_dict={self.x: x})

    def train(self, sess, x, t):
        sess.run(self.train_step,
            feed_dict={self.x: x, self.t: t})

    def read_variables(self, sess):
        with sess.as_default():
            values = [valiable.eval() for valiable in self.variable_list]
        return values

    def set_variables(self, sess, values):
        with sess.as_default():
            for valiable, value in zip(self.variable_list, values):
                valiable.load(value)
