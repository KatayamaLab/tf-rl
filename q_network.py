import tensorflow as tf

class QNetwork:
    def __init__(self, qname, phi_dim, a_dim, learning_rate=0.00025):
        def layer(input_tensor, input_dim, output_dim, name='layer', act=tf.nn.relu):
            with tf.name_scope(qname+name):
                w = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
                self.variable_list.append(w)
                b = tf.Variable(tf.truncated_normal([output_dim], stddev=0.1))
                self.variable_list.append(b)
                h = act(tf.nn.xw_plus_b(input_tensor, w, b))
                tf.summary.histogram('weight', w)
                tf.summary.histogram('bias', b)
            return h

        self.variable_list = []
        self.x = tf.placeholder(tf.float32, [None, phi_dim])
        h1 = layer(self.x, phi_dim, 128, 'Hidden1', act=tf.nn.relu)
        h2 = layer(h1, 128, 64, 'Hidden2', act=tf.nn.relu)
        self.y = layer(h2, 64, a_dim, 'Output', act=tf.identity)
        self.t = tf.placeholder(tf.float32, [None, a_dim])
        loss = tf.reduce_mean(tf.square(self.t - self.y))
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        #self.train_step = tf.train.RMSPropOptimizer(
        #    learning_rate,
        #    decay=0.95,
        #    momentum=0.95,
        #    epsilon=0.01 ).minimize(loss)

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
