import tensorflow as tf
import numpy as np


class HippocampusModel(object):
    """Model representing the Hippocampal autoencoder"""
    def __init__(self, n_inputs, n_hidden, learning_rate=0.2):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate

        with tf.variable_scope("hippocampus"):
            self._build_model()
            self._build_train_op()

    def _build_model(self):
        self.input_ph = tf.placeholder(tf.float32,
            shape=(None, self.n_inputs),
            name='input_ph')
        self.output_ph = tf.placeholder(tf.float32,
            shape=(None, 1),
            name='output_ph')

        self.layer1 = tf.Variable(
            np.random.normal(size=(self.n_inputs, self.n_hidden)) * 0.01,
            dtype=tf.float32,
            name='layer1')
        self.bias1 = tf.Variable(
            np.random.normal(size=(self.n_hidden)) * 0.01,
            dtype=tf.float32,
            name='bias1')
        # output dimension is inputs + 1 because we also predict output
        self.layer2 = tf.Variable(
            np.random.normal(size=(self.n_hidden, self.n_inputs + 1)) * 0.01,
            dtype=tf.float32,
            name='layer2')
        self.bias2 = tf.Variable(
            np.random.normal(size=(self.n_inputs + 1)) * 0.01,
            dtype=tf.float32,
            name='bias2')

    def _build_train_op(self):
        self.hidden_output = tf.matmul(self.input_ph, self.layer1)# + self.bias1
        self.encoder_output = tf.matmul(self.hidden_output, self.layer2) # + self.bias2
        expected_output = tf.concat((self.input_ph, self.output_ph), axis=1)
        self.loss = tf.reduce_mean((self.encoder_output - expected_output) ** 2.)

        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def fit(self, inputs, outputs, n_iters=100, sess=None):
        if sess is None:
            sess = tf.get_default_session()

        loss_trace = []
        for i in range(n_iters):
            loss, _ = sess.run(
                [self.loss, self.train_op],
                {
                    self.input_ph: inputs,
                    self.output_ph: outputs,
                })
            loss_trace.append(loss)

        return loss_trace

    def get_encoding (self):
        return self.hidden_output

    def predict(self, inputs, sess=None):
        if sess is None:
            sess = tf.get_default_session()
        
        outputs = sess.run(
            self.encoder_output,
            {
                self.input_ph: inputs,
            })
        return outputs



class CorticalModel(object):
    """Model representing the feed-forward cortical model"""
    def __init__(self, hipp_model, n_inputs, n_hidden, learning_rate = .2, is_lesioned=False):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.hipp_model = hipp_model
        self.is_lesioned = is_lesioned

        with tf.variable_scope("cortical"):
            self._build_model()
            self._build_train_op()


    def _build_model(self):
        self.input_ph = tf.placeholder(tf.float32,
            shape=(None, self.n_inputs),
            name='input_ph')
        self.output_ph = tf.placeholder(tf.float32,
            shape=(None, 1),
            name='output_ph')

        self.layer1 = tf.Variable(
            np.random.normal(size=(self.n_inputs, self.n_hidden)) * 0.01,
            dtype=tf.float32,
            name='layer1')
        self.bias1 = tf.Variable(
            np.random.normal(size=self.n_hidden) * 0.01,
            dtype=tf.float32,
            name='bias1')
        # output dimension is inputs + 1 because we also predict output
        self.layer2 = tf.Variable(
            np.random.normal(size=(self.n_hidden, 1)) * 0.01,
            dtype=tf.float32,
            name='layer2')
        self.bias2 = tf.Variable(
            np.random.normal(size=1) * 0.01,
            dtype=tf.float32,
            name='bias2')

    def _build_train_op(self):
        
        self.hidden_output = tf.matmul(self.input_ph,self.layer1) #+ self.bias1
        self.behavior_output = tf.identity(tf.matmul(self.hidden_output, self.layer2)) # + self.bias2)
        # print(self.hidden_output.shape)
        # print(self.behavior_output.shape)

        self.loss1 = tf.reduce_mean((self.hidden_output - self.hipp_model.get_encoding()) ** 2.)
        self.loss2 = tf.reduce_mean((self.behavior_output - self.output_ph) ** 2.)

        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_op1 = self.optimizer.minimize(self.loss1, var_list = [self.layer1])
        self.train_op2 = self.optimizer.minimize(self.loss2, var_list = [self.layer2])

    def fit(self, inputs, outputs, n_iters=100, sess=None):
        if sess is None:
            sess = tf.get_default_session()

        loss_trace = []
        output_trace = []
        if not self.is_lesioned:
            for i in range(n_iters):
                    loss1, output, _ = sess.run(
                        [self.loss1, self.behavior_output, self.train_op1],
                        {
                            self.input_ph: inputs,
                            self.output_ph: outputs,
                            self.hipp_model.input_ph: inputs
                        })
                # output_trace.append(output)

        for i in range(n_iters):
            loss2, output, _ = sess.run(
                [self.loss2, self.behavior_output, self.train_op2],
                {
                    self.input_ph: inputs,
                    self.output_ph: outputs,
                    self.hipp_model.input_ph: inputs
                })
            loss_trace.append(loss2)
            output_trace.append(output)


        return loss_trace, output_trace

    def predict(self, inputs, sess=None):
        if sess is None:
            sess = tf.get_default_session()
        
        outputs = sess.run(
            self.behavior_output,
            {
                self.input_ph: inputs,
                self.hipp_model.input_ph: inputs
            })
        return outputs


class Model(object):
    """The overall cortico-hippocampal model"""
    def __init__(self, n_inputs, n_hidden, is_lesioned=False, learning_rate=0.2):
        self.hipp_model = HippocampusModel(n_inputs, n_hidden,
            learning_rate=learning_rate)
        self.cort_model = CorticalModel(self.hipp_model, n_inputs, n_hidden,
            is_lesioned=is_lesioned,
            learning_rate=learning_rate)
        self.is_lesioned = is_lesioned

    # First trains the hippocampus and then the cortical model.
    def fit(self, inputs, outputs, n_hipp=500, n_cort=500):
        # if not self.is_lesioned:
        #     hipp_loss = self.hipp_model.fit(inputs, outputs, n_iters=n_hipp)
        # cort_loss, cort_out = self.cort_model.fit(inputs, outputs, n_iters=n_cort)

        cort_loss_trace = []
        cort_out_trace = []
        for i in range(n_hipp):
            cort_loss, cort_out = self.train(inputs, outputs)
            cort_loss_trace.append(cort_loss)
            cort_out_trace.append(cort_out)

        return cort_loss_trace, cort_out_trace

    # First trains the hippocampus and then the cortical model.
    def train(self, inputs, outputs):
        if not self.is_lesioned:
            hipp_loss = self.hipp_model.fit(inputs, outputs, n_iters=1)
        cort_loss, cort_out = self.cort_model.fit(inputs, outputs, n_iters=1)

        return cort_loss[0], cort_out[0]

    # Returns the predictions from the cortical model
    def predict(self, inputs):
        return self.cort_model.predict(inputs)


        
        
        