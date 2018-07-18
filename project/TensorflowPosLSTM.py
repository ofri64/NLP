import time
import os
import tensorflow as tf
from TensorflowAbstractModel import TensorflowAbstractModel


class TensorflowPosLSTM(TensorflowAbstractModel):

    def __init__(self, vocab_size, n_classes, max_input_length,
                 embedding_size=50, hidden_size=300, batch_size=64,
                 n_epochs=10, dropout_rate=0.5, lr=0.001):
        super(TensorflowPosLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.max_input_length = max_input_length
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.n_epochs = n_epochs
        self.lr = lr
        self.saver_path = None

    def add_placeholders(self):

        self.input_placeholder = tf.placeholder(tf.float32, shape=(None, self.max_input_length, self.vocab_size))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_input_length))
        self.mask_placeholder = tf.placeholder(tf.bool, shape=(None, self.max_input_length))
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def create_feed_dict(self, input_batch, mask_batch, labels_batch=None, dropout=0.5):
        feed_dict = {
            self.input_placeholder: input_batch,
            self.mask_placeholder: mask_batch,
            self.dropout_placeholder: 1 - dropout
        }

        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch

        return feed_dict

    def add_prediction_op(self):
        embeddings = tf.layers.dense(self.input_placeholder,
                                     units=self.embedding_size,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer())

        LSTM_cell = tf.contrib.rnn.LSTMCell(num_units=self.hidden_size,
                                            initializer=tf.contrib.layers.xavier_initializer())

        LSTM_cell = tf.contrib.rnn.DropoutWrapper(LSTM_cell, input_keep_prob=self.dropout_placeholder)

        LSTM_outputs, _ = tf.nn.dynamic_rnn(LSTM_cell, embeddings, dtype=tf.float32)

        LSTM_outputs = tf.layers.dense(LSTM_cell,
                                       units=self.n_classes,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       bias_initializer=tf.constant_initializer())
        return LSTM_outputs

    def add_loss_op(self, predictions):
        batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.boolean_mask(self.labels_placeholder, self.mask_placeholder),
                    logits=tf.boolean_mask(predictions, self.mask_placeholder),
                    name='batch_loss')
        loss = tf.reduce_mean(batch_loss)

        return loss

    def add_training_op(self, loss):
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        return train_op

    def fit(self, x_train, y_train, mask_train):
        """
        :param x_train: ndarray of shape (sample_length, max_input_length, vocab_size)
        :param y_train: ndarray of shape (sample_length, max_input_length)
        :param mask_train: ndarray of shape(sample_length, max_input_length)
        :return:
        """

        with tf.Graph().as_default():
            print("Building the graph...")
            self.build_graph()

            init = tf.global_variables_initializer()
            self.saver_path = TensorflowAbstractModel.get_save_path(model_prefix="LSTM")
            saver = tf.train.Saver()

            with tf.Session() as session:
                session.run(init)

                for epoch in range(self.n_epochs):
                    acc_loss = 0
                    print("Starting Epoch {0} out of {1}".format(epoch + 1, self.n_epochs))
                    current_step = 0
                    total_samples = x_train.shape[0]

                    for i in range(0, total_samples, self.batch_size):
                        x_train_batch = x_train[i:max(i+self.batch_size, total_samples), :, :]
                        y_train_batch = y_train[i:max(i+self.batch_size, total_samples), :]
                        mask_train_batch = mask_train[i:max(i+self.batch_size, total_samples), :]
                        loss = self.train_on_batch(session, x_train_batch, y_train_batch, mask_train_batch)
                        acc_loss += loss
                        current_step += 1

                        if current_step % 100 == 0:
                            avg_loss = acc_loss / 100
                            print("Average loss in epoch {0} and step {1} is: {2}"
                                  .format(epoch + 1, current_step, avg_loss))
                            saver.save(session, self.saver_path, global_step=current_step)
                            acc_loss = 0


