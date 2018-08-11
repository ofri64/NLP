from archive.DataProcessor import HebrewDataProcessor
import requests

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout, TimeDistributed, Bidirectional, Masking
from keras.callbacks import Callback

TRAIN_PATH = 'datasets/english/train.gold.conll'
TEST_PATH = 'datasets/english/dev.gold.conll'

HEB_TRAIN_PATH = 'datasets/hebrew/train.conllu'
HEB_TEST_PATH = 'hebrew/test.conllu'


class CloudCallback(Callback):
    def __init__(self, remote=False, slack_url='', stop_url=''):
        super(CloudCallback, self).__init__()
        self.remote = remote
        self.slack_url = slack_url
        self.stop_url = stop_url

    def stop_instance(self):
        if self.remote:
            self.send_update('Stopping instance, bye bye')
            requests.get(self.stop_url)

    def send_update(self, msg):
        print(msg)
        if self.remote:
            payload = {'message': msg, 'channel': 'nlp'}
            requests.post(self.slack_url, json=payload)

    def on_train_begin(self, logs=None):
        self.send_update('Training POS LSTM model has just started :weight_lifter:')

    def on_epoch_end(self, epoch, logs={}):
        loss = logs.get('loss')
        acc = logs.get('acc')

        self.send_update('*Epoch {0} has ended*! Loss: `{1}` - Accuracy: `{2}`'.format(epoch + 1, loss, acc))

    def on_train_end(self, logs=None):
        self.send_update('Training is done :tada:')


class POSLSTMModel(object):
    def __init__(self, vocab_size, n_classes, max_input_length, padding_index,
                 embed_size=50, hidden_size=300, batch_size=32, n_epochs=10,
                 dropout_rate=0.5, immediate_build=True):
        self.model = None
        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.max_input_length = max_input_length
        self.padding_index = padding_index
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.dropout_rate = dropout_rate

        if immediate_build:
            self.build()

    def build(self, optimizer='adam', metrics=['accuracy']):
        self.model = Sequential([Embedding(input_dim=self.vocab_size, output_dim=self.embed_size, input_length=self.max_input_length),
                                Masking(mask_value=self.padding_index),
                                Dropout(self.dropout_rate),
                                Bidirectional(LSTM(self.hidden_size, return_sequences=True)),
                                TimeDistributed(Dense(self.n_classes)),
                                Activation('softmax')])

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=metrics)

        print(self.model.summary())

    def fit(self, x_train, y_train, callbacks=None):
        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.n_epochs, callbacks=callbacks)

    def evaluate_sample(self, x_test, y_test):
        score = self.model.evaluate(x_test, y_test, batch_size=self.batch_size)
        return score


if __name__ == "__main__":
    data_processor = HebrewDataProcessor()
    data_processor.initiate_word_tags_dicts(HEB_TRAIN_PATH)
    data_processor.save_words_dict()
    x_train, y_train = data_processor.preprocess_sample_set(HEB_TRAIN_PATH)
    y_train = data_processor.transform_to_one_hot(y_train)
    x_test, y_test = data_processor.preprocess_sample_set(HEB_TEST_PATH)
    y_test = data_processor.transform_to_one_hot(y_test)

    stop_url = ''
    slack_url = ''
    cloud_callback = CloudCallback(remote=False, slack_url=slack_url, stop_url=stop_url)

    try:
        padding_index = data_processor.word2idx["PADD"]
        model = POSLSTMModel(vocab_size=data_processor.vocab_size,
                             n_classes=data_processor.n_classes,
                             max_input_length=data_processor.max_seq_len,
                             padding_index=padding_index)

        model.fit(x_train, y_train, callbacks=[cloud_callback])

        cloud_callback.send_update('Evaluation has just started.')
        score = model.evaluate_sample(x_test, y_test)
        loss, acc = score[0], score[1]
        cloud_callback.send_update('*Evaluation has ended!* Loss: `{0}` - Accuracy: `{1}`'.format(loss, acc))

    except Exception as e:
            cloud_callback.send_update(repr(e))
    finally:
            cloud_callback.stop_instance()
