import preprocessing
import requests

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout, TimeDistributed
from keras.callbacks import Callback

dropout = .5
embed_size = 50
hidden_size = 300
batch_size = 32
n_epochs = 10

stop_url = ''
slack_url = ''


class SlackProgressUpdater(Callback):
    def on_train_begin(self, logs=None):
        msg = 'Training POS LSTM model has just started :weight_lifter:'
        payload = {'message': msg, 'channel': 'nlp'}
        requests.post(slack_url, json=payload)

    def on_epoch_end(self, epoch, logs={}):
        loss = logs.get('loss')
        acc = logs.get('acc')

        msg = '*Epoch {0} ended* - Loss: `{1}` - Accuracy: `{2}`'.format(epoch + 1, loss, acc)
        payload = {'message': msg, 'channel': 'nlp'}
        requests.post(slack_url, json=payload)


if __name__ == "__main__":
    x_train, y_train, x_test, y_test, vocab_size, max_length, n_classes = preprocessing.load_data()

    model = Sequential([Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=max_length),
                        Dropout(dropout),
                        LSTM(hidden_size, return_sequences=True),
                        TimeDistributed(Dense(n_classes)),
                        Activation('softmax')])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs, callbacks=[SlackProgressUpdater()])
    model.evaluate(x_test, y_test, batch_size=batch_size)
