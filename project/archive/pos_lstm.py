import preprocessing
import requests

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout, TimeDistributed
from keras.callbacks import Callback

remote = True
stop_url = 'https://us-central1-deep-208318.cloudfunctions.net/stop-cs231'
slack_url = 'https://us-central1-deep-208318.cloudfunctions.net/notify-slack'

dropout = .5
embed_size = 50
hidden_size = 300
batch_size = 32
n_epochs = 10


def stop_instance():
    if remote:
        send_update('Stopping instance, bye bye')
        requests.get(stop_url)


def send_update(msg):
    if remote:
        payload = {'message': msg, 'channel': 'nlp'}
        requests.post(slack_url, json=payload)


class SlackProgressUpdater(Callback):
    def on_train_begin(self, logs=None):
        send_update('Training POS LSTM model has just started :weight_lifter:')

    def on_epoch_end(self, epoch, logs={}):
        loss = logs.get('loss')
        acc = logs.get('acc')

        send_update('*Epoch {0}/{1} has ended*! Loss: `{2}` - Accuracy: `{3}`'.format(epoch + 1, n_epochs, loss, acc))

    def on_train_end(self, logs=None):
        send_update('Training is done :tada:')


if __name__ == "__main__":
    try:
        x_train, y_train, x_test, y_test, vocab_size, n_classes, max_length = preprocessing.load_data()

        model = Sequential([Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=max_length),
                            Dropout(dropout),
                            LSTM(hidden_size, return_sequences=True),
                            TimeDistributed(Dense(n_classes)),
                            Activation('softmax')])

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs, callbacks=[SlackProgressUpdater()])

        send_update('Evaluation has just started.')
        score = model.evaluate(x_test, y_test, batch_size=batch_size)
        loss, acc = score[0], score[1]
        send_update('*Evaluation has ended!* Loss: `{0}` - Accuracy: `{1}`'.format(loss, acc))
    except Exception as e:
        send_update(repr(e))
    finally:
        stop_instance()

