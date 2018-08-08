from keras.callbacks import Callback
from keras.callbacks import  ModelCheckpoint
import requests


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
        loss = logs.get('pos_loss') or logs.get('loss')
        acc = logs.get('pos_acc') or logs.get('acc')

        self.send_update('*Epoch {0} has ended*! Loss: `{1}` - Accuracy: `{2}`'.format(epoch + 1, loss, acc))

    def on_train_end(self, logs=None):
        self.send_update('Training is done :tada:')


class CheckpointCallback():
    def __init__(self, save_path):
        self.checkpoint = ModelCheckpoint(save_path, monitor="loss", verbose=1, save_best_only=True, mode="min")

    def get_callback(self):
        return self.checkpoint