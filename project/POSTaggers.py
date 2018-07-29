from POSTaggerInterface import POSTaggerInterface
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout, TimeDistributed, Bidirectional, Masking, Input
from KerasCallbacks import CloudCallback, CheckpointCallback
from _datetime import datetime
import numpy as np


class KerasPOSTagger(POSTaggerInterface):

    def __init__(self, data_processor, embed_size=50, hidden_size=300, batch_size=32, n_epochs=10,
                 dropout_rate=0.5, immediate_build=True):

        self.model = None
        self.data_processor = data_processor
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.dropout_rate = dropout_rate

        if immediate_build:
            self.build()

    def build(self, optimizer='adam', metrics=['accuracy']):
        # Receive data information from processor
        word_vocab = self.data_processor.get_word2idx_vocab()
        vocab_size = len(word_vocab)
        n_classes = len(self.data_processor.get_tag2idx_vocab())
        padding_index = word_vocab["PADD"]
        input_length = self.data_processor.get_max_sequence_length()

        # Define the Functional model
        input_tensor = Input(shape=(input_length, vocab_size, ))
        embedding = Dense(units=self.embed_size)(input_tensor)
        masking = Masking(mask_value=padding_index)(embedding)
        dropout = Dropout(self.dropout_rate)(masking)
        hidden = Bidirectional(LSTM(units=self.hidden_size, return_sequences=True))(dropout)
        output = Dense(units=n_classes, activation='softmax')(hidden)

        # Compile the model
        self.model = Model(inputs=input_tensor, outputs=output)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=metrics)

        # Print model parameters
        print(self.model.summary())

    def fit(self, x_train, y_train, callbacks=None):
        if not self.model:
            self.build()

        # Add checkpoint to callbacks
        now = datetime.now()
        saver_callback = CheckpointCallback("BiLSTM_model-{0}.h5".format(now)).get_callback()
        if callbacks is None:
            callbacks = [saver_callback]
        else:
            callbacks.append(saver_callback)

        # Fit model and concatenate callbacks
        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.n_epochs, callbacks=callbacks)

    def evaluate_sample(self, x_test, y_test):
        score = self.model.evaluate(x_test, y_test, batch_size=self.batch_size)
        return score

    # def evaluate_sample_conditioned(self, x_test, y_test, condition):
    #     if condition == 'unseen':
    #
    #
    #     elif condition == 'ambiguous':
    #
    #     else:
    #         raise AttributeError("Condition must be one of: {'unseen', 'ambiguous'}")

    def predict_sentence(self, sentence):
        idx2tag = self.data_processor.get_idx2tag_vocab()
        predictions = np.argmax(self.model.predict(sentence), axis=1)
        return np.array([idx2tag[prediction] for prediction in predictions])

    def load_model_params(self, file_path):
        self.model = load_model(file_path)
