from POSTaggerInterface import POSTaggerInterface
from KerasCallbacks import CloudCallback, CheckpointCallback
from DataProcessors import HebrewBinyanDataProcessor

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout, TimeDistributed, Bidirectional, Masking, Input
from _datetime import datetime
import numpy as np


class KerasPOSTagger(POSTaggerInterface):

    def __init__(self, data_processor, embed_size=50, hidden_size=100, batch_size=32, n_epochs=10,
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
        input_tensor = Input(shape=(input_length, vocab_size,))
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

    def fit(self, x_train, y_train, name=None, callbacks=None):
        if not self.model:
            self.build()

        if not name:
            name = 'my_cool_model'

        # Add checkpoint to callbacks
        now = str(datetime.now()).split('.')[0]
        saver_callback = CheckpointCallback('{0}-{1}.h5'.format(name, now)).get_callback()
        if callbacks is None:
            callbacks = [saver_callback]
        else:
            callbacks.append(saver_callback)

        # Fit model and concatenate callbacks
        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.n_epochs, callbacks=callbacks)

    def evaluate_sample(self, x_test, y_test):
        score = self.model.evaluate(x_test, y_test, batch_size=self.batch_size)
        return score

    def evaluate_sample_conditioned(self, x_test, y_test, condition):
        x_unseen_test = []
        y_unseen_test = []
        x_test_indices = self.data_processor.transform_to_index(x_test)
        boolean_unseen_matrix = np.zeros([x_test.shape[0], x_test.shape[1]])

        if condition == 'unseen':
            num_sent = 0
            for i, sent in enumerate(x_test_indices):
                appended = False
                for j, word in enumerate(sent):
                    if word in self.data_processor.unk_indices:
                        boolean_unseen_matrix[num_sent, j] = 1
                        print('UNKNOWN WORD:', word, i)
                        if not appended:
                            x_unseen_test.append(sent)
                            y_unseen_test.append(y_test[i])
                            num_sent += 1
                            appended = True

            # delete unnecessary rows
            boolean_unseen_matrix = np.delete(boolean_unseen_matrix, [_ for _ in range(num_sent, x_test.shape[0])],
                                              axis=0)

            x_unseen_test = np.array(self.data_processor.transform_to_one_hot(x_unseen_test, x_test.shape[2]))
            y_unseen_test = self.data_processor.transform_to_index(y_unseen_test)

            print('Evaluating ', len(x_unseen_test))

            predictions = self.predict(x_unseen_test)
            acc_matrix = predictions == y_unseen_test
            acc = np.divide(np.sum(acc_matrix * boolean_unseen_matrix), np.sum(boolean_unseen_matrix))

            return acc

        # TODO: Ambiguous case

        else:
            raise AttributeError("Condition must be one of: {'unseen', 'ambiguous'}")

    def predict(self, sentences):
        predictions = self.model.predict(sentences)
        return np.argmax(predictions, axis=2)

    def load_model_params(self, file_path):
        self.model = load_model(file_path)


class MTLHebrewBinyanTagger(KerasPOSTagger):
    def __init__(self, data_processor, embed_size=50, hidden_size=100, batch_size=32,
                 n_epochs=10,
                 dropout_rate=0.5, immediate_build=True):
        super(MTLHebrewBinyanTagger, self).__init__(data_processor, embed_size, hidden_size, batch_size, n_epochs,
                                                    dropout_rate, immediate_build)

    def build(self, optimizer='adam', metrics=['accuracy']):
        # Receive data information from processor
        word_vocab = self.data_processor.get_word2idx_vocab()
        vocab_size = len(word_vocab)
        n_classes = len(self.data_processor.get_tag2idx_vocab())
        n_binyans = len(self.data_processor.get_binyan2idx_vocab())
        padding_index = word_vocab["PADD"]
        input_length = self.data_processor.get_max_sequence_length()

        sent_input = Input(shape=(input_length, vocab_size,))
        # task_input = Input(shape=(1, ))

        embedding = Dense(units=self.embed_size)(sent_input)
        masking = Masking(mask_value=padding_index)(embedding)
        dropout = Dropout(self.dropout_rate)(masking)
        hidden1 = Bidirectional(LSTM(units=self.hidden_size, return_sequences=True))(dropout)

        binyan_output = Dense(units=n_binyans, activation='softmax', name='binyan')(hidden1)
        # binyan_model = Model(inputs=sent_input, outputs=binyan_output)

        hidden2 = Bidirectional(LSTM(units=self.hidden_size, return_sequences=True))(hidden1)
        pos_output = Dense(units=n_classes, activation='softmax', name='pos')(hidden2)
        # pos_model = Model(inputs=sent_input, outputs=pos_output)

        # self.model = [binyan_model, pos_model]

        # for m in self.model:
        #     m.compile(loss='categorical_crossentropy',
        #               optimizer=optimizer,
        #               metrics=metrics)
        #     print(m.summary())

        self.model = Model(inputs=sent_input, outputs=[pos_output, binyan_output])
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=metrics)
        print(self.model.summary())

    def predict(self, sentences):
        pos_predictions, _ = self.model.predict(sentences)
        return np.argmax(pos_predictions, axis=2)

