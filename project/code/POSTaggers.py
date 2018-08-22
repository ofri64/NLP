from POSTaggerInterface import POSTaggerInterface
from KerasCallbacks import CheckpointCallback

from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Masking, Input
from datetime import datetime
import numpy as np
import os


def modelpath(subpath=''):
    return os.path.dirname(__file__) + '/../models/' + subpath


def base_network(input_length, vocab_size, embed_size, padding_index, dropout_rate=.5, hidden_size=100):
    input_layer = Input(shape=(input_length, vocab_size,))
    embedding = Dense(units=embed_size)(input_layer)
    masking = Masking(mask_value=padding_index)(embedding)
    dropout = Dropout(dropout_rate)(masking)
    bilstm = Bidirectional(LSTM(units=hidden_size, return_sequences=True))(dropout)

    return input_layer, bilstm


def output_layer(temporal_layer, n_categories, name):
    return Dense(units=n_categories, activation='softmax', name=name)(temporal_layer)


def feature_outputs(bilstm, *features_list, **features_dict):
    outputs = [output_layer(bilstm, o['n_categories'], o['name']) for o in features_list] + \
              [output_layer(bilstm, n_categories, name) for name, n_categories in features_dict.items()]

    return outputs


def build_model(input_layer, bilstm, hidden_size, n_pos, *outputs_list, **outputs_dict):
    pos_bilstm = Bidirectional(LSTM(units=hidden_size, return_sequences=True))(bilstm)
    pos_output = Dense(units=n_pos, activation='softmax', name='pos')(pos_bilstm)

    outputs = [pos_output] + feature_outputs(bilstm, *outputs_list, **outputs_dict)
    return Model(inputs=input_layer, outputs=outputs)


class SimpleTagger(POSTaggerInterface):

    def __init__(self, data_processor, embed_size=50, hidden_size=100, batch_size=32, n_epochs=10,
                 dropout_rate=0.5, immediate_build=False):

        self.model = None
        self.model_summary = None
        self.name = 'lstm_' + data_processor.get_name()

        self.data_processor = data_processor
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.dropout_rate = dropout_rate

        if immediate_build:
            self.build()

    def build_and_compile(self, optimizer='adam', metrics=['accuracy']):
        self.build(optimizer, metrics)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=metrics)
        print(self.model.summary())

    def build(self, optimizer='adam', metrics=['accuracy']):
        # Receive data information from processor
        word_dict = self.data_processor.get_word2idx_dict()
        vocab_size = len(word_dict)
        n_classes = len(self.data_processor.get_tag2idx_dict())
        padding_index = word_dict['PADD']
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
            name = self.name

        models_dir_path = os.path.dirname(__file__) + '/../models/'
        filename = name + ".h5"
        filepath = os.path.join(models_dir_path,filename)

        # if not os.path.exists(modelpath(name)):
        #     os.mkdir(modelpath(name))
        #
        # start_time = '{:%d-%m-%Y_%H:%M:%S}'.format(datetime.now())
        # os.mkdir(modelpath(name + '/' + start_time))
        # filepath = modelpath('%s/%s/model.{epoch:02d}.hdf5' % (name, start_time))

        checkpoint = ModelCheckpoint(filepath, monitor="loss", verbose=1, save_best_only=True, mode="min")
        callbacks = [checkpoint] if callbacks is None else callbacks + [checkpoint]

        # Fit model and concatenate callbacks
        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.n_epochs, callbacks=callbacks)

    def evaluate_sample(self, x_test, y_test):
        model_outputs = self.model.evaluate(x_test, y_test, batch_size=self.batch_size)
        model_outputs_names = self.model.metrics_names
        return zip(model_outputs_names, model_outputs)

    def evaluate_sample_conditioned(self, x_test, y_test, condition):
        y_test = y_test[0]
        x_unseen_test = []
        y_unseen_test = []
        x_test_indices = self.data_processor.transform_to_index(x_test)
        boolean_unseen_matrix = np.zeros([x_test.shape[0], x_test.shape[1]])

        if condition == 'unseen':
            condition_indices = self.data_processor.unk_indices
        elif condition == 'ambiguous':
            condition_indices = self.data_processor.ambig_indices
        else:
            raise AttributeError("Condition must be one of: {'unseen', 'ambiguous'}")

        num_sent = 0
        for i, sent in enumerate(x_test_indices):
            appended = False
            for j, word in enumerate(sent):
                if word in condition_indices:
                    boolean_unseen_matrix[num_sent, j] = 1
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

        print('Evaluating: {0}'.format(len(x_unseen_test)))

        predictions = self.predict(x_unseen_test)
        acc_matrix = predictions == y_unseen_test
        acc = np.divide(np.sum(acc_matrix * boolean_unseen_matrix), np.sum(boolean_unseen_matrix))

        return acc

    def predict(self, sentences):
        predictions = self.model.predict(sentences)
        return np.argmax(predictions, axis=2)

    def load_model_params(self, file_path):
        self.model = load_model(file_path)

    def predict_pos_raw_sent(self, raw_sent):
        sent = raw_sent.split(' ')
        sent_len = len(sent)
        sent = self.data_processor.preprocess_sentence(sent)

        onehot_sent = self.data_processor.transform_to_one_hot([sent], len(self.data_processor.get_word2idx_dict()))[0]

        pos_predictions = self.model.predict(np.array([onehot_sent]))[0]
        tags = np.argmax(pos_predictions, axis=1)

        return [self.data_processor.get_idx2tag_dict()[idx] for idx in tags][:sent_len]


class MTLOneFeatureTagger(SimpleTagger):
    def __init__(self, data_processor, feature, embed_size=50, hidden_size=100, batch_size=32,
                 n_epochs=10,
                 dropout_rate=0.5, immediate_build=False):
        super(MTLOneFeatureTagger, self).__init__(data_processor, embed_size, hidden_size, batch_size, n_epochs,
                                                  dropout_rate, immediate_build)

        self.feature = feature
        self.name = 'mtl_one_' + data_processor.get_name()

    def build(self, optimizer='adam', metrics=['accuracy']):
        # Receive data information from processor
        word_dict = self.data_processor.get_word2idx_dict()
        vocab_size = len(word_dict)
        n_classes = len(self.data_processor.get_tag2idx_dict())
        n_feature_values = len(self.data_processor.get_features2idx_dicts()[self.feature])
        padding_index = word_dict['PADD']
        input_length = self.data_processor.get_max_sequence_length()

        sent_input = Input(shape=(input_length, vocab_size,))
        embedding = Dense(units=self.embed_size)(sent_input)
        masking = Masking(mask_value=padding_index)(embedding)
        dropout = Dropout(self.dropout_rate)(masking)
        hidden1 = Bidirectional(LSTM(units=self.hidden_size, return_sequences=True))(dropout)

        feature_output = Dense(units=n_feature_values, activation='softmax', name=self.feature)(hidden1)

        hidden2 = Bidirectional(LSTM(units=self.hidden_size, return_sequences=True))(hidden1)
        pos_output = Dense(units=n_classes, activation='softmax', name='pos')(hidden2)

        self.model = Model(inputs=sent_input, outputs=[pos_output, feature_output])
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=metrics)
        print(self.model.summary())

    def predict(self, sentences):
        pos_predictions, _ = self.model.predict(sentences)
        return np.argmax(pos_predictions, axis=2)


class MTLAllFeaturesTagger(SimpleTagger):
    def __init__(self, data_processor, embed_size=50, hidden_size=100, batch_size=32,
                 n_epochs=10,
                 dropout_rate=0.5, immediate_build=False):
        super(MTLAllFeaturesTagger, self).__init__(data_processor, embed_size, hidden_size, batch_size, n_epochs,
                                                   dropout_rate, immediate_build)

        self.name = 'mtl_one_' + data_processor.get_name()
        self.pos_model = None

    def build(self, optimizer='adam', metrics=['accuracy']):
        # Receive data information from processor
        word_dict = self.data_processor.get_word2idx_dict()
        vocab_size = len(word_dict)
        n_classes = len(self.data_processor.get_tag2idx_dict())
        n_features_labels = {feature: len(d) for feature, d in self.data_processor.get_features2idx_dicts().items()}
        padding_index = word_dict['PADD']
        input_length = self.data_processor.get_max_sequence_length()

        sent_input = Input(shape=(input_length, vocab_size,))
        embedding = Dense(units=self.embed_size)(sent_input)
        masking = Masking(mask_value=padding_index)(embedding)
        dropout = Dropout(self.dropout_rate)(masking)

        hidden1 = Bidirectional(LSTM(units=self.hidden_size, return_sequences=True))(dropout)
        # hidden1 = Dense(units=self.hidden_size, activation='relu')(dropout)

        features_outputs = {f: Dense(units=n_labels, activation='softmax', name=f)(hidden1) for f, n_labels in
                            n_features_labels.items()}

        hidden2 = Bidirectional(LSTM(units=self.hidden_size, return_sequences=True))(hidden1)
        pos_output = Dense(units=n_classes, activation='softmax', name='pos')(hidden2)

        self.model = Model(inputs=sent_input, outputs=[pos_output] + list(features_outputs.values()))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=metrics)

        self.pos_model = Model(inputs=sent_input, outputs=pos_output)

        print(self.model.summary())

    def load_model_params(self, file_path):
        self.model = load_model(file_path)
        self.pos_model = Model(inputs=self.model.input, outputs=self.model.output[0])  # output[0] is pos...

    def predict(self, sentences):
        pos_predictions = self.pos_model.predict(sentences)
        return np.argmax(pos_predictions, axis=2)

    def predict_pos(self, raw_sent):
        sent = raw_sent.split(' ')
        sent_len = len(sent)
        sent = self.data_processor.preprocess_sentence(sent)

        onehot_sent = self.data_processor.transform_to_one_hot([sent], len(self.data_processor.get_word2idx_dict()))[0]

        pos_predictions = self.pos_model.predict(np.array([onehot_sent]))[0]
        tags = np.argmax(pos_predictions, axis=1)

        return [self.data_processor.get_idx2tag_dict()[idx] for idx in tags][:sent_len]

    def predict_pos_with_regular_model(self, raw_sent):
        sent = raw_sent.split(' ')
        sent_len = len(sent)
        sent = self.data_processor.preprocess_sentence(sent)

        onehot_sent = self.data_processor.transform_to_one_hot([sent], len(self.data_processor.get_word2idx_dict()))[0]

        pos_predictions = self.model.predict(np.array([onehot_sent]))[0][0]
        tags = np.argmax(pos_predictions, axis=1)

        return [self.data_processor.get_idx2tag_dict()[idx] for idx in tags][:sent_len]

    def predict_all(self, raw_sent):
        sent = raw_sent.split(' ')
        sent_len = len(sent)
        sent = self.data_processor.preprocess_sentence(sent)

        preds = self.model.predict(np.array([sent]))
        return preds
