from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Masking, Input, Embedding
from keras.utils import multi_gpu_model
import numpy as np
import os

from POSTaggerInterface import POSTaggerInterface


def model_path(name):
    path = os.path.join(os.path.dirname(__file__), os.pardir, 'models', "{0}.h5".format(name))
    return os.path.abspath(path)


class GeneratorPOSTagger(POSTaggerInterface):

    def __init__(self, name, data_processor, embed_size=50, hidden_size=100, steps_per_epoch=100,
                 n_epochs=10, dropout_rate=0.5, immediate_build=False, multi_gpus=True):
        self.name = name
        self.model = None
        self.data_processor = data_processor
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.dropout_rate = dropout_rate
        self.multi_gpus = multi_gpus

        if immediate_build:
            self.build()

    def build(self, optimizer='adam', metrics=['accuracy']):

        # Receive data information from processor
        word_dict = self.data_processor.word2idx
        vocab_size = len(word_dict)
        n_classes = len(self.data_processor.tag2idx)
        padding_index = word_dict['PADD']
        input_length = self.data_processor.max_seq_len

        # Define the Functional model
        input_tensor = Input(shape=(input_length,))
        masking = Masking(mask_value=padding_index)(input_tensor)
        embedding = Embedding(input_dim=vocab_size, output_dim=self.embed_size)(masking)
        dropout = Dropout(self.dropout_rate)(embedding)
        hidden = Bidirectional(LSTM(units=self.hidden_size, return_sequences=True))(dropout)
        output = Dense(units=n_classes, activation='softmax')(hidden)

        # Compile the model
        self.model = Model(inputs=input_tensor, outputs=output)

        if self.multi_gpus:
            parallel_model = multi_gpu_model(self.model, gpus=4)
            self.model = parallel_model

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=metrics)

        # Print model parameters
        print(self.model.summary())

    def fit(self, input_file, callbacks=None):
        if not self.model:
            self.build()

        # Initiate batch generator
        sample_generator = self.data_processor.get_sample_generator(input_file)

        # Define checkpoint callback to save model parameters at the end of every epoch
        checkpoint = ModelCheckpoint(model_path(self.name), monitor="loss", verbose=1, save_best_only=True, mode="min")
        callbacks_list = [checkpoint] if callbacks is None else callbacks + [checkpoint]

        self.model.fit_generator(sample_generator, steps_per_epoch=self.steps_per_epoch, epochs=self.n_epochs,
                                 callbacks=callbacks_list)

    def load_model_params(self, file_path=None):
        file_path = model_path(self.name)
        self.model = load_model(file_path)

    def predict(self, input_file, num_batch_steps):
        self.load_model_params()
        sample_generator = self.data_processor.get_sample_generator(input_file)
        _, accuracy = self.model.evaluate_generator(sample_generator, steps=num_batch_steps, verbose=1)

        return accuracy

