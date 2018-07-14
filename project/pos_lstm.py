import preprocessing
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout, TimeDistributed

dropout = .5
embed_size = 50
hidden_size = 300
batch_size = 32
n_epochs = 10


if __name__ == "__main__":
    print('Loading data...')
    x_train, y_train, x_test, y_test, vocab_size, max_length, n_classes = preprocessing.load_data()
    print('\tDONE\n')

    model = Sequential([Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=max_length),
                        Dropout(dropout),
                        LSTM(hidden_size, return_sequences=True),
                        TimeDistributed(Dense(n_classes)),
                        Activation('softmax')])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs)
    model.evaluate(x_test, y_test, batch_size=batch_size)
