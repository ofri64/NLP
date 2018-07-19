import time
import os
import numpy as np
from DataProcessor import DataProcessor
from TensorflowPosLSTM import TensorflowPosLSTM

TRAIN_PATH = 'Penn_Treebank/train.gold.conll'
TEST_PATH = 'Penn_Treebank/dev.gold.conll'


if __name__ == "__main__":
    data_processor = DataProcessor()
    x_train, y_train = data_processor.preprocess_train_set(TRAIN_PATH)
    # x_test, y_test = data_processor.preprocess_test_set(TEST_PATH)

    # create boolean mask vectors
    mask_train = data_processor.create_boolean_mask(x_train)
    # mask_test = data_processor.create_boolean_mask(x_test)

    vocab_size = data_processor.vocab_size
    n_classes = data_processor.n_classes
    max_input_length = data_processor.max_seq_len


    # # we need to transform x to one hot encoding vectors
    # # we don't need to transform y because our model uses the sparse(!) softmax
    # # which means the input is a matrix of indexes
    # x_train = tf.one_hot(indices=x_train, depth=vocab_size, axis=-1)
    # # x_test = tf.one_hot(indices=x_test, depth=vocab_size, axis=-1)

    # build model and fit using training set
    model = TensorflowPosLSTM(vocab_size, n_classes, max_input_length)
    model.fit(x_train, y_train, mask_train)




    # model = POSLSTMModel(vocab_size=data_processor.vocab_size,
    #                      n_classes=data_processor.n_classes,
    #                      max_input_length=data_processor.max_seq_len)
    #
    # model.fit(x_train, y_train, callbacks=[cloud_callback])
    #
    # cloud_callback.send_update('Evaluation has just started.')
    # score = model.evaluate_sample(x_test, y_test)
    # loss, acc = score[0], score[1]
    # cloud_callback.send_update('*Evaluation has ended!* Loss: `{0}` - Accuracy: `{1}`'.format(loss, acc))
    #
