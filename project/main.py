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
    x_test, y_test = data_processor.preprocess_test_set(TEST_PATH)

    # create boolean mask vectors
    mask_train = data_processor.create_boolean_mask(x_train)
    mask_test = data_processor.create_boolean_mask(x_test)

    vocab_size = data_processor.vocab_size
    n_classes = data_processor.n_classes
    max_input_length = data_processor.max_seq_len

    # build model and fit using training set
    save_path = "/Users/okleinfeld/Git_Projects/Natural_Language_Processing/project/LSTM/07-19-18-12:44:15/"
    # model = TensorflowPosLSTM(vocab_size, n_classes, max_input_length)
    model = TensorflowPosLSTM(vocab_size, n_classes, max_input_length, saver_path=save_path)
    # model.fit(x_train, y_train, mask_train)

    # predict over test set
    accuracy = model.evaluate_sample(x_test, y_test, mask_test)
    print(accuracy)

