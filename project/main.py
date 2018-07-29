import time
import os
import numpy as np
from DataProcessor import DataProcessor
from TensorflowPosBiLSTM import TensorflowPosBiLSTM

TRAIN_PATH = 'Penn_Treebank/train.gold.conll'
TEST_PATH = 'Penn_Treebank/dev.gold.conll'

CURR_DIR = os.getcwd()
VOCAB_PATH = os.path.join(CURR_DIR, "words_tags_dict.pickle")

if __name__ == "__main__":
    # data_processor = DataProcessor(rare_word_threshold=5)
    # data_processor.initiate_word_tags_dicts(TRAIN_PATH)

    data_processor = DataProcessor(from_file=True, save_load_path=VOCAB_PATH)

    x_train, y_train = data_processor.preprocess_sample_set(TRAIN_PATH)
    x_test, y_test = data_processor.preprocess_sample_set(TEST_PATH)

    # create boolean mask vectors
    mask_train = data_processor.create_boolean_mask(x_train)
    mask_test = data_processor.create_boolean_mask(x_test)

    vocab_size = data_processor.vocab_size
    n_classes = data_processor.n_classes
    max_input_length = data_processor.max_seq_len

    # build model and fit using training set
    # save_path = os.path.join(CURR_DIR, "LSTM", "07-19-18-15:31:48/")
    model = TensorflowPosBiLSTM(vocab_size, n_classes, max_input_length)
    # model = TensorflowPosBiLSTM(vocab_size, n_classes, max_input_length, saver_path=save_path)
    model.fit(x_train, y_train, mask_train)

    # predict over test set
    train_accuracy = model.evaluate_sample(x_train, y_train, mask_train)
    print("train accuracy is: {0}".format(train_accuracy))
    test_accuracy = model.evaluate_sample(x_test, y_test, mask_test)
    print("test accuracy is: {0}".format(test_accuracy))

