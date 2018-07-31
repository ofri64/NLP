import time
import os
import numpy as np
from DataProcessor import DataProcessor
from TensorflowPosBiLSTM import TensorflowPosBiLSTM
from DataProcessors import EnglishDataProcessor
from POSTaggers import KerasPOSTagger

TRAIN_PATH = 'datasets/english/train.gold.conll'
TEST_PATH = 'datasets/english/dev.gold.conll'

CURR_DIR = os.getcwd()
VOCAB_PATH = os.path.join(CURR_DIR, "words_tags_dict.pickle")

if __name__ == "__main__":
    # # data_processor = DataProcessor(rare_word_threshold=5)
    # # data_processor.initiate_word_tags_dicts(TRAIN_PATH)
    #
    # data_processor = DataProcessor(from_file=True, save_load_path=VOCAB_PATH)
    #
    # x_train, y_train = data_processor.preprocess_sample_set(TRAIN_PATH)
    # x_test, y_test = data_processor.preprocess_sample_set(TEST_PATH)
    #
    # # create boolean mask vectors
    # mask_train = data_processor.create_boolean_mask(x_train)
    # mask_test = data_processor.create_boolean_mask(x_test)
    #
    # vocab_size = data_processor.vocab_size
    # n_classes = data_processor.n_classes
    # max_input_length = data_processor.max_seq_len
    #
    # # build model and fit using training set
    # # save_path = os.path.join(CURR_DIR, "LSTM", "07-19-18-15:31:48/")
    # model = TensorflowPosBiLSTM(vocab_size, n_classes, max_input_length)
    # # model = TensorflowPosBiLSTM(vocab_size, n_classes, max_input_length, saver_path=save_path)
    # model.fit(x_train, y_train, mask_train)
    #
    # # predict over test set
    # train_accuracy = model.evaluate_sample(x_train, y_train, mask_train)
    # print("train accuracy is: {0}".format(train_accuracy))
    # test_accuracy = model.evaluate_sample(x_test, y_test, mask_test)
    # print("test accuracy is: {0}".format(test_accuracy))

    english_processor = EnglishDataProcessor()
    word_dict, tag_dict = english_processor.create_word_tags_dicts(TRAIN_PATH)
    x_train, y_train = english_processor.preprocess_sample(TRAIN_PATH)
    x_train, y_train = english_processor.transform_to_one_hot(x_train, len(word_dict)), \
                       english_processor.transform_to_one_hot(y_train, len(tag_dict))

    x_test, y_test = english_processor.preprocess_sample(TEST_PATH)
    x_test, y_test = english_processor.transform_to_one_hot(x_test, len(word_dict)), \
                       english_processor.transform_to_one_hot(y_test, len(tag_dict))

    english_processor.save_word_tags_dict('english_dicts2.pkl')

    print(english_processor.get_tag2idx_vocab())
    print(len(english_processor.get_word2idx_vocab()))
    print(x_test[0])
    print('------')

    tagger = KerasPOSTagger(english_processor, n_epochs=1)
    tagger.fit(x_train, y_train)
    score = tagger.evaluate_sample_conditioned(x_test, y_test, 'unseen')

    print('UNSEEN SCORE: ', score)

