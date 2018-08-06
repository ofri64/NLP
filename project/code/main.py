import os
from DataProcessors import HebrewBinyanDataProcessor, HebrewDataProcessor
from POSTaggers import MTLHebrewBinyanTagger, KerasPOSTagger


def dataset(subpath):
    return os.path.dirname(__file__) + '/../datasets/' + subpath


def pickle(name):
    return os.path.dirname(__file__) + '/../pickles/' + name + '.pickle'


TRAIN_PATH = '../datasets/english/train.gold.conll'
TEST_PATH = '../datasets/english/dev.gold.conll'

# HEBREW_TRAIN_PATH = '../datasets/hebrew/he_htb-ud-train.conllu'
# HEBREW_TEST_PATH = '../datasets/hebrew/he_htb-ud-test.conllu'
# HEBREW_DEV_PATH = '../datasets/hebrew/he_htb-ud-dev.conllu'

HEBREW_TRAIN_PATH = dataset('hebrew/he_htb-ud-train.conllu')
HEBREW_TEST_PATH = dataset('hebrew/he_htb-ud-test.conllu')
HEBREW_DEV_PATH = dataset('hebrew/he_htb-ud-dev.conllu')

CURR_DIR = os.getcwd()
VOCAB_PATH = os.path.join(CURR_DIR, "words_tags_dict.pickle")


def train(processor, dataset):
    pass


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

    # english_processor = EnglishDataProcessor()
    # word_dict, tag_dict = english_processor.create_word_tags_dicts(TRAIN_PATH)
    # x_train, y_train = english_processor.preprocess_sample(TRAIN_PATH)
    # x_train, y_train = english_processor.transform_to_one_hot(x_train, len(word_dict)), \
    #                    english_processor.transform_to_one_hot(y_train, len(tag_dict))
    #
    # x_test, y_test = english_processor.preprocess_sample(TEST_PATH)
    # x_test, y_test = english_processor.transform_to_one_hot(x_test, len(word_dict)), \
    #                    english_processor.transform_to_one_hot(y_test, len(tag_dict))
    #
    # english_processor.save_word_tags_dict('english_dicts2.pkl')
    #
    # print(english_processor.get_tag2idx_vocab())
    # print(len(english_processor.get_word2idx_vocab()))
    # print(x_test[0])
    # print('------')
    #
    # tagger = KerasPOSTagger(english_processor, n_epochs=1)
    # tagger.fit(x_train, y_train)
    # score = tagger.evaluate_sample_conditioned(x_test, y_test, 'unseen')
    #
    # print('UNSEEN SCORE: ', score)

    # hebrew_processor = HebrewBinyanDataProcessor()
    # word_dict, tag_dict, binyan_dict = hebrew_processor.create_word_tag_binyan_dicts(HEBREW_TRAIN_PATH)
    # hebrew_processor.save('hebrew_binyan_processor.pickle')
    #
    # x_train, y_pos, y_binyan = hebrew_processor.preprocess_sample(HEBREW_TRAIN_PATH)
    # x_train = hebrew_processor.transform_to_one_hot(x_train, len(word_dict))
    # y_pos = hebrew_processor.transform_to_one_hot(y_pos, len(tag_dict))
    # y_binyan = hebrew_processor.transform_to_one_hot(y_binyan, len(binyan_dict))
    #
    # model = MTLHebrewBinyanTagger(hebrew_processor, n_epochs=10)
    # model.fit(x_train, [y_pos, y_binyan], 'hebrew_binyan')


    # HEBREW REGULAR
    # Dev set - 86.70%
    # Test set - 87.98%

    hebrew_processor = HebrewDataProcessor()
    hebrew_processor.create_word_tags_dicts(HEBREW_TRAIN_PATH)
    hebrew_processor.save(pickle('hebrew_processor'))
    # hebrew_processor.load('hebrew_processor.pickle')
    word_dict = hebrew_processor.get_word2idx_vocab()
    tag_dict = hebrew_processor.get_tag2idx_vocab()

    x_train, y_train = hebrew_processor.preprocess_sample(HEBREW_TRAIN_PATH)
    x_train = hebrew_processor.transform_to_one_hot(x_train, len(word_dict))
    y_train = hebrew_processor.transform_to_one_hot(y_train, len(tag_dict))

    # x_dev, y_dev = hebrew_processor.preprocess_sample(HEBREW_DEV_PATH)
    # x_dev = hebrew_processor.transform_to_one_hot(x_dev, len(word_dict))
    # y_dev = hebrew_processor.transform_to_one_hot(y_dev, len(tag_dict))

    x_test, y_test = hebrew_processor.preprocess_sample(HEBREW_TEST_PATH)
    x_test = hebrew_processor.transform_to_one_hot(x_test, len(word_dict))
    y_test = hebrew_processor.transform_to_one_hot(y_test, len(tag_dict))

    model = KerasPOSTagger(hebrew_processor, n_epochs=10)
    model.fit(x_train, y_train, 'hebrew')
    # model.load_model_params('hebrew-2018-08-04 16:40:20.h5')

    print(model.evaluate_sample(x_test, y_test))
    print(model.model.metrics_names)

    print(model.evaluate_sample_conditioned(x_test, y_test, 'unseen'))


    # HEBREW BINYAN
    # Dev set - 86.35%
    # Test set 86.358%

    # hebrew_processor = HebrewBinyanDataProcessor()
    # hebrew_processor.load('../pickles/hebrew_binyan_processor.pickle')
    # word_dict = hebrew_processor.get_word2idx_vocab()
    # tag_dict = hebrew_processor.get_tag2idx_vocab()
    # binyan_dict = hebrew_processor.get_binyan2idx_vocab()
    #
    # x_dev, y_dev_pos, y_dev_binyan = hebrew_processor.preprocess_sample(HEBREW_TEST_PATH)
    # x_dev = hebrew_processor.transform_to_one_hot(x_dev, len(word_dict))
    # y_dev_pos = hebrew_processor.transform_to_one_hot(y_dev_pos, len(tag_dict))
    # y_dev_binyan = hebrew_processor.transform_to_one_hot(y_dev_binyan, len(binyan_dict))
    #
    # # x_test, y_pos, y_binyan = hebrew_processor.preprocess_sample(HEBREW_TEST_PATH)
    # # x_test = hebrew_processor.transform_to_one_hot(x_test, len(word_dict))
    # # y_pos = hebrew_processor.transform_to_one_hot(y_pos, len(tag_dict))
    # # y_binyan = hebrew_processor.transform_to_one_hot(y_binyan, len(binyan_dict))
    #
    # model = MTLHebrewBinyanTagger(hebrew_processor, immediate_build=False)
    # model.load_model_params('../datasets/hebrew_binyan-2018-08-04 15:46:00.h5')
    #
    # print(model.evaluate_sample(x_dev, [y_dev_pos, y_dev_binyan]))
    # print(model.model.metrics_names)
    #
    # print(model.evaluate_sample_conditioned(x_dev, y_dev_pos, 'unseen'))



