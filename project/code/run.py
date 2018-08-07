import os
from DataProcessors import SpanishDataProcessor
from POSTaggers import KerasPOSTagger

TRAIN_PATH = '../datasets/spanish/es_ancora-ud-train.conllu'
TEST_PATH = '../datasets/english/es_ancora-ud-test.conllu'


def dataset(subpath):
    path = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', subpath)
    return os.path.abspath(path)


def pickle(name):
    path = os.path.join(os.path.dirname(__file__), os.pardir, 'pickles', name + '.pickle')
    return os.path.abspath(path)


def train(data_processor, model, train_path):
    # Create dictionaries
    data_processor.create_word_tags_dicts(train_path)
    data_processor.save(pickle('spanish_processor'))
    word_dict = data_processor.get_word2idx_vocab()
    tag_dict = data_processor.get_tag2idx_vocab()

    # Create training set
    x_train, y_train = data_processor.preprocess_sample(train_path)
    x_train = data_processor.transform_to_one_hot(x_train, len(word_dict))
    y_train = data_processor.transform_to_one_hot(y_train, len(tag_dict))

    # Fit model
    model.fit(x_train, y_train)


def evaluate(data_processor, model, test_path, unseen=True):
    # Create testing set
    x_test, y_test = data_processor.preprocess_sample(test_path)
    word_dict = data_processor.get_word2idx_vocab()
    tag_dict = data_processor.get_tag2idx_vocab()
    x_test = data_processor.transform_to_one_hot(x_test, len(word_dict))
    y_test = data_processor.transform_to_one_hot(y_test, len(tag_dict))

    # Build model with the ready processor
    model.build()

    # Evaluate unseen / normal
    if unseen:
        return model.evaluate_sample_conditioned(x_test, y_test, 'unseen')
    else:
        return model.evaluate_sample(x_test, y_test)


if __name__ == "__main__":
    processor = SpanishDataProcessor()
    tagger = KerasPOSTagger(processor, immediate_build=False)
    train(processor, tagger, TRAIN_PATH)

    results = evaluate(processor, tagger, TEST_PATH)
    print(results)
