import os
from POSTaggers import KerasPOSTagger
from DataProcessorInterface import DataProcessorInterface
from config import *
from KerasCallbacks import CloudCallback


def datasets_paths(language):
    language_train_path = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', language, 'train.conllu')
    language_test_path = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', language, 'test.conllu')
    return os.path.abspath(language_train_path), os.path.abspath(language_test_path)


def pickle_path(name):
    path = os.path.join(os.path.dirname(__file__), os.pardir, 'pickles', name + '.pickle')
    return os.path.abspath(path)


def model_path(subpath):
    path = os.path.join(os.path.dirname(__file__), os.pardir, 'models', subpath)
    return os.path.abspath(path)


def run(processor, tagger, train_path, test_path, load_processor_from=None, load_tagger_from=None):
    cb = CloudCallback(remote=False, slack_url=slack_url, stop_url=stop_url)
    try:
        # Initiate processor
        if load_processor_from:
            processor.load(load_processor_from)
        else:
            processor.process(train_path)
            processor.save()

        word_dict = processor.get_word2idx_dict()
        tag_dict = processor.get_tag2idx_dict()
        feature_dicts = processor.get_features2idx_dicts().values()

        # Process training set
        x_train, y_train, y_train_features = processor.preprocess_sample(train_path)
        y_train_features = y_train_features.values()
        x_train = processor.transform_to_one_hot(x_train, len(word_dict))
        y_train = processor.transform_to_one_hot(y_train, len(tag_dict))
        y_train_features = [processor.transform_to_one_hot(y_train_feature, len(feature_dict)) for
                            y_train_feature, feature_dict in zip(y_train_features, feature_dicts)]

        # Process test set
        x_test, y_test, y_test_features = processor.preprocess_sample(test_path)
        y_test_features = y_test_features.values()
        x_test = processor.transform_to_one_hot(x_test, len(word_dict))
        y_test = processor.transform_to_one_hot(y_test, len(tag_dict))
        y_test_features = [processor.transform_to_one_hot(y_test_feature, len(feature_dict)) for
                           y_test_feature, feature_dict in zip(y_test_features, feature_dicts)]

        # Fit model
        if load_tagger_from:
            tagger.load_model_params(load_tagger_from)
        else:
            tagger.fit(x_train, y_train, callbacks=[cb])

        # Evaluate results
        cb.send_update('Evaluation has just started.')
        score = tagger.evaluate_sample(x_test, y_test)
        loss, acc = score[0], score[1]
        cb.send_update('*Regular evaluation has ended!* Loss: `{0}` - Accuracy: `{1}`'.format(loss, acc))

        # Evaluate unseen results
        unseen_acc = tagger.evaluate_sample_conditioned(x_test, y_test, 'unseen')
        cb.send_update('*Unseen Evaluation has ended!* Accuracy: `{0}`'.format(unseen_acc))

    except Exception as e:
        cb.send_update(repr(e))
    finally:
        cb.stop_instance()


if __name__ == "__main__":
    processor = DataProcessorInterface(name='english_test')
    tagger = KerasPOSTagger(processor, n_epochs=1)
    train_path, test_path = datasets_paths('hebrew')

    run(processor, tagger, train_path, test_path,
        load_processor_from=pickle_path('english_test'),
        load_tagger_from=model_path('my_cool_model/08-08-2018_17:51:44/model.01.hdf5'))
