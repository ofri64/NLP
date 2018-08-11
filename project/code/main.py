import os
import argparse

from config import *

from DataProcessor import DataProcessor
from POSTaggers import SimpleTagger, MTLOneFeatureTagger
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


def run_experiment(processor, tagger, train_path, test_path, load_processor_from=None, load_tagger_from=None,
                   features=None, remote=False):
    cb = CloudCallback(remote=remote, slack_url=slack_url, stop_url=stop_url)

    if type(features) is str:
        features = [features]
    elif not features:
        features = []

    try:
        # Initiate processor
        if load_processor_from:
            processor.load(load_processor_from)
        else:
            processor.process(train_path)
            processor.save()

        word_dict = processor.get_word2idx_dict()
        tag_dict = processor.get_tag2idx_dict()

        feature_dicts = {k: v for k, v in processor.get_features2idx_dicts().items() if k in features}
        feature_dicts = feature_dicts.values()

        # Process training set
        x_train, y_train, y_train_features = processor.preprocess_sample(train_path)

        y_train_features = {k: v for k, v in y_train_features.items() if k in features}
        y_train_features = y_train_features.values()

        x_train = processor.transform_to_one_hot(x_train, len(word_dict))
        y_train = processor.transform_to_one_hot(y_train, len(tag_dict))
        y_train_features = [processor.transform_to_one_hot(y_train_feature, len(feature_dict)) for
                            y_train_feature, feature_dict in zip(y_train_features, feature_dicts)]

        # Process test set
        x_test, y_test, y_test_features = processor.preprocess_sample(test_path)

        y_test_features = {k: v for k, v in y_test_features.items() if k in features}
        y_test_features = y_test_features.values()

        x_test = processor.transform_to_one_hot(x_test, len(word_dict))
        y_test = processor.transform_to_one_hot(y_test, len(tag_dict))
        y_test_features = [processor.transform_to_one_hot(y_test_feature, len(feature_dict)) for
                           y_test_feature, feature_dict in zip(y_test_features, feature_dicts)]

        # Fit model
        if load_tagger_from:
            tagger.load_model_params(load_tagger_from)
        else:
            tagger.fit(x_train, [y_train] + y_train_features, callbacks=[cb])

        # Evaluate results
        cb.send_update('Evaluation has just started.')
        metrics_output = tagger.evaluate_sample(x_test, [y_test] + y_test_features)
        print_str = ""
        for metric, value in metrics_output:
            if "acc" in metric:
                print_str += "{0}: {1} ".format(metric, value)
        cb.send_update('*Regular evaluation has ended!*`' + print_str)

        # Evaluate unseen results
        unseen_acc = tagger.evaluate_sample_conditioned(x_test, [y_test] + y_test_features, 'unseen')
        cb.send_update('*Unseen Evaluation has ended!* Accuracy: `{0}`'.format(unseen_acc))

    except Exception as e:
        cb.send_update(repr(e))

        import traceback
        traceback.print_exc()
    finally:
        cb.stop_instance()


def main(language, feature, n_epochs, remote, features):
    train_path, test_path = datasets_paths(language)

    if features:
        import pandas as pd
        dp = DataProcessor()
        dp.process(train_path)
        available_features = dp.get_features()
        print('\r\nAvailable features')
        print('------------------')
        print(pd.DataFrame(available_features))
        print('\r\n')
        return

    if feature:
        processor = DataProcessor(name='{0}_{1}'.format(language, feature))
        tagger = MTLOneFeatureTagger(processor, n_epochs=n_epochs, feature=feature)
    else:
        processor = DataProcessor(name=language)
        tagger = SimpleTagger(processor, n_epochs=n_epochs)

    run_experiment(processor, tagger, train_path, test_path, remote=remote)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('language', help='language to experiment on')
    parser.add_argument('-f', '--feature', help='feature to experiment on')
    parser.add_argument('-e', '--n_epochs', help='number of epochs to run', type=int, default=10)
    parser.add_argument('-r', '--remote', help='run remotely on the configured gcloud vm', action='store_true')
    parser.add_argument('-a', '--features', help='get all features of the given language', action='store_true')

    args = parser.parse_args()

    main(args.language, args.feature, args.n_epochs, args.remote, args.features)
